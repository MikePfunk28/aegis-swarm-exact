import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass

from mdap_small.adapter import OllamaAdapter
from mdap_small.models import ModelSpec
from mdap_small.red_flags import ParsedStep, has_red_flags, parse_step
from mdap_small.toolbelt import LocalToolbelt


@dataclass
class VoteResult:
    winner: ParsedStep
    rounds: int
    valid_votes: int
    invalid_votes: int
    counts: Counter


class FirstAheadByKVoter:
    def __init__(
        self,
        adapter: OllamaAdapter,
        toolbelt: LocalToolbelt | None = None,
    ):
        self.adapter = adapter
        self.toolbelt = toolbelt

    async def vote(
        self,
        prompt: str,
        models: list[ModelSpec],
        ahead_k: int,
        parallel_votes: int,
        tools_enabled: bool = False,
        tool_budget_per_vote: int = 0,
        parser_mode: str = "red_flagging",
        red_flag_token_cutoff: int = 750,
    ) -> VoteResult:
        if not models:
            raise ValueError("no models configured")

        counts: Counter = Counter()
        rounds = 0
        valid_votes = 0
        invalid_votes = 0
        idx = 0

        while True:
            rounds += 1
            batch_size = parallel_votes if rounds == 1 else 1
            tasks = []
            for _ in range(batch_size):
                model = models[idx % len(models)]
                temp = (
                    model.temperature_first
                    if rounds == 1
                    else model.temperature_followup
                )
                tasks.append(
                    self._generate_with_optional_tools(
                        model,
                        prompt,
                        temp,
                        tools_enabled=tools_enabled,
                        tool_budget=tool_budget_per_vote,
                    )
                )
                idx += 1

            outputs = await asyncio.gather(*tasks, return_exceptions=True)
            for out in outputs:
                if isinstance(out, Exception):
                    invalid_votes += 1
                    continue
                if has_red_flags(
                    out,
                    max_tokens_approx=red_flag_token_cutoff,
                    parser_mode=parser_mode,
                ):
                    invalid_votes += 1
                    continue
                step = parse_step(out, parser_mode=parser_mode)
                if not step:
                    invalid_votes += 1
                    continue
                key = (
                    step.move.disk,
                    step.move.source,
                    step.move.target,
                    step.next_state,
                )
                counts[key] += 1
                valid_votes += 1

            if not counts:
                continue
            winner, winner_count = counts.most_common(1)[0]
            second = 0
            for k, v in counts.items():
                if k != winner and v > second:
                    second = v
            if winner_count >= second + ahead_k:
                return VoteResult(
                    winner=ParsedStep(
                        move={
                            "disk": winner[0],
                            "source": winner[1],
                            "target": winner[2],
                        },
                        next_state=winner[3],
                    ),
                    rounds=rounds,
                    valid_votes=valid_votes,
                    invalid_votes=invalid_votes,
                    counts=counts,
                )

    async def _generate_with_optional_tools(
        self,
        model: ModelSpec,
        prompt: str,
        temperature: float,
        tools_enabled: bool,
        tool_budget: int,
    ) -> str:
        output = await self.adapter.generate(model, prompt, temperature)
        if not tools_enabled or tool_budget <= 0 or self.toolbelt is None:
            return output

        for _ in range(tool_budget):
            req = self._parse_tool_request(output)
            if not req:
                return output
            shell, command = req
            result = await self.toolbelt.run(shell=shell, command=command)
            prompt = (
                f"{prompt}\n"
                "Tool execution result:\n"
                f"shell={shell}; command={command}; ok={result.ok}\n"
                f"output={result.output}\n"
                "Now return only the final step result in required format."
            )
            output = await self.adapter.generate(model, prompt, temperature)
        return output

    def _parse_tool_request(self, text: str) -> tuple[str, str] | None:
        match = re.search(r"(?im)^\s*tool\s*=\s*(bash|powershell)\s*:\s*(.+)$", text)
        if match:
            return (match.group(1).lower(), match.group(2).strip())

        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        tool = payload.get("tool")
        command = payload.get("command")
        if tool in ("bash", "powershell") and isinstance(command, str):
            return (tool, command)
        return None
