from dataclasses import dataclass

from mdap_small.adapter import OllamaAdapter
from mdap_small.graph_memory import GraphMemory
from mdap_small.hanoi import HanoiState, generate_moves
from mdap_small.models import ModelSpec, RuntimeSpec
from mdap_small.prompts import build_step_prompt
from mdap_small.routing import ExpertRouter
from mdap_small.toolbelt import LocalToolbelt
from mdap_small.voting import FirstAheadByKVoter


@dataclass
class RunStats:
    steps: int = 0
    correct_steps: int = 0
    illegal_steps: int = 0
    total_rounds: int = 0
    valid_votes: int = 0
    invalid_votes: int = 0

    @property
    def accuracy(self) -> float:
        return 0.0 if self.steps == 0 else self.correct_steps / self.steps


class MakerOrchestrator:
    def __init__(self, runtime: RuntimeSpec, adapter: OllamaAdapter | None = None):
        self.runtime = runtime
        self.adapter = adapter or OllamaAdapter()
        self.router = ExpertRouter()
        self.toolbelt = LocalToolbelt(self.runtime.allowed_shell_commands)
        self.voter = FirstAheadByKVoter(self.adapter, toolbelt=self.toolbelt)
        self.graph_memory = GraphMemory() if self.runtime.graph_memory_enabled else None

    async def solve(
        self, disks: int, max_steps: int | None = None, verbose: bool = False
    ) -> RunStats:
        models = self._build_agent_pool()
        stats = RunStats()
        state = HanoiState.initial(disks)
        prev_move: tuple[int, int, int] | None = None

        for step_idx, expected in enumerate(generate_moves(disks), start=1):
            if max_steps and step_idx > max_steps:
                break

            step_models = self.router.build_pool(self.runtime, models, step_idx)
            prompt = self._build_prompt(disks, step_idx, state, step_models, prev_move)
            vote = await self.voter.vote(
                prompt=prompt,
                models=step_models,
                ahead_k=self.runtime.ahead_k,
                parallel_votes=self.runtime.parallel_votes,
                tools_enabled=self.runtime.tools_enabled,
                tool_budget_per_vote=self.runtime.tool_budget_per_vote,
                parser_mode=self.runtime.parser_mode,
                red_flag_token_cutoff=self.runtime.red_flag_token_cutoff,
            )
            pred = vote.winner.move.as_tuple()

            stats.steps += 1
            stats.total_rounds += vote.rounds
            stats.valid_votes += vote.valid_votes
            stats.invalid_votes += vote.invalid_votes
            if pred == expected:
                stats.correct_steps += 1

            try:
                next_by_rule = state.apply(pred)
                if vote.winner.next_state != next_by_rule.pegs:
                    stats.illegal_steps += 1
                    state = next_by_rule
                else:
                    state = HanoiState(vote.winner.next_state)
                prev_move = pred
            except Exception:
                stats.illegal_steps += 1

            if self.graph_memory:
                self.graph_memory.record(
                    event_type="vote_step",
                    payload={
                        "step": step_idx,
                        "winner_move": list(pred),
                        "winner_state": vote.winner.next_state,
                        "rounds": vote.rounds,
                        "valid_votes": vote.valid_votes,
                        "invalid_votes": vote.invalid_votes,
                        "counts": {str(k): v for k, v in vote.counts.items()},
                        "correct": pred == expected,
                    },
                )

            if verbose and step_idx % 100 == 0:
                print(
                    f"step={step_idx} acc={stats.accuracy:.4f} "
                    f"invalid={stats.invalid_votes} rounds={stats.total_rounds}"
                )

        return stats

    def _build_prompt(
        self,
        disks: int,
        step_idx: int,
        state: HanoiState,
        models: list[ModelSpec],
        prev_move: tuple[int, int, int] | None,
    ) -> str:
        return build_step_prompt(
            runtime=self.runtime,
            disks=disks,
            step_idx=step_idx,
            state=state,
            models=models,
            prev_move=prev_move,
        )

    def _build_agent_pool(self) -> list[ModelSpec]:
        active = self.runtime.active_models()
        if not active:
            return []

        leader = self.runtime.leader_model()
        workers = self.runtime.worker_models()

        if self.runtime.workflow_mode == "sequential":
            return [leader] if leader else [active[0]]

        if self.runtime.workflow_mode == "graph":
            target = max(self.runtime.min_agents, min(self.runtime.max_agents, 10))
            return self._expand_pool(leader, workers, target)

        if self.runtime.workflow_mode == "hivemind":
            target = max(self.runtime.min_agents, min(self.runtime.max_agents, 15))
            return self._expand_pool(leader, workers, target)

        if not self.runtime.auto_scale_agents:
            return active[: self.runtime.max_agents]

        target = max(
            self.runtime.min_agents, min(self.runtime.max_agents, len(workers) + 1)
        )
        return self._expand_pool(leader, workers, target)

    def _expand_pool(
        self,
        leader: ModelSpec | None,
        workers: list[ModelSpec],
        target: int,
    ) -> list[ModelSpec]:
        pool: list[ModelSpec] = []
        if leader:
            pool.append(leader)

        idx = 0
        while len(pool) < target:
            pool.append(workers[idx % len(workers)])
            idx += 1
        return pool
