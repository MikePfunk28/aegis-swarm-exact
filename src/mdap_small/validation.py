from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from mdap_small.adapter import OllamaAdapter
from mdap_small.hanoi import HanoiState, generate_moves
from mdap_small.maths import k_min
from mdap_small.models import ModelSpec, RuntimeSpec
from mdap_small.prompts import build_step_prompt
from mdap_small.red_flags import parse_step


@dataclass
class CalibrationResult:
    model_id: str
    disks: int
    samples: int
    parser_mode: str
    max_tokens: int
    p_step: float
    valid_rate: float
    both_wrong_collisions: int
    k_min_for_target: int


@dataclass
class CollisionResult:
    model_id: str
    parser_mode: str
    token_cutoff: int
    samples: int
    both_wrong_collisions: int


def paper_strict_profile(
    runtime: RuntimeSpec, k_value: int | None = None
) -> RuntimeSpec:
    updated = runtime.model_copy(deep=True)
    updated.parser_mode = "red_flagging"
    updated.red_flag_token_cutoff = 750
    updated.tools_enabled = False
    updated.parallel_votes = max(3, updated.parallel_votes)
    updated.ahead_k = k_value or updated.ahead_k
    for m in updated.models:
        m.temperature_first = 0.0
        m.temperature_followup = 0.1
        m.max_tokens = 750
    return updated


def paper_calibration_profile(runtime: RuntimeSpec) -> RuntimeSpec:
    updated = runtime.model_copy(deep=True)
    updated.parser_mode = "repairing"
    updated.red_flag_token_cutoff = 2048
    for m in updated.models:
        m.max_tokens = 2048
    return updated


async def estimate_single_step_success(
    runtime: RuntimeSpec,
    disks: int,
    samples: int,
    target_success: float,
    model_name: str | None = None,
    temperature: float = 0.1,
    concurrency: int = 32,
    collision_samples: int | None = None,
    request_timeout_s: float = 45.0,
) -> CalibrationResult:
    runtime = paper_calibration_profile(runtime)
    model = _select_model(runtime, model_name)
    adapter = OllamaAdapter()

    trace = _build_trace(disks)
    picks = [random.randrange(0, len(trace)) for _ in range(samples)]
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one_eval(idx: int) -> tuple[bool, bool]:
        step_idx, state, prev_move, expected_move, expected_state = trace[idx]
        prompt = build_step_prompt(
            runtime=runtime,
            disks=disks,
            step_idx=step_idx,
            state=state,
            models=[model],
            prev_move=prev_move,
        )
        async with sem:
            try:
                output = await asyncio.wait_for(
                    adapter.generate(model, prompt, temperature),
                    timeout=request_timeout_s,
                )
            except Exception:
                return (False, False)
        parsed = parse_step(output, parser_mode="repairing")
        if not parsed:
            return (False, False)
        ok = (
            parsed.move.as_tuple() == expected_move
            and parsed.next_state == expected_state
        )
        return (True, ok)

    async def one_collision(idx: int) -> bool:
        step_idx, state, prev_move, expected_move, expected_state = trace[idx]
        prompt = build_step_prompt(
            runtime=runtime,
            disks=disks,
            step_idx=step_idx,
            state=state,
            models=[model],
            prev_move=prev_move,
        )
        async with sem:
            try:
                a, b = await asyncio.gather(
                    asyncio.wait_for(
                        adapter.generate(model, prompt, temperature),
                        timeout=request_timeout_s,
                    ),
                    asyncio.wait_for(
                        adapter.generate(model, prompt, temperature),
                        timeout=request_timeout_s,
                    ),
                )
            except Exception:
                return False

        pa = parse_step(a, parser_mode="repairing")
        pb = parse_step(b, parser_mode="repairing")

        ok_a = bool(
            pa
            and pa.move.as_tuple() == expected_move
            and pa.next_state == expected_state
        )
        ok_b = bool(
            pb
            and pb.move.as_tuple() == expected_move
            and pb.next_state == expected_state
        )
        return (not ok_a) and (not ok_b)

    eval_results = await asyncio.gather(*(one_eval(i) for i in picks))
    valid = sum(1 for v, _ in eval_results if v)
    correct = sum(1 for _, ok in eval_results if ok)
    p_step = 0.0 if samples == 0 else correct / samples
    valid_rate = 0.0 if samples == 0 else valid / samples

    collision_count = collision_samples if collision_samples is not None else samples
    collision_picks = [
        random.randrange(0, len(trace)) for _ in range(max(0, collision_count))
    ]
    collision_flags = await asyncio.gather(*(one_collision(i) for i in collision_picks))
    both_wrong_collisions = sum(1 for c in collision_flags if c)

    if p_step <= 0.5:
        km = -1
    else:
        km = k_min(
            target_success=target_success,
            steps=(1 << disks) - 1,
            p_step=p_step,
            m=1,
        )
    return CalibrationResult(
        model_id=model.model_id,
        disks=disks,
        samples=samples,
        parser_mode="repairing",
        max_tokens=2048,
        p_step=p_step,
        valid_rate=valid_rate,
        both_wrong_collisions=both_wrong_collisions,
        k_min_for_target=km,
    )


async def measure_two_run_collisions(
    runtime: RuntimeSpec,
    disks: int,
    samples: int,
    model_name: str,
    parser_mode: str,
    token_cutoff: int,
    temperature_a: float = 0.1,
    temperature_b: float = 0.1,
    concurrency: int = 16,
    request_timeout_s: float = 45.0,
) -> CollisionResult:
    runtime = runtime.model_copy(deep=True)
    runtime.parser_mode = parser_mode
    runtime.red_flag_token_cutoff = token_cutoff
    for m in runtime.models:
        m.max_tokens = token_cutoff

    model = _select_model(runtime, model_name)
    adapter = OllamaAdapter()
    trace = _build_trace(disks)
    picks = [random.randrange(0, len(trace)) for _ in range(samples)]
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(idx: int) -> bool:
        step_idx, state, prev_move, expected_move, expected_state = trace[idx]
        prompt = build_step_prompt(
            runtime=runtime,
            disks=disks,
            step_idx=step_idx,
            state=state,
            models=[model],
            prev_move=prev_move,
        )
        async with sem:
            try:
                out_a, out_b = await asyncio.gather(
                    asyncio.wait_for(
                        adapter.generate(model, prompt, temperature_a),
                        timeout=request_timeout_s,
                    ),
                    asyncio.wait_for(
                        adapter.generate(model, prompt, temperature_b),
                        timeout=request_timeout_s,
                    ),
                )
            except Exception:
                return False

        pa = parse_step(out_a, parser_mode=parser_mode)
        pb = parse_step(out_b, parser_mode=parser_mode)
        ok_a = bool(
            pa
            and pa.move.as_tuple() == expected_move
            and pa.next_state == expected_state
        )
        ok_b = bool(
            pb
            and pb.move.as_tuple() == expected_move
            and pb.next_state == expected_state
        )
        return (not ok_a) and (not ok_b)

    flags = await asyncio.gather(*(one(i) for i in picks))
    collisions = sum(1 for f in flags if f)
    return CollisionResult(
        model_id=model.model_id,
        parser_mode=parser_mode,
        token_cutoff=token_cutoff,
        samples=samples,
        both_wrong_collisions=collisions,
    )


def _select_model(runtime: RuntimeSpec, model_name: str | None) -> ModelSpec:
    active = runtime.active_models()
    if not active:
        raise ValueError("no active models in runtime")
    if model_name:
        for model in active:
            if model.name == model_name or model.model_id == model_name:
                return model
        return ModelSpec(
            name=model_name.replace(":", "_"), model_id=model_name, role="worker"
        )
    leader = runtime.leader_model()
    return leader or active[0]


def _build_trace(
    disks: int,
) -> list[
    tuple[
        int,
        HanoiState,
        tuple[int, int, int] | None,
        tuple[int, int, int],
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    ]
]:
    rows = []
    state = HanoiState.initial(disks)
    prev_move: tuple[int, int, int] | None = None
    for step_idx, move in enumerate(generate_moves(disks), start=1):
        next_state = state.apply(move)
        rows.append((step_idx, state, prev_move, move, next_state.pegs))
        state = next_state
        prev_move = move
    return rows
