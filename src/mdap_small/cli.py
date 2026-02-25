import asyncio
import json
from pathlib import Path

import typer
import yaml

from mdap_small.adapter import OllamaAdapter
from mdap_small.integrations import detect_strandsagents, detect_voiceai
from mdap_small.maths import k_min
from mdap_small.models import DEFAULT_MODELS, ModelSpec, RuntimeSpec
from mdap_small.orchestrator import MakerOrchestrator
from mdap_small.strands_bridge import LocalModelSelector
from mdap_small.validation import (
    estimate_single_step_success,
    measure_two_run_collisions,
    paper_strict_profile,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_runtime(config: Path | None) -> RuntimeSpec:
    if config is None:
        return DEFAULT_MODELS
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    return RuntimeSpec(**raw)


async def _probe_model_feasibility(
    model_id: str,
    timeout_s: float,
) -> tuple[bool, str]:
    adapter = OllamaAdapter()
    probe = ModelSpec(name=model_id.replace(":", "_"), model_id=model_id, role="worker")
    prompt = "Reply with exactly: ok"
    try:
        out = await asyncio.wait_for(
            adapter.generate(probe, prompt, 0.0), timeout=timeout_s
        )
    except Exception as exc:
        msg = str(exc).strip() or exc.__class__.__name__
        if "requires more system memory" in msg.lower():
            return (False, f"insufficient_memory: {msg}")
        return (False, f"generate_failed: {msg}")

    if not out.strip():
        return (False, "empty_response")
    return (True, "ok")


@app.command()
def benchmark(
    disks: int = typer.Option(10, min=3),
    samples: int = typer.Option(500, min=10),
    ahead_k: int = typer.Option(3, min=1),
    parallel_votes: int = typer.Option(3, min=1),
    workflow_mode: str = typer.Option("swarm"),
    max_agents: int = typer.Option(15, min=3, max=40),
    moe_enabled: bool = typer.Option(True),
    tools_enabled: bool = typer.Option(False),
    tool_budget_per_vote: int = typer.Option(1, min=0, max=3),
    config: Path | None = typer.Option(None),
):
    runtime = _load_runtime(config)
    runtime.ahead_k = ahead_k
    runtime.parallel_votes = parallel_votes
    runtime.workflow_mode = workflow_mode
    runtime.max_agents = max_agents
    runtime.moe_enabled = moe_enabled
    runtime.tools_enabled = tools_enabled
    runtime.tool_budget_per_vote = tool_budget_per_vote
    orchestrator = MakerOrchestrator(runtime)
    stats = asyncio.run(
        orchestrator.solve(disks=disks, max_steps=samples, verbose=True)
    )
    print(
        "benchmark_complete "
        f"steps={stats.steps} "
        f"accuracy={stats.accuracy:.6f} "
        f"illegal={stats.illegal_steps} "
        f"valid_votes={stats.valid_votes} "
        f"invalid_votes={stats.invalid_votes}"
    )


@app.command()
def solve(
    disks: int = typer.Option(12, min=3),
    ahead_k: int = typer.Option(3, min=1),
    parallel_votes: int = typer.Option(3, min=1),
    workflow_mode: str = typer.Option("swarm"),
    max_agents: int = typer.Option(15, min=3, max=40),
    moe_enabled: bool = typer.Option(True),
    tools_enabled: bool = typer.Option(False),
    tool_budget_per_vote: int = typer.Option(1, min=0, max=3),
    config: Path | None = typer.Option(None),
):
    runtime = _load_runtime(config)
    runtime.ahead_k = ahead_k
    runtime.parallel_votes = parallel_votes
    runtime.workflow_mode = workflow_mode
    runtime.max_agents = max_agents
    runtime.moe_enabled = moe_enabled
    runtime.tools_enabled = tools_enabled
    runtime.tool_budget_per_vote = tool_budget_per_vote
    orchestrator = MakerOrchestrator(runtime)
    stats = asyncio.run(orchestrator.solve(disks=disks, max_steps=None, verbose=True))
    print(
        "solve_complete "
        f"steps={stats.steps} "
        f"accuracy={stats.accuracy:.6f} "
        f"illegal={stats.illegal_steps} "
        f"valid_votes={stats.valid_votes} "
        f"invalid_votes={stats.invalid_votes}"
    )


@app.command("estimate-k")
def estimate_k(
    p_step: float = typer.Option(..., help="Estimated single-step success rate"),
    steps: int = typer.Option(..., min=1),
    target_success: float = typer.Option(0.95),
):
    k = k_min(target_success=target_success, steps=steps, p_step=p_step, m=1)
    print(f"k_min={k} for p_step={p_step} steps={steps} target={target_success}")


@app.command()
def doctor(config: Path | None = typer.Option(None)):
    runtime = _load_runtime(config)
    adapter = OllamaAdapter()

    async def _check():
        health = []
        for model in runtime.active_models():
            health.append(await adapter.health(model))
        return health

    output = {
        "integrations": {
            "strandsagents": detect_strandsagents(),
            "voiceai": detect_voiceai(),
        },
        "models": asyncio.run(_check()),
    }
    print(json.dumps(output, indent=2))


@app.command("paper-calibrate")
def paper_calibrate(
    disks: int = typer.Option(10, min=3),
    samples: int = typer.Option(1000, min=10),
    target_success: float = typer.Option(0.95),
    model_name: str | None = typer.Option(None),
    temperature: float = typer.Option(0.1),
    concurrency: int = typer.Option(32, min=1, max=256),
    collision_samples: int = typer.Option(100, min=0),
    config: Path | None = typer.Option(None),
):
    runtime = _load_runtime(config)
    result = asyncio.run(
        estimate_single_step_success(
            runtime=runtime,
            disks=disks,
            samples=samples,
            target_success=target_success,
            model_name=model_name,
            temperature=temperature,
            concurrency=concurrency,
            collision_samples=collision_samples,
        )
    )
    print(
        "paper_calibration "
        f"model={result.model_id} "
        f"samples={result.samples} "
        f"p_step={result.p_step:.6f} "
        f"valid_rate={result.valid_rate:.6f} "
        f"collisions={result.both_wrong_collisions} "
        f"k_min={result.k_min_for_target}"
    )
    if result.k_min_for_target < 0:
        print(
            "paper_calibration_warning p_step_not_above_0.5 theorem_not_applicable=true"
        )


@app.command("paper-profile")
def paper_profile(
    output: Path = typer.Option(Path("configs/paper_strict.yaml")),
    ahead_k: int = typer.Option(3, min=1),
    config: Path | None = typer.Option(None),
):
    runtime = _load_runtime(config)
    strict = paper_strict_profile(runtime, k_value=ahead_k)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(strict.model_dump(), sort_keys=False),
        encoding="utf-8",
    )
    print(f"paper_profile_written path={output} ahead_k={strict.ahead_k}")


@app.command("paper-autotune")
def paper_autotune(
    disks: int = typer.Option(10, min=3),
    samples: int = typer.Option(120, min=10),
    collision_samples: int = typer.Option(30, min=0),
    target_success: float = typer.Option(0.95),
    temperature: float = typer.Option(0.1),
    concurrency: int = typer.Option(8, min=1, max=64),
    max_params_b: float = typer.Option(3.0, min=0.1),
    max_models: int = typer.Option(0, min=0, help="0 means no limit"),
    output: Path = typer.Option(Path("configs/paper_autotuned.yaml")),
    lock_to_best_model: bool = typer.Option(False),
    config: Path | None = typer.Option(None),
):
    runtime = _load_runtime(config)
    selector = LocalModelSelector()
    eligible = set(
        asyncio.run(selector.recommend_swarm_candidates(max_params_b=max_params_b))
    )

    active = runtime.active_models()
    if not active:
        raise typer.BadParameter("no active models in runtime")

    candidate_ids = [m.model_id for m in active if m.model_id in eligible]
    if not candidate_ids:
        candidate_ids = [m.model_id for m in active]

    deduped: list[str] = []
    seen: set[str] = set()
    for model_id in candidate_ids:
        if model_id in seen:
            continue
        deduped.append(model_id)
        seen.add(model_id)
    candidate_ids = deduped

    if max_models > 0:
        candidate_ids = candidate_ids[:max_models]

    results = []
    for model_id in candidate_ids:
        result = asyncio.run(
            estimate_single_step_success(
                runtime=runtime,
                disks=disks,
                samples=samples,
                target_success=target_success,
                model_name=model_id,
                temperature=temperature,
                concurrency=concurrency,
                collision_samples=collision_samples,
            )
        )
        results.append(result)
        print(
            "paper_autotune_eval "
            f"model={result.model_id} "
            f"p_step={result.p_step:.6f} "
            f"valid_rate={result.valid_rate:.6f} "
            f"collisions={result.both_wrong_collisions} "
            f"k_min={result.k_min_for_target}"
        )

    best = max(
        results,
        key=lambda r: (
            r.p_step,
            r.valid_rate,
            -r.both_wrong_collisions,
            -999999 if r.k_min_for_target < 0 else -r.k_min_for_target,
        ),
    )

    recommended_k = 3 if best.k_min_for_target < 0 else best.k_min_for_target
    strict = paper_strict_profile(runtime, k_value=recommended_k)
    if lock_to_best_model:
        for model in strict.models:
            model.enabled = model.model_id == best.model_id

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(strict.model_dump(), sort_keys=False),
        encoding="utf-8",
    )
    print(
        "paper_autotune_complete "
        f"best_model={best.model_id} "
        f"recommended_k={recommended_k} "
        f"profile={output}"
    )
    if best.k_min_for_target < 0:
        print(
            "paper_autotune_warning best_model_p_step_not_above_0.5 using_default_k=3"
        )


@app.command("strands-sync")
def strands_sync(config: Path | None = typer.Option(None)):
    runtime = _load_runtime(config)
    selector = LocalModelSelector()
    candidates = asyncio.run(selector.recommend_swarm_candidates(max_params_b=3.0))

    changed = 0
    candidate_set = set(candidates)
    for model in runtime.models:
        if model.model_id in candidate_set and not model.enabled:
            model.enabled = True
            changed += 1

    print(
        "strands_sync_complete "
        f"eligible={len(candidates)} "
        f"enabled_changes={changed} "
        f"models={','.join(candidates)}"
    )


@app.command("paper-validate-exact")
def paper_validate_exact(
    disks: int = typer.Option(20, min=3),
    calibration_samples: int = typer.Option(1000, min=10),
    decorrelation_samples: int = typer.Option(10000, min=100),
    target_success: float = typer.Option(0.95),
    open_source_temperature: float = typer.Option(0.1),
    concurrency: int = typer.Option(4, min=1, max=64),
    request_timeout_s: float = typer.Option(45.0, min=0.5, max=240.0),
    max_params_b: float = typer.Option(3.0, min=0.1),
    max_models: int = typer.Option(4, min=1),
    model_name: str | None = typer.Option(None),
    preflight_timeout_s: float = typer.Option(20.0, min=1.0, max=120.0),
    token_cutoffs: str = typer.Option("700,750,2048"),
    output_profile: Path = typer.Option(Path("configs/paper_validated_strict.yaml")),
    output_report: Path = typer.Option(Path("runs/paper_validation_report.json")),
    lock_to_best_model: bool = typer.Option(False),
    config: Path | None = typer.Option(None),
):
    runtime = _load_runtime(config)
    runtime.tools_enabled = False

    if model_name:
        candidates = [model_name]
    else:
        selector = LocalModelSelector()
        eligible = set(
            asyncio.run(selector.recommend_swarm_candidates(max_params_b=max_params_b))
        )
        active = runtime.active_models()
        candidates = [m.model_id for m in active if m.model_id in eligible]
        if not candidates:
            candidates = [m.model_id for m in active]

    deduped: list[str] = []
    seen: set[str] = set()
    for model_id in candidates:
        if model_id in seen:
            continue
        deduped.append(model_id)
        seen.add(model_id)
    candidates = deduped[:max_models]

    preflight = []
    feasible_candidates: list[str] = []
    for model_id in candidates:
        ok, reason = asyncio.run(
            _probe_model_feasibility(
                model_id=model_id,
                timeout_s=preflight_timeout_s,
            )
        )
        preflight.append({"model_id": model_id, "ok": ok, "reason": reason})
        if ok:
            feasible_candidates.append(model_id)
            print(f"paper_validate_preflight model={model_id} ok=true")
        else:
            print(f"paper_validate_preflight model={model_id} ok=false reason={reason}")

    if not feasible_candidates:
        report = {
            "paper_protocol": {
                "calibration": {
                    "parser_mode": "repairing",
                    "token_cutoff": 2048,
                    "temperature": open_source_temperature,
                    "samples_per_model": calibration_samples,
                },
                "decorrelation": {
                    "parser_mode": "repairing",
                    "token_cutoff": 2048,
                    "samples": decorrelation_samples,
                },
                "strict_run": {
                    "parser_mode": "red_flagging",
                    "token_cutoff": 750,
                    "temperature_first": 0.0,
                    "temperature_followup": 0.1,
                    "parallel_votes": 3,
                    "ahead_k": 3,
                },
            },
            "models_evaluated": candidates,
            "preflight": preflight,
            "warnings": ["no feasible models for exact validation"],
        }
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise typer.Exit(code=2)

    calibration_rows = []
    for model_id in feasible_candidates:
        row = asyncio.run(
            estimate_single_step_success(
                runtime=runtime,
                disks=disks,
                samples=calibration_samples,
                target_success=target_success,
                model_name=model_id,
                temperature=open_source_temperature,
                concurrency=concurrency,
                collision_samples=0,
                request_timeout_s=request_timeout_s,
            )
        )
        calibration_rows.append(row)
        print(
            "paper_validate_calibration "
            f"model={row.model_id} "
            f"p_step={row.p_step:.6f} "
            f"valid_rate={row.valid_rate:.6f} "
            f"k_min={row.k_min_for_target}"
        )

    best = max(
        calibration_rows,
        key=lambda r: (
            r.p_step,
            r.valid_rate,
            -999999 if r.k_min_for_target < 0 else -r.k_min_for_target,
        ),
    )

    decorrelation = asyncio.run(
        measure_two_run_collisions(
            runtime=runtime,
            disks=disks,
            samples=decorrelation_samples,
            model_name=best.model_id,
            parser_mode="repairing",
            token_cutoff=2048,
            temperature_a=open_source_temperature,
            temperature_b=open_source_temperature,
            concurrency=concurrency,
            request_timeout_s=request_timeout_s,
        )
    )
    print(
        "paper_validate_decorrelation "
        f"model={decorrelation.model_id} "
        f"samples={decorrelation.samples} "
        f"collisions={decorrelation.both_wrong_collisions}"
    )

    cutoff_values = []
    for part in token_cutoffs.split(","):
        part = part.strip()
        if not part:
            continue
        cutoff_values.append(int(part))

    parser_collision_rows = []
    for cutoff in cutoff_values:
        for parser_mode in ("repairing", "red_flagging"):
            row = asyncio.run(
                measure_two_run_collisions(
                    runtime=runtime,
                    disks=disks,
                    samples=max(100, min(2000, decorrelation_samples // 5)),
                    model_name=best.model_id,
                    parser_mode=parser_mode,
                    token_cutoff=cutoff,
                    temperature_a=open_source_temperature,
                    temperature_b=open_source_temperature,
                    concurrency=concurrency,
                    request_timeout_s=request_timeout_s,
                )
            )
            parser_collision_rows.append(row)
            print(
                "paper_validate_parser_compare "
                f"parser={row.parser_mode} cutoff={row.token_cutoff} "
                f"samples={row.samples} collisions={row.both_wrong_collisions}"
            )

    recommended_k = 3 if best.k_min_for_target < 0 else best.k_min_for_target
    strict = paper_strict_profile(runtime, k_value=recommended_k)
    if lock_to_best_model:
        for model in strict.models:
            model.enabled = model.model_id == best.model_id

    output_profile.parent.mkdir(parents=True, exist_ok=True)
    output_profile.write_text(
        yaml.safe_dump(strict.model_dump(), sort_keys=False),
        encoding="utf-8",
    )

    report = {
        "paper_protocol": {
            "calibration": {
                "parser_mode": "repairing",
                "token_cutoff": 2048,
                "temperature": open_source_temperature,
                "samples_per_model": calibration_samples,
            },
            "decorrelation": {
                "parser_mode": "repairing",
                "token_cutoff": 2048,
                "samples": decorrelation_samples,
            },
            "strict_run": {
                "parser_mode": "red_flagging",
                "token_cutoff": 750,
                "temperature_first": 0.0,
                "temperature_followup": 0.1,
                "parallel_votes": 3,
                "ahead_k": recommended_k,
            },
        },
        "models_evaluated": feasible_candidates,
        "preflight": preflight,
        "calibration_results": [
            {
                "model_id": r.model_id,
                "p_step": r.p_step,
                "valid_rate": r.valid_rate,
                "k_min": r.k_min_for_target,
            }
            for r in calibration_rows
        ],
        "best_model": best.model_id,
        "best_model_decorrelation": {
            "samples": decorrelation.samples,
            "both_wrong_collisions": decorrelation.both_wrong_collisions,
        },
        "parser_collision_comparison": [
            {
                "parser_mode": r.parser_mode,
                "token_cutoff": r.token_cutoff,
                "samples": r.samples,
                "both_wrong_collisions": r.both_wrong_collisions,
            }
            for r in parser_collision_rows
        ],
        "warnings": [
            "k theorem assumes p_step > 0.5; fallback to k=3 used if not satisfied"
            if best.k_min_for_target < 0
            else ""
        ],
        "output_profile": str(output_profile),
    }
    report["warnings"] = [w for w in report["warnings"] if w]

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "paper_validate_complete "
        f"best_model={best.model_id} recommended_k={recommended_k} "
        f"profile={output_profile} report={output_report}"
    )


if __name__ == "__main__":
    app()
