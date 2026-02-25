import asyncio
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path
import os

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from mdap_small.adapter import OllamaAdapter
from mdap_small.history import append_run, get_recent_runs
from mdap_small.integrations import detect_strandsagents, detect_voiceai
from mdap_small.models import DEFAULT_MODELS
from mdap_small.orchestrator import MakerOrchestrator
from mdap_small.validation_gate import load_and_check_report

app = FastAPI(title="MDAP Voice Bridge")


class VoiceRequest(BaseModel):
    transcript: str


class RunRequest(BaseModel):
    mode: str = "benchmark"
    disks: int = 10
    steps: int = 500
    ahead_k: int = 3
    parallel_votes: int = 3
    workflow_mode: str = "swarm"
    max_agents: int = 15
    moe_enabled: bool = True
    tools_enabled: bool = False
    tool_budget_per_vote: int = 1
    parser_mode: str = "red_flagging"
    red_flag_token_cutoff: int = 750
    enforce_paper_gate: bool = True


class ValidationRunRequest(BaseModel):
    model_name: str | None = None
    disks: int = 20
    calibration_samples: int = 1000
    decorrelation_samples: int = 10000
    concurrency: int = 16
    request_timeout_s: float = 8.0
    preflight_timeout_s: float = 20.0
    token_cutoffs: str = "700,750,2048"
    output_profile: str = "configs/paper_validated_strict.yaml"
    output_report: str = "runs/paper_validation_report.json"
    lock_to_best_model: bool = True
    dry_run: bool = False


_validation_job: dict = {
    "id": None,
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "command": None,
    "exit_code": None,
    "log_lines": [],
}
_validation_task: asyncio.Task | None = None


def _append_job_log(line: str) -> None:
    line = line.rstrip("\n")
    if not line:
        return
    _validation_job["log_lines"].append(line)
    if len(_validation_job["log_lines"]) > 2000:
        _validation_job["log_lines"] = _validation_job["log_lines"][-2000:]


def _on_validation_task_done(_task: asyncio.Task) -> None:
    global _validation_task
    _validation_task = None


async def _run_validation_subprocess(command: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    try:
        while True:
            chunk = await proc.stdout.readline()
            if not chunk:
                break
            _append_job_log(chunk.decode("utf-8", errors="replace"))
        exit_code = await proc.wait()
        _validation_job["exit_code"] = exit_code
        _validation_job["status"] = "completed" if exit_code == 0 else "failed"
        _validation_job["finished_at"] = datetime.now(timezone.utc).isoformat()
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        _append_job_log("job cancelled")
        _validation_job["exit_code"] = -1
        _validation_job["status"] = "cancelled"
        _validation_job["finished_at"] = datetime.now(timezone.utc).isoformat()
        raise


async def _run_validation_dry() -> None:
    _append_job_log("dry_run: starting paper validation")
    await asyncio.sleep(0.2)
    _append_job_log("dry_run: calibration complete")
    await asyncio.sleep(0.2)
    _append_job_log("dry_run: decorrelation complete")
    await asyncio.sleep(0.2)
    _append_job_log("dry_run: strict profile generated")
    _validation_job["exit_code"] = 0
    _validation_job["status"] = "completed"
    _validation_job["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/validation/cancel")
async def validation_cancel():
    global _validation_task
    if _validation_job["status"] != "running" or _validation_task is None:
        return {
            "ok": False,
            "status": _validation_job["status"],
            "message": "no running job",
        }
    _validation_task.cancel()
    return {"ok": True, "status": "cancelling", "job_id": _validation_job["id"]}


def _machine_profile() -> dict:
    return {
        "profile": "local-desktop",
        "cpu": "i7-13700K",
        "ram_gb": 32,
        "gpu": "RX 6600 8GB",
        "recommended": {
            "leader": "LFM2.5-1.2B",
            "workers": "gemma3:270m-it",
            "worker_count": 10,
            "parallel_votes": 3,
            "ahead_k": 3,
        },
    }


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    html_path = Path(__file__).resolve().parents[2] / "web" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/ui-config")
async def ui_config():
    runtime = DEFAULT_MODELS.model_copy(deep=True)
    return {
        "machine": _machine_profile(),
        "integrations": {
            "strandsagents": detect_strandsagents(),
            "voiceai": detect_voiceai(),
        },
        "default_runtime": runtime.model_dump(),
        "workflow_modes": ["sequential", "swarm", "hivemind", "graph"],
    }


@app.get("/api/runs")
async def runs(limit: int = 20):
    safe_limit = max(1, min(limit, 200))
    return {"runs": get_recent_runs(limit=safe_limit)}


@app.get("/api/health/models")
async def model_health():
    runtime = DEFAULT_MODELS.model_copy(deep=True)
    adapter = OllamaAdapter()
    rows = []
    for model in runtime.active_models():
        rows.append(await adapter.health(model))
    all_ok = all(row.get("ok", False) for row in rows)
    return {"ok": all_ok, "models": rows}


@app.get("/api/validation/status")
async def validation_status():
    gate = load_and_check_report()
    return {
        "ok": gate.ok,
        "report_path": gate.report_path,
        "messages": gate.messages,
        "summary": gate.summary,
        "checklist": gate.checklist,
        "requirements": {
            "calibration_samples_per_model_min": 1000,
            "decorrelation_samples_min": 10000,
            "strict_parser_mode": "red_flagging",
            "strict_token_cutoff": 750,
            "strict_temps": [0.0, 0.1],
            "strict_parallel_votes_min": 3,
            "strict_ahead_k_min": 3,
            "best_p_step_gt": 0.5,
        },
    }


@app.get("/api/validation/job")
async def validation_job(offset: int = 0, limit: int = 200):
    safe_offset = max(0, offset)
    safe_limit = max(1, min(limit, 1000))
    logs = _validation_job["log_lines"][safe_offset : safe_offset + safe_limit]
    return {
        "id": _validation_job["id"],
        "status": _validation_job["status"],
        "started_at": _validation_job["started_at"],
        "finished_at": _validation_job["finished_at"],
        "command": _validation_job["command"],
        "exit_code": _validation_job["exit_code"],
        "total_log_lines": len(_validation_job["log_lines"]),
        "log_lines": logs,
        "next_offset": safe_offset + len(logs),
    }


@app.post("/api/validation/run")
async def validation_run(req: ValidationRunRequest):
    global _validation_task
    if _validation_job["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "validation_job_already_running",
                "job_id": _validation_job["id"],
            },
        )

    job_id = str(uuid.uuid4())
    _validation_job["id"] = job_id
    _validation_job["status"] = "running"
    _validation_job["started_at"] = datetime.now(timezone.utc).isoformat()
    _validation_job["finished_at"] = None
    _validation_job["exit_code"] = None
    _validation_job["log_lines"] = []

    command = [
        "python",
        "-m",
        "mdap_small.cli",
        "paper-validate-exact",
        "--disks",
        str(req.disks),
        "--calibration-samples",
        str(req.calibration_samples),
        "--decorrelation-samples",
        str(req.decorrelation_samples),
        "--concurrency",
        str(req.concurrency),
        "--request-timeout-s",
        str(req.request_timeout_s),
        "--preflight-timeout-s",
        str(req.preflight_timeout_s),
        "--token-cutoffs",
        req.token_cutoffs,
        "--output-profile",
        req.output_profile,
        "--output-report",
        req.output_report,
    ]
    if req.model_name:
        command.extend(["--model-name", req.model_name])
    if req.lock_to_best_model:
        command.append("--lock-to-best-model")

    _validation_job["command"] = " ".join(shlex.quote(x) for x in command)
    _append_job_log(f"starting job {job_id}")
    _append_job_log(f"command: {_validation_job['command']}")

    if req.dry_run:
        await _run_validation_dry()
        _validation_task = None
    else:
        _validation_task = asyncio.create_task(_run_validation_subprocess(command))
        _validation_task.add_done_callback(_on_validation_task_done)

    return {
        "ok": True,
        "job_id": job_id,
        "status": _validation_job["status"],
    }


@app.post("/api/respond")
async def respond(req: VoiceRequest):
    text = req.transcript.strip()
    if not text:
        return {"reply": "I did not catch that."}

    voiceai_gateway = os.getenv("VOICEAI_GATEWAY_URL", "").strip()
    if voiceai_gateway:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                proxy = await client.post(
                    f"{voiceai_gateway.rstrip('/')}/voice/reply",
                    json={"transcript": text},
                )
                if proxy.status_code < 300:
                    data = proxy.json()
                    if isinstance(data, dict) and data.get("replyText"):
                        return {"reply": data["replyText"], "source": "voiceai"}
        except Exception:
            pass

    if "status" in text.lower():
        return {"reply": "MDAP voice bridge is running.", "source": "local"}

    return {
        "reply": (
            "Received. I can route this into MAKER voting with a 1.2B leader and "
            "simple 270M micro-agents for the next step."
        ),
        "source": "local",
    }


@app.post("/api/run")
async def run_workflow(req: RunRequest):
    if req.enforce_paper_gate:
        gate = load_and_check_report()
        if not gate.ok:
            raise HTTPException(
                status_code=428,
                detail={
                    "error": "paper_validation_gate_failed",
                    "report_path": gate.report_path,
                    "messages": gate.messages,
                    "summary": gate.summary,
                },
            )

    runtime = DEFAULT_MODELS.model_copy(deep=True)
    runtime.ahead_k = req.ahead_k
    runtime.parallel_votes = req.parallel_votes
    runtime.workflow_mode = req.workflow_mode
    runtime.max_agents = req.max_agents
    runtime.moe_enabled = req.moe_enabled
    runtime.tools_enabled = req.tools_enabled
    runtime.tool_budget_per_vote = req.tool_budget_per_vote
    runtime.parser_mode = req.parser_mode
    runtime.red_flag_token_cutoff = req.red_flag_token_cutoff

    orchestrator = MakerOrchestrator(runtime)
    max_steps = req.steps if req.mode == "benchmark" else None
    stats = await orchestrator.solve(
        disks=req.disks, max_steps=max_steps, verbose=False
    )
    result = {
        "mode": req.mode,
        "workflow_mode": req.workflow_mode,
        "steps": stats.steps,
        "accuracy": stats.accuracy,
        "illegal_steps": stats.illegal_steps,
        "valid_votes": stats.valid_votes,
        "invalid_votes": stats.invalid_votes,
    }
    append_run(
        {
            "request": req.model_dump(),
            "result": result,
        }
    )
    return result


def main():
    uvicorn.run(app, host="127.0.0.1", port=8088)


if __name__ == "__main__":
    main()
