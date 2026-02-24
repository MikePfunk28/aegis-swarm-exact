from pathlib import Path
import os

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from mdap_small.adapter import OllamaAdapter
from mdap_small.history import append_run, get_recent_runs
from mdap_small.integrations import detect_strandsagents, detect_voiceai
from mdap_small.models import DEFAULT_MODELS
from mdap_small.orchestrator import MakerOrchestrator

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
