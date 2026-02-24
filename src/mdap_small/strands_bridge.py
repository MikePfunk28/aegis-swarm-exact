from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class LocalModelInfo:
    model_id: str
    family: str
    parameter_size: str
    size_bytes: int


class LocalModelSelector:
    """Rewritten strandsagents-style selector focused on local Ollama models."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url.rstrip("/")

    async def list_available(self) -> list[LocalModelInfo]:
        async with httpx.AsyncClient(timeout=20) as client:
            res = await client.get(f"{self.base_url}/api/tags")
            res.raise_for_status()
            payload = res.json()

        out: list[LocalModelInfo] = []
        for item in payload.get("models", []):
            if not isinstance(item, dict):
                continue
            details = (
                item.get("details", {}) if isinstance(item.get("details"), dict) else {}
            )
            out.append(
                LocalModelInfo(
                    model_id=item.get("name", ""),
                    family=details.get("family", "unknown"),
                    parameter_size=details.get("parameter_size", "unknown"),
                    size_bytes=int(item.get("size", 0) or 0),
                )
            )
        return out

    async def recommend_swarm_candidates(self, max_params_b: float = 3.0) -> list[str]:
        models = await self.list_available()
        eligible: list[str] = []
        for model in models:
            if _params_b(model.parameter_size) <= max_params_b:
                eligible.append(model.model_id)
        return sorted(set(eligible))


def _params_b(raw: str) -> float:
    cleaned = raw.strip().lower().replace(" ", "")
    if cleaned.endswith("b"):
        try:
            return float(cleaned[:-1])
        except ValueError:
            return 999.0
    if cleaned.endswith("m"):
        try:
            return float(cleaned[:-1]) / 1000.0
        except ValueError:
            return 999.0
    return 999.0
