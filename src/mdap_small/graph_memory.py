from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class GraphMemory:
    """Lightweight rewritten memory graph event sink inspired by strandsagents."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (Path("data") / "graph")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.base_dir / "events.jsonl"

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.events_file.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.events_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return list(reversed(rows[-limit:]))
