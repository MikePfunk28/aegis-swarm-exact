from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNS_DIR = Path("data") / "runs"
RUNS_FILE = RUNS_DIR / "history.jsonl"


def append_run(entry: dict[str, Any]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def get_recent_runs(limit: int = 20) -> list[dict[str, Any]]:
    if not RUNS_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    with RUNS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(reversed(rows[-limit:]))
