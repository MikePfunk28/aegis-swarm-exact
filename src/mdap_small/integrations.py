from __future__ import annotations

import importlib.util
from pathlib import Path


def detect_strandsagents() -> dict:
    root = Path("M:/strandsagents")
    if not root.exists():
        return {"available": False, "reason": "path not found"}

    graph_file = root / "graph" / "enhanced_memory_graph.py"
    return {
        "available": graph_file.exists(),
        "path": str(root),
        "graph_module": str(graph_file),
    }


def detect_voiceai() -> dict:
    root = Path("M:/voiceai")
    if not root.exists():
        return {"available": False, "reason": "path not found"}
    readme = root / "README.md"
    return {
        "available": readme.exists(),
        "path": str(root),
        "readme": str(readme),
    }


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None
