from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ValidationGateStatus:
    ok: bool
    report_path: str
    messages: list[str]
    summary: dict[str, Any]
    checklist: list[dict[str, Any]]


def load_and_check_report(
    report_path: Path = Path("runs") / "paper_validation_report.json",
) -> ValidationGateStatus:
    if not report_path.exists():
        return ValidationGateStatus(
            ok=False,
            report_path=str(report_path),
            messages=["paper validation report is missing"],
            summary={},
            checklist=[],
        )

    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return ValidationGateStatus(
            ok=False,
            report_path=str(report_path),
            messages=["paper validation report is unreadable"],
            summary={},
            checklist=[],
        )

    msgs: list[str] = []
    checklist: list[dict[str, Any]] = []
    proto = raw.get("paper_protocol", {}) if isinstance(raw, dict) else {}
    calib = proto.get("calibration", {}) if isinstance(proto, dict) else {}
    deco = proto.get("decorrelation", {}) if isinstance(proto, dict) else {}
    strict = proto.get("strict_run", {}) if isinstance(proto, dict) else {}

    c = calib.get("parser_mode") == "repairing"
    checklist.append(
        {
            "key": "calib.parser",
            "label": "Calibration parser",
            "ok": c,
            "expected": "repairing",
            "actual": calib.get("parser_mode"),
        }
    )
    if not c:
        msgs.append("calibration parser_mode must be repairing")
    c = int(calib.get("token_cutoff", -1)) == 2048
    checklist.append(
        {
            "key": "calib.tokens",
            "label": "Calibration token cutoff",
            "ok": c,
            "expected": 2048,
            "actual": calib.get("token_cutoff"),
        }
    )
    if not c:
        msgs.append("calibration token_cutoff must be 2048")
    c = float(calib.get("temperature", -1)) == 0.1
    checklist.append(
        {
            "key": "calib.temp",
            "label": "Calibration temperature",
            "ok": c,
            "expected": 0.1,
            "actual": calib.get("temperature"),
        }
    )
    if not c:
        msgs.append("calibration temperature must be 0.1")
    c = int(calib.get("samples_per_model", 0)) >= 1000
    checklist.append(
        {
            "key": "calib.samples",
            "label": "Calibration samples/model",
            "ok": c,
            "expected": ">=1000",
            "actual": calib.get("samples_per_model"),
        }
    )
    if not c:
        msgs.append("calibration samples_per_model must be >= 1000")

    c = deco.get("parser_mode") == "repairing"
    checklist.append(
        {
            "key": "deco.parser",
            "label": "Decorrelation parser",
            "ok": c,
            "expected": "repairing",
            "actual": deco.get("parser_mode"),
        }
    )
    if not c:
        msgs.append("decorrelation parser_mode must be repairing")
    c = int(deco.get("token_cutoff", -1)) == 2048
    checklist.append(
        {
            "key": "deco.tokens",
            "label": "Decorrelation token cutoff",
            "ok": c,
            "expected": 2048,
            "actual": deco.get("token_cutoff"),
        }
    )
    if not c:
        msgs.append("decorrelation token_cutoff must be 2048")
    c = int(deco.get("samples", 0)) >= 10000
    checklist.append(
        {
            "key": "deco.samples",
            "label": "Decorrelation samples",
            "ok": c,
            "expected": ">=10000",
            "actual": deco.get("samples"),
        }
    )
    if not c:
        msgs.append("decorrelation samples must be >= 10000")

    c = strict.get("parser_mode") == "red_flagging"
    checklist.append(
        {
            "key": "strict.parser",
            "label": "Strict parser",
            "ok": c,
            "expected": "red_flagging",
            "actual": strict.get("parser_mode"),
        }
    )
    if not c:
        msgs.append("strict parser_mode must be red_flagging")
    c = int(strict.get("token_cutoff", -1)) == 750
    checklist.append(
        {
            "key": "strict.tokens",
            "label": "Strict token cutoff",
            "ok": c,
            "expected": 750,
            "actual": strict.get("token_cutoff"),
        }
    )
    if not c:
        msgs.append("strict token_cutoff must be 750")
    c = float(strict.get("temperature_first", -1)) == 0.0
    checklist.append(
        {
            "key": "strict.temp0",
            "label": "Strict first temperature",
            "ok": c,
            "expected": 0.0,
            "actual": strict.get("temperature_first"),
        }
    )
    if not c:
        msgs.append("strict temperature_first must be 0.0")
    c = float(strict.get("temperature_followup", -1)) == 0.1
    checklist.append(
        {
            "key": "strict.temp1",
            "label": "Strict followup temperature",
            "ok": c,
            "expected": 0.1,
            "actual": strict.get("temperature_followup"),
        }
    )
    if not c:
        msgs.append("strict temperature_followup must be 0.1")
    c = int(strict.get("parallel_votes", 0)) >= 3
    checklist.append(
        {
            "key": "strict.parallel",
            "label": "Strict parallel votes",
            "ok": c,
            "expected": ">=3",
            "actual": strict.get("parallel_votes"),
        }
    )
    if not c:
        msgs.append("strict parallel_votes must be >= 3")
    c = int(strict.get("ahead_k", 0)) >= 3
    checklist.append(
        {
            "key": "strict.k",
            "label": "Strict ahead-k",
            "ok": c,
            "expected": ">=3",
            "actual": strict.get("ahead_k"),
        }
    )
    if not c:
        msgs.append("strict ahead_k must be >= 3")

    deco_metrics = (
        raw.get("best_model_decorrelation", {}) if isinstance(raw, dict) else {}
    )
    collisions = int(deco_metrics.get("both_wrong_collisions", -1))
    c = collisions >= 0
    checklist.append(
        {
            "key": "deco.collision_metric",
            "label": "Decorrelation collision metric",
            "ok": c,
            "expected": "present",
            "actual": collisions,
        }
    )
    if collisions < 0:
        msgs.append("decorrelation collision metric missing")

    calibration_results = (
        raw.get("calibration_results", []) if isinstance(raw, dict) else []
    )
    best_p = 0.0
    if isinstance(calibration_results, list):
        for row in calibration_results:
            if isinstance(row, dict):
                try:
                    best_p = max(best_p, float(row.get("p_step", 0.0)))
                except Exception:
                    continue
    if best_p <= 0.5:
        msgs.append("no calibrated model with p_step > 0.5")
    checklist.append(
        {
            "key": "calib.p_gt_0_5",
            "label": "Best calibrated p_step",
            "ok": best_p > 0.5,
            "expected": ">0.5",
            "actual": best_p,
        }
    )

    return ValidationGateStatus(
        ok=not msgs,
        report_path=str(report_path),
        messages=msgs,
        summary={
            "best_model": raw.get("best_model"),
            "best_p_step": best_p,
            "decorrelation_collisions": collisions,
        },
        checklist=checklist,
    )
