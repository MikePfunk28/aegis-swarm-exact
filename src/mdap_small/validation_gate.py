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


def load_and_check_report(
    report_path: Path = Path("runs") / "paper_validation_report.json",
) -> ValidationGateStatus:
    if not report_path.exists():
        return ValidationGateStatus(
            ok=False,
            report_path=str(report_path),
            messages=["paper validation report is missing"],
            summary={},
        )

    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return ValidationGateStatus(
            ok=False,
            report_path=str(report_path),
            messages=["paper validation report is unreadable"],
            summary={},
        )

    msgs: list[str] = []
    proto = raw.get("paper_protocol", {}) if isinstance(raw, dict) else {}
    calib = proto.get("calibration", {}) if isinstance(proto, dict) else {}
    deco = proto.get("decorrelation", {}) if isinstance(proto, dict) else {}
    strict = proto.get("strict_run", {}) if isinstance(proto, dict) else {}

    if calib.get("parser_mode") != "repairing":
        msgs.append("calibration parser_mode must be repairing")
    if int(calib.get("token_cutoff", -1)) != 2048:
        msgs.append("calibration token_cutoff must be 2048")
    if float(calib.get("temperature", -1)) != 0.1:
        msgs.append("calibration temperature must be 0.1")
    if int(calib.get("samples_per_model", 0)) < 1000:
        msgs.append("calibration samples_per_model must be >= 1000")

    if deco.get("parser_mode") != "repairing":
        msgs.append("decorrelation parser_mode must be repairing")
    if int(deco.get("token_cutoff", -1)) != 2048:
        msgs.append("decorrelation token_cutoff must be 2048")
    if int(deco.get("samples", 0)) < 10000:
        msgs.append("decorrelation samples must be >= 10000")

    if strict.get("parser_mode") != "red_flagging":
        msgs.append("strict parser_mode must be red_flagging")
    if int(strict.get("token_cutoff", -1)) != 750:
        msgs.append("strict token_cutoff must be 750")
    if float(strict.get("temperature_first", -1)) != 0.0:
        msgs.append("strict temperature_first must be 0.0")
    if float(strict.get("temperature_followup", -1)) != 0.1:
        msgs.append("strict temperature_followup must be 0.1")
    if int(strict.get("parallel_votes", 0)) < 3:
        msgs.append("strict parallel_votes must be >= 3")
    if int(strict.get("ahead_k", 0)) < 3:
        msgs.append("strict ahead_k must be >= 3")

    deco_metrics = (
        raw.get("best_model_decorrelation", {}) if isinstance(raw, dict) else {}
    )
    collisions = int(deco_metrics.get("both_wrong_collisions", -1))
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

    return ValidationGateStatus(
        ok=not msgs,
        report_path=str(report_path),
        messages=msgs,
        summary={
            "best_model": raw.get("best_model"),
            "best_p_step": best_p,
            "decorrelation_collisions": collisions,
        },
    )
