from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synora.calibration.surrogate_fit import DEFAULT_PARAMS_PATH, load_surrogate_params
from synora.generative.design_space import ReactorDesign
from synora.generative.multizone import MultiZoneDesign
from synora.physics.label_pfr import DEFAULT_MECHANISM_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "design_reports"


def _read_mechanism_metadata(mechanism_path: str | Path) -> dict[str, str]:
    path = Path(mechanism_path)
    if not path.exists():
        return {"path": str(path), "version": "unknown"}

    cantera_version = "unknown"
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(120):
            line = handle.readline()
            if not line:
                break
            if line.strip().startswith("cantera-version:"):
                cantera_version = line.split(":", maxsplit=1)[1].strip()
                break
    return {"path": str(path), "version": cantera_version}


def _surrogate_rmse_summary(params_path: str | Path | None) -> dict[str, Any]:
    selected_path = DEFAULT_PARAMS_PATH if params_path is None else Path(params_path)
    try:
        model = load_surrogate_params(selected_path)
    except FileNotFoundError:
        return {"path": str(selected_path), "ensemble_size": 0, "rmse": {}, "rmse_std": {}}
    return {
        "path": str(selected_path),
        "ensemble_size": model.ensemble_size,
        "rmse": model.rmse,
        "rmse_std": model.rmse_std,
    }


def _serialize_design_payload(design: ReactorDesign | MultiZoneDesign) -> dict[str, Any]:
    payload = design.to_dict()
    payload["design_type"] = "multizone" if isinstance(design, MultiZoneDesign) else "single_zone"
    return payload


def generate_design_report(
    design: ReactorDesign | MultiZoneDesign,
    metrics: dict[str, Any],
    uncertainty: dict[str, float],
    ood_score: float,
    *,
    output_path: str | Path | None = None,
    surrogate_params_path: str | Path | None = None,
    mechanism_path: str | Path = DEFAULT_MECHANISM_PATH,
) -> tuple[Path, dict[str, Any]]:
    timestamp = datetime.now(tz=UTC).isoformat()
    mechanism_meta = _read_mechanism_metadata(mechanism_path)
    surrogate_meta = _surrogate_rmse_summary(surrogate_params_path)

    payload: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "design": _serialize_design_payload(design),
        "performance_metrics": metrics,
        "fouling_index": metrics.get("fouling_risk_index"),
        "profit_per_hr": metrics.get("profit_per_hr"),
        "confidence_intervals": uncertainty,
        "is_out_of_distribution": bool(metrics.get("is_out_of_distribution", ood_score > 1.0)),
        "ood_score": float(ood_score),
        "surrogate_summary": surrogate_meta,
        "mechanism": mechanism_meta,
    }

    if output_path is None:
        DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        filename = datetime.now(tz=UTC).strftime("design_report_%Y%m%dT%H%M%SZ.json")
        path = DEFAULT_REPORT_DIR / filename
    else:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, payload


__all__ = ["generate_design_report", "DEFAULT_REPORT_DIR"]
