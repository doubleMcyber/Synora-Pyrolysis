from __future__ import annotations

import json

from synora.generative.design_space import ReactorDesign
from synora.generative.report import generate_design_report


def test_generate_design_report_writes_json(tmp_path) -> None:
    design = ReactorDesign(
        length_m=1.3,
        diameter_m=0.09,
        pressure_atm=1.4,
        temp_c=970.0,
        methane_kg_per_hr=100.0,
        dilution_frac=0.78,
        carbon_removal_eff=0.65,
    )
    metrics = {
        "conversion": 0.45,
        "fouling_risk_index": 0.12,
        "profit_per_hr": 28.4,
        "is_out_of_distribution": False,
    }
    uncertainty = {
        "conversion_ci_lower": 0.40,
        "conversion_ci_upper": 0.50,
    }

    output_path = tmp_path / "design_report.json"
    written_path, payload = generate_design_report(
        design,
        metrics,
        uncertainty,
        ood_score=0.42,
        output_path=output_path,
    )

    assert written_path.exists()
    assert payload["design"]["length_m"] == design.length_m
    assert payload["performance_metrics"]["profit_per_hr"] == metrics["profit_per_hr"]
    assert "mechanism" in payload

    loaded = json.loads(written_path.read_text(encoding="utf-8"))
    assert loaded["ood_score"] == 0.42
