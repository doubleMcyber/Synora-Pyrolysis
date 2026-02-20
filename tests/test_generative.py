from __future__ import annotations

import pytest

from synora.generative.design_space import ReactorDesign
from synora.generative.objectives import evaluate_design_surrogate
from synora.generative.optimizer import propose_designs


def test_design_derived_residence_time_and_sav_bounds() -> None:
    design = ReactorDesign(
        length_m=1.4,
        diameter_m=0.09,
        pressure_atm=1.4,
        temp_c=980.0,
        methane_kg_per_hr=110.0,
        dilution_frac=0.82,
        carbon_removal_eff=0.55,
    )

    assert 1e-3 <= design.residence_time_s <= 8.0
    assert design.surface_area_to_volume == pytest.approx(4.0 / design.diameter_m, rel=1e-3)


def test_geometry_change_updates_residence_time() -> None:
    base = ReactorDesign(
        length_m=1.0,
        diameter_m=0.08,
        pressure_atm=1.2,
        temp_c=960.0,
        methane_kg_per_hr=90.0,
        dilution_frac=0.80,
        carbon_removal_eff=0.5,
    )
    larger = ReactorDesign(
        length_m=1.8,
        diameter_m=0.12,
        pressure_atm=1.2,
        temp_c=960.0,
        methane_kg_per_hr=90.0,
        dilution_frac=0.80,
        carbon_removal_eff=0.5,
    )

    assert larger.volume_m3 > base.volume_m3
    assert larger.residence_time_s > base.residence_time_s


def test_surrogate_evaluation_returns_required_metrics() -> None:
    design = ReactorDesign(
        length_m=1.2,
        diameter_m=0.08,
        pressure_atm=1.2,
        temp_c=960.0,
        methane_kg_per_hr=95.0,
        dilution_frac=0.80,
        carbon_removal_eff=0.60,
    )

    metrics = evaluate_design_surrogate(design, surrogate_params_path=None)

    required = {
        "conversion",
        "h2_rate",
        "fouling_risk_index",
        "pressure_drop_proxy",
        "heat_loss_proxy",
        "profit_per_hr",
        "constraint_violations",
    }
    assert required.issubset(set(metrics.keys()))
    assert metrics["conversion"] >= 0
    assert metrics["h2_rate"] >= 0
    assert metrics["fouling_risk_index"] >= 0
    assert metrics["pressure_drop_proxy"] >= 0
    assert metrics["heat_loss_proxy"] >= 0
    assert isinstance(metrics["constraint_violations"], list)
    assert metrics["conversion_std"] >= 0
    assert metrics["h2_yield_std"] >= 0
    assert metrics["fouling_risk_index_std"] >= 0


def test_out_of_distribution_design_is_flagged() -> None:
    extreme_design = ReactorDesign(
        length_m=3.0,
        diameter_m=0.06,
        pressure_atm=2.5,
        temp_c=1320.0,
        methane_kg_per_hr=220.0,
        dilution_frac=0.55,
        carbon_removal_eff=0.05,
    )
    metrics = evaluate_design_surrogate(extreme_design)

    assert metrics["is_out_of_distribution"] is True
    assert metrics["ood_score"] > 1.0
    assert "out_of_distribution" in metrics["constraint_violations"]


def test_optimizer_returns_top_k_and_feasible_designs(tmp_path) -> None:
    constraints = {
        "min_conversion": 0.0,
        "max_fouling_risk_index": 1.0,
        "max_pressure_drop_proxy": 1.0,
        "max_heat_loss_proxy": 1.0,
    }
    proposals = propose_designs(
        top_k=6,
        generations=5,
        population_size=60,
        seed=7,
        constraints=constraints,
        surrogate_params_path=tmp_path / "missing_surrogate_params.json",
    )

    assert len(proposals) == 6
    assert all(item.violation_count == 0 for item in proposals)
