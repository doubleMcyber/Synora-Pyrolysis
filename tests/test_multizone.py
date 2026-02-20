from __future__ import annotations

from pathlib import Path

from synora.generative.multizone import MultiZoneBounds, MultiZoneDesign, ZoneDesign
from synora.generative.objectives import evaluate_multizone_surrogate
from synora.generative.optimizer import propose_multizone_designs


def _fallback_params_path(tmp_path: Path) -> Path:
    return tmp_path / "missing_surrogate_params.json"


def test_multizone_tau_changes_with_geometry() -> None:
    base = MultiZoneDesign(
        zones=[
            ZoneDesign(temp_c=930.0, length_m=0.8, diameter_m=0.09, insulation_factor=1.0),
            ZoneDesign(temp_c=990.0, length_m=0.8, diameter_m=0.09, insulation_factor=1.0),
        ],
        methane_kg_per_hr=100.0,
        pressure_atm=1.3,
        dilution_frac=0.80,
        carbon_removal_eff=0.60,
    )
    stretched = MultiZoneDesign(
        zones=[
            ZoneDesign(temp_c=930.0, length_m=1.4, diameter_m=0.09, insulation_factor=1.0),
            ZoneDesign(temp_c=990.0, length_m=0.8, diameter_m=0.09, insulation_factor=1.0),
        ],
        methane_kg_per_hr=100.0,
        pressure_atm=1.3,
        dilution_frac=0.80,
        carbon_removal_eff=0.60,
    )

    assert stretched.zone_residence_time_s[0] > base.zone_residence_time_s[0]
    assert stretched.zone_residence_time_s[1] == base.zone_residence_time_s[1]


def test_multizone_evaluation_returns_required_keys_and_non_negative(tmp_path: Path) -> None:
    design = MultiZoneDesign(
        zones=[
            ZoneDesign(temp_c=910.0, length_m=0.9, diameter_m=0.09, insulation_factor=1.1),
            ZoneDesign(temp_c=980.0, length_m=0.9, diameter_m=0.09, insulation_factor=1.0),
        ],
        methane_kg_per_hr=95.0,
        pressure_atm=1.2,
        dilution_frac=0.78,
        carbon_removal_eff=0.55,
    )
    metrics = evaluate_multizone_surrogate(
        design,
        surrogate_params_path=_fallback_params_path(tmp_path),
    )

    required = {
        "conversion",
        "h2_rate",
        "fouling_risk_index",
        "dp_total_kpa",
        "q_required_kw",
        "q_loss_kw",
        "constraint_violations",
        "zone_1_conv_mean",
        "zone_1_conv_ci_lower",
        "zone_1_is_ood",
    }
    assert required.issubset(set(metrics.keys()))
    assert float(metrics["conversion"]) >= 0.0
    assert float(metrics["h2_rate"]) >= 0.0
    assert float(metrics["fouling_risk_index"]) >= 0.0
    assert float(metrics["dp_total_kpa"]) >= 0.0
    assert float(metrics["q_required_kw"]) >= 0.0
    assert float(metrics["q_loss_kw"]) >= 0.0
    assert isinstance(metrics["constraint_violations"], list)


def test_multizone_constraints_flag_tmax_dp_and_power(tmp_path: Path) -> None:
    params_path = _fallback_params_path(tmp_path)

    tmax_design = MultiZoneDesign(
        zones=[
            ZoneDesign(temp_c=1180.0, length_m=0.8, diameter_m=0.08, insulation_factor=1.0),
            ZoneDesign(temp_c=990.0, length_m=0.8, diameter_m=0.08, insulation_factor=1.0),
        ],
        methane_kg_per_hr=90.0,
        pressure_atm=1.2,
        dilution_frac=0.75,
        carbon_removal_eff=0.50,
        material_tmax_c=1100.0,
        dp_max_kpa=40.0,
        power_max_kw=2000.0,
    )
    tmax_metrics = evaluate_multizone_surrogate(tmax_design, surrogate_params_path=params_path)
    assert "material_temperature_limit_exceeded" in tmax_metrics["constraint_violations"]

    dp_design = MultiZoneDesign(
        zones=[
            ZoneDesign(temp_c=960.0, length_m=1.6, diameter_m=0.04, insulation_factor=1.0),
            ZoneDesign(temp_c=1020.0, length_m=1.6, diameter_m=0.04, insulation_factor=1.0),
        ],
        methane_kg_per_hr=180.0,
        pressure_atm=1.5,
        dilution_frac=0.60,
        carbon_removal_eff=0.30,
        material_tmax_c=1200.0,
        dp_max_kpa=2.0,
        power_max_kw=3000.0,
    )
    dp_metrics = evaluate_multizone_surrogate(dp_design, surrogate_params_path=params_path)
    assert "pressure_drop_limit_exceeded" in dp_metrics["constraint_violations"]

    power_design = MultiZoneDesign(
        zones=[
            ZoneDesign(temp_c=1060.0, length_m=1.2, diameter_m=0.09, insulation_factor=0.7),
            ZoneDesign(temp_c=1080.0, length_m=1.2, diameter_m=0.09, insulation_factor=0.7),
        ],
        methane_kg_per_hr=240.0,
        pressure_atm=1.3,
        dilution_frac=0.60,
        carbon_removal_eff=0.10,
        material_tmax_c=1300.0,
        dp_max_kpa=50.0,
        power_max_kw=20.0,
    )
    power_metrics = evaluate_multizone_surrogate(power_design, surrogate_params_path=params_path)
    assert "power_limit_exceeded" in power_metrics["constraint_violations"]


def test_multizone_optimizer_returns_top_k_and_sorts_feasible(tmp_path: Path) -> None:
    bounds = MultiZoneBounds(
        zone_temp_c=(880.0, 1020.0),
        zone_length_m=(0.30, 1.20),
        zone_diameter_m=(0.08, 0.16),
        zone_insulation_factor=(0.7, 1.6),
        methane_kg_per_hr=(50.0, 120.0),
        pressure_atm=(1.0, 1.8),
        dilution_frac=(0.65, 0.85),
        carbon_removal_eff=(0.2, 0.8),
        default_diameter_m=(0.08, 0.14),
        total_length_m_max=2.4,
        ambient_temp_c=25.0,
        material_tmax_c=1200.0,
        dp_max_kpa=80.0,
        power_max_kw=3000.0,
    )
    proposals = propose_multizone_designs(
        top_k=5,
        zones=2,
        generations=4,
        population_size=48,
        seed=9,
        bounds=bounds,
        surrogate_params_path=_fallback_params_path(tmp_path),
    )

    assert len(proposals) == 5
    violation_counts = [item.violation_count for item in proposals]
    assert violation_counts == sorted(violation_counts)
    if any(count == 0 for count in violation_counts):
        assert violation_counts[0] == 0
