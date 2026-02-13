from __future__ import annotations

from pathlib import Path

import numpy as np

from synora.calibration.surrogate_fit import calibrated_predict
from synora.economics.lcoh import EconInputs, hourly_economics
from synora.generative.design_space import ReactorDesign

CH4_MW_KG_PER_MOL = 0.016043
H2_MW_KG_PER_MOL = 0.002016
CARBON_FROM_CH4_MASS_RATIO = 12.011 / 16.043
AR_MW_KG_PER_MOL = 0.039948
R_UNIVERSAL = 8.314462618


DEFAULT_CONSTRAINTS: dict[str, float] = {
    "min_conversion": 0.01,
    "max_fouling_risk_index": 0.40,
    "max_pressure_drop_proxy": 0.35,
    "max_heat_loss_proxy": 0.40,
    "min_residence_time_s": 0.08,
    "max_residence_time_s": 8.0,
}


def _pressure_drop_proxy(design: ReactorDesign) -> float:
    # Darcy-like pressure loss proxy using ideal-gas density + roughness adjusted friction factor.
    temperature_k = design.temp_c + 273.15
    pressure_pa = design.pressure_atm * 101325.0
    methane_fraction = max(1e-6, 1.0 - design.dilution_frac)
    mean_mw = (methane_fraction * CH4_MW_KG_PER_MOL) + (design.dilution_frac * AR_MW_KG_PER_MOL)
    density = (pressure_pa * mean_mw) / max(R_UNIVERSAL * temperature_k, 1e-9)
    velocity = design.volumetric_flow_m3_per_s / max(design.cross_section_area_m2, 1e-9)

    roughness_ratio = (design.roughness_mm / 1000.0) / max(design.diameter_m, 1e-9)
    friction_factor = 0.015 + (0.3 * roughness_ratio)
    delta_p = friction_factor * (design.length_m / max(design.diameter_m, 1e-9))
    delta_p *= 0.5 * density * (velocity**2)
    return float(max(0.0, delta_p / max(pressure_pa, 1e-9)))


def _heat_loss_proxy(design: ReactorDesign) -> float:
    # Lumped conductive+radiative proxy; normalized by methane chemical-energy throughput.
    temperature_k = design.temp_c + 273.15
    delta_t = max(0.0, temperature_k - 298.15)
    ua_proxy = 6.0 * design.surface_area_m2 * design.emissivity / max(design.wall_thickness_m, 1e-5)
    heat_loss_watts = ua_proxy * delta_t
    methane_kg_per_s = design.methane_kg_per_hr / 3600.0
    methane_energy_watts = methane_kg_per_s * 50_000_000.0
    return float(max(0.0, heat_loss_watts / max(methane_energy_watts, 1e-9)))


def evaluate_design_surrogate(
    design: ReactorDesign,
    surrogate_params_path: str | Path | None = None,
    *,
    econ_inputs: EconInputs | None = None,
    constraints: dict[str, float] | None = None,
) -> dict[str, float | list[str]]:
    econ = econ_inputs or EconInputs()
    constraint_cfg = {**DEFAULT_CONSTRAINTS, **(constraints or {})}

    pred = calibrated_predict(
        temperature_c=design.temp_c,
        residence_time_s=design.residence_time_s,
        params_path=surrogate_params_path,
    )

    pressure_factor = np.clip(1.0 + (0.03 * (design.pressure_atm - 1.0)), 0.85, 1.15)
    conversion = float(np.clip(pred["methane_conversion"] * pressure_factor, 0.0, 0.99))
    h2_yield = float(np.clip(pred["h2_yield_mol_per_mol_ch4"] * pressure_factor, 0.0, 2.0))
    carbon_proxy = float(max(0.0, pred["carbon_formation_index"]))

    methane_mol_per_hr = design.methane_kg_per_hr / CH4_MW_KG_PER_MOL
    h2_rate_kg_per_hr = methane_mol_per_hr * h2_yield * H2_MW_KG_PER_MOL
    carbon_generation_rate = (
        design.methane_kg_per_hr
        * conversion
        * CARBON_FROM_CH4_MASS_RATIO
        * design.effective_carbon_release_factor
    )
    fouling_risk_index = carbon_proxy * design.effective_carbon_release_factor

    pressure_drop = _pressure_drop_proxy(design)
    heat_loss = _heat_loss_proxy(design)

    economics = hourly_economics(
        h2_kg_per_hr=max(0.0, h2_rate_kg_per_hr),
        carbon_kg_per_hr=max(0.0, carbon_generation_rate),
        methane_kg_per_hr=design.methane_kg_per_hr,
        econ_inputs=econ,
    )

    violations: list[str] = []
    if conversion < constraint_cfg["min_conversion"]:
        violations.append("conversion_below_min")
    if fouling_risk_index > constraint_cfg["max_fouling_risk_index"]:
        violations.append("fouling_risk_above_max")
    if pressure_drop > constraint_cfg["max_pressure_drop_proxy"]:
        violations.append("pressure_drop_above_max")
    if heat_loss > constraint_cfg["max_heat_loss_proxy"]:
        violations.append("heat_loss_above_max")
    if design.residence_time_s < constraint_cfg["min_residence_time_s"]:
        violations.append("residence_time_below_min")
    if design.residence_time_s > constraint_cfg["max_residence_time_s"]:
        violations.append("residence_time_above_max")

    return {
        "conversion": conversion,
        "h2_rate": float(max(0.0, h2_rate_kg_per_hr)),
        "fouling_risk_index": float(max(0.0, fouling_risk_index)),
        "pressure_drop_proxy": pressure_drop,
        "heat_loss_proxy": heat_loss,
        "profit_per_hr": float(economics["profit_per_hr"]),
        "lcoh_usd_per_kg": float(economics["lcoh_usd_per_kg"]),
        "carbon_generation_rate": float(max(0.0, carbon_generation_rate)),
        "constraint_violations": violations,
        "constraint_violation_count": float(len(violations)),
    }


def scalarize_metrics(metrics: dict[str, float | list[str]]) -> float:
    profit = float(metrics["profit_per_hr"])
    conversion = float(metrics["conversion"])
    fouling = float(metrics["fouling_risk_index"])
    pressure_drop = float(metrics["pressure_drop_proxy"])
    heat_loss = float(metrics["heat_loss_proxy"])
    violation_count = float(metrics["constraint_violation_count"])
    return (
        profit
        + (35.0 * conversion)
        - (18.0 * fouling)
        - (8.0 * pressure_drop)
        - (6.0 * heat_loss)
        - (60.0 * violation_count)
    )


__all__ = ["evaluate_design_surrogate", "scalarize_metrics", "DEFAULT_CONSTRAINTS"]
