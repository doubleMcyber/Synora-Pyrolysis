from __future__ import annotations

from pathlib import Path

import numpy as np

from synora.calibration.surrogate_fit import calibrated_predict
from synora.economics.lcoh import EconInputs, hourly_economics
from synora.generative.constraints import evaluate_thermal_dp_constraints
from synora.generative.design_space import ReactorDesign
from synora.generative.multizone import MultiZoneDesign

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
) -> dict[str, float | bool | list[str]]:
    econ = econ_inputs or EconInputs()
    constraint_cfg = {**DEFAULT_CONSTRAINTS, **(constraints or {})}

    pred = calibrated_predict(
        temperature_c=design.temp_c,
        residence_time_s=design.residence_time_s,
        params_path=surrogate_params_path,
    )

    conversion_mean = float(pred["methane_conversion"])
    conversion_std = float(pred.get("methane_conversion_std", 0.0))
    h2_yield_mean = float(pred["h2_yield_mol_per_mol_ch4"])
    h2_yield_std = float(pred.get("h2_yield_mol_per_mol_ch4_std", 0.0))
    carbon_proxy_mean = float(pred["carbon_formation_index"])
    carbon_proxy_std = float(pred.get("carbon_formation_index_std", 0.0))
    is_ood = bool(pred.get("is_out_of_distribution", False))
    ood_score = float(pred.get("ood_score", 0.0))

    pressure_factor = np.clip(1.0 + (0.03 * (design.pressure_atm - 1.0)), 0.85, 1.15)
    conversion = float(np.clip(conversion_mean * pressure_factor, 0.0, 0.99))
    conversion_std = float(max(0.0, conversion_std * pressure_factor))
    h2_yield = float(np.clip(h2_yield_mean * pressure_factor, 0.0, 2.0))
    h2_yield_std = float(max(0.0, h2_yield_std * pressure_factor))
    carbon_proxy = float(max(0.0, carbon_proxy_mean))
    carbon_proxy_std = float(max(0.0, carbon_proxy_std))

    methane_mol_per_hr = design.methane_kg_per_hr / CH4_MW_KG_PER_MOL
    h2_rate_kg_per_hr = methane_mol_per_hr * h2_yield * H2_MW_KG_PER_MOL
    h2_rate_std_kg_per_hr = methane_mol_per_hr * h2_yield_std * H2_MW_KG_PER_MOL
    carbon_generation_rate = (
        design.methane_kg_per_hr
        * conversion
        * CARBON_FROM_CH4_MASS_RATIO
        * design.effective_carbon_release_factor
    )
    fouling_risk_index = carbon_proxy * design.effective_carbon_release_factor
    fouling_risk_std = carbon_proxy_std * design.effective_carbon_release_factor

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
    if is_ood:
        violations.append("out_of_distribution")

    return {
        "conversion": conversion,
        "conversion_std": conversion_std,
        "conversion_ci_lower": max(0.0, conversion - (2.0 * conversion_std)),
        "conversion_ci_upper": min(1.0, conversion + (2.0 * conversion_std)),
        "h2_yield_mol_per_mol_ch4": h2_yield,
        "h2_yield_std": h2_yield_std,
        "h2_yield_ci_lower": max(0.0, h2_yield - (2.0 * h2_yield_std)),
        "h2_yield_ci_upper": min(2.0, h2_yield + (2.0 * h2_yield_std)),
        "h2_rate": float(max(0.0, h2_rate_kg_per_hr)),
        "h2_rate_std": float(max(0.0, h2_rate_std_kg_per_hr)),
        "fouling_risk_index": float(max(0.0, fouling_risk_index)),
        "fouling_risk_index_std": float(max(0.0, fouling_risk_std)),
        "fouling_risk_ci_lower": max(0.0, fouling_risk_index - (2.0 * fouling_risk_std)),
        "fouling_risk_ci_upper": max(0.0, fouling_risk_index + (2.0 * fouling_risk_std)),
        "pressure_drop_proxy": pressure_drop,
        "heat_loss_proxy": heat_loss,
        "profit_per_hr": float(economics["profit_per_hr"]),
        "lcoh_usd_per_kg": float(economics["lcoh_usd_per_kg"]),
        "carbon_generation_rate": float(max(0.0, carbon_generation_rate)),
        "is_out_of_distribution": is_ood,
        "ood_score": ood_score,
        "constraint_violations": violations,
        "constraint_violation_count": float(len(violations)),
    }


def evaluate_multizone_surrogate(
    design: MultiZoneDesign,
    surrogate_params_path: str | Path | None = None,
    *,
    econ_inputs: EconInputs | None = None,
    constraints: dict[str, float] | None = None,
) -> dict[str, float | bool | list[str]]:
    econ = econ_inputs or EconInputs()
    constraint_cfg = {**DEFAULT_CONSTRAINTS, **(constraints or {})}

    zone_taus = design.zone_residence_time_s
    total_length = max(design.total_length_m, 1e-9)
    zone_weights = [zone.length_m / total_length for zone in design.zones]

    remaining_feed = 1.0
    conversion_variance = 0.0
    h2_yield_total = 0.0
    h2_yield_variance = 0.0
    fouling_total = 0.0
    fouling_variance = 0.0
    zone_payloads: list[dict[str, float | bool]] = []
    zone_ood_scores: list[float] = []
    zone_ood_flags: list[bool] = []

    for zone_idx, (zone, tau_s, zone_weight) in enumerate(
        zip(design.zones, zone_taus, zone_weights, strict=True)
    ):
        pred = calibrated_predict(
            temperature_c=zone.temp_c,
            residence_time_s=tau_s,
            params_path=surrogate_params_path,
        )

        conversion_zone = float(np.clip(pred["methane_conversion"], 0.0, 0.99))
        conversion_zone_std = float(max(0.0, pred.get("methane_conversion_std", 0.0)))
        h2_yield_zone = float(np.clip(pred["h2_yield_mol_per_mol_ch4"], 0.0, 2.0))
        h2_yield_zone_std = float(max(0.0, pred.get("h2_yield_mol_per_mol_ch4_std", 0.0)))
        carbon_proxy_zone = float(max(0.0, pred["carbon_formation_index"]))
        carbon_proxy_zone_std = float(max(0.0, pred.get("carbon_formation_index_std", 0.0)))

        incremental_conversion = remaining_feed * conversion_zone
        incremental_conversion_std = remaining_feed * conversion_zone_std
        conversion_variance += incremental_conversion_std**2

        # Approximate zone-wise H2 contribution by weighting each zone's
        # conversion with local H2-per-converted-CH4 factor.
        h2_per_converted = float(np.clip(h2_yield_zone / max(conversion_zone, 1e-6), 0.0, 2.2))
        h2_per_converted_std = float(
            np.clip(h2_yield_zone_std / max(conversion_zone, 1e-6), 0.0, 2.2)
        )
        zone_h2_contribution = incremental_conversion * h2_per_converted
        zone_h2_contribution_std = incremental_conversion * h2_per_converted_std
        h2_yield_total += zone_h2_contribution
        h2_yield_variance += zone_h2_contribution_std**2

        zone_fouling = (
            carbon_proxy_zone * incremental_conversion * design.effective_carbon_release_factor
        )
        zone_fouling_std = (
            carbon_proxy_zone_std * incremental_conversion * design.effective_carbon_release_factor
        )
        fouling_total += zone_fouling * zone_weight
        fouling_variance += (zone_fouling_std * zone_weight) ** 2

        is_zone_ood = bool(pred.get("is_out_of_distribution", False))
        zone_ood_score = float(pred.get("ood_score", 0.0))
        zone_ood_flags.append(is_zone_ood)
        zone_ood_scores.append(zone_ood_score)

        zone_payloads.append(
            {
                "zone_index": float(zone_idx + 1),
                "temp_c": float(zone.temp_c),
                "tau_s": float(tau_s),
                "conversion": conversion_zone,
                "conversion_std": conversion_zone_std,
                "conversion_ci_lower": max(0.0, conversion_zone - (2.0 * conversion_zone_std)),
                "conversion_ci_upper": min(1.0, conversion_zone + (2.0 * conversion_zone_std)),
                "h2_yield_mol_per_mol_ch4": h2_yield_zone,
                "h2_yield_std": h2_yield_zone_std,
                "fouling_risk_index": zone_fouling,
                "fouling_risk_std": zone_fouling_std,
                "is_ood": bool(is_zone_ood),
                "ood_score": zone_ood_score,
            }
        )

        remaining_feed *= max(0.0, 1.0 - conversion_zone)

    # Sequential approximation: total conversion = 1 - product(1 - zone_conversion_i_effective)
    conversion_total = float(np.clip(1.0 - remaining_feed, 0.0, 0.995))
    conversion_std = float(np.sqrt(max(0.0, conversion_variance)))
    h2_yield_total = float(np.clip(h2_yield_total, 0.0, 2.0))
    h2_yield_std = float(max(0.0, np.sqrt(max(0.0, h2_yield_variance))))
    fouling_risk_total = float(max(0.0, fouling_total))
    fouling_risk_std = float(max(0.0, np.sqrt(max(0.0, fouling_variance))))

    pressure_factor = np.clip(1.0 + (0.03 * (design.pressure_atm - 1.0)), 0.85, 1.15)
    conversion_total = float(np.clip(conversion_total * pressure_factor, 0.0, 0.995))
    conversion_std = float(max(0.0, conversion_std * pressure_factor))
    h2_yield_total = float(np.clip(h2_yield_total * pressure_factor, 0.0, 2.0))
    h2_yield_std = float(max(0.0, h2_yield_std * pressure_factor))

    methane_mol_per_hr = design.methane_kg_per_hr / CH4_MW_KG_PER_MOL
    h2_rate_kg_per_hr = methane_mol_per_hr * h2_yield_total * H2_MW_KG_PER_MOL
    h2_rate_std_kg_per_hr = methane_mol_per_hr * h2_yield_std * H2_MW_KG_PER_MOL

    carbon_generation_rate = (
        design.methane_kg_per_hr
        * conversion_total
        * CARBON_FROM_CH4_MASS_RATIO
        * design.effective_carbon_release_factor
    )
    is_ood = any(zone_ood_flags)
    ood_score = max(zone_ood_scores) if zone_ood_scores else 0.0

    thermal_dp = evaluate_thermal_dp_constraints(
        design,
        overall_conversion=conversion_total,
        is_ood=is_ood,
    )
    dp_total_kpa = float(thermal_dp["dp_total_kpa"])
    q_loss_kw = float(thermal_dp["q_loss_kw"])
    q_required_kw = float(thermal_dp["q_required_kw"])

    pressure_drop_proxy = dp_total_kpa / max(design.dp_max_kpa, 1e-9)
    heat_loss_proxy = q_loss_kw / max(design.power_max_kw, 1e-9)

    economics = hourly_economics(
        h2_kg_per_hr=max(0.0, h2_rate_kg_per_hr),
        carbon_kg_per_hr=max(0.0, carbon_generation_rate),
        methane_kg_per_hr=design.methane_kg_per_hr,
        econ_inputs=econ,
    )

    violations = list(thermal_dp["constraint_violations"])
    if conversion_total < constraint_cfg["min_conversion"]:
        violations.append("conversion_below_min")
    if fouling_risk_total > constraint_cfg["max_fouling_risk_index"]:
        violations.append("fouling_risk_above_max")
    if pressure_drop_proxy > constraint_cfg["max_pressure_drop_proxy"]:
        violations.append("pressure_drop_above_max")
    if heat_loss_proxy > constraint_cfg["max_heat_loss_proxy"]:
        violations.append("heat_loss_above_max")
    if any(tau < constraint_cfg["min_residence_time_s"] for tau in zone_taus):
        violations.append("residence_time_below_min")
    if any(tau > constraint_cfg["max_residence_time_s"] for tau in zone_taus):
        violations.append("residence_time_above_max")

    metrics: dict[str, float | bool | list[str]] = {
        "conversion": conversion_total,
        "conversion_std": conversion_std,
        "conversion_ci_lower": max(0.0, conversion_total - (2.0 * conversion_std)),
        "conversion_ci_upper": min(1.0, conversion_total + (2.0 * conversion_std)),
        "h2_yield_mol_per_mol_ch4": h2_yield_total,
        "h2_yield_std": h2_yield_std,
        "h2_yield_ci_lower": max(0.0, h2_yield_total - (2.0 * h2_yield_std)),
        "h2_yield_ci_upper": min(2.0, h2_yield_total + (2.0 * h2_yield_std)),
        "h2_rate": float(max(0.0, h2_rate_kg_per_hr)),
        "h2_rate_std": float(max(0.0, h2_rate_std_kg_per_hr)),
        "fouling_risk_index": fouling_risk_total,
        "fouling_risk_index_std": fouling_risk_std,
        "fouling_risk_ci_lower": max(0.0, fouling_risk_total - (2.0 * fouling_risk_std)),
        "fouling_risk_ci_upper": max(0.0, fouling_risk_total + (2.0 * fouling_risk_std)),
        "pressure_drop_proxy": float(max(0.0, pressure_drop_proxy)),
        "heat_loss_proxy": float(max(0.0, heat_loss_proxy)),
        "dp_total_kpa": dp_total_kpa,
        "q_loss_kw": q_loss_kw,
        "q_required_kw": q_required_kw,
        "profit_per_hr": float(economics["profit_per_hr"]),
        "lcoh_usd_per_kg": float(economics["lcoh_usd_per_kg"]),
        "carbon_generation_rate": float(max(0.0, carbon_generation_rate)),
        "is_out_of_distribution": bool(is_ood),
        "ood_score": float(ood_score),
        "constraint_violations": sorted(set(violations)),
        "constraint_violation_count": float(len(set(violations))),
        "zone_count": float(design.zone_count),
    }

    for idx, zone_metric in enumerate(zone_payloads, start=1):
        metrics[f"zone_{idx}_temp_c"] = float(zone_metric["temp_c"])
        metrics[f"zone_{idx}_tau_s"] = float(zone_metric["tau_s"])
        metrics[f"zone_{idx}_conv_mean"] = float(zone_metric["conversion"])
        metrics[f"zone_{idx}_conv_std"] = float(zone_metric["conversion_std"])
        metrics[f"zone_{idx}_conv_ci_lower"] = float(zone_metric["conversion_ci_lower"])
        metrics[f"zone_{idx}_conv_ci_upper"] = float(zone_metric["conversion_ci_upper"])
        metrics[f"zone_{idx}_h2_yield"] = float(zone_metric["h2_yield_mol_per_mol_ch4"])
        metrics[f"zone_{idx}_fouling_risk"] = float(zone_metric["fouling_risk_index"])
        metrics[f"zone_{idx}_is_ood"] = bool(zone_metric["is_ood"])
        metrics[f"zone_{idx}_ood_score"] = float(zone_metric["ood_score"])

    return metrics


def scalarize_metrics(metrics: dict[str, float | bool | list[str]]) -> float:
    profit = float(metrics["profit_per_hr"])
    conversion = float(metrics["conversion"])
    fouling = float(metrics["fouling_risk_index"])
    pressure_drop = float(metrics["pressure_drop_proxy"])
    heat_loss = float(metrics["heat_loss_proxy"])
    ood_score = float(metrics.get("ood_score", 0.0))
    violation_count = float(metrics["constraint_violation_count"])
    return (
        profit
        + (35.0 * conversion)
        - (18.0 * fouling)
        - (8.0 * pressure_drop)
        - (6.0 * heat_loss)
        - (12.0 * ood_score)
        - (60.0 * violation_count)
    )


__all__ = [
    "evaluate_design_surrogate",
    "evaluate_multizone_surrogate",
    "scalarize_metrics",
    "DEFAULT_CONSTRAINTS",
]
