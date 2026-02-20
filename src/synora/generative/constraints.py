from __future__ import annotations

import numpy as np

from synora.generative.design_space import CH4_MW_KG_PER_MOL, GAS_CONSTANT_J_PER_MOLK
from synora.generative.multizone import MultiZoneDesign

AR_MW_KG_PER_MOL = 0.039948
DP_FRICTION_FACTOR = 0.035
U_BASE_W_PER_M2K = 38.0
CP_MIX_J_PER_KGK = 1800.0
DELTA_H_PYROLYSIS_J_PER_MOL = 75_000.0


def _mean_molecular_weight_kg_per_mol(dilution_frac: float) -> float:
    methane_fraction = max(1e-6, 1.0 - dilution_frac)
    return (methane_fraction * CH4_MW_KG_PER_MOL) + (dilution_frac * AR_MW_KG_PER_MOL)


def pressure_drop_total_kpa(design: MultiZoneDesign) -> float:
    mean_mw = _mean_molecular_weight_kg_per_mol(design.dilution_frac)
    dp_total_pa = 0.0
    for idx, zone in enumerate(design.zones):
        pressure_pa = design.zone_pressures_atm[idx] * 101325.0
        temperature_k = zone.temp_c + 273.15
        density = (pressure_pa * mean_mw) / max(GAS_CONSTANT_J_PER_MOLK * temperature_k, 1e-9)
        velocity = design.zone_volumetric_flow_m3_per_s[idx] / max(
            design.zone_cross_section_area_m2[idx], 1e-9
        )
        diameter = max(design.zone_diameters_m[idx], 1e-9)
        zone_dp = DP_FRICTION_FACTOR * (zone.length_m / diameter) * 0.5 * density * (velocity**2)
        dp_total_pa += max(0.0, float(zone_dp))
    return float(dp_total_pa / 1000.0)


def heat_loss_total_kw(design: MultiZoneDesign) -> float:
    q_loss_kw = 0.0
    for idx, zone in enumerate(design.zones):
        area_m2 = design.zone_surface_area_m2[idx]
        u_eff = U_BASE_W_PER_M2K / max(zone.insulation_factor, 0.5)
        delta_t = max(0.0, zone.temp_c - design.ambient_temp_c)
        q_loss_kw += (u_eff * area_m2 * delta_t) / 1000.0
    return float(max(0.0, q_loss_kw))


def reaction_heat_kw(design: MultiZoneDesign, overall_conversion: float) -> float:
    methane_mol_per_s = design.methane_molar_flow_mol_per_s
    return float(
        max(
            0.0,
            methane_mol_per_s * max(0.0, overall_conversion) * DELTA_H_PYROLYSIS_J_PER_MOL / 1000.0,
        )
    )


def sensible_heat_kw(design: MultiZoneDesign) -> float:
    methane_kg_per_s = design.methane_kg_per_hr / 3600.0
    total_mass_flow_kg_per_s = methane_kg_per_s / max(1.0 - design.dilution_frac, 1e-6)
    avg_temp_c = float(np.mean(design.zone_temperatures_c))
    delta_t = max(0.0, avg_temp_c - design.ambient_temp_c)
    return float(max(0.0, total_mass_flow_kg_per_s * CP_MIX_J_PER_KGK * delta_t / 1000.0))


def required_power_kw(design: MultiZoneDesign, overall_conversion: float) -> dict[str, float]:
    q_loss = heat_loss_total_kw(design)
    q_reaction = reaction_heat_kw(design, overall_conversion)
    q_sensible = sensible_heat_kw(design)
    q_required = q_reaction + q_sensible + q_loss
    return {
        "q_loss_kw": float(q_loss),
        "q_reaction_kw": float(q_reaction),
        "q_sensible_kw": float(q_sensible),
        "q_required_kw": float(q_required),
    }


def evaluate_thermal_dp_constraints(
    design: MultiZoneDesign,
    *,
    overall_conversion: float,
    is_ood: bool,
) -> dict[str, float | list[str]]:
    dp_total_kpa = pressure_drop_total_kpa(design)
    power = required_power_kw(design, overall_conversion)
    violations: list[str] = []

    if max(design.zone_temperatures_c) > design.material_tmax_c:
        violations.append("material_temperature_limit_exceeded")
    if dp_total_kpa > design.dp_max_kpa:
        violations.append("pressure_drop_limit_exceeded")
    if float(power["q_required_kw"]) > design.power_max_kw:
        violations.append("power_limit_exceeded")
    if is_ood:
        violations.append("out_of_distribution")

    return {
        "dp_total_kpa": float(dp_total_kpa),
        "q_loss_kw": float(power["q_loss_kw"]),
        "q_reaction_kw": float(power["q_reaction_kw"]),
        "q_sensible_kw": float(power["q_sensible_kw"]),
        "q_required_kw": float(power["q_required_kw"]),
        "constraint_violations": violations,
    }


__all__ = [
    "pressure_drop_total_kpa",
    "heat_loss_total_kw",
    "reaction_heat_kw",
    "sensible_heat_kw",
    "required_power_kw",
    "evaluate_thermal_dp_constraints",
]
