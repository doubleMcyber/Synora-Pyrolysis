from __future__ import annotations

import pytest

from synora.generative.constraints import (
    CP_MIX_J_PER_KGK,
    _mean_molecular_weight_kg_per_mol,
    required_power_kw,
    sensible_heat_kw,
)
from synora.generative.multizone import MultiZoneDesign, ZoneDesign


def _design() -> MultiZoneDesign:
    return MultiZoneDesign(
        zones=[
            ZoneDesign(temp_c=950.0, length_m=0.9, diameter_m=0.09, insulation_factor=1.0),
            ZoneDesign(temp_c=1000.0, length_m=0.9, diameter_m=0.09, insulation_factor=1.0),
        ],
        methane_kg_per_hr=100.0,
        pressure_atm=1.3,
        dilution_frac=0.80,
        carbon_removal_eff=0.60,
    )


def test_required_power_is_sum_of_components() -> None:
    design = _design()
    power = required_power_kw(design, overall_conversion=0.5)
    assert power["q_required_kw"] == pytest.approx(
        power["q_loss_kw"] + power["q_reaction_kw"] + power["q_sensible_kw"]
    )


def test_sensible_heat_uses_mole_weighted_mass_flow() -> None:
    # Guards the sensible_heat_kw fix: total mass flow must come from the mole-weighted
    # mean MW (dilution_frac is a MOLE fraction), not methane_mass / (1 - dilution_frac),
    # which treats it as a mass fraction and undercounts the heavy argon diluent ~2x.
    design = _design()
    avg_temp_c = sum(design.zone_temperatures_c) / len(design.zone_temperatures_c)
    correct_mass_flow = design.total_molar_flow_mol_per_s * _mean_molecular_weight_kg_per_mol(
        design.dilution_frac
    )
    expected = correct_mass_flow * CP_MIX_J_PER_KGK * (avg_temp_c - design.ambient_temp_c) / 1000.0
    assert sensible_heat_kw(design) == pytest.approx(expected)

    # The buggy mass-fraction basis would be materially smaller at high dilution.
    buggy_mass_flow = (design.methane_kg_per_hr / 3600.0) / (1.0 - design.dilution_frac)
    buggy = buggy_mass_flow * CP_MIX_J_PER_KGK * (avg_temp_c - design.ambient_temp_c) / 1000.0
    assert sensible_heat_kw(design) > 1.5 * buggy
