from __future__ import annotations

from dataclasses import dataclass

from synora.calibration.surrogate_fit import calibrated_predict

CH4_MW_KG_PER_MOL = 0.016043
H2_MW_KG_PER_MOL = 0.002016
CARBON_FROM_CH4_MASS_RATIO = 12.011 / 16.043


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


@dataclass(frozen=True)
class ReactorInputs:
    methane_kg_per_hr: float
    temp_c: float
    residence_time_s: float = 1.0
    max_conversion: float = 0.90
    surrogate_params_path: str | None = None

    def __post_init__(self) -> None:
        if self.methane_kg_per_hr < 0:
            msg = "methane_kg_per_hr must be non-negative"
            raise ValueError(msg)
        if self.residence_time_s <= 0:
            msg = "residence_time_s must be positive"
            raise ValueError(msg)
        if not 0 <= self.max_conversion <= 1:
            msg = "max_conversion must be between 0 and 1"
            raise ValueError(msg)


@dataclass(frozen=True)
class ReactorState:
    conversion: float = 0.0
    health: float = 1.0
    hours_operated: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.conversion <= 1:
            msg = "conversion must be between 0 and 1"
            raise ValueError(msg)
        if not 0 <= self.health <= 1:
            msg = "health must be between 0 and 1"
            raise ValueError(msg)
        if self.hours_operated < 0:
            msg = "hours_operated must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class ReactorOutputs:
    conversion: float
    health: float
    h2_kg_per_hr: float
    carbon_kg_per_hr: float
    carbon_rate_kg_per_hr: float
    fouling_rate_per_hr: float
    carbon_formation_index: float
    h2_yield_mol_per_mol_ch4: float
    unreacted_methane_kg_per_hr: float

    def __post_init__(self) -> None:
        if self.h2_kg_per_hr < 0:
            msg = "h2_kg_per_hr must be non-negative"
            raise ValueError(msg)
        if self.carbon_kg_per_hr < 0:
            msg = "carbon_kg_per_hr must be non-negative"
            raise ValueError(msg)
        if self.unreacted_methane_kg_per_hr < 0:
            msg = "unreacted_methane_kg_per_hr must be non-negative"
            raise ValueError(msg)
        if self.carbon_rate_kg_per_hr < 0:
            msg = "carbon_rate_kg_per_hr must be non-negative"
            raise ValueError(msg)
        if self.fouling_rate_per_hr < 0:
            msg = "fouling_rate_per_hr must be non-negative"
            raise ValueError(msg)
        if self.carbon_formation_index < 0:
            msg = "carbon_formation_index must be non-negative"
            raise ValueError(msg)
        if self.h2_yield_mol_per_mol_ch4 < 0:
            msg = "h2_yield_mol_per_mol_ch4 must be non-negative"
            raise ValueError(msg)
        if not 0 <= self.conversion <= 1:
            msg = "conversion must be between 0 and 1"
            raise ValueError(msg)
        if not 0 <= self.health <= 1:
            msg = "health must be between 0 and 1"
            raise ValueError(msg)


def simulate_step(
    inputs: ReactorInputs,
    state: ReactorState,
    dt_hr: float = 1.0,
) -> tuple[ReactorState, ReactorOutputs]:
    """Advance one deterministic twin step using calibrated surrogate predictions."""
    if dt_hr <= 0:
        msg = "dt_hr must be positive"
        raise ValueError(msg)

    surrogate = calibrated_predict(
        temperature_c=inputs.temp_c,
        residence_time_s=inputs.residence_time_s,
        params_path=inputs.surrogate_params_path,
    )
    base_conversion = _clamp(float(surrogate["methane_conversion"]), 0.0, inputs.max_conversion)
    h2_yield = _clamp(float(surrogate["h2_yield_mol_per_mol_ch4"]), 0.0, 2.0)
    carbon_proxy = max(0.0, float(surrogate["carbon_formation_index"]))

    # Health multiplies effective performance so conversion drops as the asset fouls.
    effective_conversion = _clamp(base_conversion * state.health, 0.0, inputs.max_conversion)

    methane_reacted = inputs.methane_kg_per_hr * effective_conversion
    carbon_kg_per_hr = methane_reacted * CARBON_FROM_CH4_MASS_RATIO

    methane_mol_per_hr = 0.0
    if CH4_MW_KG_PER_MOL > 0:
        methane_mol_per_hr = inputs.methane_kg_per_hr / CH4_MW_KG_PER_MOL
    h2_kg_per_hr = methane_mol_per_hr * h2_yield * state.health * H2_MW_KG_PER_MOL
    h2_kg_per_hr = min(h2_kg_per_hr, methane_reacted * 0.25)

    carbon_rate_kg_per_hr = carbon_kg_per_hr * carbon_proxy
    feed_for_scaling = max(inputs.methane_kg_per_hr, 1e-9)
    fouling_rate_per_hr = (0.05 * carbon_proxy) + (
        0.15 * (carbon_rate_kg_per_hr / feed_for_scaling)
    )

    unreacted_methane_kg_per_hr = max(0.0, inputs.methane_kg_per_hr - methane_reacted)

    next_health = _clamp(state.health - (fouling_rate_per_hr * dt_hr), 0.0, 1.0)

    next_state = ReactorState(
        conversion=effective_conversion,
        health=next_health,
        hours_operated=state.hours_operated + dt_hr,
    )
    outputs = ReactorOutputs(
        conversion=effective_conversion,
        health=next_health,
        h2_kg_per_hr=h2_kg_per_hr,
        carbon_kg_per_hr=carbon_kg_per_hr,
        carbon_rate_kg_per_hr=carbon_rate_kg_per_hr,
        fouling_rate_per_hr=fouling_rate_per_hr,
        carbon_formation_index=carbon_proxy,
        h2_yield_mol_per_mol_ch4=h2_yield,
        unreacted_methane_kg_per_hr=unreacted_methane_kg_per_hr,
    )
    return next_state, outputs


def apply_maintenance(
    state: ReactorState,
    restored_health: float = 1.0,
    *,
    reset_conversion: bool = True,
) -> ReactorState:
    """Apply maintenance and restore health while preserving runtime history."""
    if not 0 <= restored_health <= 1:
        msg = "restored_health must be between 0 and 1"
        raise ValueError(msg)
    next_conversion = state.conversion
    if reset_conversion:
        health_ratio = restored_health / max(state.health, 1e-9)
        next_conversion = _clamp(state.conversion * health_ratio, 0.0, 1.0)

    return ReactorState(
        conversion=next_conversion,
        health=restored_health,
        hours_operated=state.hours_operated,
    )


__all__ = [
    "ReactorInputs",
    "ReactorState",
    "ReactorOutputs",
    "simulate_step",
    "apply_maintenance",
]
