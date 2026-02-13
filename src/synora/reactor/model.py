from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


@dataclass(frozen=True)
class ReactorInputs:
    methane_kg_per_hr: float
    temp_c: float
    max_conversion: float = 0.90

    def __post_init__(self) -> None:
        if self.methane_kg_per_hr < 0:
            msg = "methane_kg_per_hr must be non-negative"
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
    """Advance the reactor by one timestep.

    Stoichiometry assumptions for methane pyrolysis:
    - CH4 -> C + 2H2
    - 1 kg CH4 yields 0.25 kg H2 and 0.75 kg carbon at 100% conversion.
    """
    if dt_hr <= 0:
        msg = "dt_hr must be positive"
        raise ValueError(msg)

    temp_factor = _clamp((inputs.temp_c - 650.0) / 250.0, 0.0, 1.0)
    base_conversion = inputs.max_conversion * (0.60 + (0.40 * temp_factor))
    effective_conversion = _clamp(base_conversion * state.health, 0.0, inputs.max_conversion)

    methane_reacted = inputs.methane_kg_per_hr * effective_conversion
    h2_kg_per_hr = methane_reacted * 0.25
    carbon_kg_per_hr = methane_reacted * 0.75
    unreacted_methane_kg_per_hr = max(0.0, inputs.methane_kg_per_hr - methane_reacted)

    stress_factor = 1.0 + max(0.0, (inputs.temp_c - 850.0) / 150.0)
    health_decay = 0.003 * stress_factor * dt_hr
    next_health = _clamp(state.health - health_decay, 0.0, 1.0)

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
        unreacted_methane_kg_per_hr=unreacted_methane_kg_per_hr,
    )
    return next_state, outputs


def apply_maintenance(state: ReactorState, restored_health: float = 1.0) -> ReactorState:
    """Apply maintenance and reset health while preserving operating time."""
    if not 0 <= restored_health <= 1:
        msg = "restored_health must be between 0 and 1"
        raise ValueError(msg)
    return ReactorState(
        conversion=state.conversion,
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
