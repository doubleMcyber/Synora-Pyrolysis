from __future__ import annotations

import pandas as pd

from synora.economics.lcoh import EconInputs, hourly_economics
from synora.reactor.model import ReactorInputs, ReactorState, apply_maintenance, simulate_step


def run_simulation(
    *,
    hours: int,
    methane_kg_per_hr: float,
    temp: float,
    residence_time_s: float = 1.0,
    econ_inputs: EconInputs | None = None,
    max_conversion: float = 0.90,
    initial_state: ReactorState | None = None,
    maintenance_interval_hr: int | None = None,
    surrogate_params_path: str | None = None,
) -> pd.DataFrame:
    """Run a deterministic time-stepped simulation for one reactor asset."""
    if hours <= 0:
        msg = "hours must be positive"
        raise ValueError(msg)
    if methane_kg_per_hr < 0:
        msg = "methane_kg_per_hr must be non-negative"
        raise ValueError(msg)
    if residence_time_s <= 0:
        msg = "residence_time_s must be positive"
        raise ValueError(msg)
    if maintenance_interval_hr is not None and maintenance_interval_hr <= 0:
        msg = "maintenance_interval_hr must be positive when provided"
        raise ValueError(msg)

    econ = econ_inputs or EconInputs()
    state = initial_state or ReactorState()
    rows: list[dict[str, float]] = []

    for hour in range(hours):
        maintenance_event = False
        if maintenance_interval_hr and hour > 0 and hour % maintenance_interval_hr == 0:
            state = apply_maintenance(state)
            maintenance_event = True

        reactor_inputs = ReactorInputs(
            methane_kg_per_hr=methane_kg_per_hr,
            temp_c=temp,
            residence_time_s=residence_time_s,
            max_conversion=max_conversion,
            surrogate_params_path=surrogate_params_path,
        )
        state, reactor_outputs = simulate_step(reactor_inputs, state)
        econ_outputs = hourly_economics(
            h2_kg_per_hr=reactor_outputs.h2_kg_per_hr,
            carbon_kg_per_hr=reactor_outputs.carbon_kg_per_hr,
            methane_kg_per_hr=methane_kg_per_hr,
            econ_inputs=econ,
        )
        rows.append(
            {
                "time_hr": float(hour),
                "methane_kg_per_hr": methane_kg_per_hr,
                "temp_c": temp,
                "residence_time_s": residence_time_s,
                "conversion": reactor_outputs.conversion,
                "health": reactor_outputs.health,
                "h2_kg_per_hr": reactor_outputs.h2_kg_per_hr,
                "carbon_kg_per_hr": reactor_outputs.carbon_kg_per_hr,
                "carbon_rate_kg_per_hr": reactor_outputs.carbon_rate_kg_per_hr,
                "fouling_rate_per_hr": reactor_outputs.fouling_rate_per_hr,
                "carbon_formation_index": reactor_outputs.carbon_formation_index,
                "h2_yield_mol_per_mol_ch4": reactor_outputs.h2_yield_mol_per_mol_ch4,
                "unreacted_methane_kg_per_hr": reactor_outputs.unreacted_methane_kg_per_hr,
                "cost_per_hr": econ_outputs["cost_per_hr"],
                "revenue_per_hr": econ_outputs["revenue_per_hr"],
                "profit_per_hr": econ_outputs["profit_per_hr"],
                "lcoh_usd_per_kg": econ_outputs["lcoh_usd_per_kg"],
                "maintenance_event": float(maintenance_event),
            }
        )

    df = pd.DataFrame(rows)
    df["cum_profit_usd"] = df["profit_per_hr"].cumsum()
    return df


__all__ = ["run_simulation"]
