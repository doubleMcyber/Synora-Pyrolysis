from __future__ import annotations

import pandas as pd

from synora.economics.lcoh import EconInputs, hourly_economics
from synora.reactor.model import ReactorInputs, ReactorState, simulate_step


def run_simulation(
    *,
    hours: int,
    methane_kg_per_hr: float,
    temp: float,
    econ_inputs: EconInputs | None = None,
    max_conversion: float = 0.90,
    initial_state: ReactorState | None = None,
) -> pd.DataFrame:
    """Run a deterministic time-stepped simulation for one reactor asset."""
    if hours <= 0:
        msg = "hours must be positive"
        raise ValueError(msg)
    if methane_kg_per_hr < 0:
        msg = "methane_kg_per_hr must be non-negative"
        raise ValueError(msg)

    econ = econ_inputs or EconInputs()
    state = initial_state or ReactorState()
    rows: list[dict[str, float]] = []

    for hour in range(hours):
        reactor_inputs = ReactorInputs(
            methane_kg_per_hr=methane_kg_per_hr,
            temp_c=temp,
            max_conversion=max_conversion,
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
                "conversion": reactor_outputs.conversion,
                "health": reactor_outputs.health,
                "h2_kg_per_hr": reactor_outputs.h2_kg_per_hr,
                "carbon_kg_per_hr": reactor_outputs.carbon_kg_per_hr,
                "unreacted_methane_kg_per_hr": reactor_outputs.unreacted_methane_kg_per_hr,
                "cost_per_hr": econ_outputs["cost_per_hr"],
                "revenue_per_hr": econ_outputs["revenue_per_hr"],
                "profit_per_hr": econ_outputs["profit_per_hr"],
                "lcoh_usd_per_kg": econ_outputs["lcoh_usd_per_kg"],
            }
        )

    df = pd.DataFrame(rows)
    df["cum_profit_usd"] = df["profit_per_hr"].cumsum()
    return df


__all__ = ["run_simulation"]
