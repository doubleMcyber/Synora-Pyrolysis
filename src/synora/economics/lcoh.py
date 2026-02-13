from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EconInputs:
    methane_price_usd_per_kg: float = 0.45
    hydrogen_price_usd_per_kg: float = 4.50
    carbon_price_usd_per_kg: float = 0.18
    variable_opex_usd_per_hr: float = 15.0
    fixed_opex_usd_per_hr: float = 8.0
    capex_usd: float = 500_000.0
    capex_amortization_hours: float = 5 * 365 * 24

    def __post_init__(self) -> None:
        values = {
            "methane_price_usd_per_kg": self.methane_price_usd_per_kg,
            "hydrogen_price_usd_per_kg": self.hydrogen_price_usd_per_kg,
            "carbon_price_usd_per_kg": self.carbon_price_usd_per_kg,
            "variable_opex_usd_per_hr": self.variable_opex_usd_per_hr,
            "fixed_opex_usd_per_hr": self.fixed_opex_usd_per_hr,
            "capex_usd": self.capex_usd,
            "capex_amortization_hours": self.capex_amortization_hours,
        }
        for name, value in values.items():
            if value < 0:
                msg = f"{name} must be non-negative"
                raise ValueError(msg)


def hourly_economics(
    *,
    h2_kg_per_hr: float,
    carbon_kg_per_hr: float,
    methane_kg_per_hr: float,
    econ_inputs: EconInputs,
) -> dict[str, float]:
    """Return hourly economics with deterministic, unit-consistent outputs."""
    if h2_kg_per_hr < 0 or carbon_kg_per_hr < 0 or methane_kg_per_hr < 0:
        msg = "mass flow inputs must be non-negative"
        raise ValueError(msg)

    capex_amortized_per_hr = 0.0
    if econ_inputs.capex_amortization_hours > 0:
        capex_amortized_per_hr = econ_inputs.capex_usd / econ_inputs.capex_amortization_hours

    methane_cost_per_hr = methane_kg_per_hr * econ_inputs.methane_price_usd_per_kg
    cost_per_hr = (
        methane_cost_per_hr
        + econ_inputs.variable_opex_usd_per_hr
        + econ_inputs.fixed_opex_usd_per_hr
        + capex_amortized_per_hr
    )
    revenue_per_hr = (h2_kg_per_hr * econ_inputs.hydrogen_price_usd_per_kg) + (
        carbon_kg_per_hr * econ_inputs.carbon_price_usd_per_kg
    )
    profit_per_hr = revenue_per_hr - cost_per_hr
    lcoh_usd_per_kg = float("inf")
    if h2_kg_per_hr > 0:
        lcoh_usd_per_kg = cost_per_hr / h2_kg_per_hr

    return {
        "methane_cost_per_hr": max(0.0, methane_cost_per_hr),
        "cost_per_hr": max(0.0, cost_per_hr),
        "revenue_per_hr": max(0.0, revenue_per_hr),
        "profit_per_hr": profit_per_hr,
        "lcoh_usd_per_kg": lcoh_usd_per_kg,
    }


__all__ = ["EconInputs", "hourly_economics"]
