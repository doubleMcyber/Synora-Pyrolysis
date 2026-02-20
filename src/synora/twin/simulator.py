from __future__ import annotations

import math

import pandas as pd

from synora.economics.lcoh import EconInputs, hourly_economics
from synora.reactor.model import ReactorInputs, ReactorState, apply_maintenance, simulate_step


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_visual_frame(
    row: pd.Series | dict[str, object],
    *,
    zone_count: int | None = None,
) -> dict[str, float | bool | list[float]]:
    """Map one simulation row into a renderer-ready frame payload."""
    getter = row.get  # type: ignore[union-attr]
    discovered_zone_temps = [
        _coerce_float(getter(f"zone_{idx + 1}_temp_c", None), float("nan")) for idx in range(3)
    ]
    discovered_zone_temps = [temp for temp in discovered_zone_temps if temp == temp]  # drop NaN

    inferred_zone_count = zone_count
    if inferred_zone_count is None:
        inferred_zone_count = len(discovered_zone_temps) if discovered_zone_temps else 1
    inferred_zone_count = max(1, min(3, int(inferred_zone_count)))

    fallback_temp = _coerce_float(getter("temp_c", 960.0), 960.0)
    zone_temps: list[float] = []
    zone_conversions: list[float] = []
    for idx in range(inferred_zone_count):
        temp_value = _coerce_float(getter(f"zone_{idx + 1}_temp_c", fallback_temp), fallback_temp)
        zone_temps.append(temp_value)
        zone_conversions.append(
            _clamp01(
                _coerce_float(
                    getter(f"zone_{idx + 1}_conversion", getter("conversion", 0.0)),
                    0.0,
                )
            )
        )
    if not zone_temps:
        zone_temps = [fallback_temp]

    conversion = _clamp01(_coerce_float(getter("conversion", 0.0), 0.0))
    methane_rate = max(0.0, _coerce_float(getter("methane_kg_per_hr", 0.0), 0.0))
    h2_rate = max(0.0, _coerce_float(getter("h2_kg_per_hr", 0.0), 0.0))
    carbon_rate = max(
        0.0,
        _coerce_float(
            getter("carbon_rate_kg_per_hr", getter("carbon_kg_per_hr", 0.0)),
            0.0,
        ),
    )
    fouling_index = max(
        0.0,
        _coerce_float(
            getter("fouling_index", 1.0 - _coerce_float(getter("health", 1.0), 1.0)),
            0.0,
        ),
    )
    delta_p_kpa = max(
        0.0,
        _coerce_float(getter("pressure_drop_kpa", getter("deltaP_kpa", 0.0)), 0.0),
    )
    q_required_kw = max(
        0.0,
        _coerce_float(
            getter("q_required_kw", getter("heater_power_kw", getter("power_kw", 0.0))),
            0.0,
        ),
    )
    power_kw = max(
        0.0,
        _coerce_float(
            getter("power_kw", getter("heater_power_kw", q_required_kw)),
            q_required_kw,
        ),
    )
    uncertainty_std = max(0.0, _coerce_float(getter("methane_conversion_std", 0.0), 0.0))
    ood_score = max(0.0, _coerce_float(getter("ood_score", 0.0), 0.0))
    is_ood_raw = getter("is_out_of_distribution", False)
    is_ood = bool(
        is_ood_raw if isinstance(is_ood_raw, bool) else _coerce_float(is_ood_raw, 0.0) > 0.5
    )

    confidence_default = 1.0 - min(1.0, (1.8 * uncertainty_std) + (0.20 * ood_score))
    confidence = _clamp01(
        _coerce_float(getter("confidence_index", confidence_default), confidence_default)
    )

    methane_fraction = _clamp01(
        _coerce_float(
            getter("methane_fraction", getter("ch4_fraction", max(0.0, 1.0 - conversion))),
            max(0.0, 1.0 - conversion),
        )
    )
    hydrogen_fraction = _clamp01(
        _coerce_float(
            getter(
                "hydrogen_fraction",
                getter("h2_fraction", min(1.0, h2_rate / max(methane_rate, 1e-9))),
            ),
            min(1.0, h2_rate / max(methane_rate, 1e-9)),
        )
    )
    health = _clamp01(_coerce_float(getter("health", max(0.0, 1.0 - fouling_index)), 1.0))

    return {
        "tick": _coerce_float(getter("tick", 0.0), 0.0),
        "time_hr": _coerce_float(getter("time_hr", 0.0), 0.0),
        "temp_c": _coerce_float(getter("temp_c", zone_temps[0]), zone_temps[0]),
        "zone_count": float(len(zone_temps)),
        "zone_temps_c": zone_temps,
        "zone_conversions": zone_conversions,
        "conversion": conversion,
        "h2_rate": h2_rate,
        "carbon_rate": carbon_rate,
        "fouling_index": fouling_index,
        "health": health,
        "deltaP_kpa": delta_p_kpa,
        "power_kw": power_kw,
        "q_required_kw": q_required_kw,
        "is_out_of_distribution": is_ood,
        "ood_score": ood_score,
        "confidence": confidence,
        "uncertainty_std": uncertainty_std,
        "methane_fraction": methane_fraction,
        "hydrogen_fraction": hydrogen_fraction,
        "ch4_fraction": methane_fraction,
        "h2_fraction": hydrogen_fraction,
        "validation_rmse": _coerce_float(getter("validation_rmse", float("nan")), float("nan")),
        "validation_mae": _coerce_float(getter("validation_mae", float("nan")), float("nan")),
    }


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
    ticks_per_hour: int = 1,
    convergence_tau_hr: float = 1.5,
    zone_count: int = 3,
    zone_temp_spread_c: float = 75.0,
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
    if ticks_per_hour <= 0:
        msg = "ticks_per_hour must be positive"
        raise ValueError(msg)
    if convergence_tau_hr <= 0:
        msg = "convergence_tau_hr must be positive"
        raise ValueError(msg)
    if zone_count not in {2, 3}:
        msg = "zone_count must be 2 or 3"
        raise ValueError(msg)

    econ = econ_inputs or EconInputs()
    state = initial_state or ReactorState()
    rows: list[dict[str, float]] = []
    dt_hr = 1.0 / float(ticks_per_hour)
    total_ticks = hours * ticks_per_hour
    alpha = dt_hr / (convergence_tau_hr + dt_hr)
    blended_conversion = state.conversion
    blended_h2 = 0.0
    blended_carbon = 0.0
    zone_offsets = (
        [-zone_temp_spread_c / 2.0, zone_temp_spread_c / 2.0]
        if zone_count == 2
        else [-zone_temp_spread_c, 0.0, zone_temp_spread_c]
    )

    for tick in range(total_ticks):
        hour = tick / float(ticks_per_hour)
        maintenance_event = False
        maintenance_interval_ticks = (
            maintenance_interval_hr * ticks_per_hour
            if maintenance_interval_hr is not None
            else None
        )
        if maintenance_interval_ticks and tick > 0 and tick % maintenance_interval_ticks == 0:
            state = apply_maintenance(state)
            maintenance_event = True

        reactor_inputs = ReactorInputs(
            methane_kg_per_hr=methane_kg_per_hr,
            temp_c=temp,
            residence_time_s=residence_time_s,
            max_conversion=max_conversion,
            surrogate_params_path=surrogate_params_path,
        )
        state, reactor_outputs = simulate_step(reactor_inputs, state, dt_hr=dt_hr)
        blended_conversion = (1.0 - alpha) * blended_conversion + (
            alpha * reactor_outputs.conversion
        )
        blended_h2 = (1.0 - alpha) * blended_h2 + (alpha * reactor_outputs.h2_kg_per_hr)
        blended_carbon = (1.0 - alpha) * blended_carbon + (alpha * reactor_outputs.carbon_kg_per_hr)
        process_ripple = (0.014 * math.sin((tick * 0.41) + 0.35)) + (
            0.008 * math.cos((tick * 0.17) + 0.92)
        )
        conversion_dynamic = max(
            0.0,
            min(max_conversion, blended_conversion * (1.0 + process_ripple)),
        )
        h2_dynamic = max(
            0.0,
            blended_h2 * (1.0 + (0.020 * math.sin((tick * 0.31) + 0.70))),
        )
        carbon_dynamic = max(
            0.0,
            blended_carbon * (1.0 + (0.028 * math.sin((tick * 0.47) + 1.20))),
        )
        control_progress = min(1.0, state.hours_operated / max(convergence_tau_hr, 1e-9))
        zone_temperatures = [temp + (offset * control_progress) for offset in zone_offsets]
        zone_weights_raw = [
            max(0.05, 1.0 + ((zone_temp - temp) / 220.0)) for zone_temp in zone_temperatures
        ]
        total_zone_weight = sum(zone_weights_raw)
        zone_conversion_blend = [
            conversion_dynamic * (weight / max(total_zone_weight, 1e-9))
            for weight in zone_weights_raw
        ]
        composition_total = max(1e-9, (1.0 - conversion_dynamic) + (2.0 * conversion_dynamic))
        methane_fraction = max(0.0, (1.0 - conversion_dynamic) / composition_total)
        hydrogen_fraction = max(0.0, min(1.0, (2.0 * conversion_dynamic) / composition_total))
        fouling_index = max(0.0, min(1.0, 1.0 - state.health))
        pressure_drop_kpa = max(
            0.05,
            (
                (1.8 * conversion_dynamic)
                + (5.0 * fouling_index)
                + (0.18 * residence_time_s)
                + (0.002 * max(temp - 800.0, 0.0))
            ),
        )
        pressure_drop_kpa = max(
            0.05,
            pressure_drop_kpa * (1.0 + (0.015 * math.sin((tick * 0.22) + 0.80))),
        )
        heater_power_kw = max(
            0.0,
            (
                (0.42 * methane_kg_per_hr)
                + (0.16 * max(temp - 300.0, 0.0))
                + (95.0 * conversion_dynamic)
                + (45.0 * fouling_index)
            ),
        )
        heater_power_kw = max(
            0.0,
            heater_power_kw * (1.0 + (0.012 * math.sin((tick * 0.19) + 1.10))),
        )
        ood_raw = max(0.0, float(reactor_outputs.ood_score))
        ood_scaled = (ood_raw / (ood_raw + 25.0)) * 8.0
        ood_score = max(0.0, min(10.0, ood_scaled + (1.8 * fouling_index)))
        is_ood = float(
            (ood_score >= 8.5)
            or (bool(reactor_outputs.is_out_of_distribution) and (fouling_index >= 0.35))
        )
        std_component = max(0.0, min(1.0, 4.0 * reactor_outputs.methane_conversion_std))
        confidence_index = max(
            0.0,
            min(
                1.0,
                1.0 - (0.42 * std_component) - (0.34 * fouling_index) - (0.24 * (ood_score / 10.0)),
            ),
        )
        econ_outputs = hourly_economics(
            h2_kg_per_hr=h2_dynamic,
            carbon_kg_per_hr=carbon_dynamic,
            methane_kg_per_hr=methane_kg_per_hr,
            econ_inputs=econ,
        )
        row = {
            "tick": float(tick),
            "time_hr": float(hour),
            "time_min": float(hour * 60.0),
            "methane_kg_per_hr": methane_kg_per_hr,
            "temp_c": temp,
            "residence_time_s": residence_time_s,
            "conversion": conversion_dynamic,
            "health": state.health,
            "h2_kg_per_hr": h2_dynamic,
            "carbon_kg_per_hr": carbon_dynamic,
            "carbon_rate_kg_per_hr": reactor_outputs.carbon_rate_kg_per_hr,
            "fouling_rate_per_hr": reactor_outputs.fouling_rate_per_hr,
            "carbon_formation_index": reactor_outputs.carbon_formation_index,
            "fouling_index": fouling_index,
            "methane_fraction": methane_fraction,
            "hydrogen_fraction": hydrogen_fraction,
            "h2_yield_mol_per_mol_ch4": reactor_outputs.h2_yield_mol_per_mol_ch4,
            "unreacted_methane_kg_per_hr": reactor_outputs.unreacted_methane_kg_per_hr,
            "methane_conversion_std": reactor_outputs.methane_conversion_std,
            "is_out_of_distribution": is_ood,
            "ood_score": ood_score,
            "pressure_drop_kpa": pressure_drop_kpa,
            "heater_power_kw": heater_power_kw,
            "confidence_index": confidence_index,
            "cost_per_hr": econ_outputs["cost_per_hr"],
            "revenue_per_hr": econ_outputs["revenue_per_hr"],
            "profit_per_hr": econ_outputs["profit_per_hr"],
            "lcoh_usd_per_kg": econ_outputs["lcoh_usd_per_kg"],
            "maintenance_event": float(maintenance_event),
        }
        for zone_idx in range(zone_count):
            row[f"zone_{zone_idx + 1}_temp_c"] = zone_temperatures[zone_idx]
            row[f"zone_{zone_idx + 1}_conversion"] = zone_conversion_blend[zone_idx]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["cum_profit_usd"] = df["profit_per_hr"].cumsum()
    return df


def build_simulation_context(
    *,
    hours: int,
    methane_kg_per_hr: float,
    temp: float,
    residence_time_s: float = 1.0,
    econ_inputs: EconInputs | None = None,
    maintenance_interval_hr: int | None = None,
    surrogate_params_path: str | None = None,
    ticks_per_hour: int = 12,
    zone_count: int = 3,
) -> dict[str, object]:
    frames = run_simulation(
        hours=hours,
        methane_kg_per_hr=methane_kg_per_hr,
        temp=temp,
        residence_time_s=residence_time_s,
        econ_inputs=econ_inputs,
        maintenance_interval_hr=maintenance_interval_hr,
        surrogate_params_path=surrogate_params_path,
        ticks_per_hour=ticks_per_hour,
        zone_count=zone_count,
    )
    latest = frames.iloc[-1].to_dict() if not frames.empty else {}
    return {
        "frames": frames,
        "latest": latest,
        "zone_count": zone_count,
        "ticks_per_hour": ticks_per_hour,
        "hours": hours,
    }


__all__ = ["run_simulation", "build_simulation_context", "build_visual_frame"]
