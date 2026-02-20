from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd

from synora.economics.lcoh import EconInputs
from synora.generative.design_space import DesignBounds, ReactorDesign, mutate_design
from synora.generative.multizone import (
    MultiZoneBounds,
    MultiZoneDesign,
    ZoneDesign,
    mutate_multizone_design,
)
from synora.generative.objectives import (
    evaluate_design_surrogate,
    evaluate_multizone_surrogate,
    scalarize_metrics,
)


@dataclass
class DesignEvaluation:
    design: ReactorDesign
    metrics: dict[str, float | bool | list[str]]
    score: float

    @property
    def violation_count(self) -> float:
        return float(self.metrics["constraint_violation_count"])

    def to_dict(self) -> dict[str, float | str | bool]:
        payload: dict[str, float | str | bool] = design_to_flat_dict(self.design)
        payload["score"] = self.score
        for key, value in self.metrics.items():
            if key == "constraint_violations":
                payload[key] = ",".join(value) if isinstance(value, list) else str(value)
            elif isinstance(value, bool):
                payload[key] = bool(value)
            elif isinstance(value, Number):
                payload[key] = float(value)
            else:
                payload[key] = str(value)
        return payload


@dataclass
class MultiZoneEvaluation:
    design: MultiZoneDesign
    metrics: dict[str, float | bool | list[str]]
    score: float

    @property
    def violation_count(self) -> float:
        return float(self.metrics["constraint_violation_count"])

    def to_dict(self) -> dict[str, float | str | bool]:
        payload: dict[str, float | str | bool] = multizone_to_flat_dict(self.design)
        payload["score"] = self.score
        for key, value in self.metrics.items():
            if key == "constraint_violations":
                payload[key] = ",".join(value) if isinstance(value, list) else str(value)
            elif isinstance(value, bool):
                payload[key] = bool(value)
            elif isinstance(value, Number):
                payload[key] = float(value)
            else:
                payload[key] = str(value)
        return payload


def design_to_flat_dict(design: ReactorDesign) -> dict[str, float]:
    payload = design.to_dict()
    return {key: float(value) for key, value in payload.items()}


def multizone_to_flat_dict(design: MultiZoneDesign) -> dict[str, float]:
    payload: dict[str, float] = {
        "zone_count": float(design.zone_count),
        "total_length_m": float(design.total_length_m),
        "methane_kg_per_hr": float(design.methane_kg_per_hr),
        "pressure_atm": float(design.pressure_atm),
        "dilution_frac": float(design.dilution_frac),
        "carbon_removal_eff": float(design.carbon_removal_eff),
        "ambient_temp_c": float(design.ambient_temp_c),
        "material_tmax_c": float(design.material_tmax_c),
        "dp_max_kpa": float(design.dp_max_kpa),
        "power_max_kw": float(design.power_max_kw),
        "default_diameter_m": float(design.default_diameter_m),
    }
    for idx, zone in enumerate(design.zones, start=1):
        payload[f"zone_{idx}_temp_c"] = float(zone.temp_c)
        payload[f"zone_{idx}_length_m"] = float(zone.length_m)
        payload[f"zone_{idx}_diameter_m"] = float(design.zone_diameters_m[idx - 1])
        payload[f"zone_{idx}_insulation_factor"] = float(zone.insulation_factor)
        payload[f"zone_{idx}_pressure_atm"] = float(design.zone_pressures_atm[idx - 1])
        payload[f"zone_{idx}_tau_s"] = float(design.zone_residence_time_s[idx - 1])
    return payload


def _evaluate_population(
    population: list[ReactorDesign],
    *,
    surrogate_params_path: str | Path | None,
    econ_inputs: EconInputs | None,
    constraints: dict[str, float] | None,
) -> list[DesignEvaluation]:
    evaluations: list[DesignEvaluation] = []
    for design in population:
        metrics = evaluate_design_surrogate(
            design,
            surrogate_params_path=surrogate_params_path,
            econ_inputs=econ_inputs,
            constraints=constraints,
        )
        evaluations.append(
            DesignEvaluation(
                design=design,
                metrics=metrics,
                score=scalarize_metrics(metrics),
            )
        )
    return evaluations


def _evaluate_multizone_population(
    population: list[MultiZoneDesign],
    *,
    surrogate_params_path: str | Path | None,
    econ_inputs: EconInputs | None,
    constraints: dict[str, float] | None,
    delta_t_max: float,
    smoothness_penalty_weight: float,
    uncertainty_penalty_weight: float,
) -> list[MultiZoneEvaluation]:
    evaluations: list[MultiZoneEvaluation] = []
    for design in population:
        metrics = evaluate_multizone_surrogate(
            design,
            surrogate_params_path=surrogate_params_path,
            econ_inputs=econ_inputs,
            constraints=constraints,
        )
        base_score = scalarize_metrics(metrics)
        zone_temps = design.zone_temperatures_c
        smoothness_excess = sum(
            max(0.0, abs(zone_temps[idx] - zone_temps[idx - 1]) - delta_t_max)
            for idx in range(1, len(zone_temps))
        )
        uncertainty_penalty = (
            float(metrics.get("conversion_std", 0.0))
            + (0.5 * float(metrics.get("h2_yield_std", 0.0)))
            + float(metrics.get("fouling_risk_index_std", 0.0))
        )
        score = (
            base_score
            - (smoothness_penalty_weight * smoothness_excess)
            - (uncertainty_penalty_weight * uncertainty_penalty)
        )
        metrics["temp_smoothness_penalty"] = float(smoothness_excess)
        metrics["uncertainty_penalty"] = float(uncertainty_penalty)
        evaluations.append(
            MultiZoneEvaluation(
                design=design,
                metrics=metrics,
                score=float(score),
            )
        )
    return evaluations


def _sort_evaluations(evaluations: list[DesignEvaluation]) -> list[DesignEvaluation]:
    return sorted(
        evaluations,
        key=lambda item: (item.violation_count, -item.score),
    )


def _sort_multizone_evaluations(
    evaluations: list[MultiZoneEvaluation],
) -> list[MultiZoneEvaluation]:
    return sorted(
        evaluations,
        key=lambda item: (item.violation_count, -item.score),
    )


def _crossover_multizone(
    parent_a: MultiZoneDesign,
    parent_b: MultiZoneDesign,
    rng: np.random.Generator,
) -> MultiZoneDesign:
    zones: list[ZoneDesign] = []
    for idx in range(parent_a.zone_count):
        source = parent_a if rng.random() < 0.5 else parent_b
        zone = source.zones[idx]
        zones.append(
            ZoneDesign(
                temp_c=zone.temp_c,
                length_m=zone.length_m,
                diameter_m=zone.diameter_m,
                insulation_factor=zone.insulation_factor,
                pressure_atm_override=zone.pressure_atm_override,
            )
        )
    return MultiZoneDesign(
        zones=zones,
        methane_kg_per_hr=float((parent_a.methane_kg_per_hr + parent_b.methane_kg_per_hr) / 2.0),
        pressure_atm=float((parent_a.pressure_atm + parent_b.pressure_atm) / 2.0),
        dilution_frac=float((parent_a.dilution_frac + parent_b.dilution_frac) / 2.0),
        carbon_removal_eff=float((parent_a.carbon_removal_eff + parent_b.carbon_removal_eff) / 2.0),
        ambient_temp_c=parent_a.ambient_temp_c,
        material_tmax_c=parent_a.material_tmax_c,
        dp_max_kpa=parent_a.dp_max_kpa,
        power_max_kw=parent_a.power_max_kw,
        default_diameter_m=float((parent_a.default_diameter_m + parent_b.default_diameter_m) / 2.0),
    )


def propose_designs(
    *,
    top_k: int = 10,
    generations: int = 10,
    population_size: int = 120,
    elite_fraction: float = 0.20,
    seed: int = 42,
    bounds: DesignBounds | None = None,
    surrogate_params_path: str | Path | None = None,
    econ_inputs: EconInputs | None = None,
    constraints: dict[str, float] | None = None,
) -> list[DesignEvaluation]:
    if top_k <= 0:
        msg = "top_k must be positive"
        raise ValueError(msg)
    if population_size < max(8, top_k):
        msg = "population_size must be >= max(8, top_k)"
        raise ValueError(msg)
    if generations <= 0:
        msg = "generations must be positive"
        raise ValueError(msg)
    if not 0 < elite_fraction < 1:
        msg = "elite_fraction must be in (0, 1)"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    search_bounds = bounds or DesignBounds()
    population = [search_bounds.sample(rng) for _ in range(population_size)]

    for _ in range(generations):
        evaluations = _sort_evaluations(
            _evaluate_population(
                population,
                surrogate_params_path=surrogate_params_path,
                econ_inputs=econ_inputs,
                constraints=constraints,
            )
        )
        elite_count = max(2, int(population_size * elite_fraction))
        elites = evaluations[:elite_count]

        next_population: list[ReactorDesign] = [item.design for item in elites]
        while len(next_population) < population_size:
            if rng.random() < 0.70:
                parent = elites[int(rng.integers(0, len(elites)))].design
                child = mutate_design(parent, search_bounds, rng)
            else:
                child = search_bounds.sample(rng)
            next_population.append(child)
        population = next_population

    final_evaluations = _sort_evaluations(
        _evaluate_population(
            population,
            surrogate_params_path=surrogate_params_path,
            econ_inputs=econ_inputs,
            constraints=constraints,
        )
    )
    feasible = [item for item in final_evaluations if item.violation_count == 0]
    if len(feasible) >= top_k:
        return feasible[:top_k]
    return final_evaluations[:top_k]


def propose_multizone_designs(
    *,
    top_k: int = 10,
    zones: int = 2,
    generations: int = 10,
    population_size: int = 120,
    elite_fraction: float = 0.20,
    seed: int = 42,
    bounds: MultiZoneBounds | None = None,
    surrogate_params_path: str | Path | None = None,
    econ_inputs: EconInputs | None = None,
    constraints: dict[str, float] | None = None,
    delta_t_max: float = 180.0,
    smoothness_penalty_weight: float = 0.30,
    uncertainty_penalty_weight: float = 20.0,
) -> list[MultiZoneEvaluation]:
    if top_k <= 0:
        msg = "top_k must be positive"
        raise ValueError(msg)
    if zones not in {2, 3}:
        msg = "zones must be 2 or 3"
        raise ValueError(msg)
    if population_size < max(12, top_k):
        msg = "population_size must be >= max(12, top_k)"
        raise ValueError(msg)
    if generations <= 0:
        msg = "generations must be positive"
        raise ValueError(msg)
    if not 0 < elite_fraction < 1:
        msg = "elite_fraction must be in (0, 1)"
        raise ValueError(msg)
    if delta_t_max <= 0:
        msg = "delta_t_max must be positive"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    search_bounds = bounds or MultiZoneBounds()
    population = [search_bounds.sample(zones, rng) for _ in range(population_size)]

    for _ in range(generations):
        evaluations = _sort_multizone_evaluations(
            _evaluate_multizone_population(
                population,
                surrogate_params_path=surrogate_params_path,
                econ_inputs=econ_inputs,
                constraints=constraints,
                delta_t_max=delta_t_max,
                smoothness_penalty_weight=smoothness_penalty_weight,
                uncertainty_penalty_weight=uncertainty_penalty_weight,
            )
        )
        elite_count = max(3, int(population_size * elite_fraction))
        elites = evaluations[:elite_count]

        next_population: list[MultiZoneDesign] = [item.design for item in elites]
        while len(next_population) < population_size:
            draw = rng.random()
            if draw < 0.55:
                parent = elites[int(rng.integers(0, len(elites)))].design
                child = mutate_multizone_design(parent, search_bounds, rng)
            elif draw < 0.80 and len(elites) >= 2:
                parent_a = elites[int(rng.integers(0, len(elites)))].design
                parent_b = elites[int(rng.integers(0, len(elites)))].design
                child = _crossover_multizone(parent_a, parent_b, rng)
            else:
                child = search_bounds.sample(zones, rng)
            next_population.append(child)
        population = next_population

    final_evaluations = _sort_multizone_evaluations(
        _evaluate_multizone_population(
            population,
            surrogate_params_path=surrogate_params_path,
            econ_inputs=econ_inputs,
            constraints=constraints,
            delta_t_max=delta_t_max,
            smoothness_penalty_weight=smoothness_penalty_weight,
            uncertainty_penalty_weight=uncertainty_penalty_weight,
        )
    )
    feasible = [item for item in final_evaluations if item.violation_count == 0]
    if len(feasible) >= top_k:
        return feasible[:top_k]
    return final_evaluations[:top_k]


def evaluations_to_frame(evaluations: list[DesignEvaluation]) -> pd.DataFrame:
    rows = [item.to_dict() for item in evaluations]
    return pd.DataFrame(rows)


def multizone_evaluations_to_frame(evaluations: list[MultiZoneEvaluation]) -> pd.DataFrame:
    rows = [item.to_dict() for item in evaluations]
    return pd.DataFrame(rows)


__all__ = [
    "DesignEvaluation",
    "MultiZoneEvaluation",
    "propose_designs",
    "propose_multizone_designs",
    "evaluations_to_frame",
    "multizone_evaluations_to_frame",
    "design_to_flat_dict",
    "multizone_to_flat_dict",
]
