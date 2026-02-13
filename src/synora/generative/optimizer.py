from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd

from synora.economics.lcoh import EconInputs
from synora.generative.design_space import DesignBounds, ReactorDesign, mutate_design
from synora.generative.objectives import evaluate_design_surrogate, scalarize_metrics


@dataclass
class DesignEvaluation:
    design: ReactorDesign
    metrics: dict[str, float | list[str]]
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


def design_to_flat_dict(design: ReactorDesign) -> dict[str, float]:
    payload = design.to_dict()
    return {key: float(value) for key, value in payload.items()}


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


def _sort_evaluations(evaluations: list[DesignEvaluation]) -> list[DesignEvaluation]:
    return sorted(
        evaluations,
        key=lambda item: (item.violation_count, -item.score),
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


def evaluations_to_frame(evaluations: list[DesignEvaluation]) -> pd.DataFrame:
    rows = [item.to_dict() for item in evaluations]
    return pd.DataFrame(rows)


__all__ = ["DesignEvaluation", "propose_designs", "evaluations_to_frame", "design_to_flat_dict"]
