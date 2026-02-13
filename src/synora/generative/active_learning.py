from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from synora.calibration.surrogate_fit import (
    DEFAULT_PARAMS_PATH,
    DEFAULT_PHYSICS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    fit_surrogate,
    latest_physics_dataset,
    save_surrogate_params,
)
from synora.economics.lcoh import EconInputs
from synora.generative.design_space import DesignBounds, ReactorDesign
from synora.generative.optimizer import DesignEvaluation, propose_designs
from synora.physics.label_pfr import DEFAULT_MECHANISM_PATH, PFRLabeler

LabelFunction = Callable[[ReactorDesign], dict[str, float]]

REQUIRED_DATASET_COLUMNS = (
    "temperature_c",
    "residence_time_s",
    "pressure_atm",
    "methane_conversion",
    "h2_yield_mol_per_mol_ch4",
    "carbon_formation_index",
)


@dataclass
class ActiveLearningIteration:
    iteration: int
    candidates_generated: int
    candidates_verified: int
    rmse_conversion: float
    rmse_h2_yield: float
    rmse_carbon_proxy: float


@dataclass
class ActiveLearningResult:
    dataset_path: Path
    surrogate_params_path: Path
    iterations: list[ActiveLearningIteration]


def _resolve_dataset_path(dataset_path: str | Path | None) -> Path:
    if dataset_path is not None:
        return Path(dataset_path)
    try:
        return latest_physics_dataset(DEFAULT_PHYSICS_DIR)
    except FileNotFoundError:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        return DEFAULT_PHYSICS_DIR / f"active_learning_seed_{timestamp}.parquet"


def _empty_dataset() -> pd.DataFrame:
    columns = list(dict.fromkeys((*REQUIRED_DATASET_COLUMNS, *FEATURE_COLUMNS, *TARGET_COLUMNS)))
    return pd.DataFrame(columns=columns)


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_dataset()
    loaded = pd.read_parquet(path)
    for column in REQUIRED_DATASET_COLUMNS:
        if column not in loaded.columns:
            loaded[column] = pd.NA
    return loaded


def _default_label_function(mechanism_path: str | Path = DEFAULT_MECHANISM_PATH) -> LabelFunction:
    labeler = PFRLabeler(mechanism_path=mechanism_path)

    def _label(design: ReactorDesign) -> dict[str, float]:
        return labeler.label_case(
            temperature_c=design.temp_c,
            residence_time_s=design.residence_time_s,
            pressure_atm=design.pressure_atm,
            dilution_frac=design.dilution_frac,
            methane_kg_per_hr=design.methane_kg_per_hr,
        )

    return _label


def _evaluation_to_label_row(
    evaluation: DesignEvaluation,
    label: dict[str, float],
) -> dict[str, float]:
    return {
        "temperature_c": float(label["temperature_c"]),
        "residence_time_s": float(label["residence_time_s"]),
        "pressure_atm": float(label["pressure_atm"]),
        "methane_conversion": float(label["methane_conversion"]),
        "h2_yield_mol_per_mol_ch4": float(label["h2_yield_mol_per_mol_ch4"]),
        "carbon_formation_index": float(label["carbon_formation_index"]),
        "design_length_m": evaluation.design.length_m,
        "design_diameter_m": evaluation.design.diameter_m,
        "design_methane_kg_per_hr": evaluation.design.methane_kg_per_hr,
        "design_dilution_frac": evaluation.design.dilution_frac,
        "design_carbon_removal_eff": evaluation.design.carbon_removal_eff,
    }


def run_active_learning(
    *,
    iterations: int,
    candidates_per_iter: int,
    verify_top_n: int,
    dataset_path: str | Path | None = None,
    surrogate_params_path: str | Path = DEFAULT_PARAMS_PATH,
    bounds: DesignBounds | None = None,
    econ_inputs: EconInputs | None = None,
    constraints: dict[str, float] | None = None,
    label_function: LabelFunction | None = None,
    seed: int = 42,
    ensemble_size: int = 7,
) -> ActiveLearningResult:
    if iterations <= 0:
        msg = "iterations must be positive"
        raise ValueError(msg)
    if candidates_per_iter <= 0:
        msg = "candidates_per_iter must be positive"
        raise ValueError(msg)
    if verify_top_n <= 0 or verify_top_n > candidates_per_iter:
        msg = "verify_top_n must be in [1, candidates_per_iter]"
        raise ValueError(msg)

    dataset_file = _resolve_dataset_path(dataset_path)
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    params_file = Path(surrogate_params_path)
    params_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(dataset_file)
    active_label_fn = label_function or _default_label_function()
    search_bounds = bounds or DesignBounds()
    iteration_summaries: list[ActiveLearningIteration] = []

    for iter_idx in range(iterations):
        proposals = propose_designs(
            top_k=candidates_per_iter,
            generations=8,
            population_size=max(60, candidates_per_iter * 8),
            seed=seed + iter_idx,
            bounds=search_bounds,
            surrogate_params_path=params_file if params_file.exists() else None,
            econ_inputs=econ_inputs,
            constraints=constraints,
        )
        selected = proposals[:verify_top_n]

        labeled_rows: list[dict[str, float]] = []
        for evaluation in selected:
            label = active_label_fn(evaluation.design)
            labeled_rows.append(_evaluation_to_label_row(evaluation, label))

        new_rows = pd.DataFrame(labeled_rows)
        if dataset.empty:
            dataset = new_rows
        else:
            dataset = pd.concat([dataset, new_rows], ignore_index=True)
        dataset.to_parquet(dataset_file, index=False)

        model = fit_surrogate(
            dataset,
            degree=2,
            ensemble_size=ensemble_size,
            random_seed=seed + iter_idx,
        )
        save_surrogate_params(model, params_path=params_file)

        iteration_summaries.append(
            ActiveLearningIteration(
                iteration=iter_idx + 1,
                candidates_generated=len(proposals),
                candidates_verified=len(selected),
                rmse_conversion=model.rmse["methane_conversion"],
                rmse_h2_yield=model.rmse["h2_yield_mol_per_mol_ch4"],
                rmse_carbon_proxy=model.rmse["carbon_formation_index"],
            )
        )

    return ActiveLearningResult(
        dataset_path=dataset_file,
        surrogate_params_path=params_file,
        iterations=iteration_summaries,
    )


__all__ = ["run_active_learning", "ActiveLearningResult", "ActiveLearningIteration"]
