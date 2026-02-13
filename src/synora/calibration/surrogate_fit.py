from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHYSICS_DIR = PROJECT_ROOT / "data" / "processed" / "physics_runs"
DEFAULT_PARAMS_PATH = DEFAULT_PHYSICS_DIR / "surrogate_params.json"

FEATURE_COLUMNS = ("temperature_c", "residence_time_s")
TARGET_COLUMNS = (
    "methane_conversion",
    "h2_yield_mol_per_mol_ch4",
    "carbon_formation_index",
)

_MODEL_CACHE: dict[str, SurrogateModel] = {}


@dataclass(frozen=True)
class SurrogateModel:
    degree: int
    temperature_mean: float
    temperature_std: float
    residence_time_mean: float
    residence_time_std: float
    coefficients: dict[str, list[float]]
    rmse: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "degree": self.degree,
            "temperature_mean": self.temperature_mean,
            "temperature_std": self.temperature_std,
            "residence_time_mean": self.residence_time_mean,
            "residence_time_std": self.residence_time_std,
            "coefficients": self.coefficients,
            "rmse": self.rmse,
        }

    @staticmethod
    def from_dict(payload: dict[str, object]) -> SurrogateModel:
        coefficients = {
            key: [float(value) for value in values]
            for key, values in dict(payload["coefficients"]).items()
        }
        rmse = {key: float(value) for key, value in dict(payload["rmse"]).items()}
        return SurrogateModel(
            degree=int(payload["degree"]),
            temperature_mean=float(payload["temperature_mean"]),
            temperature_std=float(payload["temperature_std"]),
            residence_time_mean=float(payload["residence_time_mean"]),
            residence_time_std=float(payload["residence_time_std"]),
            coefficients=coefficients,
            rmse=rmse,
        )


def _poly_features(temp_norm: np.ndarray, tau_norm: np.ndarray, degree: int) -> np.ndarray:
    terms = [np.ones_like(temp_norm)]
    for total_degree in range(1, degree + 1):
        for temp_power in range(total_degree, -1, -1):
            tau_power = total_degree - temp_power
            terms.append((temp_norm**temp_power) * (tau_norm**tau_power))
    return np.column_stack(terms)


def _safe_scale(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-9:
        std = 1.0
    return mean, std


def _validate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in (*FEATURE_COLUMNS, *TARGET_COLUMNS) if column not in dataset
    ]
    if missing_columns:
        msg = f"Dataset is missing required columns: {missing_columns}"
        raise ValueError(msg)
    cleaned = dataset[list((*FEATURE_COLUMNS, *TARGET_COLUMNS))].dropna().copy()
    if cleaned.empty:
        msg = "Dataset is empty after dropping missing rows"
        raise ValueError(msg)
    return cleaned


def fit_surrogate(dataset: pd.DataFrame, degree: int = 2) -> SurrogateModel:
    if degree < 1:
        msg = "degree must be at least 1"
        raise ValueError(msg)

    cleaned = _validate_dataset(dataset)

    temperature = cleaned["temperature_c"].to_numpy(dtype=float)
    residence_time = cleaned["residence_time_s"].to_numpy(dtype=float)

    temperature_mean, temperature_std = _safe_scale(temperature)
    residence_time_mean, residence_time_std = _safe_scale(residence_time)

    temp_norm = (temperature - temperature_mean) / temperature_std
    tau_norm = (residence_time - residence_time_mean) / residence_time_std
    feature_matrix = _poly_features(temp_norm, tau_norm, degree)

    coefficients: dict[str, list[float]] = {}
    rmse: dict[str, float] = {}
    for target in TARGET_COLUMNS:
        y = cleaned[target].to_numpy(dtype=float)
        solved, *_ = np.linalg.lstsq(feature_matrix, y, rcond=None)
        y_hat = feature_matrix @ solved
        coefficients[target] = solved.astype(float).tolist()
        rmse[target] = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    return SurrogateModel(
        degree=degree,
        temperature_mean=temperature_mean,
        temperature_std=temperature_std,
        residence_time_mean=residence_time_mean,
        residence_time_std=residence_time_std,
        coefficients=coefficients,
        rmse=rmse,
    )


def predict_with_model(
    model: SurrogateModel,
    temperature_c: float | np.ndarray,
    residence_time_s: float | np.ndarray,
) -> dict[str, float | np.ndarray]:
    temp_array = np.asarray(temperature_c, dtype=float)
    tau_array = np.asarray(residence_time_s, dtype=float)
    scalar_output = temp_array.ndim == 0 and tau_array.ndim == 0
    temp_array = np.atleast_1d(temp_array)
    tau_array = np.atleast_1d(tau_array)
    if temp_array.shape != tau_array.shape:
        msg = "temperature_c and residence_time_s must have matching shapes"
        raise ValueError(msg)

    temp_norm = (temp_array - model.temperature_mean) / model.temperature_std
    tau_norm = (tau_array - model.residence_time_mean) / model.residence_time_std
    feature_matrix = _poly_features(temp_norm, tau_norm, model.degree)

    outputs: dict[str, float | np.ndarray] = {}
    for target, coeffs in model.coefficients.items():
        prediction = feature_matrix @ np.asarray(coeffs, dtype=float)
        outputs[target] = float(prediction[0]) if scalar_output else prediction
    return outputs


def latest_physics_dataset(data_dir: str | Path = DEFAULT_PHYSICS_DIR) -> Path:
    physics_dir = Path(data_dir)
    parquet_files = sorted(physics_dir.glob("*.parquet"), key=lambda path: path.stat().st_mtime)
    if not parquet_files:
        msg = f"No parquet files found in {physics_dir}"
        raise FileNotFoundError(msg)
    return parquet_files[-1]


def load_pfr_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    selected_path = latest_physics_dataset() if dataset_path is None else Path(dataset_path)
    if not selected_path.exists():
        msg = f"Dataset path does not exist: {selected_path}"
        raise FileNotFoundError(msg)
    return pd.read_parquet(selected_path)


def save_surrogate_params(
    model: SurrogateModel,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> Path:
    path = Path(params_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    _MODEL_CACHE[str(path.resolve())] = model
    return path


def load_surrogate_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> SurrogateModel:
    path = Path(params_path)
    if not path.exists():
        msg = f"Surrogate parameter file does not exist: {path}"
        raise FileNotFoundError(msg)

    cache_key = str(path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    payload = json.loads(path.read_text(encoding="utf-8"))
    model = SurrogateModel.from_dict(payload)
    _MODEL_CACHE[cache_key] = model
    return model


def _fallback_predict(temperature_c: float, residence_time_s: float) -> dict[str, float]:
    normalized_temp = np.clip((temperature_c - 850.0) / 250.0, 0.0, 1.0)
    tau_factor = 1.0 - np.exp(-max(residence_time_s, 0.0) / 1.5)
    conversion = float(np.clip((0.32 + (0.56 * normalized_temp)) * tau_factor, 0.0, 0.95))
    h2_yield = float(np.clip(2.0 * conversion * (0.90 + (0.10 * normalized_temp)), 0.0, 2.0))
    carbon_proxy = float(max(0.0, 0.03 + (0.20 * conversion) + (0.04 * normalized_temp)))
    return {
        "methane_conversion": conversion,
        "h2_yield_mol_per_mol_ch4": h2_yield,
        "carbon_formation_index": carbon_proxy,
    }


def calibrated_predict(
    temperature_c: float,
    residence_time_s: float,
    params_path: str | Path | None = None,
) -> dict[str, float]:
    try:
        model = load_surrogate_params(DEFAULT_PARAMS_PATH if params_path is None else params_path)
        prediction = predict_with_model(model, temperature_c, residence_time_s)
        conversion = float(np.clip(prediction["methane_conversion"], 0.0, 1.0))
        h2_yield = float(np.clip(prediction["h2_yield_mol_per_mol_ch4"], 0.0, 2.0))
        carbon_proxy = float(max(0.0, prediction["carbon_formation_index"]))
        return {
            "methane_conversion": conversion,
            "h2_yield_mol_per_mol_ch4": h2_yield,
            "carbon_formation_index": carbon_proxy,
        }
    except FileNotFoundError:
        return _fallback_predict(temperature_c, residence_time_s)


def calibrate_and_store(
    *,
    dataset_path: str | Path | None = None,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    degree: int = 2,
    verbose: bool = True,
) -> SurrogateModel:
    dataset = load_pfr_dataset(dataset_path)
    model = fit_surrogate(dataset, degree=degree)
    saved_path = save_surrogate_params(model, params_path=params_path)
    if verbose:
        print(f"Saved surrogate parameters to: {saved_path}")
        print("RMSE:")
        for target, value in model.rmse.items():
            print(f"  {target}: {value:.6f}")
    return model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and store methane pyrolysis surrogate parameters."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a parquet dataset. If omitted, uses latest in data/processed/physics_runs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_PARAMS_PATH),
        help="Path to save surrogate parameters as JSON.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Polynomial degree for surrogate fitting.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    calibrate_and_store(
        dataset_path=args.dataset,
        params_path=args.output,
        degree=args.degree,
        verbose=True,
    )


if __name__ == "__main__":
    main()
