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
FORMAT_VERSION = 2

_MODEL_CACHE: dict[str, SurrogateModel] = {}


@dataclass(frozen=True)
class SurrogateMember:
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
    def from_dict(payload: dict[str, object]) -> SurrogateMember:
        return SurrogateMember(
            degree=int(payload["degree"]),
            temperature_mean=float(payload["temperature_mean"]),
            temperature_std=float(payload["temperature_std"]),
            residence_time_mean=float(payload["residence_time_mean"]),
            residence_time_std=float(payload["residence_time_std"]),
            coefficients={
                key: [float(value) for value in values]
                for key, values in dict(payload["coefficients"]).items()
            },
            rmse={key: float(value) for key, value in dict(payload.get("rmse", {})).items()},
        )


@dataclass(frozen=True)
class SurrogateModel:
    format_version: int
    ensemble_size: int
    members: list[SurrogateMember]
    rmse: dict[str, float]
    rmse_std: dict[str, float]
    feature_bounds: dict[str, list[float]]
    feature_mean: dict[str, float]
    feature_cov_inv: list[list[float]]
    ood_mahalanobis_threshold: float = 3.0

    @property
    def degree(self) -> int:
        return self.members[0].degree

    @property
    def coefficients(self) -> dict[str, list[float]]:
        return self.members[0].coefficients

    @property
    def temperature_mean(self) -> float:
        return self.members[0].temperature_mean

    @property
    def temperature_std(self) -> float:
        return self.members[0].temperature_std

    @property
    def residence_time_mean(self) -> float:
        return self.members[0].residence_time_mean

    @property
    def residence_time_std(self) -> float:
        return self.members[0].residence_time_std

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "ensemble_size": self.ensemble_size,
            "members": [member.to_dict() for member in self.members],
            "rmse": self.rmse,
            "rmse_std": self.rmse_std,
            "feature_bounds": self.feature_bounds,
            "feature_mean": self.feature_mean,
            "feature_cov_inv": self.feature_cov_inv,
            "ood_mahalanobis_threshold": self.ood_mahalanobis_threshold,
        }

    @staticmethod
    def _from_legacy_dict(payload: dict[str, object]) -> SurrogateModel:
        member = SurrogateMember.from_dict(payload)
        t_mean = member.temperature_mean
        t_std = max(member.temperature_std, 1e-6)
        tau_mean = member.residence_time_mean
        tau_std = max(member.residence_time_std, 1e-6)
        return SurrogateModel(
            format_version=1,
            ensemble_size=1,
            members=[member],
            rmse={key: float(value) for key, value in dict(payload.get("rmse", {})).items()},
            rmse_std={target: 0.0 for target in TARGET_COLUMNS},
            feature_bounds={
                "temperature_c": [t_mean - (3.0 * t_std), t_mean + (3.0 * t_std)],
                "residence_time_s": [tau_mean - (3.0 * tau_std), tau_mean + (3.0 * tau_std)],
            },
            feature_mean={
                "temperature_c": t_mean,
                "residence_time_s": tau_mean,
            },
            feature_cov_inv=[
                [1.0 / (t_std**2), 0.0],
                [0.0, 1.0 / (tau_std**2)],
            ],
            ood_mahalanobis_threshold=3.0,
        )

    @staticmethod
    def from_dict(payload: dict[str, object]) -> SurrogateModel:
        if "members" not in payload:
            return SurrogateModel._from_legacy_dict(payload)
        members = [SurrogateMember.from_dict(item) for item in list(payload["members"])]
        if not members:
            msg = "Surrogate model payload must contain at least one member"
            raise ValueError(msg)
        return SurrogateModel(
            format_version=int(payload.get("format_version", FORMAT_VERSION)),
            ensemble_size=int(payload.get("ensemble_size", len(members))),
            members=members,
            rmse={key: float(value) for key, value in dict(payload.get("rmse", {})).items()},
            rmse_std={
                key: float(value)
                for key, value in dict(
                    payload.get("rmse_std", {target: 0.0 for target in TARGET_COLUMNS})
                ).items()
            },
            feature_bounds={
                key: [float(v) for v in values]
                for key, values in dict(payload["feature_bounds"]).items()
            },
            feature_mean={
                key: float(value) for key, value in dict(payload["feature_mean"]).items()
            },
            feature_cov_inv=[
                [float(value) for value in row] for row in list(payload["feature_cov_inv"])
            ],
            ood_mahalanobis_threshold=float(payload.get("ood_mahalanobis_threshold", 3.0)),
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


def _training_stats(
    cleaned: pd.DataFrame,
) -> tuple[dict[str, list[float]], dict[str, float], list[list[float]]]:
    temperature = cleaned["temperature_c"].to_numpy(dtype=float)
    residence_time = cleaned["residence_time_s"].to_numpy(dtype=float)
    feature_bounds = {
        "temperature_c": [float(np.min(temperature)), float(np.max(temperature))],
        "residence_time_s": [float(np.min(residence_time)), float(np.max(residence_time))],
    }
    feature_mean = {
        "temperature_c": float(np.mean(temperature)),
        "residence_time_s": float(np.mean(residence_time)),
    }
    features = np.column_stack([temperature, residence_time])
    cov = np.cov(features.T, ddof=0)
    cov += np.eye(2) * 1e-9
    cov_inv = np.linalg.pinv(cov)
    return feature_bounds, feature_mean, cov_inv.astype(float).tolist()


def _predict_member(
    member: SurrogateMember,
    temperature: np.ndarray,
    residence_time: np.ndarray,
) -> dict[str, np.ndarray]:
    temp_norm = (temperature - member.temperature_mean) / member.temperature_std
    tau_norm = (residence_time - member.residence_time_mean) / member.residence_time_std
    feature_matrix = _poly_features(temp_norm, tau_norm, member.degree)
    return {
        target: feature_matrix @ np.asarray(coeffs, dtype=float)
        for target, coeffs in member.coefficients.items()
    }


def _predict_member_ensemble(
    model: SurrogateModel,
    temperature: np.ndarray,
    residence_time: np.ndarray,
) -> dict[str, np.ndarray]:
    stacked: dict[str, list[np.ndarray]] = {target: [] for target in TARGET_COLUMNS}
    for member in model.members:
        prediction = _predict_member(member, temperature, residence_time)
        for target in TARGET_COLUMNS:
            stacked[target].append(prediction[target])
    return {target: np.vstack(values) for target, values in stacked.items()}


def fit_surrogate(
    dataset: pd.DataFrame,
    degree: int = 2,
    *,
    ensemble_size: int = 7,
    random_seed: int = 42,
) -> SurrogateModel:
    if degree < 1:
        msg = "degree must be at least 1"
        raise ValueError(msg)
    if ensemble_size < 1:
        msg = "ensemble_size must be at least 1"
        raise ValueError(msg)

    cleaned = _validate_dataset(dataset)
    n_rows = len(cleaned)
    temperature = cleaned["temperature_c"].to_numpy(dtype=float)
    residence_time = cleaned["residence_time_s"].to_numpy(dtype=float)
    t_mean, t_std = _safe_scale(temperature)
    tau_mean, tau_std = _safe_scale(residence_time)

    full_temp_norm = (temperature - t_mean) / t_std
    full_tau_norm = (residence_time - tau_mean) / tau_std
    full_features = _poly_features(full_temp_norm, full_tau_norm, degree)

    rng = np.random.default_rng(random_seed)
    members: list[SurrogateMember] = []
    for _model_idx in range(ensemble_size):
        if ensemble_size == 1:
            sample_idx = np.arange(n_rows)
        else:
            sample_idx = rng.choice(n_rows, size=n_rows, replace=True)

        fit_temp = temperature[sample_idx]
        fit_tau = residence_time[sample_idx]
        fit_matrix = _poly_features(
            (fit_temp - t_mean) / t_std, (fit_tau - tau_mean) / tau_std, degree
        )

        member_coeffs: dict[str, list[float]] = {}
        member_rmse: dict[str, float] = {}
        for target in TARGET_COLUMNS:
            y_fit = cleaned[target].to_numpy(dtype=float)[sample_idx]
            y_all = cleaned[target].to_numpy(dtype=float)
            solved, *_ = np.linalg.lstsq(fit_matrix, y_fit, rcond=None)
            y_hat = full_features @ solved
            member_coeffs[target] = solved.astype(float).tolist()
            member_rmse[target] = float(np.sqrt(np.mean((y_all - y_hat) ** 2)))

        members.append(
            SurrogateMember(
                degree=degree,
                temperature_mean=t_mean,
                temperature_std=t_std,
                residence_time_mean=tau_mean,
                residence_time_std=tau_std,
                coefficients=member_coeffs,
                rmse=member_rmse,
            )
        )

    member_predictions = _predict_member_ensemble(
        SurrogateModel(
            format_version=FORMAT_VERSION,
            ensemble_size=ensemble_size,
            members=members,
            rmse={},
            rmse_std={},
            feature_bounds={},
            feature_mean={},
            feature_cov_inv=[],
        ),
        temperature,
        residence_time,
    )
    rmse: dict[str, float] = {}
    rmse_std: dict[str, float] = {}
    for target in TARGET_COLUMNS:
        y_true = cleaned[target].to_numpy(dtype=float)
        y_pred_mean = np.mean(member_predictions[target], axis=0)
        rmse[target] = float(np.sqrt(np.mean((y_true - y_pred_mean) ** 2)))
        rmse_std[target] = float(np.std([member.rmse[target] for member in members]))

    feature_bounds, feature_mean, feature_cov_inv = _training_stats(cleaned)
    return SurrogateModel(
        format_version=FORMAT_VERSION,
        ensemble_size=ensemble_size,
        members=members,
        rmse=rmse,
        rmse_std=rmse_std,
        feature_bounds=feature_bounds,
        feature_mean=feature_mean,
        feature_cov_inv=feature_cov_inv,
        ood_mahalanobis_threshold=3.0,
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

    member_predictions = _predict_member_ensemble(model, temp_array, tau_array)
    outputs: dict[str, float | np.ndarray] = {}
    for target in TARGET_COLUMNS:
        mean_pred = np.mean(member_predictions[target], axis=0)
        outputs[target] = float(mean_pred[0]) if scalar_output else mean_pred
    return outputs


def predict_with_uncertainty(
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

    member_predictions = _predict_member_ensemble(model, temp_array, tau_array)
    outputs: dict[str, float | np.ndarray] = {}
    for target in TARGET_COLUMNS:
        mean_pred = np.mean(member_predictions[target], axis=0)
        std_pred = np.std(member_predictions[target], axis=0)
        low = mean_pred - (2.0 * std_pred)
        high = mean_pred + (2.0 * std_pred)

        if target == "methane_conversion":
            mean_pred = np.clip(mean_pred, 0.0, 1.0)
            low = np.clip(low, 0.0, 1.0)
            high = np.clip(high, 0.0, 1.0)
        elif target == "h2_yield_mol_per_mol_ch4":
            mean_pred = np.clip(mean_pred, 0.0, 2.0)
            low = np.clip(low, 0.0, 2.0)
            high = np.clip(high, 0.0, 2.0)
        else:
            mean_pred = np.clip(mean_pred, 0.0, None)
            low = np.clip(low, 0.0, None)
            high = np.clip(high, 0.0, None)

        outputs[target] = float(mean_pred[0]) if scalar_output else mean_pred
        outputs[f"{target}_std"] = float(std_pred[0]) if scalar_output else std_pred
        outputs[f"{target}_ci_lower"] = float(low[0]) if scalar_output else low
        outputs[f"{target}_ci_upper"] = float(high[0]) if scalar_output else high
    return outputs


def _ood_metrics(
    model: SurrogateModel,
    temperature_c: float | np.ndarray,
    residence_time_s: float | np.ndarray,
) -> tuple[bool | np.ndarray, float | np.ndarray]:
    temp = np.asarray(temperature_c, dtype=float)
    tau = np.asarray(residence_time_s, dtype=float)
    scalar_output = temp.ndim == 0 and tau.ndim == 0
    temp = np.atleast_1d(temp)
    tau = np.atleast_1d(tau)

    t_min, t_max = model.feature_bounds["temperature_c"]
    tau_min, tau_max = model.feature_bounds["residence_time_s"]
    t_span = max(t_max - t_min, 1e-9)
    tau_span = max(tau_max - tau_min, 1e-9)

    t_dist = np.where(
        temp < t_min, (t_min - temp) / t_span, np.where(temp > t_max, (temp - t_max) / t_span, 0.0)
    )
    tau_dist = np.where(
        tau < tau_min,
        (tau_min - tau) / tau_span,
        np.where(tau > tau_max, (tau - tau_max) / tau_span, 0.0),
    )
    bbox_distance = np.sqrt((t_dist**2) + (tau_dist**2))

    mean_vec = np.array(
        [
            model.feature_mean["temperature_c"],
            model.feature_mean["residence_time_s"],
        ],
        dtype=float,
    )
    cov_inv = np.asarray(model.feature_cov_inv, dtype=float)
    delta = np.column_stack([temp, tau]) - mean_vec
    mahal = np.sqrt(np.sum((delta @ cov_inv) * delta, axis=1))
    normalized_mahal = mahal / max(model.ood_mahalanobis_threshold, 1e-9)

    ood_score = np.maximum(bbox_distance, normalized_mahal)
    is_ood = ood_score > 1.0
    if scalar_output:
        return bool(is_ood[0]), float(ood_score[0])
    return is_ood, ood_score


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


def _fallback_predict(temperature_c: float, residence_time_s: float) -> dict[str, float | bool]:
    normalized_temp = np.clip((temperature_c - 850.0) / 250.0, 0.0, 1.0)
    tau_factor = 1.0 - np.exp(-max(residence_time_s, 0.0) / 1.5)
    conversion = float(np.clip((0.32 + (0.56 * normalized_temp)) * tau_factor, 0.0, 0.95))
    h2_yield = float(np.clip(2.0 * conversion * (0.90 + (0.10 * normalized_temp)), 0.0, 2.0))
    carbon_proxy = float(max(0.0, 0.03 + (0.20 * conversion) + (0.04 * normalized_temp)))
    conversion_std = 0.03
    h2_std = 0.05
    carbon_std = 0.02
    is_ood = not (850.0 <= temperature_c <= 1100.0 and 0.1 <= residence_time_s <= 5.0)
    ood_score = 2.0 if is_ood else 0.4
    return {
        "methane_conversion": conversion,
        "methane_conversion_std": conversion_std,
        "methane_conversion_ci_lower": max(0.0, conversion - (2.0 * conversion_std)),
        "methane_conversion_ci_upper": min(1.0, conversion + (2.0 * conversion_std)),
        "h2_yield_mol_per_mol_ch4": h2_yield,
        "h2_yield_mol_per_mol_ch4_std": h2_std,
        "h2_yield_mol_per_mol_ch4_ci_lower": max(0.0, h2_yield - (2.0 * h2_std)),
        "h2_yield_mol_per_mol_ch4_ci_upper": min(2.0, h2_yield + (2.0 * h2_std)),
        "carbon_formation_index": carbon_proxy,
        "carbon_formation_index_std": carbon_std,
        "carbon_formation_index_ci_lower": max(0.0, carbon_proxy - (2.0 * carbon_std)),
        "carbon_formation_index_ci_upper": carbon_proxy + (2.0 * carbon_std),
        "is_out_of_distribution": is_ood,
        "ood_score": ood_score,
    }


def calibrated_predict(
    temperature_c: float,
    residence_time_s: float,
    params_path: str | Path | None = None,
) -> dict[str, float | bool]:
    try:
        model = load_surrogate_params(DEFAULT_PARAMS_PATH if params_path is None else params_path)
        prediction = predict_with_uncertainty(model, temperature_c, residence_time_s)
        is_ood, ood_score = _ood_metrics(model, temperature_c, residence_time_s)
        prediction["is_out_of_distribution"] = bool(is_ood)
        prediction["ood_score"] = float(ood_score)
        return prediction
    except FileNotFoundError:
        return _fallback_predict(temperature_c, residence_time_s)


def calibrate_and_store(
    *,
    dataset_path: str | Path | None = None,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    degree: int = 2,
    ensemble_size: int = 7,
    random_seed: int = 42,
    verbose: bool = True,
) -> SurrogateModel:
    dataset = load_pfr_dataset(dataset_path)
    model = fit_surrogate(
        dataset,
        degree=degree,
        ensemble_size=ensemble_size,
        random_seed=random_seed,
    )
    saved_path = save_surrogate_params(model, params_path=params_path)
    if verbose:
        print(f"Saved surrogate parameters to: {saved_path}")
        print(f"Ensemble size: {model.ensemble_size}")
        print("RMSE:")
        for target, value in model.rmse.items():
            print(f"  {target}: {value:.6f} (+/- {model.rmse_std.get(target, 0.0):.6f})")
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
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=7,
        help="Bootstrap ensemble size for uncertainty estimation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used for bootstrap resampling.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    calibrate_and_store(
        dataset_path=args.dataset,
        params_path=args.output,
        degree=args.degree,
        ensemble_size=args.ensemble_size,
        random_seed=args.random_seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
