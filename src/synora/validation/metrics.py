from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from synora.calibration.surrogate_fit import calibrated_predict

REQUIRED_EXPERIMENT_COLUMNS = (
    "time_s",
    "temperature_c",
    "methane_conversion_exp",
    "hydrogen_mol_percent",
    "methane_mole_fraction",
    "hydrogen_mole_fraction",
    "helium_mole_fraction",
)


def _hydrogen_mol_percent_from_surrogate(
    *,
    methane_conversion: float,
    h2_yield_mol_per_mol_ch4: float,
    methane_baseline_mole_fraction: float,
    hydrogen_baseline_mole_fraction: float,
    helium_baseline_mole_fraction: float,
) -> float:
    methane_baseline = max(methane_baseline_mole_fraction, 1e-9)
    methane_left = methane_baseline * max(0.0, 1.0 - methane_conversion)
    hydrogen_total = max(0.0, hydrogen_baseline_mole_fraction) + (
        methane_baseline * max(0.0, methane_conversion) * max(0.0, h2_yield_mol_per_mol_ch4)
    )
    total_moles_proxy = methane_left + hydrogen_total + max(0.0, helium_baseline_mole_fraction)
    if total_moles_proxy <= 1e-12:
        return float("nan")
    return float(np.clip((hydrogen_total / total_moles_proxy) * 100.0, 0.0, 100.0))


def compare_experiment_to_surrogate(
    df_exp: pd.DataFrame,
    surrogate_params_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    missing = [column for column in REQUIRED_EXPERIMENT_COLUMNS if column not in df_exp.columns]
    if missing:
        msg = f"Experimental dataframe missing required columns: {missing}"
        raise ValueError(msg)
    if df_exp.empty:
        msg = "Experimental dataframe is empty"
        raise ValueError(msg)

    earliest_time_s = float(df_exp["time_s"].min())
    baseline_rows = df_exp[df_exp["time_s"] == earliest_time_s]
    methane_baseline = float(baseline_rows["methane_mole_fraction"].max())
    hydrogen_baseline = float(baseline_rows["hydrogen_mole_fraction"].max())
    helium_baseline = float(baseline_rows["helium_mole_fraction"].max())

    rows: list[dict[str, float | bool]] = []
    for _, row in df_exp.sort_values("time_s").iterrows():
        residence_time_proxy_s = float(row.get("residence_time_proxy_s", row["time_s"]))
        pred = calibrated_predict(
            temperature_c=float(row["temperature_c"]),
            residence_time_s=residence_time_proxy_s,
            params_path=surrogate_params_path,
        )
        conv_mean = float(pred["methane_conversion"])
        conv_std = float(pred.get("methane_conversion_std", 0.0))
        conv_ci_lower = float(pred.get("methane_conversion_ci_lower", conv_mean - (2.0 * conv_std)))
        conv_ci_upper = float(pred.get("methane_conversion_ci_upper", conv_mean + (2.0 * conv_std)))

        h2_pred_mean = float("nan")
        h2_pred_ci_lower = float("nan")
        h2_pred_ci_upper = float("nan")
        hydrogen_prediction_available = False
        if "hydrogen_mol_percent" in pred:
            h2_pred_mean = float(pred["hydrogen_mol_percent"])
            h2_pred_ci_lower = h2_pred_mean
            h2_pred_ci_upper = h2_pred_mean
            hydrogen_prediction_available = True
        elif "h2_yield_mol_per_mol_ch4" in pred:
            h2_yield_mean = float(pred["h2_yield_mol_per_mol_ch4"])
            h2_yield_ci_lower = float(pred.get("h2_yield_mol_per_mol_ch4_ci_lower", h2_yield_mean))
            h2_yield_ci_upper = float(pred.get("h2_yield_mol_per_mol_ch4_ci_upper", h2_yield_mean))
            h2_pred_mean = _hydrogen_mol_percent_from_surrogate(
                methane_conversion=conv_mean,
                h2_yield_mol_per_mol_ch4=h2_yield_mean,
                methane_baseline_mole_fraction=methane_baseline,
                hydrogen_baseline_mole_fraction=hydrogen_baseline,
                helium_baseline_mole_fraction=helium_baseline,
            )
            h2_pred_ci_lower = _hydrogen_mol_percent_from_surrogate(
                methane_conversion=conv_ci_lower,
                h2_yield_mol_per_mol_ch4=h2_yield_ci_lower,
                methane_baseline_mole_fraction=methane_baseline,
                hydrogen_baseline_mole_fraction=hydrogen_baseline,
                helium_baseline_mole_fraction=helium_baseline,
            )
            h2_pred_ci_upper = _hydrogen_mol_percent_from_surrogate(
                methane_conversion=conv_ci_upper,
                h2_yield_mol_per_mol_ch4=h2_yield_ci_upper,
                methane_baseline_mole_fraction=methane_baseline,
                hydrogen_baseline_mole_fraction=hydrogen_baseline,
                helium_baseline_mole_fraction=helium_baseline,
            )
            hydrogen_prediction_available = True

        rows.append(
            {
                "time_s": float(row["time_s"]),
                "residence_time_proxy_s": residence_time_proxy_s,
                "temperature_c": float(row["temperature_c"]),
                "pressure_kpa": float(row.get("pressure_kpa", np.nan)),
                "methane_conversion_exp": float(row["methane_conversion_exp"]),
                "methane_conversion_pred_mean": conv_mean,
                "pred_ci_lower": conv_ci_lower,
                "pred_ci_upper": conv_ci_upper,
                "is_out_of_distribution": bool(pred.get("is_out_of_distribution", False)),
                "ood_score": float(pred.get("ood_score", 0.0)),
                "hydrogen_mol_percent_exp": float(row["hydrogen_mol_percent"]),
                "hydrogen_mol_percent_pred_mean": h2_pred_mean,
                "hydrogen_pred_ci_lower": h2_pred_ci_lower,
                "hydrogen_pred_ci_upper": h2_pred_ci_upper,
                "hydrogen_prediction_available": bool(hydrogen_prediction_available),
            }
        )

    overlay = pd.DataFrame(rows).sort_values("time_s").reset_index(drop=True)
    error = overlay["methane_conversion_pred_mean"] - overlay["methane_conversion_exp"]
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    ood_count = int(overlay["is_out_of_distribution"].sum())
    n_samples = int(len(overlay))
    ood_fraction = float(ood_count / max(n_samples, 1))

    summary = {
        "rmse_conversion": rmse,
        "mae_conversion": mae,
        "bias_conversion": bias,
        "ood_count": float(ood_count),
        "ood_fraction": ood_fraction,
        "n_samples": float(n_samples),
    }
    return overlay, summary


__all__ = ["compare_experiment_to_surrogate"]
