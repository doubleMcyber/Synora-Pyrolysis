from __future__ import annotations

import numpy as np
import pandas as pd

from synora.calibration.surrogate_fit import (
    calibrated_predict,
    fit_surrogate,
    predict_with_model,
    predict_with_uncertainty,
    save_surrogate_params,
)


def _synthetic_dataset() -> pd.DataFrame:
    temperatures = np.linspace(850.0, 1100.0, 9)
    residence_times = np.linspace(0.1, 5.0, 11)
    rows: list[dict[str, float]] = []
    for temperature in temperatures:
        for residence_time in residence_times:
            temp_norm = (temperature - 975.0) / 125.0
            tau_norm = (residence_time - 2.55) / 2.45

            conversion = np.clip(
                0.58
                + (0.20 * temp_norm)
                + (0.16 * tau_norm)
                - (0.06 * temp_norm * tau_norm)
                - (0.05 * tau_norm**2),
                0.0,
                1.0,
            )
            h2_yield = np.clip((1.85 * conversion) + (0.08 * temp_norm), 0.0, 2.0)
            carbon_proxy = max(
                0.0,
                0.04 + (0.18 * conversion) + (0.05 * temp_norm**2) + (0.02 * tau_norm),
            )

            rows.append(
                {
                    "temperature_c": float(temperature),
                    "residence_time_s": float(residence_time),
                    "methane_conversion": float(conversion),
                    "h2_yield_mol_per_mol_ch4": float(h2_yield),
                    "carbon_formation_index": float(carbon_proxy),
                }
            )
    return pd.DataFrame(rows)


def test_surrogate_calibration_converges() -> None:
    dataset = _synthetic_dataset()
    model = fit_surrogate(dataset, degree=2, ensemble_size=6, random_seed=3)

    assert model.degree == 2
    assert model.ensemble_size == 6
    assert model.temperature_std > 0
    assert model.residence_time_std > 0
    for coefficients in model.coefficients.values():
        coeff_array = np.asarray(coefficients, dtype=float)
        assert np.isfinite(coeff_array).all()
    for target in ("methane_conversion", "h2_yield_mol_per_mol_ch4", "carbon_formation_index"):
        assert model.rmse_std[target] >= 0


def test_surrogate_error_below_threshold(tmp_path) -> None:
    dataset = _synthetic_dataset()
    model = fit_surrogate(dataset, degree=2, ensemble_size=1, random_seed=11)

    predictions = predict_with_model(
        model,
        dataset["temperature_c"].to_numpy(dtype=float),
        dataset["residence_time_s"].to_numpy(dtype=float),
    )

    conversion_rmse = float(
        np.sqrt(
            np.mean(
                (
                    dataset["methane_conversion"].to_numpy(dtype=float)
                    - np.asarray(predictions["methane_conversion"], dtype=float)
                )
                ** 2
            )
        )
    )
    h2_rmse = float(
        np.sqrt(
            np.mean(
                (
                    dataset["h2_yield_mol_per_mol_ch4"].to_numpy(dtype=float)
                    - np.asarray(predictions["h2_yield_mol_per_mol_ch4"], dtype=float)
                )
                ** 2
            )
        )
    )

    assert conversion_rmse < 1e-8
    assert h2_rmse < 1e-8

    params_path = tmp_path / "surrogate_params.json"
    save_surrogate_params(model, params_path)
    probe = calibrated_predict(temperature_c=960.0, residence_time_s=1.8, params_path=params_path)

    assert 0 <= probe["methane_conversion"] <= 1
    assert 0 <= probe["h2_yield_mol_per_mol_ch4"] <= 2
    assert probe["carbon_formation_index"] >= 0


def test_uncertainty_std_non_negative() -> None:
    dataset = _synthetic_dataset()
    model = fit_surrogate(dataset, degree=2, ensemble_size=7, random_seed=7)

    pred = predict_with_uncertainty(
        model,
        np.array([900.0, 980.0]),
        np.array([1.2, 2.8]),
    )

    assert (np.asarray(pred["methane_conversion_std"]) >= 0).all()
    assert (np.asarray(pred["h2_yield_mol_per_mol_ch4_std"]) >= 0).all()
    assert (np.asarray(pred["carbon_formation_index_std"]) >= 0).all()


def test_ensemble_has_stable_rmse_vs_members() -> None:
    dataset = _synthetic_dataset()
    model = fit_surrogate(dataset, degree=2, ensemble_size=8, random_seed=19)

    for target in ("methane_conversion", "h2_yield_mol_per_mol_ch4", "carbon_formation_index"):
        member_rmses = [member.rmse[target] for member in model.members]
        assert model.rmse[target] <= np.mean(member_rmses) + 1e-9
