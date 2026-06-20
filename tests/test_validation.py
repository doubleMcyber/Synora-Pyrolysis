from __future__ import annotations

import pandas as pd
import pytest

from synora.calibration.surrogate_fit import calibrated_predict
from synora.validation.experimental import (
    DEFAULT_EXPERIMENTAL_DATASET_PATH,
    load_cv_reactor_experiment,
)
from synora.validation.metrics import (
    _hydrogen_mol_percent_from_surrogate,
    compare_experiment_to_surrogate,
)


def test_load_cv_reactor_experiment_returns_expected_columns() -> None:
    if not DEFAULT_EXPERIMENTAL_DATASET_PATH.exists():
        pytest.skip("Experimental dataset not available (data/processed/experimental/)")
    df = load_cv_reactor_experiment()

    expected_columns = {
        "time_s",
        "residence_time_proxy_s",
        "temperature_c",
        "pressure_kpa",
        "methane_mol_percent",
        "hydrogen_mol_percent",
        "helium_mol_percent",
        "methane_conversion_exp",
        "methane_mole_fraction",
        "hydrogen_mole_fraction",
        "helium_mole_fraction",
    }
    assert expected_columns.issubset(set(df.columns))
    assert len(df) > 0
    assert df["methane_conversion_exp"].notna().all()


def test_compare_experiment_to_surrogate_returns_overlay_and_metrics() -> None:
    if not DEFAULT_EXPERIMENTAL_DATASET_PATH.exists():
        pytest.skip("Experimental dataset not available (data/processed/experimental/)")
    df_exp = load_cv_reactor_experiment().head(25).copy()
    overlay, metrics = compare_experiment_to_surrogate(df_exp)

    expected_overlay_columns = {
        "time_s",
        "methane_conversion_exp",
        "methane_conversion_pred_mean",
        "pred_ci_lower",
        "pred_ci_upper",
        "is_out_of_distribution",
        "hydrogen_mol_percent_exp",
        "hydrogen_mol_percent_pred_mean",
    }
    expected_metric_keys = {
        "rmse_conversion",
        "mae_conversion",
        "bias_conversion",
        "ood_count",
        "ood_fraction",
        "n_samples",
    }
    assert expected_overlay_columns.issubset(set(overlay.columns))
    assert expected_metric_keys.issubset(set(metrics.keys()))
    assert float(metrics["rmse_conversion"]) >= 0.0
    assert float(metrics["mae_conversion"]) >= 0.0


def test_validation_baseline_uses_single_inlet_row() -> None:
    # Two rows at the earliest time where the max-methane row (A) and the max-hydrogen row
    # (B) are different physical samples. The fix selects one consistent inlet row (A);
    # the old per-column-max would mix A's methane with B's hydrogen/helium.
    df = pd.DataFrame(
        [
            {  # row A: inlet (most methane, least reacted)
                "time_s": 0.5,
                "temperature_c": 1000.0,
                "methane_conversion_exp": 0.1,
                "hydrogen_mol_percent": 2.0,
                "methane_mole_fraction": 0.90,
                "hydrogen_mole_fraction": 0.02,
                "helium_mole_fraction": 0.08,
            },
            {  # row B: more reacted, highest hydrogen at the same earliest time
                "time_s": 0.5,
                "temperature_c": 1000.0,
                "methane_conversion_exp": 0.5,
                "hydrogen_mol_percent": 30.0,
                "methane_mole_fraction": 0.50,
                "hydrogen_mole_fraction": 0.30,
                "helium_mole_fraction": 0.20,
            },
            {
                "time_s": 1.5,
                "temperature_c": 1050.0,
                "methane_conversion_exp": 0.2,
                "hydrogen_mol_percent": 10.0,
                "methane_mole_fraction": 0.70,
                "hydrogen_mole_fraction": 0.15,
                "helium_mole_fraction": 0.15,
            },
        ]
    )
    overlay, _ = compare_experiment_to_surrogate(df)
    row0 = overlay.iloc[0]
    pred = calibrated_predict(temperature_c=1000.0, residence_time_s=0.5)
    conv = float(pred["methane_conversion"])
    h2_yield = float(pred["h2_yield_mol_per_mol_ch4"])

    expected_single_row = _hydrogen_mol_percent_from_surrogate(
        methane_conversion=conv,
        h2_yield_mol_per_mol_ch4=h2_yield,
        methane_baseline_mole_fraction=0.90,
        hydrogen_baseline_mole_fraction=0.02,
        helium_baseline_mole_fraction=0.08,
    )
    mixed_per_column_max = _hydrogen_mol_percent_from_surrogate(
        methane_conversion=conv,
        h2_yield_mol_per_mol_ch4=h2_yield,
        methane_baseline_mole_fraction=0.90,
        hydrogen_baseline_mole_fraction=0.30,
        helium_baseline_mole_fraction=0.20,
    )
    got = float(row0["hydrogen_mol_percent_pred_mean"])
    assert got == pytest.approx(expected_single_row)
    assert got != pytest.approx(mixed_per_column_max)
