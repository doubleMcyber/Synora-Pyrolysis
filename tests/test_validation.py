from __future__ import annotations

import pytest

from synora.validation.experimental import (
    DEFAULT_EXPERIMENTAL_DATASET_PATH,
    load_cv_reactor_experiment,
)
from synora.validation.metrics import compare_experiment_to_surrogate


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
