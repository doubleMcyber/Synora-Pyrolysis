from __future__ import annotations

import pandas as pd

from synora.generative.active_learning import run_active_learning
from synora.generative.design_space import ReactorDesign


def _mock_label_function(design: ReactorDesign) -> dict[str, float]:
    temp_norm = (design.temp_c - 850.0) / 250.0
    tau_norm = (design.residence_time_s - 0.1) / 4.9
    conversion = max(0.0, min(0.95, 0.05 + (0.2 * temp_norm) + (0.4 * tau_norm)))
    h2_yield = max(0.0, min(2.0, (1.7 * conversion) + (0.05 * temp_norm)))
    carbon_proxy = max(0.0, 0.03 + (0.15 * conversion) + (0.03 * tau_norm))
    return {
        "temperature_c": design.temp_c,
        "residence_time_s": design.residence_time_s,
        "pressure_atm": design.pressure_atm,
        "methane_conversion": conversion,
        "h2_yield_mol_per_mol_ch4": h2_yield,
        "carbon_formation_index": carbon_proxy,
    }


def test_active_learning_single_iteration_updates_artifacts(tmp_path) -> None:
    dataset_path = tmp_path / "active_learning_dataset.parquet"
    params_path = tmp_path / "surrogate_params.json"

    result = run_active_learning(
        iterations=1,
        candidates_per_iter=6,
        verify_top_n=3,
        dataset_path=dataset_path,
        surrogate_params_path=params_path,
        label_function=_mock_label_function,
        seed=11,
    )

    assert result.dataset_path.exists()
    assert result.surrogate_params_path.exists()
    assert len(result.iterations) == 1

    dataset = pd.read_parquet(result.dataset_path)
    assert len(dataset) == 3
    assert {"temperature_c", "residence_time_s", "methane_conversion"}.issubset(dataset.columns)
