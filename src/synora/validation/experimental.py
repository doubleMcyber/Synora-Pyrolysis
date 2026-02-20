from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENTAL_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "experimental" / "cv_reactor_table2_normalized.parquet"
)

REQUIRED_COLUMNS = (
    "time_s",
    "temperature_k",
    "pressure_kpa",
    "oven_setpoint_k",
    "initial_pressure_kpa",
    "methane_mol_percent",
    "hydrogen_mol_percent",
    "helium_mol_percent",
    "ethane_mol_percent",
    "ethylene_mol_percent",
    "propane_mol_percent",
    "benzene_mol_percent",
)

MOL_PERCENT_COLUMNS = tuple(
    column for column in REQUIRED_COLUMNS if column.endswith("_mol_percent")
)
VALIDATION_AXIS_NOTE = (
    "Experimental time_s is used as a proxy for residence_time_s during surrogate validation. "
    "This is a pragmatic axis-mapping for overlay analysis and not a strict hydrodynamic equivalence."
)


def load_cv_reactor_experiment(
    dataset_path: str | Path = DEFAULT_EXPERIMENTAL_DATASET_PATH,
) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        msg = f"Experimental dataset path does not exist: {path}"
        raise FileNotFoundError(msg)

    loaded = pd.read_parquet(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in loaded.columns]
    if missing:
        msg = f"Experimental dataset missing required columns: {missing}"
        raise ValueError(msg)

    df = loaded[list(REQUIRED_COLUMNS)].copy()
    df = df.sort_values("time_s").reset_index(drop=True)
    df["temperature_c"] = df["temperature_k"] - 273.15

    earliest_time_s = float(df["time_s"].min())
    baseline_rows = df[df["time_s"] == earliest_time_s]
    methane_baseline = float(baseline_rows["methane_mol_percent"].max())
    if methane_baseline <= 0:
        msg = "methane baseline must be positive to derive methane_conversion_exp"
        raise ValueError(msg)

    df["methane_conversion_exp"] = 1.0 - (df["methane_mol_percent"] / methane_baseline)
    df["methane_baseline_mol_percent"] = methane_baseline
    df["residence_time_proxy_s"] = df["time_s"].astype(float)

    for column in MOL_PERCENT_COLUMNS:
        df[column.replace("_mol_percent", "_mole_fraction")] = df[column] / 100.0

    ordered_columns = [
        "time_s",
        "residence_time_proxy_s",
        "temperature_c",
        "pressure_kpa",
        "oven_setpoint_k",
        "initial_pressure_kpa",
        "methane_conversion_exp",
        "methane_baseline_mol_percent",
        *MOL_PERCENT_COLUMNS,
        *(column.replace("_mol_percent", "_mole_fraction") for column in MOL_PERCENT_COLUMNS),
    ]
    df = df[ordered_columns].copy()

    df.attrs["dataset_path"] = str(path.resolve())
    df.attrs["validation_axis_note"] = VALIDATION_AXIS_NOTE
    df.attrs["methane_baseline_mol_percent"] = methane_baseline
    return df


__all__ = [
    "DEFAULT_EXPERIMENTAL_DATASET_PATH",
    "VALIDATION_AXIS_NOTE",
    "load_cv_reactor_experiment",
]
