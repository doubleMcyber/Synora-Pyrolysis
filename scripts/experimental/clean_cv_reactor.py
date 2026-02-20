from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV = Path("data/raw/experimental/methane_cv_reactor.csv")
OUT_DIR = Path("data/processed/experimental")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _clean_numeric(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.upper() == "ND":
        return np.nan
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def main() -> None:
    # Read without headers so we can build them ourselves
    raw = pd.read_csv(RAW_CSV, header=None)
    # Row 0: group headers ("Reaction time (s)", "Temperature (K)", ...)
    # Row 1: species names ("Methane", "Ethane", ...)
    # Data begins at row 2
    group = raw.iloc[0].tolist()
    names = raw.iloc[1].tolist()
    data = raw.iloc[2:].copy()

    # Build stable column names by combining group+name when needed
    cols = []
    for g, n in zip(group, names, strict=False):
        g = "" if pd.isna(g) else str(g).strip()
        n = "" if pd.isna(n) else str(n).strip()
        if n and g and n != g:
            cols.append(f"{g}__{n}")
        elif g:
            cols.append(g)
        elif n:
            cols.append(n)
        else:
            cols.append("")

    data.columns = cols

    # Identify the split point between Table 1 and Table 2 using the repeated "Reaction time (s)"
    rt_cols = [c for c in data.columns if "Reaction time (s)" in c]
    if len(rt_cols) < 2:
        raise RuntimeError(f"Expected 2 Reaction time columns, found {len(rt_cols)}: {rt_cols}")

    rt2 = rt_cols[1]
    idx_split = data.columns.get_loc(rt2)

    t1 = data.iloc[:, :idx_split].copy()
    t2 = data.iloc[:, idx_split:].copy()

    # Standardize column names
    def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        new_cols = []
        for c in df.columns:
            c = c.replace("Molar Concentration (%)__", "")
            c = c.replace("Molar Concentration (%)", "")
            c = c.replace("Final Pressure (kPa)", "pressure_kpa")
            c = c.replace("Temperature (K)", "temperature_k")
            c = c.replace("Reaction time (s)", "time_s")
            c = c.replace("Scaling factor*__", "scaling_factor")
            c = c.replace("Scaling factor*", "scaling_factor")
            c = c.replace("Sum", "sum_percent")
            c = c.replace("__", "_")
            c = c.strip().lower().replace(" ", "_").replace("-", "_")
            new_cols.append(c)
        df.columns = new_cols
        return df

    t1 = normalize_cols(t1)
    t2 = normalize_cols(t2)

    # Clean numeric fields: convert ND -> NaN, strings -> floats
    for df in (t1, t2):
        for c in df.columns:
            df[c] = df[c].map(_clean_numeric)

        # Drop fully empty rows
        df.dropna(how="all", inplace=True)

    # Add experiment metadata (from the table title you showed)
    # Oven setpoint 873 K, initial pressure 399 kPa (final pressure varies per row)
    for df in (t1, t2):
        df["oven_setpoint_k"] = 873.0
        df["initial_pressure_kpa"] = 399.0

    # Save
    t1_path = OUT_DIR / "cv_reactor_table1_measured.parquet"
    t2_path = OUT_DIR / "cv_reactor_table2_normalized.parquet"

    t1.to_parquet(t1_path, index=False)
    t2.to_parquet(t2_path, index=False)

    print("Wrote:")
    print(" -", t1_path.as_posix(), "shape=", t1.shape, "cols=", len(t1.columns))
    print(" -", t2_path.as_posix(), "shape=", t2.shape, "cols=", len(t2.columns))
    print("\nPreview table1:")
    print(t1.head(5).to_string(index=False))
    print("\nPreview table2:")
    print(t2.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
