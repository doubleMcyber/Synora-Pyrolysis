from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from synora.physics.label_pfr import DEFAULT_MECHANISM_PATH, PFRLabeler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "physics_runs"

TEMPERATURE_RANGE_C = (850.0, 1100.0)
RESIDENCE_TIME_RANGE_S = (0.1, 5.0)
DEFAULT_PRESSURE_ATM = 1.0


def generate_pfr_dataset(
    *,
    mechanism_path: Path,
    output_dir: Path,
    n_temperature_points: int,
    n_residence_points: int,
    pressure_atm: float,
    dilution_frac: float,
) -> Path:
    if not mechanism_path.exists():
        msg = f"Mechanism file not found: {mechanism_path}"
        raise FileNotFoundError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    labeler = PFRLabeler(mechanism_path=mechanism_path)

    temperatures_c = np.linspace(
        TEMPERATURE_RANGE_C[0], TEMPERATURE_RANGE_C[1], n_temperature_points
    )
    residence_times_s = np.linspace(
        RESIDENCE_TIME_RANGE_S[0], RESIDENCE_TIME_RANGE_S[1], n_residence_points
    )

    rows: list[dict[str, float]] = []
    for temperature_c in temperatures_c:
        for residence_time_s in residence_times_s:
            row = labeler.label_case(
                temperature_c=float(temperature_c),
                residence_time_s=float(residence_time_s),
                pressure_atm=pressure_atm,
                dilution_frac=dilution_frac,
            )
            rows.append(row)

    df = (
        pd.DataFrame(rows).sort_values(["temperature_c", "residence_time_s"]).reset_index(drop=True)
    )

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / (
        f"pfr_dataset_{timestamp}_{n_temperature_points}x{n_residence_points}.parquet"
    )
    df.to_parquet(output_path, index=False)

    elapsed_s = time.perf_counter() - start_time
    print(f"Generated {len(df)} PFR points in {elapsed_s:.2f}s")
    print(f"Wrote dataset to: {output_path}")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate methane pyrolysis PFR dataset with Cantera."
    )
    parser.add_argument(
        "--mechanism",
        default=str(DEFAULT_MECHANISM_PATH),
        help="Path to Cantera mechanism YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where parquet output is written.",
    )
    parser.add_argument(
        "--n-temperature-points",
        type=int,
        default=9,
        help="Number of temperature points between 850 and 1100 C.",
    )
    parser.add_argument(
        "--n-residence-points",
        type=int,
        default=14,
        help="Number of residence time points between 0.1 and 5.0 s.",
    )
    parser.add_argument(
        "--pressure-atm",
        type=float,
        default=DEFAULT_PRESSURE_ATM,
        help="Operating pressure in atm.",
    )
    parser.add_argument(
        "--dilution-frac",
        type=float,
        default=0.8,
        help="Inlet diluent fraction (remaining fraction is methane).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    generate_pfr_dataset(
        mechanism_path=Path(args.mechanism),
        output_dir=Path(args.output_dir),
        n_temperature_points=args.n_temperature_points,
        n_residence_points=args.n_residence_points,
        pressure_atm=args.pressure_atm,
        dilution_frac=args.dilution_frac,
    )


if __name__ == "__main__":
    main()
