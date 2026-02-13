from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import cantera as ct
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MECHANISM_PATH = PROJECT_ROOT / "data" / "raw" / "mechanisms" / "gri30.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "physics_runs"

TEMPERATURE_RANGE_C = (850.0, 1100.0)
RESIDENCE_TIME_RANGE_S = (0.1, 5.0)
DEFAULT_PRESSURE_ATM = 1.0
INITIAL_MOLE_FRACTIONS = "CH4:1.0, AR:4.0"


def _carbon_formation_index(
    *,
    x_c2h2: float,
    x_c2h4: float,
    x_c2h6: float,
    heavier_hydrocarbons: float,
) -> float:
    return max(
        0.0,
        (3.0 * x_c2h2) + (1.7 * x_c2h4) + (1.2 * x_c2h6) + (0.6 * heavier_hydrocarbons),
    )


def _build_heavy_hydrocarbon_indices(gas: ct.Solution) -> list[int]:
    indices: list[int] = []
    for idx, _species_name in enumerate(gas.species_names):
        carbon_atoms = gas.n_atoms(idx, "C")
        hydrogen_atoms = gas.n_atoms(idx, "H")
        if carbon_atoms >= 3 and hydrogen_atoms >= 1:
            indices.append(idx)
    return indices


def simulate_pfr_case(
    *,
    gas: ct.Solution,
    temperature_c: float,
    residence_time_s: float,
    pressure_atm: float,
    heavy_hydrocarbon_indices: list[int],
) -> dict[str, float]:
    temperature_k = temperature_c + 273.15
    pressure_pa = pressure_atm * ct.one_atm
    gas.TPX = temperature_k, pressure_pa, INITIAL_MOLE_FRACTIONS

    reactor = ct.IdealGasConstPressureReactor(gas, clone=False)
    network = ct.ReactorNet([reactor])

    n_total_in = reactor.mass / gas.mean_molecular_weight
    x_ch4_in = float(gas["CH4"].X[0])
    n_ch4_in = max(1e-18, x_ch4_in * n_total_in)

    network.advance(residence_time_s)
    gas_out = reactor.phase

    n_total_out = reactor.mass / gas_out.mean_molecular_weight
    x_ch4 = float(gas_out["CH4"].X[0])
    x_h2 = float(gas_out["H2"].X[0])
    x_c2h2 = float(gas_out["C2H2"].X[0])
    x_c2h4 = float(gas_out["C2H4"].X[0])
    x_c2h6 = float(gas_out["C2H6"].X[0])

    n_ch4_out = x_ch4 * n_total_out
    n_h2_out = x_h2 * n_total_out

    methane_conversion = float(np.clip((n_ch4_in - n_ch4_out) / n_ch4_in, 0.0, 1.0))
    h2_yield = float(max(0.0, n_h2_out / n_ch4_in))

    x_values = gas_out.X
    heavier_hydrocarbons = float(np.sum(x_values[heavy_hydrocarbon_indices]))
    carbon_proxy = _carbon_formation_index(
        x_c2h2=x_c2h2,
        x_c2h4=x_c2h4,
        x_c2h6=x_c2h6,
        heavier_hydrocarbons=heavier_hydrocarbons,
    )

    return {
        "temperature_c": temperature_c,
        "pressure_atm": pressure_atm,
        "residence_time_s": residence_time_s,
        "methane_conversion": methane_conversion,
        "h2_yield_mol_per_mol_ch4": h2_yield,
        "x_ch4": x_ch4,
        "x_h2": x_h2,
        "x_c2h2": x_c2h2,
        "x_c2h4": x_c2h4,
        "x_c2h6": x_c2h6,
        "x_heavier_hydrocarbons": heavier_hydrocarbons,
        "carbon_formation_index": carbon_proxy,
    }


def generate_pfr_dataset(
    *,
    mechanism_path: Path,
    output_dir: Path,
    n_temperature_points: int,
    n_residence_points: int,
    pressure_atm: float,
) -> Path:
    if not mechanism_path.exists():
        msg = f"Mechanism file not found: {mechanism_path}"
        raise FileNotFoundError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    gas = ct.Solution(str(mechanism_path))
    heavy_hydrocarbon_indices = _build_heavy_hydrocarbon_indices(gas)

    temperatures_c = np.linspace(
        TEMPERATURE_RANGE_C[0], TEMPERATURE_RANGE_C[1], n_temperature_points
    )
    residence_times_s = np.linspace(
        RESIDENCE_TIME_RANGE_S[0], RESIDENCE_TIME_RANGE_S[1], n_residence_points
    )

    rows: list[dict[str, float]] = []
    for temperature_c in temperatures_c:
        for residence_time_s in residence_times_s:
            row = simulate_pfr_case(
                gas=gas,
                temperature_c=float(temperature_c),
                residence_time_s=float(residence_time_s),
                pressure_atm=pressure_atm,
                heavy_hydrocarbon_indices=heavy_hydrocarbon_indices,
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
    )


if __name__ == "__main__":
    main()
