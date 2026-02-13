from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CH4_MW_KG_PER_MOL = 0.016043
H2_MW_KG_PER_MOL = 0.002016
CARBON_FROM_CH4_MASS_RATIO = 12.011 / 16.043

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MECHANISM_PATH = PROJECT_ROOT / "data" / "raw" / "mechanisms" / "gri30.yaml"


def _load_cantera() -> Any:
    try:
        import cantera as ct
    except ModuleNotFoundError as exc:
        msg = (
            "Cantera is required for physics labeling. "
            "Install optional dependency with: pip install -e '.[physics]'"
        )
        raise ModuleNotFoundError(msg) from exc
    return ct


def _carbon_formation_index(
    *,
    x_c2h2: float,
    x_c2h4: float,
    x_c2h6: float,
    heavier_hydrocarbons: float,
) -> float:
    # Proxy for solid carbon tendency from unsaturated and heavier hydrocarbons.
    return max(
        0.0,
        (3.0 * x_c2h2) + (1.7 * x_c2h4) + (1.2 * x_c2h6) + (0.6 * heavier_hydrocarbons),
    )


def _build_heavy_hydrocarbon_indices(gas: Any) -> list[int]:
    indices: list[int] = []
    for idx, _species_name in enumerate(gas.species_names):
        carbon_atoms = gas.n_atoms(idx, "C")
        hydrogen_atoms = gas.n_atoms(idx, "H")
        if carbon_atoms >= 3 and hydrogen_atoms >= 1:
            indices.append(idx)
    return indices


@dataclass
class PFRLabeler:
    mechanism_path: str | Path = DEFAULT_MECHANISM_PATH

    def __post_init__(self) -> None:
        self._ct = _load_cantera()
        path = Path(self.mechanism_path)
        if not path.exists():
            msg = f"Mechanism file not found: {path}"
            raise FileNotFoundError(msg)
        self._gas = self._ct.Solution(str(path))
        self._heavy_hydrocarbon_indices = _build_heavy_hydrocarbon_indices(self._gas)

    @staticmethod
    def mixture_from_dilution(
        dilution_frac: float,
        *,
        diluent_species: str = "AR",
    ) -> dict[str, float]:
        if not 0 <= dilution_frac < 1:
            msg = "dilution_frac must be in [0, 1)"
            raise ValueError(msg)
        methane_frac = max(1e-9, 1.0 - dilution_frac)
        diluent_frac = max(1e-9, dilution_frac)
        return {"CH4": methane_frac, diluent_species: diluent_frac}

    def label_case(
        self,
        *,
        temperature_c: float,
        residence_time_s: float,
        pressure_atm: float = 1.0,
        dilution_frac: float = 0.8,
        diluent_species: str = "AR",
        methane_kg_per_hr: float | None = None,
    ) -> dict[str, float]:
        if residence_time_s <= 0:
            msg = "residence_time_s must be positive"
            raise ValueError(msg)
        if pressure_atm <= 0:
            msg = "pressure_atm must be positive"
            raise ValueError(msg)

        temperature_k = temperature_c + 273.15
        pressure_pa = pressure_atm * self._ct.one_atm
        mixture = self.mixture_from_dilution(dilution_frac, diluent_species=diluent_species)
        self._gas.TPX = temperature_k, pressure_pa, mixture

        reactor = self._ct.IdealGasConstPressureReactor(self._gas, clone=False)
        network = self._ct.ReactorNet([reactor])

        n_total_in = reactor.mass / self._gas.mean_molecular_weight
        x_ch4_in = float(self._gas["CH4"].X[0])
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
        heavier_hydrocarbons = float(np.sum(x_values[self._heavy_hydrocarbon_indices]))
        carbon_proxy = _carbon_formation_index(
            x_c2h2=x_c2h2,
            x_c2h4=x_c2h4,
            x_c2h6=x_c2h6,
            heavier_hydrocarbons=heavier_hydrocarbons,
        )

        result: dict[str, float] = {
            "temperature_c": float(temperature_c),
            "pressure_atm": float(pressure_atm),
            "residence_time_s": float(residence_time_s),
            "dilution_frac": float(dilution_frac),
            "methane_conversion": methane_conversion,
            "h2_yield_mol_per_mol_ch4": h2_yield,
            "x_ch4": x_ch4,
            "x_h2": x_h2,
            "x_c2h2": x_c2h2,
            "x_c2h4": x_c2h4,
            "x_c2h6": x_c2h6,
            "x_heavier_hydrocarbons": heavier_hydrocarbons,
            "carbon_formation_index": carbon_proxy,
            "fouling_risk_index": carbon_proxy,
        }

        if methane_kg_per_hr is not None:
            methane_mol_per_hr = methane_kg_per_hr / CH4_MW_KG_PER_MOL
            h2_kg_per_hr = methane_mol_per_hr * h2_yield * H2_MW_KG_PER_MOL
            carbon_kg_per_hr = methane_kg_per_hr * methane_conversion * CARBON_FROM_CH4_MASS_RATIO
            result["methane_kg_per_hr"] = float(methane_kg_per_hr)
            result["h2_kg_per_hr"] = float(max(0.0, h2_kg_per_hr))
            result["carbon_kg_per_hr"] = float(max(0.0, carbon_kg_per_hr))

        return result


def label_pfr_case(
    *,
    temperature_c: float,
    residence_time_s: float,
    pressure_atm: float = 1.0,
    dilution_frac: float = 0.8,
    diluent_species: str = "AR",
    methane_kg_per_hr: float | None = None,
    mechanism_path: str | Path = DEFAULT_MECHANISM_PATH,
) -> dict[str, float]:
    labeler = PFRLabeler(mechanism_path=mechanism_path)
    return labeler.label_case(
        temperature_c=temperature_c,
        residence_time_s=residence_time_s,
        pressure_atm=pressure_atm,
        dilution_frac=dilution_frac,
        diluent_species=diluent_species,
        methane_kg_per_hr=methane_kg_per_hr,
    )


__all__ = ["PFRLabeler", "label_pfr_case", "DEFAULT_MECHANISM_PATH"]
