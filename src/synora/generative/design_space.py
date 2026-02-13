from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi

import numpy as np

CH4_MW_KG_PER_MOL = 0.016043
GAS_CONSTANT_J_PER_MOLK = 8.314462618


@dataclass(frozen=True)
class ReactorDesign:
    length_m: float
    diameter_m: float
    pressure_atm: float
    temp_c: float
    methane_kg_per_hr: float
    dilution_frac: float
    carbon_removal_eff: float
    wall_thickness_m: float = 0.008
    emissivity: float = 0.85
    roughness_mm: float = 0.2

    def __post_init__(self) -> None:
        if self.length_m <= 0:
            msg = "length_m must be positive"
            raise ValueError(msg)
        if self.diameter_m <= 0:
            msg = "diameter_m must be positive"
            raise ValueError(msg)
        if self.pressure_atm <= 0:
            msg = "pressure_atm must be positive"
            raise ValueError(msg)
        if self.methane_kg_per_hr <= 0:
            msg = "methane_kg_per_hr must be positive"
            raise ValueError(msg)
        if not 0 <= self.dilution_frac < 1:
            msg = "dilution_frac must be in [0, 1)"
            raise ValueError(msg)
        if not 0 <= self.carbon_removal_eff <= 1:
            msg = "carbon_removal_eff must be in [0, 1]"
            raise ValueError(msg)
        if not 0 < self.emissivity <= 1:
            msg = "emissivity must be in (0, 1]"
            raise ValueError(msg)
        if self.wall_thickness_m <= 0:
            msg = "wall_thickness_m must be positive"
            raise ValueError(msg)
        if self.roughness_mm <= 0:
            msg = "roughness_mm must be positive"
            raise ValueError(msg)

    @property
    def cross_section_area_m2(self) -> float:
        return pi * ((self.diameter_m / 2.0) ** 2)

    @property
    def volume_m3(self) -> float:
        return self.cross_section_area_m2 * self.length_m

    @property
    def surface_area_m2(self) -> float:
        return pi * self.diameter_m * self.length_m

    @property
    def surface_area_to_volume(self) -> float:
        return self.surface_area_m2 / max(self.volume_m3, 1e-12)

    @property
    def methane_molar_flow_mol_per_s(self) -> float:
        methane_kg_per_s = self.methane_kg_per_hr / 3600.0
        return methane_kg_per_s / CH4_MW_KG_PER_MOL

    @property
    def total_molar_flow_mol_per_s(self) -> float:
        methane_fraction = max(1e-6, 1.0 - self.dilution_frac)
        return self.methane_molar_flow_mol_per_s / methane_fraction

    @property
    def volumetric_flow_m3_per_s(self) -> float:
        temperature_k = self.temp_c + 273.15
        pressure_pa = self.pressure_atm * 101325.0
        return (self.total_molar_flow_mol_per_s * GAS_CONSTANT_J_PER_MOLK * temperature_k) / max(
            pressure_pa, 1e-9
        )

    @property
    def residence_time_s(self) -> float:
        return self.volume_m3 / max(self.volumetric_flow_m3_per_s, 1e-12)

    @property
    def effective_carbon_release_factor(self) -> float:
        return 1.0 - self.carbon_removal_eff

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["residence_time_s"] = self.residence_time_s
        payload["surface_area_to_volume"] = self.surface_area_to_volume
        return payload


@dataclass(frozen=True)
class DesignBounds:
    length_m: tuple[float, float] = (0.6, 3.0)
    diameter_m: tuple[float, float] = (0.04, 0.20)
    pressure_atm: tuple[float, float] = (0.8, 3.0)
    temp_c: tuple[float, float] = (850.0, 1100.0)
    methane_kg_per_hr: tuple[float, float] = (40.0, 240.0)
    dilution_frac: tuple[float, float] = (0.55, 0.92)
    carbon_removal_eff: tuple[float, float] = (0.0, 0.95)
    wall_thickness_m: tuple[float, float] = (0.004, 0.020)
    emissivity: tuple[float, float] = (0.6, 0.95)
    roughness_mm: tuple[float, float] = (0.05, 1.0)

    @staticmethod
    def _sample_range(bounds: tuple[float, float], rng: np.random.Generator) -> float:
        return float(rng.uniform(bounds[0], bounds[1]))

    def sample(self, rng: np.random.Generator) -> ReactorDesign:
        return ReactorDesign(
            length_m=self._sample_range(self.length_m, rng),
            diameter_m=self._sample_range(self.diameter_m, rng),
            pressure_atm=self._sample_range(self.pressure_atm, rng),
            temp_c=self._sample_range(self.temp_c, rng),
            methane_kg_per_hr=self._sample_range(self.methane_kg_per_hr, rng),
            dilution_frac=self._sample_range(self.dilution_frac, rng),
            carbon_removal_eff=self._sample_range(self.carbon_removal_eff, rng),
            wall_thickness_m=self._sample_range(self.wall_thickness_m, rng),
            emissivity=self._sample_range(self.emissivity, rng),
            roughness_mm=self._sample_range(self.roughness_mm, rng),
        )


def mutate_design(
    design: ReactorDesign,
    bounds: DesignBounds,
    rng: np.random.Generator,
    mutation_scale: float = 0.08,
) -> ReactorDesign:
    payload = design.to_dict()
    mutable = {
        "length_m": bounds.length_m,
        "diameter_m": bounds.diameter_m,
        "pressure_atm": bounds.pressure_atm,
        "temp_c": bounds.temp_c,
        "methane_kg_per_hr": bounds.methane_kg_per_hr,
        "dilution_frac": bounds.dilution_frac,
        "carbon_removal_eff": bounds.carbon_removal_eff,
        "wall_thickness_m": bounds.wall_thickness_m,
        "emissivity": bounds.emissivity,
        "roughness_mm": bounds.roughness_mm,
    }
    for field_name, field_bounds in mutable.items():
        span = field_bounds[1] - field_bounds[0]
        delta = rng.normal(0.0, mutation_scale * span)
        value = float(payload[field_name] + delta)
        payload[field_name] = float(np.clip(value, field_bounds[0], field_bounds[1]))
    return ReactorDesign(
        length_m=payload["length_m"],
        diameter_m=payload["diameter_m"],
        pressure_atm=payload["pressure_atm"],
        temp_c=payload["temp_c"],
        methane_kg_per_hr=payload["methane_kg_per_hr"],
        dilution_frac=payload["dilution_frac"],
        carbon_removal_eff=payload["carbon_removal_eff"],
        wall_thickness_m=payload["wall_thickness_m"],
        emissivity=payload["emissivity"],
        roughness_mm=payload["roughness_mm"],
    )


__all__ = ["ReactorDesign", "DesignBounds", "mutate_design"]
