from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from synora.generative.design_space import CH4_MW_KG_PER_MOL, GAS_CONSTANT_J_PER_MOLK, ReactorDesign


@dataclass(frozen=True)
class ZoneDesign:
    temp_c: float
    length_m: float
    insulation_factor: float = 1.0
    diameter_m: float | None = None
    pressure_atm_override: float | None = None

    def __post_init__(self) -> None:
        if self.length_m <= 0:
            msg = "zone length_m must be positive"
            raise ValueError(msg)
        if self.diameter_m is not None and self.diameter_m <= 0:
            msg = "zone diameter_m must be positive when provided"
            raise ValueError(msg)
        if self.pressure_atm_override is not None and self.pressure_atm_override <= 0:
            msg = "zone pressure_atm_override must be positive when provided"
            raise ValueError(msg)
        if not 0.5 <= self.insulation_factor <= 2.0:
            msg = "zone insulation_factor must be in [0.5, 2.0]"
            raise ValueError(msg)


@dataclass(frozen=True)
class MultiZoneDesign:
    zones: list[ZoneDesign]
    methane_kg_per_hr: float
    pressure_atm: float
    dilution_frac: float
    carbon_removal_eff: float
    ambient_temp_c: float = 25.0
    material_tmax_c: float = 1150.0
    dp_max_kpa: float = 35.0
    power_max_kw: float = 1500.0
    default_diameter_m: float = 0.09

    def __post_init__(self) -> None:
        if not 2 <= len(self.zones) <= 3:
            msg = "MultiZoneDesign supports 2 or 3 zones for MVP"
            raise ValueError(msg)
        if self.methane_kg_per_hr <= 0:
            msg = "methane_kg_per_hr must be positive"
            raise ValueError(msg)
        if self.pressure_atm <= 0:
            msg = "pressure_atm must be positive"
            raise ValueError(msg)
        if not 0 <= self.dilution_frac < 1:
            msg = "dilution_frac must be in [0, 1)"
            raise ValueError(msg)
        if not 0 <= self.carbon_removal_eff <= 1:
            msg = "carbon_removal_eff must be in [0, 1]"
            raise ValueError(msg)
        if self.default_diameter_m <= 0:
            msg = "default_diameter_m must be positive"
            raise ValueError(msg)
        if self.dp_max_kpa <= 0:
            msg = "dp_max_kpa must be positive"
            raise ValueError(msg)
        if self.power_max_kw <= 0:
            msg = "power_max_kw must be positive"
            raise ValueError(msg)
        if self.material_tmax_c <= self.ambient_temp_c:
            msg = "material_tmax_c must exceed ambient_temp_c"
            raise ValueError(msg)

    @property
    def zone_count(self) -> int:
        return len(self.zones)

    @property
    def total_length_m(self) -> float:
        return float(sum(zone.length_m for zone in self.zones))

    @property
    def zone_temperatures_c(self) -> list[float]:
        return [float(zone.temp_c) for zone in self.zones]

    @property
    def zone_diameters_m(self) -> list[float]:
        return [
            float(zone.diameter_m if zone.diameter_m is not None else self.default_diameter_m)
            for zone in self.zones
        ]

    @property
    def zone_pressures_atm(self) -> list[float]:
        return [
            float(
                zone.pressure_atm_override
                if zone.pressure_atm_override is not None
                else self.pressure_atm
            )
            for zone in self.zones
        ]

    @property
    def zone_cross_section_area_m2(self) -> list[float]:
        return [pi * ((diameter / 2.0) ** 2) for diameter in self.zone_diameters_m]

    @property
    def zone_volume_m3(self) -> list[float]:
        return [
            area * zone.length_m
            for area, zone in zip(self.zone_cross_section_area_m2, self.zones, strict=True)
        ]

    @property
    def zone_surface_area_m2(self) -> list[float]:
        return [
            pi * diameter * zone.length_m
            for diameter, zone in zip(self.zone_diameters_m, self.zones, strict=True)
        ]

    @property
    def methane_molar_flow_mol_per_s(self) -> float:
        methane_kg_per_s = self.methane_kg_per_hr / 3600.0
        return methane_kg_per_s / CH4_MW_KG_PER_MOL

    @property
    def total_molar_flow_mol_per_s(self) -> float:
        methane_fraction = max(1e-6, 1.0 - self.dilution_frac)
        return self.methane_molar_flow_mol_per_s / methane_fraction

    @property
    def zone_volumetric_flow_m3_per_s(self) -> list[float]:
        flows: list[float] = []
        for temp_c, pressure_atm in zip(
            self.zone_temperatures_c, self.zone_pressures_atm, strict=True
        ):
            temperature_k = temp_c + 273.15
            pressure_pa = pressure_atm * 101325.0
            flow = (
                self.total_molar_flow_mol_per_s * GAS_CONSTANT_J_PER_MOLK * temperature_k
            ) / max(pressure_pa, 1e-9)
            flows.append(float(flow))
        return flows

    @property
    def zone_residence_time_s(self) -> list[float]:
        return [
            volume / max(flow, 1e-12)
            for volume, flow in zip(
                self.zone_volume_m3, self.zone_volumetric_flow_m3_per_s, strict=True
            )
        ]

    @property
    def effective_carbon_release_factor(self) -> float:
        return 1.0 - self.carbon_removal_eff

    def to_single_zone_equivalent(self) -> ReactorDesign:
        total_volume = float(sum(self.zone_volume_m3))
        total_length = max(self.total_length_m, 1e-9)
        equivalent_diameter = sqrt((4.0 * total_volume) / (pi * total_length))
        weighted_temp = float(
            sum(zone.temp_c * zone.length_m for zone in self.zones) / max(total_length, 1e-9)
        )
        weighted_pressure = float(
            sum(
                pressure * zone.length_m
                for pressure, zone in zip(self.zone_pressures_atm, self.zones, strict=True)
            )
            / max(total_length, 1e-9)
        )
        return ReactorDesign(
            length_m=total_length,
            diameter_m=equivalent_diameter,
            pressure_atm=weighted_pressure,
            temp_c=weighted_temp,
            methane_kg_per_hr=self.methane_kg_per_hr,
            dilution_frac=self.dilution_frac,
            carbon_removal_eff=self.carbon_removal_eff,
        )

    def to_dict(self) -> dict[str, object]:
        zone_rows: list[dict[str, float]] = []
        for idx, zone in enumerate(self.zones):
            zone_rows.append(
                {
                    "index": float(idx + 1),
                    "temp_c": float(zone.temp_c),
                    "length_m": float(zone.length_m),
                    "diameter_m": float(self.zone_diameters_m[idx]),
                    "pressure_atm": float(self.zone_pressures_atm[idx]),
                    "insulation_factor": float(zone.insulation_factor),
                    "residence_time_s": float(self.zone_residence_time_s[idx]),
                    "surface_area_m2": float(self.zone_surface_area_m2[idx]),
                    "volume_m3": float(self.zone_volume_m3[idx]),
                }
            )
        return {
            "zone_count": float(self.zone_count),
            "methane_kg_per_hr": float(self.methane_kg_per_hr),
            "pressure_atm": float(self.pressure_atm),
            "dilution_frac": float(self.dilution_frac),
            "carbon_removal_eff": float(self.carbon_removal_eff),
            "ambient_temp_c": float(self.ambient_temp_c),
            "material_tmax_c": float(self.material_tmax_c),
            "dp_max_kpa": float(self.dp_max_kpa),
            "power_max_kw": float(self.power_max_kw),
            "default_diameter_m": float(self.default_diameter_m),
            "total_length_m": float(self.total_length_m),
            "zones": zone_rows,
        }


@dataclass(frozen=True)
class MultiZoneBounds:
    zone_temp_c: tuple[float, float] = (850.0, 1100.0)
    zone_length_m: tuple[float, float] = (0.25, 1.80)
    zone_diameter_m: tuple[float, float] = (0.05, 0.18)
    zone_insulation_factor: tuple[float, float] = (0.5, 2.0)
    methane_kg_per_hr: tuple[float, float] = (40.0, 240.0)
    pressure_atm: tuple[float, float] = (0.9, 3.0)
    dilution_frac: tuple[float, float] = (0.55, 0.92)
    carbon_removal_eff: tuple[float, float] = (0.0, 0.95)
    default_diameter_m: tuple[float, float] = (0.06, 0.16)
    total_length_m_max: float = 4.5
    ambient_temp_c: float = 25.0
    material_tmax_c: float = 1150.0
    dp_max_kpa: float = 35.0
    power_max_kw: float = 1500.0

    @staticmethod
    def _sample(bounds: tuple[float, float], rng: np.random.Generator) -> float:
        return float(rng.uniform(bounds[0], bounds[1]))

    def sample(self, zones_count: int, rng: np.random.Generator) -> MultiZoneDesign:
        if not 2 <= zones_count <= 3:
            msg = "zones_count must be 2 or 3"
            raise ValueError(msg)
        zones = [
            ZoneDesign(
                temp_c=self._sample(self.zone_temp_c, rng),
                length_m=self._sample(self.zone_length_m, rng),
                diameter_m=self._sample(self.zone_diameter_m, rng),
                insulation_factor=self._sample(self.zone_insulation_factor, rng),
            )
            for _ in range(zones_count)
        ]
        lengths = np.array([zone.length_m for zone in zones], dtype=float)
        if float(np.sum(lengths)) > self.total_length_m_max:
            scale = self.total_length_m_max / float(np.sum(lengths))
            zones = [
                ZoneDesign(
                    temp_c=zone.temp_c,
                    length_m=float(zone.length_m * scale),
                    diameter_m=zone.diameter_m,
                    insulation_factor=zone.insulation_factor,
                    pressure_atm_override=zone.pressure_atm_override,
                )
                for zone in zones
            ]

        return MultiZoneDesign(
            zones=zones,
            methane_kg_per_hr=self._sample(self.methane_kg_per_hr, rng),
            pressure_atm=self._sample(self.pressure_atm, rng),
            dilution_frac=self._sample(self.dilution_frac, rng),
            carbon_removal_eff=self._sample(self.carbon_removal_eff, rng),
            ambient_temp_c=self.ambient_temp_c,
            material_tmax_c=self.material_tmax_c,
            dp_max_kpa=self.dp_max_kpa,
            power_max_kw=self.power_max_kw,
            default_diameter_m=self._sample(self.default_diameter_m, rng),
        )


def mutate_multizone_design(
    design: MultiZoneDesign,
    bounds: MultiZoneBounds,
    rng: np.random.Generator,
    *,
    mutation_scale: float = 0.08,
) -> MultiZoneDesign:
    def _clip(value: float, value_bounds: tuple[float, float]) -> float:
        return float(np.clip(value, value_bounds[0], value_bounds[1]))

    mutated_zones: list[ZoneDesign] = []
    for zone in design.zones:
        zone_temp_span = bounds.zone_temp_c[1] - bounds.zone_temp_c[0]
        zone_length_span = bounds.zone_length_m[1] - bounds.zone_length_m[0]
        zone_diam_span = bounds.zone_diameter_m[1] - bounds.zone_diameter_m[0]
        zone_ins_span = bounds.zone_insulation_factor[1] - bounds.zone_insulation_factor[0]

        diameter = zone.diameter_m if zone.diameter_m is not None else design.default_diameter_m
        mutated_zones.append(
            ZoneDesign(
                temp_c=_clip(
                    zone.temp_c + rng.normal(0.0, mutation_scale * zone_temp_span),
                    bounds.zone_temp_c,
                ),
                length_m=_clip(
                    zone.length_m + rng.normal(0.0, mutation_scale * zone_length_span),
                    bounds.zone_length_m,
                ),
                diameter_m=_clip(
                    diameter + rng.normal(0.0, mutation_scale * zone_diam_span),
                    bounds.zone_diameter_m,
                ),
                insulation_factor=_clip(
                    zone.insulation_factor + rng.normal(0.0, mutation_scale * zone_ins_span),
                    bounds.zone_insulation_factor,
                ),
                pressure_atm_override=zone.pressure_atm_override,
            )
        )

    lengths = np.array([zone.length_m for zone in mutated_zones], dtype=float)
    if float(np.sum(lengths)) > bounds.total_length_m_max:
        scale = bounds.total_length_m_max / float(np.sum(lengths))
        mutated_zones = [
            ZoneDesign(
                temp_c=zone.temp_c,
                length_m=float(zone.length_m * scale),
                diameter_m=zone.diameter_m,
                insulation_factor=zone.insulation_factor,
                pressure_atm_override=zone.pressure_atm_override,
            )
            for zone in mutated_zones
        ]

    methane_span = bounds.methane_kg_per_hr[1] - bounds.methane_kg_per_hr[0]
    pressure_span = bounds.pressure_atm[1] - bounds.pressure_atm[0]
    dilution_span = bounds.dilution_frac[1] - bounds.dilution_frac[0]
    carbon_span = bounds.carbon_removal_eff[1] - bounds.carbon_removal_eff[0]
    default_diam_span = bounds.default_diameter_m[1] - bounds.default_diameter_m[0]

    return MultiZoneDesign(
        zones=mutated_zones,
        methane_kg_per_hr=_clip(
            design.methane_kg_per_hr + rng.normal(0.0, mutation_scale * methane_span),
            bounds.methane_kg_per_hr,
        ),
        pressure_atm=_clip(
            design.pressure_atm + rng.normal(0.0, mutation_scale * pressure_span),
            bounds.pressure_atm,
        ),
        dilution_frac=_clip(
            design.dilution_frac + rng.normal(0.0, mutation_scale * dilution_span),
            bounds.dilution_frac,
        ),
        carbon_removal_eff=_clip(
            design.carbon_removal_eff + rng.normal(0.0, mutation_scale * carbon_span),
            bounds.carbon_removal_eff,
        ),
        ambient_temp_c=design.ambient_temp_c,
        material_tmax_c=design.material_tmax_c,
        dp_max_kpa=design.dp_max_kpa,
        power_max_kw=design.power_max_kw,
        default_diameter_m=_clip(
            design.default_diameter_m + rng.normal(0.0, mutation_scale * default_diam_span),
            bounds.default_diameter_m,
        ),
    )


__all__ = [
    "ZoneDesign",
    "MultiZoneDesign",
    "MultiZoneBounds",
    "mutate_multizone_design",
]
