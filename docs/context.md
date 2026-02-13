# Synora Product Context

## Vision

Synora is building a physics-in-the-loop generative design engine for modular
high-temperature reactors. Methane pyrolysis is the v1 wedge where the
platform couples fast surrogate evaluation with selective physics labeling.

## Core Definitions

- **Surrogate model**: a fast approximation trained on physics-generated data.
  It is used online for optimization, control, and dashboard interactivity.
- **Physics labeling**: offline high-fidelity evaluation (Cantera PFR in v1)
  used to verify candidates and improve surrogate quality.
- **Fouling Risk Index**: renamed carbon proxy that estimates carbon deposition
  tendency from hydrocarbon species signatures.

## What Is A Design

A design is the joint specification of reactor geometry and operating proxies.
At minimum it includes:

- Geometry: `length_m`, `diameter_m`, and related surface/volume structure.
- Operating conditions: `temp_c`, `pressure_atm`, `methane_kg_per_hr`,
  `dilution_frac`.
- Maintenance/carbon handling proxy: `carbon_removal_eff`.

Derived properties such as `residence_time_s` and `surface_area_to_volume`
are computed from the design and feed downstream objectives.

## v1 Workflow

1. Physics dataset generation (`temp`, `tau`, `pressure`) with Cantera PFR.
2. Surrogate fitting on generated labels.
3. Generative optimizer proposes candidate designs using surrogate metrics.
4. Active learning verifies top candidates with physics labeling.
5. Surrogate refits with appended verified data.
