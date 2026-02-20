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
At minimum it includes (single-zone or multi-zone):

- Geometry: `length_m`, `diameter_m`, and related surface/volume structure.
- Multi-zone geometry (Sprint 3): 2-3 serial zones with per-zone
  `temp_c`, `length_m`, `diameter_m`, `insulation_factor`.
- Operating conditions: `temp_c`, `pressure_atm`, `methane_kg_per_hr`,
  `dilution_frac`.
- Maintenance/carbon handling proxy: `carbon_removal_eff`.

Derived properties such as `residence_time_s` (or per-zone `residence_time_s`)
and `surface_area_to_volume` are computed from geometry + flow and feed
downstream objectives.

## Multi-Zone Scope (CFD-lite)

Multi-zone is the first geometry step toward richer reactor architecture,
without introducing heavy CFD dependencies.

- Uses sequential surrogate evaluation by zone.
- Uses fast proxy constraints for thermal duty and pressure drop.
- Does not solve full Navier-Stokes, radiation transport, or detailed wall
  conduction PDEs in real time.

## v1 Workflow

1. Physics dataset generation (`temp`, `tau`, `pressure`) with Cantera PFR.
2. Surrogate fitting on generated labels.
3. Generative optimizer proposes candidate designs (single-zone or multi-zone)
   using surrogate + proxy metrics.
4. Active learning verifies top candidates with physics labeling.
5. Surrogate refits with appended verified data.
