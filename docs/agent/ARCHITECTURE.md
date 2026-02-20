# Synora Architecture

Synora is a digital twin platform designed to model, simulate,
and optimize distributed methane pyrolysis hydrogen networks.

This architecture defines module boundaries, public APIs,
data schemas, and invariants for the MVP.

## Module Contracts

### src/synora/reactor
Purpose: Surrogate model for methane pyrolysis reactor behavior.
Public API:
from synora.reactor.model import ReactorInputs, ReactorState, ReactorOutputs, simulate_step, apply_maintenance

Data schemas:
- ReactorInputs
- ReactorState
- ReactorOutputs

Invariants:
- 0 ≤ conversion ≤ max_conversion
- health ∈ [0, 1]

### src/synora/economics
Purpose: Economic evaluation of hourly reactor outputs.
Public API:
from synora.economics.lcoh import EconInputs, hourly_economics

Data schemas:
- EconInputs with prices and cost parameters

Invariants:
- cost ≥ 0
- revenue ≥ 0

### src/synora/twin
Purpose: Runs time-stepped simulations of asset behavior.
Public API:
from synora.twin.simulator import run_simulation

Output:
- DataFrame with time series columns (h2_kg_per_hr, carbon_kg_per_hr, profit_per_hr, lcoh_usd_per_kg, etc.)

### src/synora/calibration
Purpose: Surrogate training, persistence, and prediction utilities.
Public API:
from synora.calibration.surrogate_fit import fit_surrogate, calibrated_predict, calibrate_and_store

Contracts:
- Supports bootstrap surrogate ensembles for uncertainty estimation.
- Stores training-domain metadata for OOD detection.
- Maintains backward compatibility with legacy single-model parameter files.

### src/synora/physics
Purpose: Offline physics labeling via Cantera PFR.
Public API:
from synora.physics.label_pfr import PFRLabeler, label_pfr_case

Constraints:
- Cantera usage is offline labeling only.
- Online runtime paths must not require Cantera import.

### src/synora/generative
Purpose: LEAP-71-inspired parametric design exploration with active learning.
Public API:
from synora.generative.design_space import ReactorDesign, DesignBounds
from synora.generative.multizone import ZoneDesign, MultiZoneDesign, MultiZoneBounds
from synora.generative.objectives import evaluate_design_surrogate
from synora.generative.objectives import evaluate_multizone_surrogate
from synora.generative.optimizer import propose_designs
from synora.generative.optimizer import propose_multizone_designs
from synora.generative.active_learning import run_active_learning
from synora.generative.report import generate_design_report

Contracts:
- Design evaluation uses surrogate and analytic proxies only.
- Multi-zone evaluation runs zones sequentially and aggregates conversion/yield with documented approximations.
- Thermal and pressure-drop checks use fast CFD-lite proxies (Darcy/UA/energy balance style), not CFD solvers.
- Optimizer returns deterministic results for a fixed seed.
- Active learning performs: propose -> physics label -> append -> refit.
- Design report generation exports JSON artifacts for decision traceability.

### src/synora/generative/constraints.py
Purpose: Thermal and pressure-drop proxy models for feasibility checks.
Public API:
from synora.generative.constraints import evaluate_thermal_dp_constraints

Contracts:
- Evaluates `dp_total_kpa`, `q_loss_kw`, and `q_required_kw` with lightweight proxies.
- Flags hard limits: `material_tmax_c`, `dp_max_kpa`, and `power_max_kw`.
- Executes in sub-second candidate loops.

### scripts/physics/generate_pfr_dataset.py
Purpose: Batch physics label generation for calibration datasets.
Contract:
- Calls shared physics wrapper (`synora.physics.label_pfr`) instead of duplicating Cantera logic.
- Writes parquet artifacts to `data/processed/physics_runs/`.

### Dashboard (apps/dashboard/app.py)
- Streamlit UI
- Uses Plotly for charts
- Imports simulation and generative logic from synora package
- Includes design exploration views (leaderboard, Pareto, candidate inspection)
- Supports on-demand physics verification from selected design.
- Supports design report export with uncertainty and OOD context.

## Generative Data Flow

dataset -> fit -> propose -> label -> refit

1. `scripts/physics/generate_pfr_dataset.py` or active-learning labeling produces parquet physics data.
2. `synora.calibration.surrogate_fit` fits surrogate parameters.
3. `synora.generative.optimizer.propose_designs` explores design space with surrogate objectives.
4. `synora.physics.label_pfr` verifies selected candidates with physics.
5. `synora.generative.active_learning.run_active_learning` appends labels and refits surrogate.

## Multi-Zone Evaluation Flow (Sprint 3)

1. Build `MultiZoneDesign` with 2-3 serial zones and global feed/limits.
2. Derive per-zone residence times from geometry + flow (no direct tau inputs).
3. Evaluate each zone with `calibrated_predict(temp, tau)` and aggregate outcomes.
4. Evaluate thermal/DeltaP constraints with `synora.generative.constraints`.
5. Optimize with `propose_multizone_designs` under objective and feasibility penalties.

## Units
- Mass: kg
- Time: hours
- Cost: USD
- Energy: kWh

## Invariants Across Modules
- All public APIs must be importable without error.
- All numeric units follow the above conventions.
- Results must be reproducible given the same inputs.