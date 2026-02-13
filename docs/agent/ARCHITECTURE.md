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
from synora.generative.objectives import evaluate_design_surrogate
from synora.generative.optimizer import propose_designs
from synora.generative.active_learning import run_active_learning

Contracts:
- Design evaluation uses surrogate and analytic proxies only.
- Optimizer returns deterministic results for a fixed seed.
- Active learning performs: propose -> physics label -> append -> refit.

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

## Generative Data Flow

dataset -> fit -> propose -> label -> refit

1. `scripts/physics/generate_pfr_dataset.py` or active-learning labeling produces parquet physics data.
2. `synora.calibration.surrogate_fit` fits surrogate parameters.
3. `synora.generative.optimizer.propose_designs` explores design space with surrogate objectives.
4. `synora.physics.label_pfr` verifies selected candidates with physics.
5. `synora.generative.active_learning.run_active_learning` appends labels and refits surrogate.

## Units
- Mass: kg
- Time: hours
- Cost: USD
- Energy: kWh

## Invariants Across Modules
- All public APIs must be importable without error.
- All numeric units follow the above conventions.
- Results must be reproducible given the same inputs.