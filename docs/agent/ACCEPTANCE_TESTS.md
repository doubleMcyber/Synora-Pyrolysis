# Synora Acceptance Tests — Onion Layers

These tests define done criteria for each incremental layer.

---

## Layer 0 — Repository & Environment
- Python 3.11 in `.venv`
- Editable install works (`pip install -e .`)
- CI executes lint and tests

---

## Layer 1 — Reactor + Economics + Dashboard Slice
Acceptance:
1. `python -c "import synora; print(synora.__version__)"`
2. `run_simulation(hours=24, methane_kg_per_hr=100, temp=800, default econ)` returns expected columns.
3. Dashboard controls update KPIs and plots with no runtime errors.

Tests:
- Unit test for non-negative reactor outputs
- Unit test for deterministic hourly economics

---

## Layer 2 — Fouling + Physics Dataset + Surrogate
Acceptance:
1. Cantera dataset generation script produces parquet under `data/processed/physics_runs/`.
2. Surrogate calibration reports RMSE for conversion, H2 yield, and Fouling Risk Index.
3. Health decays with fouling and can be restored with maintenance.

Tests:
- Test fouling decreases health
- Test maintenance restores health
- Test calibration convergence and error threshold on synthetic data

---

## Layer 3 — Generative Design Engine (LEAP-71 Inspired MVP)
Acceptance:
1. A parametric design schema exists with geometry + operating fields and derived proxies (`residence_time_s`, `surface_area_to_volume`).
2. Surrogate design evaluation returns:
   - conversion
   - h2_rate
   - fouling_risk_index
   - pressure_drop_proxy
   - heat_loss_proxy
   - profit_per_hr
   - constraint_violations
3. Optimizer returns top-K designs that prioritize feasibility.
4. Active-learning loop executes:
   - propose -> label -> append -> refit
   and updates dataset + surrogate artifacts.

Tests:
- Design proxy bounds test
- Surrogate objective keys/non-negative test
- Optimizer top-K and constraints test
- Active-learning one-iteration mock-label test

---

## Layer 4 — Dashboard Design Explorer
Acceptance:
1. Dashboard includes `Design Explorer` tab.
2. Optimization can be launched interactively.
3. Leaderboard table and Pareto frontier render without errors.
4. Candidate inspection shows detailed metrics and time-series simulation.

Tests:
- Smoke run for dashboard start with no runtime exceptions in logs

---

# Minimum Passing Criteria Across All Layers
- `ruff check .` passes
- `pytest -q` passes
- CI is green