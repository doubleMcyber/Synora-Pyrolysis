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

## Layer 5 — Credibility & Confidence Upgrade (Sprint 2)
Acceptance:
1. Surrogate supports bootstrap ensemble uncertainty and returns:
   - mean prediction
   - std deviation
   - +/- 2sigma interval for conversion, H2 yield, and carbon index
2. OOD detection is available on calibrated predictions and in design metrics:
   - `is_out_of_distribution`
   - `ood_score`
3. Design Explorer supports "Physics Verify Selected Design" workflow with clear physics vs surrogate labeling.
4. Design Explorer supports report export for selected design to JSON artifact.
5. Geometry-driven design definition remains intact:
   - residence time is derived from geometry and flow (not direct input).

Tests:
- Uncertainty standard deviations are non-negative
- Ensemble behavior improves RMSE stability vs individual members
- OOD design triggers OOD flag
- Geometry change alters derived residence time
- Physics-verify independent active-learning path still runs in mocked test mode
- Report artifact generation test passes

---

## Layer 6 — Multi-Zone Architecture + Thermal/DeltaP Constraints (Sprint 3)
Acceptance:
1. Multi-zone schema exists:
   - `ZoneDesign`
   - `MultiZoneDesign`
   - supports 2-3 zones and derived per-zone residence times from geometry + flow
2. Multi-zone surrogate objective evaluation returns:
   - overall conversion/H2/fouling metrics
   - zone-wise conversion + confidence fields
   - aggregated OOD status/score
3. Thermal and pressure-drop constraints are enforced via proxy models:
   - `dp_total_kpa`
   - `q_loss_kw`
   - `q_required_kw`
   - violation reasons for material/power/DeltaP limits
4. Multi-zone optimizer returns top-K candidates and prioritizes feasible designs.
5. Dashboard includes `Reactor Architecture` tab with:
   - single-zone vs multi-zone toggle
   - zone profile + uncertainty display
   - multi-zone physics verify flow
   - report export including zone/constraint context

Tests:
- Geometry change updates derived zone residence time
- Multi-zone objective keys/non-negative checks
- Material/DeltaP/power limit violations trigger correctly
- Multi-zone optimizer top-K with feasibility-prioritized ordering
- Import contracts include multi-zone APIs

---

# Minimum Passing Criteria Across All Layers
- `ruff check .` passes
- `pytest -q` passes
- CI is green