# Synora Development Roadmap

## Phase 1 — Professional Environment Setup
- Pin Python version
- src-layout packaging fixed
- Editable install working
- Ruff + Pytest + Pre-commit configured
- CI green

## Phase 2 — Thin Vertical Slice
- Reactor simulation
- Economics model
- Twin simulator loop
- Streamlit demo

## Phase 3 — Optimization Layer
- Health decay modeling ✅ (`synora.reactor.model`)
- Maintenance scheduling ✅ (`synora.twin.simulator`)
- Linear programming dispatch — deferred/descoped. The generative design layer
  (`synora.generative.optimizer`, surrogate-objective search) replaced the
  originally-planned LP optimization layer.

## Phase 4 — Multi-node Network — deferred/descoped
- Carbon routing, capacity constraints, and multi-node profit maximization were
  not built. Single-asset design exploration + economics is the current scope.
  (The `logistics`/`optimization` stub packages for this phase were removed.)

Each phase must produce:
- Working demo capability
- Passing tests
- No environment regressions
