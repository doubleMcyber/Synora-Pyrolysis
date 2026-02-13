# Synora Context

Synora is building an industrial-grade digital twin + optimization platform
for distributed methane pyrolysis hydrogen networks.

Product vision (current):
- Physics-in-the-loop generative design engine for modular high-temperature reactors.
- Methane pyrolysis is the v1 wedge.

This repository contains the MVP implementation.

## Architecture Overview

- Python 3.11.9 pinned via pyenv (.python-version).
- src-layout packaging:
  - All product code lives in src/synora/.
  - The package name must remain `synora`.
- Dashboard:
  - Streamlit app located in apps/dashboard/.
  - The dashboard must import logic from src/synora.
- CI:
  - GitHub Actions must pass.
  - Linting and tests are mandatory.

## Principles

- Clean modular architecture.
- Deterministic simulation unless explicitly modeling uncertainty.
- Surrogate-first online evaluation; physics labeling offline.
- Small vertical slices.
- All functionality must be runnable locally.
- Editable install required (pip install -e .).
- Do not introduce heavy frameworks.
- Do not refactor folder structure unless explicitly instructed.

## Goal

Environment must be industrial-grade:
- Pinned Python version
- Ruff for lint/format
- Pytest for testing
- Pre-commit hooks enforced
- CI gates clean

## Shared Vocabulary

- Surrogate model: fast approximation used for online design and simulation.
- Physics labeling: offline Cantera evaluation used to verify/refit surrogate.
- Fouling Risk Index: carbon-formation proxy used for health/constraint logic.
- Design: parametric reactor geometry + operating condition tuple with derived proxies.
