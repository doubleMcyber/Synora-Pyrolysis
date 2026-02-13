# Synora Context

Synora is building an industrial-grade digital twin + optimization platform
for distributed methane pyrolysis hydrogen networks.

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
