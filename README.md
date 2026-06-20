# Synora-Pyrolysis

Physics-informed digital twin platform for distributed methane pyrolysis hydrogen
production, integrating reactor simulation, economic modeling, and generative
reactor design into a virtual hydrogen plant.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

Optional physics extra (Cantera) — required only to regenerate the PFR dataset /
run physics labeling:

```bash
pip install -e ".[physics]"
```

## Run the dashboard

```bash
streamlit run apps/dashboard/app.py
```

## Command-line tools

Installed as console scripts by `pip install -e .`:

| Command | Module | Purpose |
|---|---|---|
| `synora-generate-pfr` | `synora.physics.dataset_cli` | Generate the Cantera PFR dataset (needs `.[physics]`) |
| `synora-calibrate` | `synora.calibration.surrogate_fit` | Fit and store surrogate parameters from the latest dataset |

```bash
synora-generate-pfr        # writes data/processed/physics_runs/pfr_dataset_*.parquet
synora-calibrate           # writes data/processed/physics_runs/surrogate_params.json
```

## Development

```bash
ruff format .
ruff check .
pytest -q
```

See `CONTRIBUTING.md` for the full workflow and `docs/agent/` for architecture,
roadmap, and data-source notes.
