# Physics-generated dataset (Cantera)

This dataset is generated locally using Cantera reactor simulations for methane
pyrolysis. The PFR is labeled **isothermally** (the reaction temperature is held
fixed) so `temperature_c` matches the surrogate's reaction-temperature feature.

Generate it with the packaged CLI:

```bash
pip install -e ".[physics]"        # installs Cantera
synora-generate-pfr                # or: python -m synora.physics.dataset_cli
```

Output parquet artifacts are written to `data/processed/physics_runs/`.
