# Acceptance Tests for Environment Setup

Environment is considered complete only when:

1. `python --version` inside .venv shows 3.11.x.
2. `pip install -e ".[dev]"` succeeds without errors.
3. `ruff format .` runs successfully.
4. `ruff check .` passes with no errors.
5. `pytest -q` passes.
6. CI pipeline passes on GitHub.
7. Editable install allows:
   python -c "import synora; print(synora.__version__)"
