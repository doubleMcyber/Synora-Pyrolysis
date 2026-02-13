# Synora Engineering Decisions

1. Python version pinned via pyenv:
   - Required version: 3.11.x
   - Must not use global system Python or conda base.

2. src-layout required:
   - Package root: src/synora
   - Must use editable install.

3. Dependency policy:
   - No GPL dependencies.
   - Keep dependency surface minimal.
   - Scientific stack allowed (numpy, pandas, scipy).

4. Tooling policy:
   - Ruff required for lint and formatting.
   - Pytest required for testing.
   - Pre-commit must enforce format rules.
   - CI must block on lint/test failures.

5. Development constraints:
   - No code generation outside repo.
   - No sweeping refactors without explicit instruction.
   - No architectural drift.
