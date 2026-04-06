# Argus Project Guidelines

## Code Style
- Prefer small, focused changes that preserve the existing module boundaries under src/argus and web/src.
- Follow the current Python style from pyproject.toml: Python 3.11 target, Ruff line length 100.
- Prefer extending existing services, models, and tests instead of introducing parallel abstractions.
- For frontend work in web/, keep using Vue 3, TypeScript, Vite, Ant Design Vue, and the existing API/composable patterns.

## Architecture
- Treat this repo as two applications: the Python detection backend in src/argus and the Vue dashboard in web/.
- Keep pipeline changes aligned with the documented stage boundaries in [docs/architecture.md](../docs/architecture.md): capture, prefilter, person, anomaly, core orchestration, alerts, storage, and dashboard.
- Configuration is YAML plus Pydantic models. Update both schema and defaults together when adding config.
- Prefer linking to existing design and requirement details instead of restating them. Use [docs/architecture.md](../docs/architecture.md), [docs/requirements.md](../docs/requirements.md), [README.md](../README.md), and [configs/default.yaml](../configs/default.yaml) as the primary references.

## Build And Test
- Use the workspace's configured Python environment and treat [CLAUDE.md](../CLAUDE.md) as the source of truth for local workflow details.
- Do not create or switch Python environments unless the task requires it. Avoid reinstalling the package after simple source-file changes because editable install and worktree behavior are already documented in [CLAUDE.md](../CLAUDE.md).
- Common backend commands: pip install -e ".[dev]", python -m argus, python -m pytest tests/ -v
- Run targeted tests for changed areas when practical, then expand to broader coverage if the change affects shared pipeline behavior.
- Lint Python only if dev dependencies are present: ruff check src/ tests/
- For frontend work in web/: npm install, npm run dev, npm run build

## Conventions
- Check [CLAUDE.md](../CLAUDE.md) before changing workflows or environment assumptions; it documents repo-specific development notes, editable-install behavior, shell expectations, and known test pitfalls.
- There is extensive existing test coverage in tests/unit and tests/integration. Add or update tests when behavior changes, especially for pipeline, alerting, config, and dashboard API logic.
- Avoid copying large documentation blocks into code or instructions. Link to the canonical docs instead.
- Ignore .claude/worktrees when searching for source patterns unless the task is specifically about agent worktrees.
- Treat runtime data under data/ as environment state, not source code to refactor casually.