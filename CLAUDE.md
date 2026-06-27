# Job Radar — Claude guidance

Observable, evaluated agentic system that finds remote dev roles, drafts applications
under human approval, and tracks them — exposed over MCP. The centerpiece is
**reliability** (evaluation, observability, human-in-the-loop, guardrails), not retrieval.

**`docs/DESIGN.md` is the source of truth** for stack, repo layout, env/config, and phased
delivery. Honor its decisions; do not relitigate them. (During scaffolding the design lives
at the repo root as `DESIGN.md` until it moves to `docs/` in commit 2.)

## Stack

Python 3.12 · FastAPI · LangGraph · FastMCP · Postgres + pgvector (hybrid search: vector +
FTS + RRF) · SQLAlchemy/Alembic · Langfuse + OpenTelemetry · APScheduler. Local Ollama
(`nomic-embed-text`, local generation model) for dev; paid LLM API for the final quality pass.
The whole stack is **async**.

## Environment & tooling

- **uv** owns the env and deps. Python is pinned to 3.12 via `requires-python` — uv
  provisions it regardless of the system Python. Never `pip install` globally.
  - `uv sync --all-groups` — set up / update the env
  - `uv run <cmd>` — run inside the project env
  - Add deps in `pyproject.toml`; commit `uv.lock`. Don't over-declare — libraries are
    added per-feature on their own branches, not pre-stubbed.
- **ruff** is both linter and formatter (config in `pyproject.toml`).
- **pytest** + **pytest-asyncio** (`asyncio_mode = "auto"`).
- **mypy** is configured but **not** a hard gate yet.
- **just** is the task runner: `just lint`, `just fmt`, `just test` (DB targets land later).
- **pre-commit** mirrors CI and adds a gitleaks secret scan.

## Common commands

```bash
just lint        # uv run ruff check .
just fmt         # uv run ruff format . && uv run ruff check --fix .
just test        # uv run pytest
uv run pre-commit run --all-files
```

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, `pytest`, and a
gitleaks scan on every push to `main` and every PR. Keep every commit lint-clean and CI-green.

## Layout (target, from DESIGN.md §19)

`app/` FastAPI + web UI + MCP entrypoint · `src/job_radar/` package (agents, retrieval, fit,
application, guardrails) · `eval/` labeled sets + metrics + golden queries · `infra/` Docker +
scheduler · `docs/` design + diagrams. Subpackages land **with their features**, not as empty
stubs up front.

## Hygiene — load-bearing, this project's whole thesis is safety/observability

- **Never commit** real secrets, `.env`, `.venv/`, data dumps, model weights (`*.gguf`/
  `*.safetensors`), large fixtures, or **any real PII**. No CV, name-as-data, or email in the
  repo. Profile data is runtime data in Postgres, never in git.
- `.env` is gitignored; **`.env.example` is the committed contract** — every required var with
  a safe placeholder. gitleaks runs in pre-commit *and* CI as the backstop.
- Don't fabricate metrics in the README — real numbers land when they exist.

## Keep it intentional — no boilerplate

This repo is a portfolio centerpiece; every committed file is read as a hiring signal. Write
only what the project actually uses. A lean, deliberate file beats a comprehensive template
every time.

- **No generator dumps or template kitchen sinks.** Don't paste the stock GitHub
  `Python.gitignore`, a framework starter, or any "just in case" boilerplate. `.gitignore`
  lists only artifacts *this* stack can produce (uv, ruff, mypy, pytest, build) — not
  pipenv/poetry/pdm/pixi/django/celery/etc.
- **No commented-out template sections** left in config files. If a block isn't used, delete
  it, don't comment it.
- **Don't pre-declare dependencies or stub subpackages** before a feature needs them. Libraries
  and packages land on their own branches with the code that uses them.
- **No dead code, placeholder functions, or `example`/`TODO` scaffolding** left lying around.
- When adding a file, ask: is every line here load-bearing for this project? Cut what isn't.
