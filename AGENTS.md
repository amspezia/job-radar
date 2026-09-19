# Job Radar — Codex guidance

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

Embedding-model eval harness (runbook: `docs/EVAL.md`; needs `EVALS_DATABASE_URL`):

```bash
just evals-db-init                          # create + migrate the EVALS database (idempotent)
just evals-db-migrate                       # alembic upgrade head on the EVALS database
just evals-db-status                        # current revision + embedding_* row counts
just embedding-import --tier A --name <t>   # read-only copy of the prod pool, HyDE texts, labels
just embedding-embed --topic <t>            # fill the vector cache
just embedding-run --topic <t> --embedder all --per-text   # exact-cosine rankings per candidate
just embedding-verify --topic <t>           # parity checks against prod's dense arm
just embedding-judge --topic <t> --model <m>               # Gemma silver labels for every job
just embedding-judge-calibrate --topic <t> --judge-run <id>   # agreement with human labels (gate)
just embedding-evaluate --topic <t> --view all --judge-run <id>   # metrics, intervals, controls
just embedding-topics                       # list topics: pool size, tier, labels by source
just embedding-labels-export --topic <t>    # human labels to JSON (backup)
just embedding-labels-import --topic <t> --file <path>     # restore them
```

The existing test suite writes to `DATABASE_URL` when run locally (`just test` included), so run
only the harness tests (`uv run pytest tests/test_evals_*.py tests/test_embedding_*.py`); they use
throwaway `job_radar_evals_test_*` databases.

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, `pytest`, and a
gitleaks scan on every push to `main` and every PR. Keep every commit lint-clean and CI-green.

## Layout (target, from DESIGN.md §19)

`app/` FastAPI + web UI + MCP entrypoint · `src/job_radar/` package (agents, retrieval, fit,
application, guardrails) · `eval/` labeled sets + metrics + golden queries · `infra/` Docker +
scheduler · `docs/` design + diagrams. Subpackages land **with their features**, not as empty
stubs up front.

- `eval/evals_db/` — infrastructure for the separate EVALS database (settings, admin, read-only
  prod reader); its migrations live in `alembic_evals/`.
- `eval/embedding/` — the dense-only embedding-model comparison harness (tiers, embedders, judge,
  scoring, `evaluate`); candidates are defined in `eval/embedders.toml`.

## Hygiene — load-bearing, this project's whole thesis is safety/observability

- **Never commit** real secrets, `.env`, `.venv/`, data dumps, model weights (`*.gguf`/
  `*.safetensors`), large fixtures, or **any real PII**. No CV, name-as-data, or email in the
  repo. Profile data is runtime data in Postgres, never in git.
- `.env` is gitignored; **`.env.example` is the committed contract** — every required var with
  a safe placeholder. gitleaks runs in pre-commit *and* CI as the backstop.
- The embedding-eval harness only ever **reads** prod, through a read-only connection (this
  guards against accidents, not a deliberate override). Its data lives in the separate EVALS
  database and gitignored `data/`, and `cv_text` is scrubbed of PII (best effort) before it is
  stored.
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
