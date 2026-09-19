# Embedding Eval — Implementation Plan, Stage 1

Locks down [EMBEDDING_EVAL_DESIGN.md](EMBEDDING_EVAL_DESIGN.md) **Stage 1**: build the EVALS
database and the Tier-A dataset for the real profile, embed its ~1.4k-job pool with every
candidate embedder, rank, verify, label **every** job with the Gemma judge, and evaluate. Fact
references (F1…F22) point at the design doc's verified-facts table; `R-…` IDs point at
[EMBEDDING_EVAL_REVIEW_REPORT.md](EMBEDDING_EVAL_REVIEW_REPORT.md), whose findings this revision
folds in.

**Not in this plan:** the full blind-labeling queue (Stage 2; the first ~50 blind labels *are* in it via `label-blind`), the pre-registered decision rule
(Stage 2); Tier B (Stage 3); Tier C; cutover (Stage 4). The code is shaped so each is an addition
(design §3.2, §7 below). Stage 1 **does** ship a paired document bootstrap and a query-sensitivity
table (R-M9): without an interval a one-profile comparison is not readable.

## 1. Done means

On the real profile's frozen pool (1,395 jobs on 2026-09-18), in the EVALS database:

1. **Prod is untouched** — imported through a read-only connection whose guard is tested; prod row
   counts (`jobs`, `profile`, `eval_labels`, `fit_judgments`) and the prod Alembic revision are
   identical before and after. (The guard stops accidents, not a deliberate `SET` — R-m4.)
2. **Topic imported and frozen:** every eligible job and the 3 frozen HyDE texts; **no labels are
   imported** (the prod `eval_labels` were produced by the fit LLM, not a human — R-M10). `cv_text` is
   scrubbed of name, email, phone and URLs (R-M6).
3. **Embedded and ranked with every candidate** in `eval/embedders.toml` (`nomic-v1.5`,
   `nomic-v1.5-hyde-query`, `qwen3-0.6b-noinst`, `qwen3-0.6b-instruct`,
   `qwen3-0.6b-instruct-d768`, `bge-m3`), all on the production text: full rankings stored,
   docs/s and per-document `truncated` counts recorded.
4. **Both parity checks pass** (`parity_ranking`, `parity_reconstruction`).
5. **~50 blind human labels** (`human_blind`, ~30 pooled + ~20 random, made with `label-blind`) exist;
   **Gemma labeled all jobs** (`llm_judge`) with a frozen prompt version; failures retried to zero
   or listed; a `judge_calibration` check against the blind labels exists for that judge run (passed or
   failed — R-M4).
6. `evaluate` prints and writes, under `silver`, `human` (condensed only) and `effective`: nDCG@100,
   P@100, AP, nDCG@10, Recall@100 **with its ceiling**, bpref, judged@k, slices, paired-bootstrap
   intervals vs the incumbent, negative controls, query sensitivity, check status and run metadata.
   Judge-based views without a passed calibration are stamped `UNCALIBRATED`.
7. `just lint` is clean and **no test that passed at baseline fails** (baseline: 322 passed,
   1 known failure — R-M8); the contract tests (§6) and the isolation tests pass.

## 2. Conventions

- Match the repo: Python 3.12, fully async (except admin/CLI entry points), SQLAlchemy 2.0
  `Mapped[...]`, `psycopg`, `uv`, ruff (line length 100), `pytest` with `asyncio_mode = "auto"`.
  No boilerplate: every file is load-bearing, no placeholder functions, no commented-out sections,
  no unused protocol or column (CLAUDE.md, R-m8).
- **Gates mean "no new failures".** Baseline `pytest -q` on a migrated empty DB: 322 passed, 1
  failed (`tests/test_eval_gate.py::test_ndcg_meets_golden_threshold`, pre-existing, R-M8).
- **Never write prod, in code or tests.** The repo's *existing* suite writes to `DATABASE_URL`
  (R-M8) — run only your own test files locally. Every new prod-facing test targets the
  **throwaway EVALS database** as a stand-in for prod (same guard mechanism, zero real-prod risk).
- Dependencies: nothing new in `[project.dependencies]`. Declare **`numpy`** in the `dev` group
  beside `ranx` **and run `uv lock`** in the same change (CI uses `--locked`, R-m5). `httpx`,
  `alembic`, `pgvector`, `pydantic-settings`, `psycopg` are already declared; `tomllib` is stdlib.
- **No `ci.yml` change.** Test fixtures create throwaway databases on the server named by
  `DATABASE_URL` (superuser `job_radar`, same image in CI); `EVALS_DATABASE_URL` is only for the
  CLI (R-m5).
- Git: one branch, `feat/embedding-eval-stage-1`; single-line `feat:`/`fix:` subjects, no trailers
  (project memory). Agents **never commit**; the controller decides.
- Ollama for anything that embeds or judges: `just` recipes point at the eval instance
  (`OLLAMA_BASE_URL={{_EVAL_OLLAMA_URL}}`, port 11435). Judge and embedders run one after the
  other (Gemma is ~8 GB). Tests never call a real Ollama.
- The harness commands `import-topic` and `verify` import prod modules, which need the four prod
  variables (`DATABASE_URL`, `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`, `GENERATION_MODEL`) — they are
  in `.env` (R-M5). Everything else needs only `EVALS_DATABASE_URL`.

## 3. Layout and dependency rules

```
alembic_evals.ini · alembic_evals/{env.py, script.py.mako, versions/}
eval/
  evals_db/     __init__ settings base admin prod_reader cli
  llm/          __init__ ollama
  embedding/
    __init__ models labels ranking scoring cache runner evaluate checks parity labels_io label_blind cli
    tiers/      __init__ base tier_a
    embedders/  __init__ base ollama registry
    methods/    __init__ base dense
    judge/      __init__ base prompt fewshot calibration ollama_judge runner
  embedders.toml
tests/          test_evals_*.py · test_embedding_*.py · test_embed_text.py
```

**Import rules** (enforced by tests, §4 WP2 and §6):

- Only `evals_db/prod_reader.py`, `embedding/tiers/tier_a.py` and `embedding/parity.py` may import
  `job_radar.*`, and only from `job_radar.db.models`, `job_radar.retrieval.vector`,
  `job_radar.retrieval.filters` and `job_radar.ingest.embed_text` — **never**
  `job_radar.db.base`, `job_radar.config`, `job_radar.fit`, `job_radar.adapters`. Nothing in the
  harness may import `eval.run`, `eval.qrels` or `eval.label` (F13).
- Nothing else may import those three modules at import time: `embedding/cli.py` imports
  `tier_a` / `parity` **lazily, inside the command function**.
- Enforced two ways: (a) an AST test of direct imports; (b) a **subprocess test** that imports every
  other harness module in a clean interpreter with the four prod variables stripped and asserts no
  `job_radar*` / `langfuse` module was loaded (R-M5: the AST test alone misses transitive and
  dynamic imports).
- `eval/metrics.py` stays pure; the harness imports it, never the reverse.

## 4. Work packages

### WP0 — Prerequisites (manual, once)

1. `docker compose -f infra/docker-compose.yml up -d` (Postgres up).
2. `ollama pull qwen3-embedding:0.6b && ollama pull bge-m3`; `ollama list` shows
   `qwen3-embedding:0.6b` (Q8_0) and `bge-m3:latest` (F16). (Done on the reviewer's machine.)
3. Add to local `.env`:
   `EVALS_DATABASE_URL=postgresql+psycopg://job_radar:job_radar@localhost:5432/job_radar_evals`.
4. The Qwen3 end-of-text diagnostic is **answered** (R-resolved: Ollama vs HF reference cosine
   ≥ 0.9994, 30 texts); nothing to run.

### WP1 — Two small, safe prod-side changes

**Goal:** the harness can build the *exact* text production embedded, and score AP.

- `src/job_radar/ingest/embed_text.py`:
  `build_embed_text(title, description, requirements, responsibilities) -> str` — title +
  requirements + responsibilities joined by `\n`, empties dropped (`filter(None, …)`, which also
  drops an empty title, as prod does), when either extracted field is truthy, else
  `f"{title}\n{description}"`. Accepts `None` for the two fields (the DB stores `x or None`, F3). Pure —
  **no imports from `job_radar`**. `ingest/pipeline.py::_prepare` calls it; behavior unchanged.
- `eval/metrics.py`: add `average_precision(ranking, labels, rel_threshold=2)` — mean of precision at
  each relevant document's rank, over all relevant documents in `labels` (0.0 with none).
- Tests: `tests/test_embed_text.py` (both branches; `None` and empty fields; empty title; equals the
  legacy inline expression over a table of cases); `average_precision` cases in
  `tests/test_eval_metrics.py` (hand-computed values, no relevant docs → 0.0, perfect → 1.0).

**Gate:** `tests/test_ingest*`, `test_pipeline`, `test_eval_metrics` still pass; ruff clean.

### WP2 — EVALS database infrastructure

**Goal:** `just evals-db-init` creates and migrates `job_radar_evals`, idempotently, without ever
touching the prod database; tests get throwaway databases.

- `eval/evals_db/settings.py` — `EvalsSettings(BaseSettings)`, `env_file=".env"`, `extra="ignore"`:
  `evals_database_url: str`, `database_url: str | None = None` (prod; used by `import-topic` and
  `verify`), `ollama_base_url: str = "http://localhost:11434"`. A validator **rejects an
  `evals_database_url` whose database name equals `database_url`'s** (and a full-URL match).
  `get_settings()` is `lru_cache`d. Does not import `job_radar.config`. Note `job_radar.config`
  calls `load_dotenv()` at import, so in a pytest process `.env` values are in `os.environ` —
  settings tests pass explicit kwargs and `_env_file=None`.
- `eval/evals_db/base.py` — *written by the controller before the agents start* (see §8): `EvalsBase`,
  `make_engine(url)`, lazily cached `get_engine()`, and `evals_session(url=None)` async context
  manager. No engine at import time.
- `eval/evals_db/admin.py` — **synchronous**, `psycopg` with `autocommit=True`:
  `database_name(url)`, `with_database(url, name)`, `ensure_database(url) -> bool` (created?; the
  identifier is validated against `[A-Za-z0-9_]+` and quoted — `CREATE DATABASE` can't be
  parameterized), `drop_database(url)` (**refuses any name without `_test_`**),
  `migrate(url=None)` (runs `alembic upgrade head` through `alembic.command` with
  `alembic_evals.ini` located via `Path(__file__)`, not the CWD, `sqlalchemy.url` set on the
  `Config`).
- `alembic_evals.ini` + `alembic_evals/env.py` + `alembic_evals/script.py.mako` — mirror the prod
  async `env.py` (NullPool, `run_sync`), but `target_metadata = EvalsBase.metadata`, import
  `eval.embedding.models`, URL = the config's `sqlalchemy.url` when set (tests/`migrate`), else
  `EvalsSettings.evals_database_url`. **The mako must `import pgvector.sqlalchemy`** (R-m1: the
  autogenerated migration references `pgvector.sqlalchemy.vector.VECTOR()`).
  Migration `0001_enable_pgvector`: `CREATE EXTENSION IF NOT EXISTS vector`. Migration
  `0002_embedding_tables`: autogenerated from `eval/embedding/models.py` (schema contract, §7.1),
  read line by line.
- `eval/evals_db/cli.py` — `job-radar-evals-db`: `init` (ensure database, then migrate), `migrate`,
  `status` (current revision, row counts of `embedding_*`). Synchronous.
- `pyproject.toml`: script entries `job-radar-evals-db = "eval.evals_db.cli:main"` and
  `job-radar-eval-embedding = "eval.embedding.cli:main"`; `numpy` in `dev`; **`uv lock`**.
  `justfile`: `evals-db-init`, `evals-db-migrate`, `evals-db-status`.
  `.env.example`: `EVALS_DATABASE_URL` with a comment that it is a **separate database the harness
  owns; prod is only ever read**.
- **Test fixtures** (`tests/conftest.py`, owned by this WP): `evals_db_url` — session-scoped, *sync*:
  creates `job_radar_evals_test_<uuid8>` on the server of `DATABASE_URL`, runs `migrate` **in a
  worker thread** (verified: Alembic's `asyncio.run` inside a thread works even under a running
  loop, R-m1; a plain call would be hostile to pytest-asyncio's loop), yields the URL, drops the
  database at teardown (`drop_database`, which refuses non-`_test_` names). `evals_session` —
  function-scoped async: asserts the name contains `_test_`, `TRUNCATE`s every table in
  `EvalsBase.metadata.sorted_tables` (`RESTART IDENTITY CASCADE`), yields an `AsyncSession`. Neither
  fixture is autouse.
- **Isolation tests** (`tests/test_evals_isolation.py`, written here because they scan the tree
  dynamically): AST rule of direct imports + the **subprocess** check of §3.
- Tests: settings validator (same URL, same name, different name; explicit kwargs); `ensure_database`
  idempotent against a throwaway name; `drop_database` refuses a non-`_test_` name; `migrate` on a
  throwaway DB reaches head; `alembic check` clean.

**Gate:** `just evals-db-init` twice in a row succeeds; `job_radar_evals` has the extension and head;
prod `alembic current` unchanged; the new tests pass; no `ci.yml` diff.

### WP3 — Read-only prod reader

**Goal:** the only way the harness reads prod is a connection that cannot write by accident.

- `eval/evals_db/prod_reader.py` — `prod_session(url=None)` async context manager: a **`NullPool`**
  engine (no state leaks between sessions) from `EvalsSettings.database_url` (clear error if unset)
  with `connect_args={"options": "-c default_transaction_read_only=on"}`; on entry execute
  `SHOW transaction_read_only` and raise `ProdNotReadOnly` unless `on`; roll back on exit, never
  commit. Also `table_counts(session, tables=("jobs","profile","eval_labels","fit_judgments")) ->
  dict[str, int]` and `alembic_revision(session) -> str | None` for the before/after report.
- Test `tests/test_evals_prod_reader.py` — **the target is the throwaway EVALS database** passed as
  the "prod" URL (R-M8): the setting is `on`; a plain `SELECT` works; `CREATE TABLE _ro_probe_x`,
  `INSERT`, `UPDATE … WHERE false`, `DELETE … WHERE false` and `TRUNCATE` on a **nonexistent** table
  each raise `DBAPIError` matching `read-only transaction`; a monkeypatched engine without the option
  makes `prod_session()` raise `ProdNotReadOnly`; a connection left with
  `SET default_transaction_read_only=off` is rejected by the entry check (documents R-m4).

**Gate:** tests green; on the dev DB, prod row counts unchanged afterwards.

### WP4 — Models and migration

**Goal:** the ten `embedding_*` tables of the schema contract (§7.1) — design §2.2 minus the columns
cut in R-m8.

- `eval/embedding/models.py` — *written by the controller before the agents start* (§8), so every
  other package can code against it. Decisions: UUID primary keys (`as_uuid=True`);
  `embedding_job.id` = deterministic `uuid5` of `(origin, origin_id, embed_text_sha)` under a fixed
  namespace; `embedding_vector.vector` and `embedding_job.prod_embedding` are
  `pgvector.sqlalchemy.Vector()` **with no dimension** (verified: round-trips 3-d and 1024-d in one
  column; `alembic check` clean); `embedding_label.id` and `embedding_check.id` are `BIGINT IDENTITY`
  so "latest" is unambiguous (`now()` is constant inside a transaction, R-M4);
  `embedding_label.grade` has `CHECK (grade BETWEEN 0 AND 3)`; `source` is plain `TEXT` validated in
  `labels.py`; JSON columns are `JSONB`; timestamps `timestamptz` default `now()`;
  `embedding_run_ranking` is unique on `(run_id, job_id)`.
- Migration `0002_embedding_tables` (owned by WP2's agent).
- Tests (`tests/test_embedding_models.py`, throwaway DB): insert/read every table; duplicate
  document key rejected; grade 4 rejected; **a 3-dim and a 1024-dim vector round-trip in the same
  column**; `alembic check` reports no drift.

### WP5 — Labels (pure)

**Goal:** the single place that defines sources, precedence, views and effective-label resolution.

- `eval/embedding/labels.py` — `SOURCES` in precedence order (`human_blind`, `constructed`,
  `llm_judge`), `HUMAN_SOURCES` (the first two), `VIEWS` (`silver`, `human`, `effective`);
  `LabelRow(job_id, grade, source, judge_run_id, id)` (`id` orders rows: larger = newer);
  `resolve_effective(rows, view, judge_run_id=None) -> dict[job_id, grade]`: per view keep only its
  sources; **`silver` and `effective` require `judge_run_id`** (raise `ValueError` otherwise) and
  count only `llm_judge` rows of that run, so prompt versions never mix (R-M4); within a source the
  row with the largest `id` wins; across sources the highest precedence wins. `is_partial(view)`
  (`human` only). No prod-label parser: nothing imports prod labels.
- Tests: precedence; latest-wins; a retest supersedes the older row (even lowering the grade); view
  exclusion; judge-run pinning; missing pin raises; an `llm_judge` row from another run is ignored;
  unknown source rejected.

**Gate:** pure tests green, no DB.

### WP6 — Topic framework, Tier-A builder, import

**Goal:** `import-topic --tier A` copies the pool, frozen HyDE texts and labels into the EVALS
database without writing prod, through a tier-agnostic core.

- `tiers/base.py` — dataclasses `JobRecord`, `LabelRecord`, `TopicPayload` (§7.2); the `TopicBuilder`
  protocol (`tier: str`, `async build(name, **opts) -> TopicPayload`); `job_uuid(origin, origin_id,
  embed_text_sha)`; and **`persist_topic(session, payload) -> PersistSummary`** — one transaction:
  raise `TopicExists` if the name exists; upsert documents (`INSERT … ON CONFLICT (id) DO NOTHING`);
  insert membership and labels; set `status='frozen'`; commit. Tier-agnostic: nothing in it mentions a
  profile, HyDE or eligibility.
- `tiers/tier_a.py` (**prod-touching, allow-listed**) — `TierABuilder.build(name, profile_id=None)`:
  1. Inside `prod_session()`: load the profile (`source='real'` unless an id is given).
  2. `query_inputs = {"hyde_texts": [...]}` by parsing `profile.dense_query_cache` JSON directly —
     **never** `build_hyde_embedding`, which writes (F19); a clear "run `job-radar-fit` once to
     populate the cache" error if empty.
  3. Corpus = `Job.id` where `build_profile_filter(profile)` **and** `Job.embedding IS NOT NULL`,
     ordered by id; per job `embed_text = build_embed_text(...)`, `embed_text_sha`,
     `prod_embedding = Job.embedding`.
  4. `profile_snapshot`: `target_titles`, `seniority`, `years_experience`, `tech_stack`, `domains`,
     `location_rules`, `remote_required`, `salary_floor`, `currency`, `work_history` reduced to role +
     years, and `cv_text` **scrubbed** by `scrub_cv_text(cv_text, full_name)`: email addresses, phone-like
     numbers, URLs and the name's tokens removed (R-M6; verified the raw CV contains all four). This
     shape is the `CandidateBrief` schema the judge consumes.
  5. Labels: none. The profile's prod `eval_labels` were produced by the fit LLM, not a human, so they
     are never read (R-M10); `payload.labels` is `[]` and gold comes from `label-blind`.
  All of it is fetched first into plain dataclasses and assembled by a **pure `assemble()`**
  function, so the logic is unit-testable without prod.
- `import_topic(name, tier, **opts)` (called by the CLI): builds, persists, and prints pool size,
  HyDE-text count, and prod row counts (incl. `eval_labels`) and Alembic revision **before and
  after** (aborting on any difference).
- `labels_io.py` — `export_labels(session, topic_name, path)` writes the **human-source** label rows
  (`human_blind`, `constructed`) as JSON keyed by `origin_id` + `embed_text_sha` (no
  text); `import_labels(session, topic_name, path)` re-inserts them (R-m6: the design's promised
  backup path).
- Tests: `assemble()` with fake rows (**value-level PII test**: a fake CV containing a name, email,
  phone and URL — none survives in the snapshot; no labels in the payload; empty-cache
  error); `persist_topic` (throwaway DB): a second persist of the same name fails, a shared posting
  is one document across two topics, a changed `embed_text` becomes a new document id, ≥ 1,400-row
  batch inserts succeed; `labels_io` round trip.
  **The real-data import is the manual acceptance run.**

**Gate:** on real data — pool count matches F10, prod counts unchanged.

### WP7 — Ollama client, embedders, cache, dense method, runner

**Goal:** `run` embeds the pool with any configured embedder and stores its full ranking.

- `eval/llm/ollama.py` — `OllamaClient(base_url, transport=None)` over `httpx.AsyncClient`:
  `embed(model, text, *, num_ctx, truncate=True) -> EmbedResponse(vector, prompt_eval_count)`;
  a 400 whose body contains `exceeds the context length` raises `ContextOverflow`;
  `chat_json(model, messages, schema, *, options) -> ChatResult(data, prompt_eval_count, eval_count,
  prompt_eval_seconds, seconds)` (`/api/chat`, `stream=False`, `format=schema`);
  `tags() -> dict[str, TagInfo(digest, quantization, context_length)]` with names normalized so
  `"bge-m3"` and `"bge-m3:latest"` both resolve; `version()`; `warm_embedder(model, num_ctx)` —
  a real `/api/embed` call, because `/api/generate` **400s for embedding models** (R-m2);
  `warm_chat(model)` via `/api/generate`. Three attempts with backoff on connect/read timeouts, 5xx
  **and a 400 whose body contains `EOF`** (observed when a resident runner changes `num_ctx`).
  Timeouts sized for a cold load. No Langfuse, no `job_radar` imports (F12).
- `embedders/base.py` — `EmbedderSpec` (frozen; `dim` requires `mrl`; `hyde_prefix=None` means "same
  as `doc_prefix`"), `EmbedResult(vector, n_tokens, truncated)`, the `Embedder` protocol (`spec`,
  `async ready()`, `async embed(text) -> EmbedResult`, `describe()`), `apply_dim(vec, dim)` (slice,
  then re-normalize — F4), `fingerprint(backend, digest, num_ctx)`.
- `embedders/ollama.py` — `OllamaEmbedder`: **truncation policy (R-M7):** embed with
  `truncate=False`; on `ContextOverflow` retry with `truncate=True` and mark the result
  `truncated=True`. (`prompt_eval_count` is recorded as `n_tokens` but **never** used to detect
  truncation: nomic and bge-m3 saturate at 2,048 whatever `num_ctx` says.) `ready()` confirms the
  tag is pulled (actionable `ollama pull …` error) and warms it; `describe()` returns digest,
  quantization, Ollama version and the requested `num_ctx`.
- `embedders/registry.py` — `load_specs(path=None)` reads `eval/embedders.toml` (`tomllib`); validates
  unique names, **exactly one `incumbent`**, a known `backend` (`ollama`), `dim`/`mrl` consistency;
  `build_embedder(spec, client)` (a plain function — one backend, no dispatch table).
- `eval/embedders.toml` — the six variants (design §3.3). Qwen3's hyde prefix is
  `"Instruct: Given a job description, retrieve similar job postings\nQuery:"` — no space after
  `Query:`; `nomic-v1.5-hyde-query` uses `hyde_prefix = "search_query: "`.
- `ranking.py` — pure: `cosine_rank(matrix, ids, query) -> list[(id, score)]` sorted by
  `(-score, id)` (F6), and `compare_rankings(a, b, k, eps)` (§7.3).
- `cache.py` — `VectorCache`: `get_many(fingerprint, shas)`, `put_many(...)` (`ON CONFLICT DO
  NOTHING`; stores `n_tokens` and `truncated`), `ensure_embedder_row(...)`. Vectors are cached
  **native-dimension**; read back through the ORM column (`ndarray[float32]`).
- `methods/base.py` — the `RankingMethod` protocol (§7.4), `CorpusDoc`, `TopicView`,
  `load_topic_view(session, name)`.
- `methods/dense.py` — `DenseMethod(embedder, cache, query_recipe="hyde_mean")`. `prepare`:
  `text_sha = sha256(prefix + text)`; fetch cached; embed only the missing at concurrency 8, progress
  every 25 (F8); embed the HyDE texts with `hyde_prefix`; record docs/s and the count of
  `truncated` vectors. `rank`: apply `dim`, the query is the **mean of the HyDE vectors**
  (`query_recipe="hyde_mean"`) or a single text (`"hyde_text:<i>"` — the free query-sensitivity
  runs, R-M9), `cosine_rank`. The method fingerprint covers the embedder fingerprint, both prefixes,
  `dim` and the recipe.
- `runner.py` — `run_method(session, topic_name, method, git_sha=None) -> run_id`: create
  `embedding_run` (`running`), `prepare`, `rank`, bulk-insert `embedding_run_ranking`, close the run
  with timings and stats. Method-agnostic.
- Tests (throwaway DB, **fake embedder**, no Ollama): `cosine_rank` vs a brute-force reference
  **including an all-identical-vectors case** (F6); `apply_dim`; registry validation; a second
  `prepare` makes **zero** embed calls; a changed prefix changes the sha and the method
  fingerprint; the `d768` spec re-uses cached native vectors (call counter); a fake embedder that
  overflows once yields `truncated=True` and one retry; `OllamaClient` request shapes, the 400
  mapping, retry and tag normalization via `httpx.MockTransport`.

**Gate (manual, real data):** all six variants complete; docs/s per embedder recorded (measured on
2026-09-18: nomic ≈ 109, bge-m3 ≈ 41, Qwen3 ≈ 30); the second run of any variant is a pure cache hit.

### WP8 — Verify: parity checks

**Goal:** prove the imported dataset and the harness reproduce production's dense arm before any
comparison is trusted.

- `checks.py` — `record_check(session, topic_id, name, passed, detail, judge_run_id=None)`,
  `latest_checks(session, topic_id)`, `REQUIRED_CHECKS = (parity_ranking, parity_reconstruction)`.
- `parity.py` (**prod-touching, allow-listed**) — `verify_topic(name)`:
  - **`parity_ranking`:** take the incumbent's query vector from the harness; inside `prod_session()`
    call prod's own `search_vector(session, vec, 100, filter)` with the profile filter **and
    `Job.id IN (topic's origin ids)`**; rank the same vector over the imported `prod_embedding`
    vectors with `cosine_rank`; `compare_rankings(..., k=100, eps=1e-5)` (§7.3: verified on real
    data — sets equal, order differs only across byte-identical ties, max |Δ| 2.4e-7). Never calls
    `build_hyde_embedding`.
  - **`parity_reconstruction`:** with the incumbent's cached vectors, the fraction of documents with
    cosine ≥ 0.999 to `prod_embedding` must be ≥ 99% (measured: all 1,395 at ≥ 0.999999); misses
    listed by `source` and embed-text branch. Also stores the stored-vs-re-embedded metric delta as
    the noise floor.
- Tests: `compare_rankings` (equal, permuted ties accepted, a swapped non-tie rejected, set mismatch,
  score drift, tie group crossing rank `k`); reconstruction on synthetic vectors in the throwaway DB.
  The prod comparison itself is manual acceptance.

**Gate:** both checks pass on real data. If `parity_reconstruction` fails, **stop**.

### WP9 — The Gemma judge

**Goal:** a complete, reproducible, calibrated silver labeling of the pool.

- `judge/base.py` — `CandidateBrief.from_snapshot`, `JobView`, `Judgment(grade, rationale,
  prompt_eval_seconds)`, and the `Judge` protocol (`name`, `describe()`, `async grade(job)` — the
  brief and few-shot are fixed at construction so the prefix is stable).
- `judge/prompt.py` — `PROMPT_VERSION`, the rubric (EVAL.md grade definitions), `JUDGMENT_SCHEMA`
  (`reason` first, `grade` an integer enum 0–3), `build_static_prefix(brief, fewshot)` and
  `render_label_view(job)`. **Static content first, job last**, so the prefix is byte-identical
  across calls. **The job section mirrors `eval/label.py::_show_job` (R-M3):** title, company,
  source, seniority, `requirements` / `responsibilities` **each clipped to 400 chars** (with `…`),
  the description clipped to 400 only when both are empty — **no location**. That is what the user's
  `label-blind` labels are graded on; the embedders and the fit stage see the text unclipped (85% of pool
  documents are clipped in this view; a full-text view is the Stage-2 diagnostic). `prompt_sha`
  hashes the static prefix template.
- `judge/fewshot.py` — `select_fewshot(labels, per_grade, seed)` and `split_dev_test(ids, seed)`,
  pure and deterministic; exemplars are excluded from calibration.
- `judge/calibration.py` — pure: quadratic-weighted κ, confusion matrix, exact and within-1
  agreement, binary (≥ 2) precision/recall/F1.
- `judge/ollama_judge.py` — `OllamaJudge(client, model, brief, fewshot, seed=1, num_ctx=8192,
  num_predict=300)`; system message = static prefix, user message = job section; temperature 0;
  invalid JSON or an out-of-range grade is a failure for that job (retried), never a silent default;
  never receives HyDE text, ranks or scores. Verified in the review: 100/100 schema-valid outputs;
  the prefix cache works sequentially (`prompt_eval_duration` 3.35 s → 0.24–0.45 s; use the
  *duration*, `prompt_eval_count` still reports the full count).
- `judge/runner.py` — `label_topic(session, topic_id, judge, fewshot_ids, concurrency, limit,
  resume_run_id) -> judge_run_id`: create an `embedding_judge_run` (model, digest, prompt
  version/sha, params incl. the clip constant, few-shot); label every corpus job lacking a label in
  that run; write `llm_judge` label rows in batches of 25; progress every 25; count failures;
  resumable. `calibrate(session, topic_name, judge_run_id, min_kappa, use_split=False)` computes the
  calibration on the human view (`human_blind` / `constructed`) **excluding the few-shot exemplars**
  (default: all of them; `use_split=True` / `--held-out-half` uses the held-out half), stores
  `judge_calibration` (with `judge_run_id`, threshold, `passed`), and prints the caveats: the labels
  are the user's own blind grades on the 400-char view, and with n ≈ 46 the κ_w SE ≈ 0.1 (the actual
  n is stated). `build_judge` draws the exemplars (default 1 per grade = 4) from the same human view
  and raises, naming the grades that lack labels, until `label-blind` has covered all four.
- **Prompt discipline:** iterate on the labels not used as exemplars (with ≈ 50 labels the held-out half is opt-in and too small to be decisive); freeze `PROMPT_VERSION` before the full run.
  *Indicative single-shot probe (R-M3): QWK 0.41, top-grade compression — expect several rounds.*
- Tests: few-shot selection deterministic and covers all four grades; calibration vs hand-computed κ;
  the static prefix byte-identical across jobs; **`render_label_view` byte-equals a reference
  rendering of `_show_job`'s fields** (clips, no location); the runner with a **fake judge**
  (resume skips done jobs, failures counted, concurrency cap respected, rows carry the judge run);
  `OllamaJudge` with a mocked client (schema sent; bad JSON retried, then a counted failure).

**Gate (manual, real data):** (1) `judge --limit 24` at concurrency 1, 2 and 4 — measure jobs/s
(sequential measured 2.55 s/job; the concurrency claim of F18 is unverified) and pick the
concurrency; (2) calibrate, look at the confusion matrix, iterate the prompt against the blind labels;
(3) freeze and run the full pool (~1 h at 2.6 s/job); every job labeled or listed as failed;
(4) a `judge_calibration` check exists for the run — **if κ_w < 0.6, the silver view is stamped
`UNCALIBRATED`; switch model/prompt (a paid `Judge` behind the same protocol) rather than trust it.**

### WP10 — Evaluate and report

**Goal:** the metrics table the whole effort exists for — one that cannot be misread (R-M1/M2/M9).

- `scoring.py` — pure, on top of `eval/metrics.py`: `compute_metrics(ranking, labels, partial)` —
  **headline** nDCG@100, P@100, AP (grade ≥ 2); **secondary** nDCG@10, Recall@50, Recall@100,
  bpref, P@10, judged@10/@100; `recall_ceiling(labels, k=100)` = `min(1, k/R)` and `R`. **For a
  partial view (`human`) only *condensed* metrics (unjudged removed), bpref and judged@k are
  computed — raw nDCG/Recall with unjudged = 0 are never produced** (R-M1). Also
  `paired_bootstrap(rankings, labels, baseline, metric, n, seed)` (seeded, vectorized document
  bootstrap; returns mean, SD and a 95% interval of the difference vs the baseline),
  `random_ranking(ids, seed)` and `shuffled_tail(ranking, keep, seed)` (**negative controls** scored
  beside the candidates: a comparison where a control beats the incumbent is flagged),
  `order_stability(per_recipe_scores)`.
- `evaluate.py` — `evaluate_topic(session, topic_name, view, judge_run_id=None, bootstrap=1000,
  seed=0, allow_unverified=False)`: for each embedder the latest completed run per method
  fingerprint; effective labels via `labels.resolve_effective`; per-run metrics; slices by extraction
  branch, descriptions > 4,000 chars, one-field-only extractions and source (F20); the recipe
  table (`hyde_mean` vs each `hyde_text:<i>`; embedder order per recipe); bootstrap intervals vs the
  incumbent; the negative controls. **It refuses** a comparison missing a required check
  (`REQUIRED_CHECKS`; for `silver`/`effective` also a *passed* `judge_calibration` for the pinned
  judge run) unless `--allow-unverified`, which stamps the output `UNVERIFIED`/`UNCALIBRATED`.
  Output: a console table per view, and `eval/results/embedding-<topic>-<view>-<ts>.json`
  (gitignored) with run metadata (digest, quantization, Ollama version, docs/s, truncated docs, git
  sha) and the checks. Metrics are functions of `(ranking, labels)`, recomputed on demand; a
  metric-name → function table makes a new metric one entry.
- CLI `evaluate --topic NAME --view silver|human|effective|all [--judge-run ID] [--bootstrap N]
  [--allow-unverified]`; `topics` lists topics with pool size, tier and label counts by source.
- Tests: metric values on a tiny hand-checked fixture; view resolution; condensed variants; the raw
  partial metrics are absent; ceiling; bootstrap determinism and coverage on a synthetic case;
  controls; refusal without checks; output schema.

**Gate:** on real data, `evaluate --view all` prints the three tables with checks and intervals.

### WP11 — Documentation and wrap-up

- `.env.example` (done in WP2), `CLAUDE.md` (the `evals-*` and `embedding-*` commands under "Common
  commands"; a hygiene line that the harness only ever reads prod and its data lives in gitignored
  `data/` and the EVALS database), `docs/EVAL.md`, `docs/COMPONENTS.md`, `docs/STATUS.md` — flip
  "planned" to built with the real command names and the measured findings.
- The isolation tests (WP2) and the contract tests (§6) — verified here.
- Final `just lint`, the harness tests, `uv run pre-commit run --all-files`.

## 5. Where the next stages plug in

| Later work | Where it attaches | Stage 1 groundwork |
|---|---|---|
| Blind human labeling (S2) | A labeling CLI writing `human_blind` rows; migration adds `embedding_label_task` and the nullable `selection` column | `human_blind` exists in `labels.py`; precedence already ranks human above the judge; `evaluate` needs no change. **Highest-value first target: the union of the six variants' top-30 = 73 documents (R report §5).** |
| A `basis = full` label set (S2) | Migration adds a nullable `basis` column; `render_label_view` gets a sibling | The judge's clip constant is recorded in `params` |
| A different document representation | A `doc_text` option in `DenseMethod` (deferred, R-m8) | `description`/`requirements`/`responsibilities` are stored; only ~16 pool docs exceed 2,048 tokens as full description |
| Judge-error sensitivity, decision rule (S2) | New functions over stored rankings and label rows | Rankings stored in full; `judge_calibration` confusion matrix stored |
| Tier B (S3) | `tiers/tier_b.py` + a lazy CLI branch | `persist_topic`, the tier-agnostic contract test, `origin='authored'`, `constructed` source |
| A new embedder | An `[[embedder]]` table + `ollama pull` | Vectors cached per fingerprint |
| A new ranking method (hybrid, reranker) | A `RankingMethod` | Runs keyed by method fingerprint; `FakeMethod` contract test |
| A new judge (paid API) | A `Judge` implementation | `ChatClient` typing is the only Ollama coupling; runs record model + prompt hash |
| Another eval family | `<family>_*` tables + migration + `eval/<family>/` | `evals_db/` is family-agnostic; `alembic_evals/env.py` imports the models module (add an aggregation import) |
| Cutover (S4) | Migration + re-embed + fixture/golden regeneration + `quality/relevance.py` recalibration | Design §6 |

## 6. Generality is tested, not asserted

Stage 1 ships two **contract tests** using fakes registered only in the test:

- **Tier-agnostic:** a `FakeTierBuilder` builds a topic with `origin='authored'`, a closed corpus and
  `constructed` labels through `persist_topic`; then `run`, the judge runner (fake judge) and
  `evaluate` execute **unchanged**, with a Tier-A-shaped topic alongside it in the same database.
  Nothing may branch on `tier`.
- **Method-agnostic:** a `FakeMethod` (a deterministic non-embedding ranker) goes through `run_method`
  and `evaluate` with no embedder involved.

## 7. Interface contracts (what parallel agents code against)

These are binding: agents implement **exactly** these names and signatures so packages written in
parallel fit together. Anything not listed is the owner's private business. `UUID` = `uuid.UUID`;
`np` = numpy; async functions take the `AsyncSession` explicitly (dependency injection — no
module-level sessions).

### 7.1 Schema (`eval/embedding/models.py`, controller-written)

`EvalsBase` from `eval/evals_db/base.py`. Classes and columns (all tables prefixed `embedding_`):

| Class / table | Columns |
|---|---|
| `EmbeddingJob` `embedding_job` | `id` uuid PK · `origin` text · `origin_id` text · `source` text · `title` text · `company` text · `url` text · `location` text? · `remote` bool? · `seniority` text? · `description` text · `requirements` text? · `responsibilities` text? · `content_hash` text? · `embed_text` text · `embed_text_sha` text · `prod_embedding` `Vector()`? · unique(`origin`,`origin_id`,`embed_text_sha`) · index(`embed_text_sha`) |
| `EmbeddingTopic` `embedding_topic` | `id` uuid PK · `tier` text · `name` text unique · `status` text (`draft`\|`frozen`) · `profile_snapshot` JSONB · `query_inputs` JSONB · `builder` text · `notes` text? · `created_at` |
| `EmbeddingTopicJob` `embedding_topic_job` | `topic_id` FK · `job_id` FK · PK(`topic_id`,`job_id`) |
| `EmbeddingJudgeRun` `embedding_judge_run` | `id` uuid PK · `topic_id` FK · `model` · `model_digest` text? · `prompt_version` text · `prompt_sha` text · `params` JSONB · `fewshot` JSONB · `n_labeled` int? · `n_failed` int? · `seconds` float? · `status` text · `created_at` |
| `EmbeddingLabel` `embedding_label` | `id` BIGINT identity PK · `topic_id` FK · `job_id` FK · `grade` int CHECK 0..3 · `source` text · `judge_run_id` FK? · `rationale` text? · `created_at` · index(`topic_id`,`job_id`) |
| `EmbeddingEmbedder` `embedding_embedder` | `fingerprint` text PK · `name` · `model` · `digest` · `quantization` text? · `runtime_version` text? · `config` JSONB · `created_at` |
| `EmbeddingVector` `embedding_vector` | `fingerprint` FK · `text_sha` text · `vector` `Vector()` · `n_tokens` int? · `truncated` bool default false · PK(`fingerprint`,`text_sha`) |
| `EmbeddingRun` `embedding_run` | `id` uuid PK · `topic_id` FK · `name` text (variant name) · `embedder_fingerprint` FK? · `method_fingerprint` text · `method_config` JSONB · `git_sha` text? · `status` text (`running`\|`completed`\|`failed`) · `docs_per_s` float? · `n_truncated` int? · `seconds` float? · `error` text? · `started_at` · `finished_at`? |
| `EmbeddingRunRanking` `embedding_run_ranking` | `run_id` FK · `job_id` FK · `rank` int · `score` float · PK(`run_id`,`job_id`) · index(`run_id`,`rank`) |
| `EmbeddingCheck` `embedding_check` | `id` BIGINT identity PK · `topic_id` FK · `name` text · `passed` bool · `detail` JSONB · `judge_run_id` FK? · `created_at` |

### 7.2 Dataclasses and functions shared across packages

```python
# eval/embedding/tiers/base.py
@dataclass(frozen=True) class JobRecord:      # → EmbeddingJob; id is computed by job_uuid()
    origin: str; origin_id: str; source: str; title: str; company: str; url: str
    location: str | None; remote: bool | None; seniority: str | None
    description: str; requirements: str | None; responsibilities: str | None
    content_hash: str | None; embed_text: str; embed_text_sha: str
    prod_embedding: list[float] | None
@dataclass(frozen=True) class LabelRecord: origin_id: str; grade: int; source: str; rationale: str | None = None
@dataclass class TopicPayload:
    tier: str; name: str; profile_snapshot: dict; query_inputs: dict
    jobs: list[JobRecord]; labels: list[LabelRecord]; builder: str; notes: str | None = None
    # membership = every job in `jobs`
def job_uuid(origin: str, origin_id: str, embed_text_sha: str) -> UUID   # uuid5, fixed namespace
class TopicExists(Exception)
@dataclass class PersistSummary: topic_id: UUID; n_jobs_new: int; n_jobs_reused: int; n_labels: int
async def persist_topic(session, payload: TopicPayload) -> PersistSummary
class TopicBuilder(Protocol): tier: str; async def build(self, name: str, **opts) -> TopicPayload

# eval/embedding/labels.py
SOURCES = ("human_blind", "constructed", "llm_judge"); HUMAN_SOURCES = ("human_blind", "constructed"); VIEWS = ("silver", "human", "effective")
@dataclass(frozen=True) class LabelRow: job_id: UUID; grade: int; source: str; judge_run_id: UUID | None; id: int
def resolve_effective(rows: Iterable[LabelRow], view: str, judge_run_id: UUID | None = None) -> dict[UUID, int]
def is_partial(view: str) -> bool

# eval/embedding/ranking.py            (pure)
def cosine_rank(matrix: np.ndarray, ids: Sequence[UUID], query: np.ndarray) -> list[tuple[UUID, float]]
def compare_rankings(a: Sequence[tuple[UUID, float]], b: Sequence[tuple[UUID, float]], k: int, eps: float) -> CompareResult
    # CompareResult(ok: bool, reasons: list[str]); rule in §7.3

# eval/llm/ollama.py
class ContextOverflow(Exception)
@dataclass class EmbedResponse: vector: list[float]; prompt_eval_count: int | None
@dataclass class ChatResult: data: dict; prompt_eval_count: int | None; eval_count: int | None; prompt_eval_seconds: float | None; seconds: float
@dataclass class TagInfo: digest: str; quantization: str | None; context_length: int | None
class OllamaClient:
    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None) -> None
    async def embed(self, model: str, text: str, *, num_ctx: int, truncate: bool = True) -> EmbedResponse
    async def chat_json(self, model: str, messages: list[dict], schema: dict, *, options: dict) -> ChatResult
    async def tags(self) -> dict[str, TagInfo]
    async def version(self) -> str
    async def warm_embedder(self, model: str, num_ctx: int) -> None
    async def warm_chat(self, model: str) -> None
    async def aclose(self) -> None

# eval/embedding/embedders/base.py
@dataclass(frozen=True) class EmbedderSpec:
    name: str; model: str; backend: str = "ollama"; num_ctx: int = 8192; doc_prefix: str = ""
    hyde_prefix: str | None = None; mrl: bool = False; dim: int | None = None; incumbent: bool = False
    # property effective_hyde_prefix -> hyde_prefix if not None else doc_prefix
@dataclass class EmbedResult: vector: np.ndarray; n_tokens: int | None; truncated: bool   # float32, native dim
class Embedder(Protocol): spec: EmbedderSpec; async def ready(self) -> None; async def embed(self, text: str) -> EmbedResult; def describe(self) -> dict
def apply_dim(vec: np.ndarray, dim: int | None) -> np.ndarray      # slice then L2-normalize; None → unchanged
def fingerprint(backend: str, digest: str, num_ctx: int) -> str
# eval/embedding/embedders/registry.py:  load_specs(path: Path | None = None) -> list[EmbedderSpec]; build_embedder(spec, client) -> Embedder

# eval/embedding/methods/base.py
@dataclass(frozen=True) class CorpusDoc: id: UUID; embed_text: str; title: str; description: str
    requirements: str | None; responsibilities: str | None; source: str; seniority: str | None
@dataclass class TopicView: topic_id: UUID; name: str; query_inputs: dict; profile_snapshot: dict; docs: list[CorpusDoc]
async def load_topic_view(session, name: str) -> TopicView
class RankingMethod(Protocol):
    name: str
    def fingerprint(self) -> str
    def describe(self) -> dict                        # → embedding_run.method_config
    async def prepare(self, topic: TopicView) -> None
    def rank(self, topic: TopicView) -> list[tuple[UUID, float]]
    def stats(self) -> dict                           # optional keys: docs_per_s, n_truncated, embedder_fingerprint
# methods/dense.py: class DenseMethod(embedder, cache, query_recipe="hyde_mean")   # name = spec.name (+ ":" + recipe if not hyde_mean)
# runner.py:  async def run_method(session, topic_name: str, method: RankingMethod, git_sha: str | None = None) -> UUID

# eval/embedding/cache.py
class VectorCache:
    def __init__(self, session)
    async def ensure_embedder_row(self, fingerprint: str, spec: EmbedderSpec, describe: dict) -> None
    async def get_many(self, fingerprint: str, shas: Sequence[str]) -> dict[str, tuple[np.ndarray, int | None, bool]]
    async def put_many(self, fingerprint: str, items: Sequence[tuple[str, np.ndarray, int | None, bool]]) -> None

# eval/embedding/judge/*
@dataclass(frozen=True) class CandidateBrief: seniority: str; years_experience: float | None; target_titles: list[str]
    tech_stack: list[str]; domains: list[str]; remote_required: bool; cv_text: str
    @classmethod def from_snapshot(cls, snapshot: dict) -> "CandidateBrief"
@dataclass(frozen=True) class JobView: title: str; company: str; source: str; seniority: str | None
    requirements: str | None; responsibilities: str | None; description: str
@dataclass class Judgment: grade: int; rationale: str; prompt_eval_seconds: float | None = None
class Judge(Protocol): name: str; def describe(self) -> dict; async def grade(self, job: JobView) -> Judgment
def render_label_view(job: JobView) -> str          # mirrors eval/label.py::_show_job (§WP9)
def select_fewshot(labels: dict[Hashable, int], per_grade: int, seed: int) -> list[Hashable]
def split_dev_test(ids: Sequence[Hashable], seed: int) -> tuple[list, list]
def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int], k: int = 4) -> float
# judge/runner.py: async def build_judge(session, topic_name, client, model, *, per_grade=1, fewshot_seed=13, seed=1) -> tuple[OllamaJudge, list[UUID]]
#                    # brief from the topic's profile_snapshot; few-shot = per_grade exemplars per grade drawn (seeded) from the topic's human view (`human_blind` / `constructed`), raising if a grade has too few; returns the judge and the exemplar job ids
#                  async def label_topic(session, topic_id, judge, fewshot_ids, concurrency=4, limit=None, resume_run_id=None) -> UUID
#                  async def calibrate(session, topic_name, judge_run_id, min_kappa=0.6, use_split=False) -> CalibrationResult   # dataclass: kappa, passed, n, exact, within1, binary, confusion, split

# eval/embedding/scoring.py            (pure)
def compute_metrics(ranking: Sequence[UUID], labels: dict[UUID, int], partial: bool) -> dict[str, float]
def recall_ceiling(labels: dict[UUID, int], k: int = 100, rel_threshold: int = 2) -> float
def paired_bootstrap(rankings: dict[str, Sequence[UUID]], labels: dict[UUID, int], baseline: str, metric: str, n: int, seed: int) -> dict[str, BootstrapResult]
def random_ranking(ids: Sequence[UUID], seed: int) -> list[UUID]
def shuffled_tail(ranking: Sequence[UUID], keep: int, seed: int) -> list[UUID]

# eval/embedding/checks.py
REQUIRED_CHECKS = ("parity_ranking", "parity_reconstruction")
async def record_check(session, topic_id: UUID, name: str, passed: bool, detail: dict, judge_run_id: UUID | None = None) -> None
@dataclass(frozen=True) class CheckRow: id: int; name: str; passed: bool; detail: dict; judge_run_id: UUID | None; created_at: datetime
async def latest_checks(session, topic_id: UUID) -> dict[str, CheckRow]      # newest (max id) per name
async def calibration_for_run(session, topic_id: UUID, judge_run_id: UUID) -> CheckRow | None   # newest judge_calibration for that run
# eval/embedding/evaluate.py: async def evaluate_topic(session, topic_name, view, judge_run_id=None, bootstrap=1000, seed=0, allow_unverified=False) -> dict   # JSON-serializable report
#                             def format_report(report: dict) -> str ; def write_report(report: dict, out_dir: Path) -> Path ; async def list_topics(session) -> list[dict]
#                             class MissingChecks(Exception)
#   evaluate_topic with view silver/effective and judge_run_id=None picks the newest completed judge run of the topic that has a PASSED calibration, else the newest completed run (which then fails the calibration requirement -> MissingChecks unless allow_unverified)
# Embedder.describe() must include the keys: model, digest, quantization, runtime_version, num_ctx, fingerprint
# eval/embedding/parity.py:   async def verify_topic(name: str) -> dict          (opens its own prod_session and evals_session)
# eval/embedding/tiers/tier_a.py: assemble(...) ; scrub_cv_text(cv_text, full_name) -> str ; async def import_topic(name, tier, **opts) -> PersistSummary
# eval/embedding/label_blind.py: async def label_blind(session, topic_name, n=50, bucket="mixed", seed=0, input_fn=input, output_fn=print) -> dict
#   # the user's blind grading: candidates = topic docs with no human label, ordered by `seed` alone (so a rerun continues), `pooled` (in the top-30 of some but not all latest completed primary runs) / `random` / `mixed` (~60/40); shows the rubric and `render_label_view` only (never a grade, score, rank, run or bucket); one `human_blind` row per label, committed at once (bucket + note in `rationale`); `s` skips, `q`/EOF/Ctrl-C ends; returns {labeled, skipped, remaining, buckets}
# eval/embedding/labels_io.py: async def export_labels(session, topic_name, path) -> int ; async def import_labels(session, topic_name, path) -> int
# eval/evals_db/prod_reader.py: class ProdNotReadOnly(RuntimeError); prod_session(url=None) (async cm); table_counts(session, tables=...) ; alembic_revision(session)
# eval/evals_db/admin.py: database_name, with_database, ensure_database, drop_database, migrate
# tests fixtures: evals_db_url (session, sync) -> str ; evals_session (function, async) -> AsyncSession
```

### 7.3 `compare_rankings` (R-m3)

Inputs are two `[(id, score)]` lists, each sorted best-first. `ok` iff, for the first `k` positions:
(1) the **score sequences** agree position-wise within `eps` (ties make id order arbitrary but score
order is invariant); (2) partition the reference `a` into *eps-chains* (consecutive scores differing
by ≤ `eps`); every chain that lies fully inside the first `k` positions has equal **id sets** in `a`
and `b`; (3) the chain crossing position `k` may differ in ids but `b`'s members of that chain must
score within `eps` of `a`'s chain. Verified against real data (top-100 set equal; order differs only
inside byte-identical tie groups; max |Δscore| 2.4e-7; no tie at the rank-100 boundary).

### 7.4 CLI (`eval/embedding/cli.py`, `job-radar-eval-embedding`)

Synchronous `main()`, one `asyncio.run` per command, sub-commands `import-topic --tier A --name N
[--profile-id]`, `embed`, `run --topic N --embedder NAME|all [--per-text]` (`--per-text` also runs
the `hyde_text:<i>` recipes), `verify --topic N`, `judge --topic N --model M [--limit N]
[--resume ID] [--concurrency N] [--fewshot-per-grade 1]`, `label-blind --topic N [--n 50]
[--bucket mixed|pooled|random] [--seed 0]`, `judge-calibrate --topic N --judge-run ID [--min-kappa 0.6]
[--held-out-half]`, `evaluate --topic N --view … [--judge-run ID] [--bootstrap N] [--allow-unverified]`,
`topics`, `labels-export --topic N [--out P]`, `labels-import --topic N --file P`. `import-topic` and
`verify` import prod modules lazily inside the command.

## 8. Phased parallel execution

The work is split so that everything inside a phase runs **concurrently by independent agents**, each
owning a disjoint set of files (a file is never written by two agents). Phases are gated by the
controller (verification = diff review, `ruff`, the owner's tests, contract conformance) before the
next starts. Agents never commit, never touch files outside their ownership, and run only their own
test files (the repo's existing suite writes to `DATABASE_URL`, R-M8).

**Phase 0 — controller, serial, small.** Write the shared contract files: `eval/evals_db/__init__.py`,
`eval/evals_db/base.py`, `eval/embedding/__init__.py`, `eval/embedding/models.py` (§7.1),
`eval/llm/__init__.py`, the empty package `__init__.py`s. Verify the import graph is clean (no
`job_radar` import). This is the only serial step and it unblocks everything.

**Phase 1 — six agents in parallel** (no dependency between them beyond Phase 0):

| Agent | Work packages | Owns (create/edit) |
|---|---|---|
| **A** | WP1 | `src/job_radar/ingest/embed_text.py`, `src/job_radar/ingest/pipeline.py` (the `_prepare` call only), `eval/metrics.py`, `tests/test_embed_text.py`, `tests/test_eval_metrics.py` |
| **B** | WP5 | `eval/embedding/labels.py`, `tests/test_embedding_labels.py` |
| **C** | WP7a — client, embedders | `eval/llm/ollama.py`, `eval/embedding/embedders/*`, `eval/embedders.toml`, `tests/test_evals_ollama.py`, `tests/test_embedding_embedders.py` |
| **D** | WP9a — pure judge | `eval/embedding/judge/{__init__,base,prompt,fewshot,calibration,ollama_judge}.py`, `tests/test_embedding_judge_*.py` (uses a fake `ChatClient` with `chat_json` — signature §7.2) |
| **E** | WP2 + WP3 + WP4 (migrations) + isolation tests | `eval/evals_db/{settings,admin,prod_reader,cli}.py`, `alembic_evals.ini`, `alembic_evals/**`, `pyproject.toml`, `uv.lock`, `justfile` (`evals-db-*`), `.env.example`, `tests/conftest.py`, `tests/test_evals_*.py` (settings, admin, prod_reader, isolation), `tests/test_embedding_models.py` |
| **F** | WP7b-pure + WP10a | `eval/embedding/ranking.py`, `eval/embedding/scoring.py`, `tests/test_embedding_ranking.py`, `tests/test_embedding_scoring.py` |

**Phase 2 — seven agents in parallel** (need Phase 0 + the Phase 1 outputs they import):

| Agent | Work packages | Owns | Needs from |
|---|---|---|---|
| **H** | WP7 rest — cache, methods, runner | `eval/embedding/cache.py`, `methods/*`, `runner.py`, `tests/test_embedding_{cache,dense,runner}.py` | models, C, F, E fixtures |
| **I** | WP6 — topics, Tier A, labels I/O | `eval/embedding/tiers/*`, `labels_io.py`, `tests/test_embedding_{tiers,tier_a,labels_io}.py` | models, A, B, E |
| **J** | WP9b — judge runner + calibrate | `eval/embedding/judge/runner.py`, `tests/test_embedding_judge_runner.py` | models, B, D |
| **K** | WP8 — checks + parity | `eval/embedding/checks.py`, `parity.py`, `tests/test_embedding_{checks,parity}.py` | models, F, E |
| **L** | WP10b — evaluate | `eval/embedding/evaluate.py`, `tests/test_embedding_evaluate.py` | models, B, F, K's `checks` signatures (§7.2) |
| **M** | CLI + `just` | `eval/embedding/cli.py`, `justfile` (`embedding-*` recipes), `tests/test_embedding_cli.py` (argument parsing with the entry functions monkeypatched) | contracts only |
| **O** | WP11 docs | `CLAUDE.md`, `docs/EVAL.md`, `docs/COMPONENTS.md`, `docs/STATUS.md` | contracts only |

**Phase 3 — verification and integration** (controller + one agent):

| Agent | Work | Owns |
|---|---|---|
| **N** | The two contract tests and an end-to-end fake-pipeline test (`FakeTierBuilder` → `persist_topic` → `run_method` with a fake embedder → fake judge → `calibrate` → `evaluate`), plus fixes of integration mismatches reported by the controller | `tests/test_embedding_contract.py`, `tests/test_embedding_e2e.py` |
| Controller | Full harness test run on a throwaway DB, `ruff check`/`format --check`, the isolation tests, contract review, fix-ups (by re-tasking the owning agent) | — |

**Phase 4 — real-data acceptance (controller).** `evals-db-init`; `import-topic`; `run --embedder all
--per-text`; `verify`; a small judge spike; `evaluate` under the `human` view (condensed) — the parts
that need no hour-long judge run. The full-pool Gemma run and the prompt iteration (WP9 gate) are
operator steps documented in the runbook.

**Ownership rules for every agent.** (1) Only create/edit files in your Owns column. (2) Code against
§7 exactly; if a contract seems wrong, keep it and say so in your report. (3) No boilerplate, no
speculative abstraction (CLAUDE.md). (4) `uv run ruff check <your files>` and `uv run ruff format
<your files>` clean; `uv run pytest <your test files> -q -p no:cacheprovider` green — and nothing
else (never the whole suite). (5) Tests use the `evals_session` / `evals_db_url` fixtures, fakes,
`httpx.MockTransport`; **never** the real prod database, `db_session`, or a real Ollama. (6) Never
`git commit`. (7) Report in ≤ 300 words: files created, deviations from the contract, anything the
controller must fix.

## 9. Runbook (real data, after Phase 4)

```bash
docker compose -f infra/docker-compose.yml up -d
just evals-db-init
just embedding-import --tier A --name A-real-2026-09-18   # read-only from prod
just eval-ollama-start                                    # optional dedicated Ollama
just embedding-run --topic A-real-2026-09-18 --embedder all --per-text
just embedding-verify --topic A-real-2026-09-18           # both parity checks must pass
just embedding-label --topic A-real-2026-09-18 --n 50     # blind-label ~50 jobs yourself (~30 pooled + ~20 random); rerun to resume
just embedding-judge --topic A-real-2026-09-18 --model Gemma3:12b --limit 24   # timing spike
just embedding-judge-calibrate --topic A-real-2026-09-18 --judge-run <id>      # against your blind labels
just embedding-judge --topic A-real-2026-09-18 --model Gemma3:12b              # full pool, ~1 h
just embedding-judge-calibrate --topic A-real-2026-09-18 --judge-run <id>
just embedding-evaluate --topic A-real-2026-09-18 --view all --judge-run <id>
just embedding-labels-export --topic A-real-2026-09-18    # backup of the human labels
```

## 10. Risks and things to watch

- **The judge is the critical path.** Single-shot QWK was 0.41 (R-M3). Budget several prompt rounds;
  a paid judge behind the same protocol is the fallback. Nothing in `silver`/`effective` is
  reportable without a passed calibration.
- **Judge prompt overfitting to ≈ 50 labels.** Only 46 remain after the 4 exemplars (κ_w SE ≈ 0.1);
  bounded by the freeze and the opt-in `--held-out-half`; the random calibration sample in Stage 2
  is the unbiased check.
- **Recall@100 is not a quality score here** (ceiling 0.15–0.23, R-M2); never quote it without the
  ceiling.
- **Silent truncation.** Effective ceilings: nomic 2,048, bge-m3 2,048 (an Ollama cap), Qwen3 ≥ 8,192
  (R-M7). Detection is `truncate=false`, not `prompt_eval_count`.
- **Qwen3 runner crash on very long input** (HTTP 400 `EOF` at ≳ 3,000 tokens on Ollama 0.30.11, R-m11): the pool's longest Qwen3 document is 2,024 tokens, so Stage 1 is safe; retries can't fix a deterministic crash, so a failing document fails the run loudly.
- **Quantization confound.** Qwen3 is Q8_0, nomic and bge-m3 F16 — recorded per run, not absorbed.
- **The extraction step is frozen and lossy (F20)** and the labelers saw an even more clipped view
  (F21, R-M3): the harness measures embedders *given* this text.
- **Long full-pool runs die.** Resumable runs, per-job retry, batch commits.
- **Memory:** Gemma3:12b (~8 GB) plus an embedder — run them sequentially.
- **Prod drift after the freeze.** The topic is frozen and parity restricts prod to the frozen ids; a
  refresh is a *new* topic.
- **PII stays local.** `cv_text` is scrubbed (best effort) and lives only in the EVALS database's
  `profile_snapshot`, used by the local judge; nothing exports it (`labels-export` writes grades and
  ids only).
