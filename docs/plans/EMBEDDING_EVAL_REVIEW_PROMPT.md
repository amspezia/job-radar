# Prompt: Independent Review of the Embedding-Eval Design and Stage-1 Plan

> Hand this file, verbatim, to a fresh AI session that has **read access to the repo and a
> shell**. It is written to be self-contained: the reviewer has none of the conversation that
> produced the plans. If the reviewer has no repo access, attach the files listed in §3.

---

## 1. Your mission

You are a senior engineer with deep experience in **information-retrieval evaluation**
(test-collection construction, LLM-as-judge, embedding benchmarks) **and** in building
reproducible, database-backed evaluation tooling in Python/Postgres. You are the independent
reviewer of two planning documents for **Job Radar**, a personal job-matching system:

- `docs/plans/EMBEDDING_EVAL_DESIGN.md` — the *what and why*.
- `docs/plans/EMBEDDING_EVAL_IMPLEMENTATION_PLAN.md` — *Stage 1*, the *how*, as eleven ordered
  work packages (WP0–WP11).

Your job is to find out — by reading the real code, querying the real (read-only) data, and
running small experiments — **whether the plan is correct, feasible, aligned with best practice,
and will actually work as written.** Both documents were written by an AI in one long session;
treat every claim as a hypothesis, including the ones marked "verified".

Success is **not** approval. Success is a report that would let the developer start work with
confidence, or that stops them from wasting days on something that won't work. Be direct. Do not
be diplomatic. If something is fine, say so briefly and move on; spend your effort where the
plan is most likely to be wrong.

## 2. Ground rules (read before touching anything)

**Safety — non-negotiable.** The whole design rests on *never changing production*.

- **Do not write to the production database (`job_radar`).** Read only, and prefer a connection
  that enforces it: `create_async_engine(url, connect_args={"options": "-c
  default_transaction_read_only=on"})`. Never run `alembic upgrade` against it, never the
  `just eval-reset-labels`, `eval-teardown-synthetic`, `eval-inject-synthetic`, `eval-commit-golden`
  recipes, and never the ingest, scheduler, or fit pipelines (they write).
- If an experiment needs a database, create a **throwaway** one named
  `job_radar_evals_test_<random>` on the same server and **drop it when finished**. Confirm the
  name before every `DROP`.
- **Do not modify, create or delete any file in the repository**, and do not commit. Put scratch
  scripts in your session's scratchpad/temp directory. Your only output is the report (§6).
- Never print secrets. `.env` holds DB credentials; read values programmatically, don't `cat` it.
- Ollama is available locally (default `http://localhost:11434`). Running embed/chat calls is
  fine. `qwen3-embedding:0.6b` and `bge-m3` are **not yet pulled** (~2 GB together): pulling them
  is acceptable unless your environment forbids it — if you skip, mark the dependent items
  UNVERIFIED rather than guessing.

**Honesty rules.**

- Label every finding **VERIFIED** (you read it or ran it — show the evidence), **PLAUSIBLE**
  (reasoned but not tested), or **UNVERIFIED** (you couldn't check). Quote file paths with line
  numbers and paste the relevant command output. **Never invent a number.**
- If your own experiment contradicts a documented fact, say so prominently — that is the most
  valuable thing you can find.
- Distinguish "this is wrong" from "I would have done it differently". Only the first is a
  finding; the second belongs in a short "opinions" list.

## 3. What to read, in order

1. `CLAUDE.md` — project rules (hygiene, "keep it intentional — no boilerplate", no PII in git,
   no fabricated metrics, tooling, CI).
2. `docs/plans/EMBEDDING_EVAL_DESIGN.md` — read the verified-facts table (F1–F22) closely.
3. `docs/plans/EMBEDDING_EVAL_IMPLEMENTATION_PLAN.md`.
4. `docs/EVAL.md`, `docs/STATUS.md` (§4, the label-history note), `docs/COMPONENTS.md`
   (Evaluation Harness section), `docs/plans/SYNTHETIC_EVAL_DESIGN.md`.
5. The production and eval code the plan builds on or touches — **read the actual code, not the
   plan's description of it**:
   - Embedding path: `src/job_radar/ingest/pipeline.py` (`_prepare`), `ingest/extract.py`,
     `ingest/dedup.py`, `adapters/embeddings.py`, `adapters/providers.py`.
   - Retrieval: `retrieval/vector.py`, `retrieval/filters.py`, `retrieval/geo.py`,
     `retrieval/seniority.py`, `retrieval/bm25.py`, `retrieval/fusion.py`.
   - HyDE / query side: `fit/pipeline.py` (`_HYDE_PROMPT`, `_generate_hyde_posting`,
     `build_hyde_embedding`), `fit/analyze.py`.
   - Persistence and config: `db/base.py`, `db/models.py`, `config.py`, `alembic/env.py`,
     `alembic.ini`, `alembic/versions/*`.
   - Existing eval: `eval/metrics.py`, `eval/qrels.py`, `eval/label.py`, `eval/run.py`,
     `eval/run_synthetic.py`, `eval/inject_synthetic.py`, `eval/personas_lib.py`,
     `eval/golden/result.json`, and (gitignored, local) `eval/results/RETRIEVAL_ANALYSIS.md`.
   - Tooling and CI: `pyproject.toml` (packaging, scripts, ruff, pytest config), `justfile`,
     `.github/workflows/ci.yml`, `infra/docker-compose.yml`, `.env.example`,
     `tests/conftest.py` (note the autouse fixtures), `tests/test_eval_gate.py`,
     `tests/test_eval_metrics.py`.

## 4. Environment you can query

- Repo: `/Users/afonsospezia/Projects/job-radar`; Python 3.12 via `uv run …`; `just` is the task
  runner.
- Postgres 16 (`paradedb/paradedb:pg16`, Docker) at `localhost:5432`; connection string in
  `.env` as `DATABASE_URL` (the user is a superuser, per `infra/docker-compose.yml`). Live data as
  of 2026-09-18: 16,389 jobs, one real profile, 221 human labels in `eval_labels`, pgvector 0.8.2.
- Ollama 0.30.11; models present: `nomic-embed-text`, `Gemma3:12b` (use the tag exactly as
  `ollama list` prints it), `qwen2.5`, `qwen2.5:3b`, `qwen3:4b`, and others.
- The developer's constraints, from the conversation that produced the plans: a **separate EVALS
  database** with `embedding_`-prefixed tables (it may host other eval families later); prod is
  **imported from, never changed**; **Tier A first** (the real profile over its ~1.4k-job eligible
  pool), Tier B (synthetic personas) later, Tier C deferred; embed the pool with each candidate
  embedder; label the **whole** pool with a local Gemma judge; blind human labels come in a later
  stage; the harness must be **dense-only** (vary only the embedder), runnable independently of
  the production app, and **extensible** (new embedders by config; new tiers/methods/judges
  without rewriting the core). This is a portfolio project — every committed file is read as a
  hiring signal, hence CLAUDE.md's "no boilerplate".

## 5. What to evaluate

Work through every dimension. Each has concrete questions; they are prompts, not limits.

### A. Are the documented facts true?

Re-derive the **load-bearing** facts independently — don't reuse the author's scripts. At minimum:

| Fact | What to reproduce / challenge |
|---|---|
| F1 | `EXPLAIN` of the `search_vector` query shape uses a Seq Scan, not the HNSW index. Does the *real* query (with the profile filter and a real vector) change the plan? |
| F3 | Rebuild `embed_text` from stored columns for a sample (both branches of the fallback) and re-embed through Ollama with the `search_document: ` prefix; compare to the stored vector. Also test rows the author did **not** sample (long descriptions, both fields NULL). |
| F6 | Count byte-identical embeddings; check whether ties actually reach a top-100 boundary for the real HyDE query. |
| F10 | Recompute the eligible pool for the real profile and how many of the 221 labels fall inside it. |
| F16 | Reproduce the read-only guard. Then attack it: connection reuse in a pool, `SET TRANSACTION READ WRITE`, `pool_pre_ping`. State the limitation honestly (a guard against accidents, not against a determined override). |
| F17 | Can the `job_radar` role `CREATE DATABASE` and `CREATE EXTENSION vector` in a new DB? Does CI's Postgres allow the same? |
| F18 | Time Gemma judging. The author measured **without few-shot**; the plan adds few-shot. |
| F20 | Recompute extraction-branch shares, length ratios and the >4,000-char share. |
| F22 | Recompute the HyDE-text and embedding similarity of the three cached texts. |

Also check the smaller factual claims scattered through both documents (file names, line
references, function names, default values, model tags and sizes, what `label.py` displays, what
`analyze.py` reads).

### B. Will each work package work as specified?

For **every WP0–WP11**, judge: *could a competent engineer implement it from the text alone,
and would its gate pass?* Look for hidden coupling, missing steps, wrong ordering, and
unspecified behavior. Specific areas the author is least sure about:

1. **Alembic for a second database** (`alembic_evals/`, `alembic_evals.ini`): does the proposed
   `env.py` work with `pgvector.sqlalchemy.Vector()` **with no dimension**? Does autogenerate
   render it and does `alembic check` come out clean, or will it report perpetual drift? Does
   running Alembic (which starts its own event loop) from a worker thread inside an async pytest
   fixture behave?
2. **`EvalsSettings`**: a pydantic-settings class that must **not** import `job_radar.config`,
   reads `.env`, and rejects an evals URL equal to the prod URL. Does it behave in CI (env vars
   only, no `.env`)? Any interaction with `job_radar.config`'s `load_dotenv()`?
3. **The import-isolation AST test**: is an allow-list of three modules enough? Transitive
   imports — `job_radar.retrieval.vector` and `filters` import `job_radar.db.models`, which imports
   `job_radar.db.base`, which **builds a read-write engine at import**. Does that undermine the
   "prod is read-only" story for the allow-listed modules? Is the AST approach the right tool?
4. **The read-only prod reader**: with SQLAlchemy async + psycopg, does the `options` connect arg
   survive pooling and reconnects? Could a `Session` flush something? Is `SHOW
   transaction_read_only` the right assertion?
5. **`persist_topic` and deterministic `uuid5` document IDs**: idempotency, changed-text
   behavior, two topics sharing a posting, foreign-key ordering, bulk insert sizes for ~1.4k
   documents each carrying `description`, `embed_text` and a vector.
6. **The vector cache** (`embedding_vector`): unconstrained `vector` column, `ON CONFLICT DO
   NOTHING`, reading ~1.4k×1024 vectors back into numpy — round-trip dtype/precision, and does
   psycopg return them in a form the plan assumes?
7. **The dense method and cache keys**: `text_sha = sha256(prefix + text)` with the fingerprint
   excluding prefixes; Matryoshka variants sharing native vectors; the claim that changing a prefix
   changes the sha but not the fingerprint. Any collision or stale-cache hole?
8. **The Ollama client**: `/api/embed` request/response shapes, `prompt_eval_count` as a
   truncation detector (does Ollama silently truncate to `num_ctx`, and does `prompt_eval_count`
   then equal `num_ctx`?), `/api/tags` digest and quantization fields, `/api/chat` with a JSON
   `format` schema, behavior on cold model load.
9. **The judge**: is the prompt-prefix-caching claim true in Ollama (does a byte-identical system
   prompt across requests actually reuse the KV cache, given parallel slots and `num_ctx`)?
   Schema adherence rate? Does `reason` before `grade` help or hurt?
10. **`parity_ranking`**: feeding the same query vector to prod's `search_vector` restricted to
    the frozen ids — is the tie-aware comparison correctly specified? Does the plan's claim that it
    never calls `build_hyde_embedding` hold given what it imports?
11. **CI**: adding `EVALS_DATABASE_URL` and an init step; per-session throwaway databases in
    tests; does the existing `tests/conftest.py` (autouse Langfuse fake patching specific module
    paths, a `db_session` fixture importing `job_radar.db.base`) interact badly with the new
    fixtures? Is every commit CI-green as claimed?
12. **Packaging**: `pyproject.toml` packages `["src/job_radar", "eval"]` only — does a top-level
    `alembic_evals/` directory or `eval/embedders.toml` get packaged/found the way the plan
    assumes? Do the new console scripts resolve?

### C. Is the evaluation *methodology* valid — will it answer the question?

This matters more than any code detail. Challenge:

1. **Can the metrics discriminate?** One profile; a 1,395-document pool; roughly 100–200 relevant
   documents (unknown). The plan's primary metrics are nDCG@100, Recall@100 and AP. If more than
   100 documents are relevant, **Recall@100 has a ceiling below 1** — does that make it a poor
   headline? Would R-precision, nDCG@R, or AP be better? The existing golden nDCG@10 is
   saturated: prove or disprove that the new family isn't.
2. **Complete silver labels + sparse gold:** is the plan's mixture sound? The `effective` view
   puts 212 imported human labels *above* LLM labels for the rest — mixing two labelers whose
   biases differ. Could that create artifacts (e.g. human-labeled documents systematically
   different in relevance rate from the silver-only remainder)? What does the metric mean under
   each of the three views, and is "a claim must agree across views" a workable rule?
3. **Judge validity.** Calibration is on the *held-out half of the 212 imported labels*, which
   were drawn from the incumbent's retrieval pool (optimistic at the top of the ranking, blind to
   the tail). Few-shot exemplars come from the same pool. The judge sees the candidate's CV text.
   Is κ ≥ 0.6 a sensible gate? Is quadratic-weighted κ the right statistic with 4 ordinal grades
   and skewed prevalence? Is 212 enough to split into dev/test and still say anything? Where does
   **circularity or leakage** remain (HyDE and judge both LLM-based; imported labels possibly
   anchored to LLM suggestions; the fewshot/dev/test split)?
4. **Label basis.** Grades are judged on the *production view* (LLM-extracted fields, ~half the
   text, tails truncated at 4,000 chars) so that they match the 221 imported labels. Does that
   silently make the eval measure "agreement with the extractor's view of relevance"? Is the
   reasoning that this is fair to embedders correct, or does it hide the most important variable?
5. **One frozen query.** The three HyDE samples are near-identical (F22), so the query is
   effectively one vector; every embedder is scored against it. How much could a single query
   text dominate the ordering? Is the "robustness check" deferred to Stage 2 adequate, or should
   something cheaper be in Stage 1?
6. **Inference.** Stage 1 reports point estimates. With one topic, is *anything* in the output
   defensible as a comparison? What is the smallest metric difference that would be meaningful,
   and can you estimate it (e.g. by bootstrapping documents on the existing data)?
7. **Pool and sampling design.** The plan uses the whole eligible pool, silver labels for
   everything, then (Stage 2) pooled/calibration/retest human buckets. Is that the best use of
   human time versus alternatives (e.g. adaptive labeling of only rank-disagreements first)? Any
   flaw in the stratified random calibration sample and its use?
8. **Confounds.** Quantization, runtime version, `num_ctx` (same 8192 for all — is that fair to a
   32K model?), prefixes/instructions (tuning budget per model), embedding of `hyde_prefix` vs
   `doc_prefix`, dimension truncation. Anything not controlled?

### D. Best-practice alignment

Compare against established practice and cite sources (papers, standards, well-known tools —
e.g. TREC pooling and reusability, BEIR/MTEB custom-task methodology, Cranfield paradigm caveats,
LLM-judge calibration and prediction-powered inference, statistical testing in IR, experiment
tracking and reproducibility, data immutability/versioning, evaluation-harness design in mature
libraries). Where the plan **departs** from practice, say whether the departure is justified. Also
assess: database schema design (normalization, keys, indexes, `TEXT` + code-side validation vs
`CHECK`/enums, `JSONB` use, append-only labels), migration strategy, idempotency, resumability of
long jobs, logging/observability, secrets/PII handling (`cv_text` copied into the evals DB), and
testing strategy (fakes, integration tests against a throwaway DB, contract tests).

### E. Do the two documents agree with each other and themselves?

Look for contradictions, stale remnants of superseded designs (an earlier on-disk "bundle", Tier C,
`E1–E5` naming, `label.py` changes that were dropped), mismatched table/column/field names,
numbers that differ between sections, fact references (`F1…F22`) that don't support the claim
they're cited for, section references that resolve to the wrong place, things the design promises
that no work package schedules, and work-package steps with no design basis.

### F. Are the extensibility claims real — and is it over-engineered?

The design lists "seams" (tier builders, ranking methods, embedders, judges, label sources,
metrics, other eval families). **Mentally implement four additions**: (1) Tier B personas with
closed corpora; (2) a hybrid or reranker ranking method; (3) a paid-API judge; (4) an unrelated
second eval family sharing the EVALS database. For each, list what would *actually* have to
change and where the abstraction would leak. Then apply CLAUDE.md's "no boilerplate, no
speculative abstraction": which protocols, tables, columns (`basis`, `selection`, `role`,
`method_kind`…) or modules are not load-bearing for Stage 1 and should be cut or deferred?

### G. Does it fit *this* repository?

Conventions, ruff configuration (line length, rules), naming, `eval` as a package name, existing
tests and fixtures that the new code could break, the existing `eval/` scripts and `just` recipes,
CI timing/cost, dependency declarations (`numpy` in the dev group, `tomllib`, `httpx`), and
whether any proposed change to production code (`build_embed_text` extraction) is really safe and
minimal.

### H. Pre-mortem

Assume Stage 1 was implemented **exactly as written** and the final `evaluate` output is
**misleading, unusable, or wrong**. List the 5–10 most likely causes, ranked by likelihood ×
damage, each with the earliest observable symptom and the cheapest guard that would catch it.

## 6. Experiments to run (prioritized, safe)

Do as many as you can, in this order; record commands and outputs (the report's appendix).

1. **EVALS-DB mechanics** in a throwaway database: create it from Python with `AUTOCOMMIT` as
   the `job_radar` user; `CREATE EXTENSION vector`; a table with an *unconstrained* `vector`
   column; insert 3-d and 1024-d vectors into it; read them back as numpy; then build a tiny
   Alembic environment mirroring the plan's (async, own `alembic_evals.ini`), autogenerate a
   migration with `Vector()` and run `alembic check`. Run the upgrade from a worker thread inside
   an async test. Drop the database.
2. **Read-only guard**: reproduce F16 and probe its limits (pooling, override, ORM flush).
3. **Import-graph reality check**: trace what importing `job_radar.retrieval.vector` and
   `job_radar.retrieval.filters` pulls in (`python -X importtime` or a module-list diff), and
   prototype the AST allow-list test. Report whether it holds.
4. **Embedders through Ollama** (pull the two models if permitted): dimensions, unit norm,
   determinism, prefix sensitivity, what happens beyond `num_ctx`, `prompt_eval_count` semantics,
   throughput. Check the Qwen3 end-of-text fidelity question against the reference implementation
   with `uv run --with sentence-transformers …` on ~30 texts.
5. **Gemma judge**: ~20 jobs from the real pool with a few-shot static prefix; seconds per job
   with and without prefix reuse; JSON-schema adherence; a rough agreement figure against the
   human labels on the jobs you judged (indicative only, say so).
6. **Metric discrimination and ceiling**: from the stored production vectors, the cached HyDE
   texts and the 212 imported labels (all read-only), compute rankings and nDCG@10 / nDCG@100 /
   Recall@100 / AP for nomic; then for **deliberately degraded variants** (no doc prefix; MRL
   truncation to 256/128 dims; a shuffled-tail ranking) and see whether the metric family
   separates them, at what magnitude, and with what document-bootstrap spread. Estimate the
   number of grade ≥ 2 documents in the pool from a small silver sample and state the resulting
   Recall@100 ceiling.
7. **`resolve_effective` semantics**: write the intended precedence/latest-wins/basis logic in a
   scratch script and probe edge cases (a retest that lowers a grade; two judge runs; a `full`
   basis row alongside a `production` one) for ambiguity in the spec.
8. **Repo checks**: run `uv run ruff check .` and `uv run pytest -q` **read-only** to learn the
   baseline (note pre-existing failures — `docs/STATUS.md` mentions one); inspect how the autouse
   fixtures in `tests/conftest.py` would treat new modules.

## 7. Deliverable

Your final message is the report. **Do not edit repo files.** Structure:

1. **Verdict** (≤ 8 lines): GO / GO-WITH-CHANGES / NO-GO for starting WP1, and the three things
   that matter most.
2. **Findings**, each with: an ID, severity, category (A–H), **VERIFIED / PLAUSIBLE /
   UNVERIFIED**, evidence (paths, line numbers, command output), impact, and the concrete fix
   (an exact edit to the design or plan where possible, as a short diff-style snippet).
   Severity scale:
   - **BLOCKER** — would not work as written, would touch production, or would yield invalid
     conclusions.
   - **MAJOR** — likely rework, a wrong result under plausible conditions, or a materially
     misleading claim.
   - **MINOR** — real but cheap to fix or unlikely to bite.
   - **NIT** — wording, naming, polish. Keep these to one short list; don't pad.
3. **Fact-check table** for F1–F22: confirmed / contradicted / unverified, with your evidence.
4. **Per-work-package verdict**: WP0–WP11, each GO / GO-WITH-CHANGES / NO-GO with one line why.
5. **Methodology assessment**: will Stage 1's output support any comparison of embedders? What
   would it *not* be able to tell the developer? What is the minimum addition that would fix the
   biggest gap?
6. **Best-practice gaps**, with citations.
7. **Cut / defer list** (over-engineering under CLAUDE.md's rules) and **missing list** (things
   the plan needs but doesn't say).
8. **Open questions for the developer** (decisions only they can make).
9. **A prioritized "do before starting" checklist.**
10. **Appendix**: every experiment run — command, output excerpt, conclusion.

Length: as long as the evidence needs, no longer. Lead with what matters; put detail in the
appendix. Prefer three well-evidenced findings to fifteen speculative ones.

## 8. Things the author already knows — assess, don't re-litigate

- The **decisions** above (§4) come from the developer. Judge whether the plan *achieves* them,
  not whether they are the right goals — except where a decision makes the plan unable to answer
  its own question, which you should say loudly.
- Out of scope for Stage 1 by design: blind human labeling, statistical inference, Tier B,
  Tier C, cutover. Do assess whether Stage 1 leaves them *cheaply reachable*.
- Proposed numbers awaiting real data (judge κ threshold 0.6; few-shot 2 per grade; concurrency
  4/8; top-30 pooling depth) are proposals. Say whether they are reasonable, not that they are
  unproven.
- The author's own suspected weak spots — start here, but do not stop here: (a) Recall@100 with
  possibly >100 relevant documents; (b) the mixed-source `effective` view; (c) calibrating the
  judge on labels drawn from the incumbent's pool; (d) whether one effectively-single query can
  support any embedder ranking; (e) pgvector's unconstrained `vector` with Alembic; (f) the
  import-isolation guarantee given `job_radar.db.base`'s import-time engine; (g) Ollama's
  handling of Qwen3-Embedding; (h) whether the labeling `basis` column and other generality
  hooks are worth their cost now.
