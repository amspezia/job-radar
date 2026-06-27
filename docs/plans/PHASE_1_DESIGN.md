# Phase 1 — Discovery + reliability spine — Implementation Design

> High-level design for Phase 1 only. Source of truth is
> [DESIGN.md](../../DESIGN.md); this document sequences the build and names
> interfaces, responsibilities, and verification — not code. Keep every committed
> file load-bearing (CLAUDE.md no-boilerplate rule); subpackages and dependencies
> land **with their milestone**, never as empty stubs up front.

Phase 1 delivers the Discovery capability (§3.1) over the reliability spine
(§3, evaluation/observability/guardrails). It is a complete standalone deliverable
(§17): ingest → index → hybrid-search → grounded fit, exposed over MCP and a thin
web UI, with eval and observability gates in CI and a runnable deployment.

---

## 1. Scope

**In scope (§17 Phase 1).**
- Ingestion: source-adapter interface, a small enabled starter set, normalization,
  cross-source dedup, embedding cache, ingest-by-URL, APScheduler daily batch.
- Indexing: pgvector vectors + Postgres FTS over the corpus.
- Hybrid search: vector + FTS fused with RRF (§19).
- MCP tool surface for the Discovery capability (FastMCP).
- LangGraph search/fit graph: Supervisor → Interpreter → Retrieval → Fit (§7.1).
- Evaluation harness: labeled fit set, retrieval + fit metrics, golden-query CI gate.
- Observability: OTel GenAI traces → Langfuse with PII redaction; cost panel.
- Deployment: Docker/Compose + Fly.io; README with real numbers only.

**Non-goals (deferred — do not build in Phase 1).**
- Requirements / Drafting / Critic agents, HITL interrupt/resume, Submission
  Handler (Phase 2, §7.2, §8).
- Injection screen, grounding guardrail (Phase 2, §11). *PII redaction in traces
  ships in Phase 1* because tracing ships in Phase 1 (§13).
- `APPLICATION` and `ANSWER` entities, tracking board, lifecycle state (Phase 2/3, §9, §7.3).
- Autonomous submission — out of scope by design at every phase (§2.4, §10).
- Paid-API final quality pass is dev-on-Ollama by default; the local-vs-API
  comparison is set up here but reported fully in later phases (§13).

---

## 2. Data model subset (§9)

Only three entities, built via SQLAlchemy/Alembic migrations.

**`JOB`** — full entity per §9: `id`, `source`, `source_type`
(`board`·`ats_api`·`aggregator`·`community`·`manual`), `ingested_via`
(adapter | `manual`), `source_id`, `url`, `title`, `company`, `description`,
`salary_min`, `salary_max`, `currency`, `location`, `remote`, `job_type`,
`published_at`, `collected_at`, `embedding` (pgvector), `content_hash`.
`content_hash` drives dedup **and** the embedding cache (unchanged posting never
re-embedded) and collapses the same role across sources to one record (§9).
Missing fields (salary, remote) are stored unknown, never fabricated (Appendix A.1).

**`PROFILE`** — single row: `full_name`, `email`, `links`, `work_history`,
`cv_text`, `cv_embedding`, `target_titles`, `seniority`, `domains_keywords`,
`salary_floor`, `currency`, `location_rules`, `remote_required`. `cv_embedding`
grounds fit analysis (§9). Profile data is runtime data in Postgres, **never in
git** (CLAUDE.md).

**`EVAL_LABEL`** — `job_id`, `label` (good/bad/neutral), `labeled_by`, `notes`;
the ground truth for retrieval and fit metrics (§12).

Not built in Phase 1: `APPLICATION`, `ANSWER`.

---

## 3. MCP tools & LangGraph nodes in scope

**MCP tools (FastMCP, typed/discoverable — §5, §8).**
- `search_jobs(criteria) -> ranked candidates` — hybrid search entry; read-only
  over postings (least privilege, §11).
- `analyze_fit(posting, profile) -> {matches, gaps, score}` — the Fit Agent's
  tool (§8), grounded in posting + profile.
- `ingest_url(url) -> job` — on-demand ingest-by-URL (Appendix A.3).

**LangGraph nodes (§6, §7.1) — search/fit graph only.**
- **Supervisor** — owns the state machine and routing; grounded in graph state (§8).
- **Interpreter** — NL request + profile → structured criteria (terms + filters).
- **Retrieval** — hybrid search + RRF over pgvector and FTS.
- **Fit Analyzer** — per-candidate RAG fit vs profile; returns matches, gaps,
  grounded score. *If results are thin, report the thin result; filters are not
  relaxed to invent matches* (§7.1).

Out of scope this phase: Requirements, Drafting, Critic, HITL Gateway, Submission
Handler nodes (§8, Phase 2).

---

## 4. Build sequence (milestones)

Each milestone is independently verifiable and walks skeleton → working search/fit
→ eval + obs gates. Repo paths are from §18 / CLAUDE.md (`src/job_radar/...`).

### M0 — Persistence foundation
**Build:** Async SQLAlchemy models + Alembic migrations for `JOB`, `PROFILE`,
`EVAL_LABEL`; pgvector extension + vector columns; Postgres FTS (`tsvector` +
GIN index); settings via `pydantic-settings`. **Realizes:** §9 subset, retrieval
substrate scaffolding (§6). **Lands:** `src/job_radar/db/` (models, migrations),
`infra/` Postgres. **Verify:** migration applies on a clean Postgres+pgvector;
round-trip insert/select test for each entity; `just test` green.

### M1 — Ingestion + dedup + embedding cache
**Build:** Source-adapter interface (`fetch`/`map`/`capabilities`/`terms`,
Appendix A.1); the enabled starter adapters (set chosen at build time — see Open
Questions); normalization to the `JOB` schema; cross-source dedup and embedding
cache keyed on `content_hash`; ingest-by-URL (A.3); APScheduler daily batch,
idempotent and retried (§16). Embeddings via `nomic-embed-text` on Ollama (§14).
**Realizes:** Ingestion Worker (§5), `JOB` population (§9), Appendix A.1/A.3.
**Lands:** `src/job_radar/ingest/` (adapters, scheduler, embedding), `infra/`
scheduler. **Verify:** re-running the batch yields **zero duplicates incl.
cross-source** and re-embeds nothing unchanged (§15); ingest-by-URL produces one
`ingested_via=manual` record; adapter contract tests with recorded fixtures (no
live calls, no real PII).

### M2 — Hybrid retrieval (vector + FTS + RRF)
**Build:** Vector similarity query (pgvector), FTS query (Postgres), and RRF
fusion of the two rankings (§6, §19). **Realizes:** Retrieval component (§6),
retrieval substrate (§6). **Lands:** `src/job_radar/retrieval/`. **Verify:**
unit tests for RRF fusion on known rankings; retrieval-only latency p95 < 500 ms
on the corpus (§15); returns ranked candidates for a sample query.

### M3 — Fit analysis (RAG)
**Build:** Fit Analyzer — RAG over profile + `cv_embedding` vs a posting,
returning grounded matches, gaps, and a score (§6, §8). **Realizes:** `fit/`
capability (§3.1, §6). **Lands:** `src/job_radar/fit/`. **Verify:** fit output
cites profile/posting sources; agrees with human labels on the `EVAL_LABEL` set
(measured in M6); thin/empty handling does not fabricate.

### M4 — LangGraph search/fit graph
**Build:** Wire Supervisor → Interpreter → Retrieval → Fit into a LangGraph graph
implementing §7.1; nodes that are only a prompt over the same context stay
functions, not agents (§8). **Realizes:** Agent Service search path (§5, §6, §7.1).
**Lands:** `src/job_radar/agents/`. **Verify:** end-to-end graph test from NL
query → ranked, grounded matches with reasoning + sources; thin-result path
reports thin without relaxing filters (§7.1).

### M5 — Surfaces (MCP + FastAPI + thin Web UI)
**Build:** FastMCP server exposing `search_jobs`, `analyze_fit`, `ingest_url`
(§5, §8); FastAPI endpoints + orchestration entry; thin web UI for profile
editing and ranked results (§5). **Realizes:** MCP Server, FastAPI app, Web UI
(§5). **Lands:** `app/`. **Verify:** MCP client lists and calls the tools; an API
request runs the full graph; web UI renders profile + results; retrieval tools are
read-only (least privilege, §11).

### M6 — Evaluation harness + CI gate
**Build:** Labeled fit set (versioned, in `eval/`); retrieval precision/recall
reported for hybrid **and separately** vector-only and keyword-only; fit-vs-label
agreement; golden queries + a representative safety case run in GitHub Actions; a
regression fails the build (§12). **Realizes:** evaluation plan retrieval/fit
slices (§12). **Lands:** `eval/` (labeled sets, metrics, golden queries, runner),
`.github/workflows/`. **Verify:** `eval` runner emits metrics; hybrid vs component
comparison reported honestly even where hybrid does not win on this corpus (§12);
CI fails on an injected regression.

### M7 — Observability + cost panel
**Build:** OTel GenAI-semantic-convention tracing → Langfuse: what was retrieved,
each node's decision, every tool call's I/O, per-step latency and token cost —
**PII redacted at the export boundary** (§13). Per-query cost decomposed by step;
content-hash embedding-cache cost-per-query reported before/after (§13).
**Realizes:** Observability container (§5), §13. **Lands:**
`src/job_radar/obs/`, integrated across nodes/tools. **Verify:** a query produces
a complete trace in Langfuse; redaction test proves **zero PII** in URLs, query
strings, or trace export (§11, §15); cost panel reports per-query decomposition.

### M8 — Deployment + README
**Build:** Docker/Compose (app + agent + ingestion + Postgres/pgvector volume),
Fly.io config, health checks, env-based config; `.env.example` as the committed
contract (§16, CLAUDE.md). README documents run/eval/observe with **real numbers
only** (no fabricated metrics, CLAUDE.md). **Realizes:** §16 deployment.
**Lands:** `infra/`, repo root. **Verify:** `docker compose up` brings the stack
healthy; demo query runs end-to-end; no secrets/PII committed (gitleaks in
pre-commit + CI).

---

## 5. Evaluation & observability deliverables required for "Phase 1 done"

Phase 1 does not count until these are green (§12, §13, §15, §17):
- Retrieval precision/recall over the labeled fit set, hybrid vs vector-only vs
  keyword-only, reported honestly (§12).
- Fit judgments agree with human labels on the eval set (§12).
- Golden queries + a representative safety case gate CI; regression fails the
  build (§12).
- Full traces in Langfuse with **zero PII** at the export boundary (§13, §15).
- Cost reported per query, decomposed by step, with embedding-cache before/after
  (§13).
- NFRs observed and reported: retrieval p95 < 500 ms; end-to-end query p95 < 8 s;
  daily batch < 15 min, zero duplicates on re-run (§15).

---

## 6. Open questions / decisions deferred to implementation

Per Appendix A.4, these depend on live targeting and terms verified at build time:
1. **Enabled source set and ordering** (A.4.1) — recommended starting backbone is
   a handful of per-ATS boards for target companies plus one or two remote
   aggregators, but the set is configuration. Decide which adapters ship in M1.
2. **Company seed list for ATS boards** (A.4.2) — derived from the user's target
   companies; expected to grow.
3. **Whether any aggregator is enabled** (A.4.3) — decided per source after
   reading current terms; off by default.
4. **All terms, rate limits, and field availability must be re-verified against
   each provider's official docs before an adapter ships** (Appendix A.4).
5. **Dev generation model** — local Qwen/Llama choice for any LLM-judge/fit
   reasoning (§14); paid-API comparison is set up but reported later (§13).

### Gaps / notes flagged against DESIGN.md
- **Fit Analyzer needs an LLM for grounded reasoning, but §14 specifies the
  generation model only "for dev (Qwen/Llama)".** Phase 1 runs fit reasoning on
  the local generation model; this is consistent with §13's local-dev posture,
  noted here because §17's Phase-1 list emphasizes embeddings over generation.
- **LLM-judge belongs to draft-quality eval (§12), which is Phase 2.** Phase 1's
  fit evaluation uses the human-labeled `EVAL_LABEL` set directly (precision/recall,
  agreement) and does not require a validated LLM-judge — that is a Phase 2
  deliverable. Called out so the M6 harness is not over-built.
