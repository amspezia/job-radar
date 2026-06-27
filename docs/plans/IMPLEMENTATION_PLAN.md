# Job Radar — Implementation Plan

> Table of contents for the build. The three phases and their boundaries are
> fixed by [DESIGN.md §17](../../DESIGN.md). This document maps each phase to the
> containers it touches, its workstreams in dependency order, and the
> evaluation/safety gates that must be green for the phase to count. Per the
> design's thesis (§2.2, §12, §13), evaluation, observability, and guardrails are
> deliverables in every phase, not a final pass.

The protected core in every phase is **the working path plus its evaluation and
safety gates**; breadth (extra sources, larger eval sets, extra optimizations) is
cut before the core path (§17). Honor the stack and repo layout already fixed by
DESIGN.md (§14, §18, §19) and CLAUDE.md — subpackages and dependencies land
**with their features**, never as empty stubs.

---

## Phase 1 — Discovery + reliability spine (v1)

**Goal / done.** A standalone deliverable a user (or an MCP client) can ask
"what's worth applying to this week?" and get back ranked, grounded matches with
reasoning and sources (§7.1). "Done" means: daily ingestion populates the corpus
idempotently; hybrid search (vector + FTS + RRF) serves ranked candidates;
the LangGraph search/fit graph runs interpret → retrieve → fit; the MCP tool
surface and a thin web UI expose it; the evaluation harness reports
retrieval and fit metrics in CI; traces and a cost panel are live; and the demo
deploys. See [PHASE_1_DESIGN.md](PHASE_1_DESIGN.md) for the milestone-level design.

**Containers / sections touched.** Ingestion Worker, Database (Postgres +
pgvector + FTS), Agent Service (Supervisor, Interpreter, Retrieval, Fit), MCP
Server, FastAPI app + thin Web UI, Observability (§5). Realizes workflow §7.1,
data-model entities `JOB`, `PROFILE`, `EVAL_LABEL` (§9), retrieval substrate (§6),
the evaluation plan's retrieval/fit slices (§12), and observability + cost (§13).

**Workstreams (dependency order).**
1. Persistence foundation — SQLAlchemy/Alembic models for `JOB`, `PROFILE`,
   `EVAL_LABEL`; pgvector + FTS columns and indexes (§9).
2. Ingestion — source-adapter interface (Appendix A.1), the enabled starter
   adapters, normalization, content-hash dedup + embedding cache, ingest-by-URL
   (A.3), APScheduler daily batch (§5, §16).
3. Retrieval — vector + FTS queries fused with RRF (§6, §19).
4. Fit — RAG fit analysis grounded against profile/CV embedding (§6, §8).
5. Agent graph — Supervisor + Interpreter + Retrieval + Fit nodes in LangGraph
   (§6, §7.1).
6. Surfaces — MCP tools (`search_jobs`, `analyze_fit`, `ingest_url`) via FastMCP;
   FastAPI endpoints; thin web profile/results UI (§5).
7. Observability + cost — OTel GenAI tracing → Langfuse with PII redaction;
   per-query cost decomposition; embedding-cache before/after (§13).
8. Evaluation harness — labeled fit set, retrieval precision/recall (hybrid vs
   vector-only vs keyword-only), fit-vs-label agreement, golden-query CI gate (§12).
9. Deployment + README — Docker/Compose, Fly.io, health checks, env config (§16);
   real-numbers-only README (CLAUDE.md hygiene).

**Eval + safety gates (must be green to ship).**
- Retrieval precision/recall reported over the labeled fit set, with hybrid
  compared against vector-only and keyword-only; result reported honestly even
  where hybrid does not win on this corpus (§12).
- Fit judgments agree with human labels on the eval set (§12 agent task success).
- Golden queries pass in GitHub Actions; a regression fails the build (§12).
- **Privacy invariant:** zero PII in URLs, query strings, or unredacted traces —
  redaction tested at the export boundary (§11, §13, §15).
- NFR targets observed and reported: retrieval p95 < 500 ms, end-to-end query
  p95 < 8 s, daily batch < 15 min with zero duplicates on re-run (§15).

**Dependencies / deferred.** No prior-phase dependency (foundation phase).
Deferred to Phase 2: requirements/drafting/critic agents, HITL interrupt/resume,
submission handler, injection/grounding guardrails, draft-quality eval. Deferred
to Phase 3: tracking board and lifecycle state. The `APPLICATION` and `ANSWER`
entities (§9) are **not** built in Phase 1.

---

## Phase 2 — Assisted application + multi-agent (v2)

**Goal / done.** Given a chosen role, the system parses requirements, detects
custom questions, collects personal raw material via an HITL interrupt, drafts
and critiques on-voice answers, and assembles an approved package handed to the
user for submission — **never submitting autonomously** (§7.2, §10). "Done"
means the assisted-application workflow reaches an approvable package without
manual repair, with the injection/grounding guardrails and draft-quality
evaluation gating CI.

**Containers / sections touched.** Agent Service grows the Requirements,
Drafting, Critic agents, the HITL Gateway, and the Submission Handler (§6, §8);
the `guardrails/` package (injection screen, grounding check, PII redaction, §11);
Web UI gains draft review/approval (§5, §10). Realizes workflow §7.2, data-model
entities `APPLICATION` and `ANSWER` (§9), the multi-agent topology (§8), the HITL
boundary (§10), the guardrails (§11), and draft-quality evaluation (§12).

**Workstreams (dependency order).**
1. `APPLICATION` + `ANSWER` persistence with `status_history` audit (§9).
2. Requirements Agent — parse posting to required fields, detect questions,
   identify ATS type, treating posting text strictly as data (§8, §11).
3. HITL Gateway — LangGraph interrupt/resume with state persisted across the
   pause; challenge-and-response approval surface (§10).
4. Drafting Agent — tailored, on-voice answers grounded only in profile/CV +
   user raw input (§8).
5. Critic Agent — grounding/voice/relevance verdict; ungrounded claim →
   regenerate or surface (§8, §11).
6. Submission Handler — assemble + pre-fill the approved package; `prepare_submission`
   tool; **no submit path** (§8, §10).
7. Guardrails — injection screen, grounding guardrail, PII redaction, least
   privilege, as testable invariants (§11).
8. Draft-quality evaluation — human-labeled seed + LLM-judge validated against
   labels, scored on grounding/voice/relevance (§12).

**Eval + safety gates.**
- **Safety invariant:** exactly zero autonomous external form submissions —
  enforced in code and CI (§11, §15).
- **Grounding:** zero approved drafts with unsupported claims about the user in
  the eval set; injection-laced posting caught in CI (§11, §12, §15).
- Draft-quality scores (grounding/voice/relevance) reported with the LLM-judge
  validated against human labels (§12).
- Failure-mode taxonomy documented with an example of each and the guardrail
  that catches it (§12).
- Approval gates and redaction covered by tests as policy-as-code (§11).

**Dependencies / deferred.** Depends on Phase 1's profile store, corpus, agent
graph, observability, and eval harness. Tracking board UI and lifecycle state
remain deferred to Phase 3.

---

## Phase 3 — Tracking + polish

**Goal / done.** Each application is persisted and tracked through the lifecycle
state machine (§7.3) on an editable tracking board, with timestamps recorded on
every transition; the system is packaged end-to-end. "Done" means a user can move
an application Drafted → Applied → … → Offer/Refused, edit state at any point, and
the full Discovery → Assisted application → Tracking path runs as one product.

**Containers / sections touched.** Web UI gains the tracking board; Agent
Service records lifecycle transitions; Database uses `APPLICATION.status` and
`status_history` (§9). Realizes the tracking lifecycle (§7.3) and the third
capability of the solution overview (§3).

**Workstreams (dependency order).**
1. Lifecycle state machine — transitions and timestamping per §7.3.
2. Tracking board UI — editable state, history view (§5, §7.3).
3. End-to-end packaging — wire Discovery → Application → Tracking; final
   local-vs-API cost/quality comparison as a reported deliverable (§13).

**Eval + safety gates.**
- Lifecycle transitions recorded with correct timestamps; `status_history` audit
  preserved (§9).
- Local-vs-paid-API quality and cost comparison reported (§13).
- No regression in Phase 1/2 gates; full CI suite green.

**Dependencies / deferred.** Depends on Phase 2's `APPLICATION`/`ANSWER` records.
**Deferred beyond v2 (out of scope, §17):** autonomous form submission, additional
sources, multi-user, cross-session memory beyond the application workflow.

---

## Cross-phase invariants

These hold from the first commit and are re-verified at every phase gate:
- **Assisted, not autonomous** — the agent prepares; the human submits (§3, §10).
- **No real PII or secrets in git;** `.env.example` is the committed contract
  (§11, CLAUDE.md hygiene).
- **No boilerplate** — only load-bearing files; subpackages and deps land with
  their features (CLAUDE.md, §18).
- **Every commit lint-clean and CI-green** (CLAUDE.md).
