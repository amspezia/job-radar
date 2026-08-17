# Phase C — Observability Design (Langfuse + OpenTelemetry)

> Design-level doc, matching this project's own convention (a `*_DESIGN.md` explores the
> space and locks decisions; a follow-up `*_IMPLEMENTATION_PLAN.md` turns it into a
> file-by-file build, the way `M3_FIT_DESIGN.md` preceded `M3_IMPLEMENTATION_PLAN.md`).
> Written to be read for understanding, not just as a spec — this is the "study" pass.

---

## 1. Where this sits, and why it's next

Per [`docs/RECOMMENDATION.md`](../RECOMMENDATION.md)'s sequencing: Phase A (mechanical
fixes) and most of Phase B (measurement-integrity — the synthetic persona eval, the
CV-parsing eval) are now built *and validated* — not just built. Phase C is next, before
the LangGraph/agent work, for a reason that stopped being abstract partway through this
session and became lived experience:

- The HyDE partial-generation-failure bug (A2/B1's investigation) was only found because
  I happened to grep raw log lines by hand during a debugging session. A trace would have
  shown it as a first-class event with a retry/failure count, immediately, on every run —
  not something you notice by accident.
- `FIT_THROUGHPUT_PLAN.md`'s whole performance investigation was built on **one manually
  instrumented call**, reading Ollama's `prompt_eval_count`/`eval_count` fields by hand,
  once, as a special exercise. That's exactly what tracing gives you continuously, for
  every call, for free — not a one-off measurement script.
- The retrieval root-cause investigation (median rank per grade, top-10 source
  breakdown) was done via throwaway Python scripts written on the spot. With retrieval
  spans recording what was retrieved, from which arm, at what rank, that kind of
  debugging becomes "read the trace," not "write a new diagnostic script."

This is also precisely the Anthropic engineering lesson already cited in
[`docs/STATUS.md`](../STATUS.md) §5.5: *"full production tracing is what makes multi-step
failures diagnosable at all."* Building the LangGraph orchestration layer next, on a
system that still has zero instrumentation, means every new failure mode the graph
introduces becomes indistinguishable from the ones already latent in retrieval/fit
today. Instrument first.

---

## 2. The tech stack, and what each piece actually is

### 2.1 OpenTelemetry (OTel) — the vendor-neutral instrumentation standard

OTel defines a common vocabulary for describing "what happened while my code ran":

- **Trace** — one end-to-end operation (e.g., one `run_fit_pipeline()` call).
- **Span** — one step inside a trace, with a start/end time, a name, and key-value
  **attributes** (e.g., a `retrieve` span, a `generate(FitJudgment)` span). Spans nest —
  a trace is a tree of spans.
- **Semantic conventions** — standardized attribute *names*, so a "which model, how many
  tokens" question is asked the same way regardless of which vendor's tooling reads the
  trace. The **GenAI semantic conventions** (`gen_ai.*`) are OTel's LLM-specific vocabulary
  — `gen_ai.request.model`, `gen_ai.usage.input_tokens`/`gen_ai.usage.output_tokens`,
  `gen_ai.response.finish_reasons`, `gen_ai.input.messages`/`gen_ai.output.messages`. As
  of 2026 these are still marked experimental by OTel itself (attribute names can still
  shift), but they're the direction every major observability vendor has converged on —
  worth using even at "experimental" status, since the alternative is inventing your own
  vocabulary that's guaranteed less portable.

OTel itself is a **standard and an SDK**, not a product — it doesn't give you a UI, a
dashboard, or storage. Something has to *receive* the traces it produces.

### 2.2 Langfuse — the LLM-observability platform that receives them

Langfuse is where traces actually get looked at: a trace explorer, cost/token
dashboards, and a place to attach eval scores to individual generations. What's changed
since `DESIGN.md` was written (commit 1, never revised — see `docs/STATUS.md`'s
staleness note): **Langfuse's Python SDK (v3+, GA mid-2025; v4 released March 2026) is
now built natively on OpenTelemetry.** Initializing the Langfuse client registers its own
OTel span processor; any other OTel-instrumented spans in the same process nest into the
same trace automatically. `DESIGN.md`'s tech-stack line ("Langfuse + OpenTelemetry") reads
as two separate technologies wired together — in current practice they're one integration,
not two. ([Langfuse OTel Python SDK](https://langfuse.com/integrations/native/opentelemetry),
[OTel GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai))

**Practical consequence for this project's `.env.example`:** it declares
`OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`, implying a local OTel Collector
sitting between the app and Langfuse. That's the right shape for a multi-service org
fanning traces out to several backends. For one app exporting to one destination, it's a
middleman this project doesn't need — the Langfuse SDK can export directly. Recommend
dropping `OTEL_EXPORTER_OTLP_ENDPOINT` from the active config (keep it documented as an
option, not a requirement) unless a second consumer of these traces shows up later.

### 2.3 Self-hosted vs. Langfuse Cloud — a decision `.env.example` has already half-made, worth revisiting

`.env.example`'s placeholder, `LANGFUSE_HOST=http://localhost:3000`, is the self-hosted
convention (Langfuse's own docker-compose UI runs on :3000 locally). Current self-hosting
requirements: Postgres + **ClickHouse + Redis + S3-compatible object storage** — three
additional stateful services beyond what this project already runs, to observe a
single-user local dev workload. That's a real operational cost with no corresponding
benefit here (no data-sovereignty requirement, no team to share a self-hosted instance
with). **Recommendation: use Langfuse Cloud instead** (`LANGFUSE_HOST` pointed at
Langfuse's hosted endpoint), and update `.env.example`'s placeholder accordingly. This is
a one-line config decision, not an architecture change — nothing about the instrumentation
code differs between the two, only where `LANGFUSE_HOST` points.

---

## 3. What gets instrumented — fewer places than it sounds like

`docs/COMPONENTS.md`'s cross-cutting note already established the relevant fact: **every
LLM/embedding call in this codebase goes through exactly two functions**,
`adapters/generation.py::generate()` and `adapters/embeddings.py::embed()`. That's not a
coincidence for Phase C — it means instrumenting those two functions covers every current
call site (`profile/parse.py`, `ingest/extract.py`, `fit/pipeline.py`'s HyDE synthesis,
`fit/analyze.py`) at once, the same way `adapters/retry.py` added retry/backoff to all of
them by touching only those two files. This is the second time this exact seam has paid
for itself this session.

**Two files to touch for LLM/embedding tracing**, not four-plus:

- `adapters/generation.py::generate()` — wrap the `with_retry(_call, ...)` call in a
  Langfuse generation observation. Attributes: `gen_ai.request.model`, the Pydantic
  schema name (`gen_ai.output.type` or a custom `schema` attribute — this codebase's
  entire generation surface is schema-constrained, worth its own attribute), retry count
  (from `with_retry`'s own attempt loop), and token usage from Ollama's own response body
  (`prompt_eval_count`→`gen_ai.usage.input_tokens`, `eval_count`→
  `gen_ai.usage.output_tokens` — already present in every Ollama response, currently
  read only for the `TruncatedGeneration` check).
- `adapters/embeddings.py::embed()` — same shape, simpler (no schema, no truncation
  concern): model, task (`query`/`document`), token count if Ollama's embed response
  exposes one, retry count.

**A distinct second concern: retrieval spans.** `DESIGN.md` §13 explicitly wants "what
was retrieved" traced, not just LLM calls — retrieval has no LLM call in it (BM25/vector
search are pure DB queries) but is exactly the kind of step whose behavior needs to be
inspectable when a ranking looks wrong (as it did, repeatedly, during this session's
investigation). Instrument `retrieval/search.py::search()` (or `eval/qrels.py::build_run`
for the eval path) with a span per arm (`bm25_arm`, `hyde_arm`) recording candidate count
and top score, plus a `fuse` span recording the RRF weights actually used and the final
ranking size. This directly replaces the ad hoc "pull the top 10, print source and grade"
scripts written by hand during the retrieval investigation with something inspectable on
every run, not just when someone thinks to write a diagnostic script.

**Not in scope for Phase C:** LangGraph node-level tracing (there's no graph yet — that's
Phase D), distributed tracing across services (this is one process), and any
Langfuse-side alerting/paging setup — this phase is "make failures diagnosable after the
fact," not "get paged in real time."

---

## 4. PII and safety — this is not optional, and the boundary already exists

`DESIGN.md` §11/§13 are explicit: PII must be redacted at the trace-export boundary,
"never let observability become a PII leak." This project already drew exactly this
boundary once, for a different but structurally identical problem — the golden-fixture
export in A1 (`scripts/export_golden_fixture.py`'s docstring: real name/email/CV
text/work history never leave the script; structured search-criteria fields do). Reuse
the same boundary here:

| Safe to trace fully | Must be redacted/excluded |
|---|---|
| Model name, schema name, token counts, latency, retry count | Raw prompt text (contains CV text, profile fields) |
| `target_titles`, `tech_stack`, `seniority`, `domains` (structured criteria) | `full_name`, `email`, `cv_text` |
| Retrieval scores, arm names, RRF weights, ranking size | Individual job posting full text (less sensitive than PII, but still unnecessary to store verbatim in a third-party trace store) |
| Grade/verdict/score outputs (`FitAssessment`, not the evidence quotes) | `Evidence.quote` fields specifically — these are verbatim CV/posting excerpts by design (`fit/schema.py`'s own docstring) |

Concretely: pass `input`/`output` to Langfuse's generation observations as **redacted
summaries** (model, schema name, byte length) by default, not the raw prompt/response —
opt into full payload capture only behind an explicit debug flag, off by default. This
mirrors the project's existing logging discipline (`profile/loader.py`,
`profile/parse.py` already log sizes/counts, never CV content) — Phase C extends a
pattern that already exists, doesn't invent a new one.

---

## 5. Cost tracking — how it actually works for local Ollama, concretely

Langfuse infers cost automatically for known hosted-model names, but self-hosted/local
models aren't in its default price list — this project needs a **custom model
definition** (Project Settings → Models in the Langfuse UI, or via API) for each Ollama
model in use (`nomic-embed-text`, the configured `generation_model`/`extraction_model`/
`fit_model`), with per-token price set to $0 (or a nominal figure representing amortized
local compute, if a non-zero comparison point is ever wanted). Token *counts* still come
through accurately either way, since Ollama's response body already includes them —
only the dollar figure needs a manual price definition, once, per model.

This sets up — but does not build — the local-vs-paid-API comparison `DESIGN.md` calls a
"reported deliverable" and `RECOMMENDATION.md` deferred to Phase E (no provider
abstraction exists yet; `generate()` is Ollama-specific down to the payload shape).
Phase C's job is making sure that whenever Phase E adds a second provider, the cost
picture is already visible per-call, not something built from scratch at that point.

---

## 6. A worked example — what one real trace looks like

Reading a trace should answer "why did this fit score come out the way it did," end to
end. For one `run_fit_pipeline()` call against one job:

```
trace: fit_pipeline_run (profile_id=..., job_count=100)
├─ span: build_lexical_query                         [12ms]
├─ span: build_hyde_embedding
│  ├─ (cache hit: 0 children — or, on a miss:)
│  ├─ generation: generate(_HyDEPosting) × up to 3    [model, retry_count, tokens]
│  └─ embedding: embed(task=document) × N             [model, tokens]
├─ span: retrieve
│  ├─ span: bm25_arm      [candidate_count, top_score]
│  ├─ span: hyde_arm      [candidate_count, top_score]
│  └─ span: fuse          [weights=[2.0,1.0], k=60, result_count]
└─ span: analyze_fit × 100 (one per job, cache-hit ones show 0 LLM children)
   └─ generation: generate(FitJudgment)                [model, retry_count, tokens, verdict]
```

Given today's investigations, this shape would have made three of this session's
findings visible immediately instead of requiring a special investigation: the HyDE
partial-failure (visible as a generation span with a non-zero retry count feeding an
undersized `build_hyde_embedding` result), the domains-missing-from-retrieval bug
(visible by inspecting the `bm25_arm`/`fuse` span's actual query attributes), and the
RRF weight mismatch between eval and production (visible by comparing the `fuse` span's
recorded `weights` attribute across two trace sources instead of grepping two files).

---

## 7. Suggested build order (a future `PHASE_C_IMPLEMENTATION_PLAN.md`'s job to lock)

1. Langfuse Cloud project + API keys; update `.env.example`'s `LANGFUSE_HOST` placeholder
   and drop `OTEL_EXPORTER_OTLP_ENDPOINT` from the required set (§2.3).
2. Instrument `generate()`/`embed()` (§3) — two files, covers every current LLM call site.
3. Add the redaction boundary (§4) as part of the same change, not a follow-up — tracing
   PII even briefly during development is the failure mode to design out from the start,
   not patch after the fact.
4. Add retrieval spans (§3's second concern) — a distinct, smaller change once step 2 is
   live and the pattern is established.
5. Define custom model prices for the local Ollama models (§5); confirm token counts
   flow through correctly on a real `job-radar-fit` run.
6. Validation: deliberately reproduce one of this session's already-diagnosed bugs (e.g.,
   force a HyDE partial failure) and confirm it's visible in a trace without writing a
   throwaway script to find it — that's the actual acceptance bar for this phase, not
   "traces exist."

---

## Sources

- [Langfuse — OpenTelemetry (OTEL) for LLM Observability](https://langfuse.com/integrations/native/opentelemetry)
- [Langfuse — OTEL-based Python SDK v3](https://langfuse.com/changelog/2025-05-23-otel-based-python-sdk)
- [Langfuse — Token & Cost Tracking](https://langfuse.com/docs/observability/features/token-and-cost-tracking)
- [OpenTelemetry — GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai)
- [OpenTelemetry blog — Inside the LLM Call: GenAI Observability with OpenTelemetry (2026)](https://opentelemetry.io/blog/2026/genai-observability/)
