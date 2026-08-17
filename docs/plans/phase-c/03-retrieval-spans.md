# C.3 — Retrieval Spans

**Goal:** "what was retrieved, from which arm, at what rank, under what weights" becomes
readable from a trace instead of requiring a throwaway diagnostic script — directly
closing the gap that made the retrieval-eval investigation (median rank per grade,
top-10 source breakdown) take a dedicated session to answer by hand.

## Why this is separate from C.2

Retrieval has no LLM call in it — BM25 and vector search are plain DB queries. It needs
plain OTel spans, not Langfuse generations, and it's a genuinely different kind of
instrumentation decision (what to record about a *ranking*, not a *model call*). Keeping
it a separate phase means C.2 stays focused on the one pattern (wrap `generate()`/
`embed()`) that covers the most ground for the least effort, before moving to this
smaller, more bespoke piece.

## Files touched

- `src/job_radar/retrieval/search.py::search()` — the production path.
- `eval/qrels.py::build_run()` — the eval path (deliberately bypasses `search()` per its
  own docstring, so it needs its own spans, not inherited ones from `search()`).

## Steps

1. Add a `retrieve` parent span around `search()`'s body.
2. Add a child span per active arm (`bm25_arm`, `hyde_arm` — matching the arm names
   already used in `SearchConfig.arms`), each recording candidate count and top score.
3. Add a `fuse` span around the `reciprocal_rank_fusion()` call, recording the actual
   `weights` used (or `None`/equal-weight if not passed — this is exactly the attribute
   that would have made the `HYBRID` vs. `HYBRID_PROD` weight-mismatch bug from the
   retrieval investigation visible by comparing two trace attributes instead of grepping
   two files) and the final result count.
4. Repeat the same three-span shape in `eval/qrels.py::build_run()` — don't try to share
   the span-creation code between `search()` and `build_run()` just because the shape is
   similar; they're deliberately separate code paths for eval-parameter-flexibility
   reasons (see `PHASE_C_OBSERVABILITY_DESIGN.md`'s citation of `docs/COMPONENTS.md`'s
   note on this), and sharing instrumentation code would quietly re-couple them.
5. Confirm nesting: run `job-radar-fit`, confirm each `analyze_fit` trace's parent
   `fit_pipeline_run` trace shows a `retrieve` span with correctly nested arm/fuse
   children — this is where context propagation (studied in C.1) actually gets exercised
   for the first time in this codebase.

## Core concepts to understand before writing this code

- **Plain OTel spans vs. Langfuse generations** — no token/cost semantics here, this is
  the "just a span with attributes" case C.1's `start_as_current_span()` reference
  covers; don't reach for Langfuse's generation-specific API for this phase.
- **Span attributes vs. span events** — a candidate count is a stable attribute (known at
  span end); if you ever want per-candidate detail (e.g., every job ID considered, not
  just the count), that's an *event* on the span, not an attribute — worth knowing the
  distinction exists even if this phase only needs attributes.

## Study references

- [OpenTelemetry — Instrumentation (Python)](https://opentelemetry.io/docs/languages/python/instrumentation/) —
  same reference as C.1; re-read specifically for the plain-span (non-GenAI) API, since
  this phase doesn't use the GenAI semantic conventions at all.
- [opentelemetry.trace package reference](https://opentelemetry-python.readthedocs.io/en/latest/api/trace.html) —
  the actual API docs for `set_attribute`, `add_event`, and span status — useful once
  you're past the tutorial stage and need the exact method signatures.
- Re-read [`src/job_radar/retrieval/fusion.py`](../../../src/job_radar/retrieval/fusion.py)
  and [`eval/qrels.py`](../../../eval/qrels.py) — not external references, but you need
  both files' exact current arm/weight handling in front of you while deciding what to
  name and record on the `fuse` span; guessing from memory risks attribute names that
  don't actually match what the code does.

## Acceptance for this phase

One `job-radar-fit` run produces a trace where the `retrieve` span's children show
correct per-arm candidate counts and the `fuse` span shows the actual RRF weights used
for that run. Separately, one `eval-run-synthetic` run shows the same shape for
`build_run()`, confirming both retrieval code paths are covered independently.
