# C.4 — Cost Tracking

**Goal:** every trace's token counts resolve to a dollar figure in the Langfuse
dashboards, for the local Ollama models actually in use — laying the groundwork for the
local-vs-paid-API comparison `DESIGN.md` calls a reported deliverable, without building
that comparison now (no second provider exists yet — that's Phase E, deliberately out of
scope here per `docs/RECOMMENDATION.md`).

## Why this needs a manual step, not just code

Langfuse can infer cost automatically for model names it recognizes (hosted models like
`gpt-4o`), but a local Ollama model name (`qwen2.5:7b`, `nomic-embed-text`, whatever's
configured) isn't in its default price list. Token *counts* already flow through
correctly once C.2 is done (they come from Ollama's own response, not from Langfuse
inference) — this phase is specifically about getting a dollar figure attached to those
counts, which requires defining the model once in Langfuse's UI.

## Steps

1. Confirm C.2 is fully done and token counts are visible and correct on real traces
   first — this phase adds pricing on top of counts that must already be right.
2. In Langfuse's Project Settings → Models (or via API — check current UI/API shape, it
   may have moved since these docs were written), add a custom model definition for each
   Ollama model actually configured in `.env` (`embedding_model`, `generation_model`,
   and any of `extraction_model`/`fit_model` that differ from the default).
3. Decide the price-per-token value: `$0` is the honest number for genuinely local,
   already-owned compute — resist the temptation to invent an amortized-hardware-cost
   number unless you actually want that comparison point; `$0` keeps the dashboard
   meaningful as "which calls are expensive in *token volume*," which is still useful
   signal on its own (ties directly back to `FIT_THROUGHPUT_PLAN.md`'s own finding that
   decode/output tokens are ~93% of a fit run's wall-clock cost — that's a real cost even
   at $0 nominal price, and now visible per-call instead of requiring a special
   instrumented measurement session the way it did in that plan).
4. Run `job-radar-fit` against a real batch; confirm the Langfuse dashboard shows
   token-volume breakdowns per model/schema, matching what you'd expect from
   `FIT_THROUGHPUT_PLAN.md`'s own numbers (~800 output tokens/job for `analyze_fit`) —
   this is a good cross-check that C.2's token-count wiring is actually correct, not
   just present.

## Core concepts to understand before doing this

- **Why cost tracking is a pricing-table problem, not an instrumentation problem** — the
  hard part (getting accurate token counts per call) is already solved by C.2; this
  phase is closer to data entry than engineering. Don't over-build here.
- **Unit economics vs. token economics** — Langfuse's free-tier "unit" (traces +
  observations + scores, from `00-overview.md`) is a different accounting axis from
  token/dollar cost. Don't conflate "this run is expensive in Langfuse units" with "this
  run is expensive in tokens" — they're tracking different things and can move
  independently.

## Study references

- [Langfuse — Token & Cost Tracking](https://langfuse.com/docs/observability/features/token-and-cost-tracking) —
  the primary reference for this whole phase: how custom model definitions work, how
  Langfuse falls back to inference when usage isn't provided, and why providing usage
  explicitly (as C.2 does) is more reliable for a model Langfuse doesn't already know.
- Re-read [`docs/plans/FIT_THROUGHPUT_PLAN.md`](../FIT_THROUGHPUT_PLAN.md) — not an
  external reference, but the step 4 cross-check depends on remembering that plan's own
  measured numbers (93% decode-bound, ~800 output tokens/job) to know what "looks right"
  means when you check the dashboard.

## Acceptance for this phase

Langfuse's cost/token dashboard shows a non-zero token-volume breakdown per model for a
real `job-radar-fit` run, roughly consistent with `FIT_THROUGHPUT_PLAN.md`'s own prior
measurements. Dollar figures show as $0 (or your chosen nominal value) rather than
blank/unknown.
