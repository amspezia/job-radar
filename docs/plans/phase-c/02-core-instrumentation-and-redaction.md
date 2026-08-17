# C.2 — Core Instrumentation (`generate()` / `embed()`) + Redaction Boundary

**Goal:** every LLM/embedding call in the codebase produces a trace, with PII redaction
built in from the first line of code, not added afterward. This is the phase that does
most of the actual work — because `generate()`/`embed()` are the sole external-I/O seam
for every LLM call in this codebase (`docs/COMPONENTS.md`'s own cross-cutting note),
touching these two files covers `profile/parse.py`, `ingest/extract.py`,
`fit/pipeline.py`'s HyDE synthesis, and `fit/analyze.py` all at once.

## Why redaction is *in* this phase, not a follow-up

`PHASE_C_OBSERVABILITY_DESIGN.md` §4 is explicit about this and it's worth restating:
tracing PII even briefly during development is the failure mode to design out from the
start. If redaction lands in a later phase, every trace produced between C.2 and that
later phase potentially has raw CV text sitting in Langfuse's cloud storage. Build the
redacted-by-default version first; there is no "off by default, opt in to safety" here —
it's "safe by default, opt in to full payload capture behind an explicit debug flag."

## Files touched

- `src/job_radar/adapters/generation.py` — wrap the existing `with_retry(_call, ...)`
  call.
- `src/job_radar/adapters/embeddings.py` — same shape, simpler (no schema/truncation).

## Steps

1. Read `src/job_radar/adapters/generation.py` and `embeddings.py` as they exist today —
   both already have the `with_retry`-wrapped `_call()` closure pattern from the earlier
   retry work; the Langfuse observation wraps around that same call, doesn't replace it.
2. Decide observation type: `generate()` calls are Langfuse **generations** (LLM calls
   specifically, get token/cost fields); `embed()` calls are **generations** too in
   Langfuse's model (embeddings are a generation type), or plain **spans** if you'd
   rather keep them out of the cost/token dashboards — decide based on whether you want
   embedding cost visible alongside generation cost (recommend: yes, same dashboard,
   consistent picture).
3. Attributes to set on `generate()`'s observation — redacted by default per the table
   below: `gen_ai.request.model`, schema name (`schema.__name__` — already available),
   retry count (thread through from `with_retry`'s attempt loop or capture via a
   log-count side channel), `gen_ai.usage.input_tokens`/`output_tokens` (from Ollama's
   `prompt_eval_count`/`eval_count`, already read once for the `TruncatedGeneration`
   check — reuse that same response parse, don't call Ollama twice).
4. Attributes for `embed()`: model, task (`query`/`document`), retry count, token count
   if Ollama's embed response exposes one (check the actual response shape — don't
   assume it matches the chat endpoint's field names).
5. Implement the redaction table from the design doc directly:

   | Trace fully | Redact/exclude |
   |---|---|
   | Model, schema name, tokens, latency, retry count | Raw prompt text |
   | Structured criteria (target_titles, tech_stack, seniority, domains) | full_name, email, cv_text |
   | Scores/verdicts | `Evidence.quote` fields specifically |

   Concretely: pass `input`/`output` to the observation as a redacted summary (model,
   schema name, byte length of the prompt) by default; add an explicit, off-by-default
   env flag (e.g. `LANGFUSE_CAPTURE_FULL_PAYLOAD=false`) for anyone who wants full
   prompt/response capture during local debugging, never as a shipped default.
6. Run the existing test suites for both adapters (`tests/test_generation.py`,
   `tests/test_embeddings.py`) — they already mock `httpx.AsyncClient`, so this is where
   you'll find out whether the Langfuse wrapper interferes with that mocking. Extend them
   with a test asserting no raw prompt text reaches whatever object gets passed to the
   Langfuse client (mock the Langfuse client the same way `httpx` is already mocked).
7. Run one real `job-radar-profile` and one real `job-radar-fit` against a couple of
   jobs; confirm traces appear in Langfuse Cloud with the expected attributes and, most
   importantly, manually inspect one trace's `input`/`output` fields in the UI to
   **visually confirm no CV text or PII is present** before considering this phase done.

## Core concepts to understand before writing this code

- **Generations vs. spans in Langfuse's data model** — not the same as OTel's flat
  span-only model; Langfuse layers "generation" (an LLM/embedding call with token/cost
  semantics) on top of the OTel span concept. Understand this distinction before deciding
  observation types in step 2.
- **The OTel GenAI semantic conventions' attribute names** — using `gen_ai.request.model`
  instead of inventing your own `model_name` attribute is what makes these traces
  portable if you ever move off Langfuse. Worth doing right even though the convention is
  still "experimental" status.
- **Why token counts come from Ollama's response, not from Langfuse inference** — Langfuse
  can *infer* token counts for known hosted models from prompt/response text, but won't
  reliably do that for a local Ollama model it doesn't recognize; passing the counts
  Ollama already computed is more accurate and cheaper than relying on inference.

## Study references

- [OpenTelemetry — GenAI semantic conventions, spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) —
  the actual attribute-name reference (`gen_ai.request.model`, `gen_ai.usage.*`,
  `gen_ai.response.finish_reasons`) — keep this open while writing step 3/4's attribute
  code, don't invent names from memory.
- [Langfuse — Instrument your application with the Langfuse SDKs](https://langfuse.com/docs/sdk/python/decorators) —
  same reference from C.1, re-read with step 2's decision in mind (generation vs. span
  creation specifically).
- [Langfuse — Observation Types](https://langfuse.com/docs/observability/features/observation-types) —
  the generation/span/event distinction referenced above, directly informs step 2.
- [Langfuse — Log Levels](https://langfuse.com/docs/observability/features/log-levels) —
  relevant for deciding whether a retried-but-succeeded call should be logged at a
  different level than a clean first-try success, useful context for the retry-count
  attribute in step 3.
- Re-read [`src/job_radar/adapters/retry.py`](../../../src/job_radar/adapters/retry.py)
  and its tests (`tests/test_retry.py`) — not an external reference, but the exact shape
  the Langfuse wrapper needs to compose with; understand `with_retry`'s attempt-count
  bookkeeping before deciding how to surface it as a trace attribute.

## Acceptance for this phase

Run `job-radar-profile`, `job-radar-ingest` (a small batch), and `job-radar-fit` against
a handful of jobs. All three produce traces in Langfuse Cloud with correct model/token/
retry attributes. Manually inspect at least one trace from each and confirm zero raw CV
text, name, or email is visible anywhere in it.
