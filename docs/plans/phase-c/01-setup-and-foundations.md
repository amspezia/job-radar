# C.1 — Setup & Foundations

**Goal:** zero application code changed yet. Get a Langfuse Cloud project live, the SDK
installed and configured, and one trivial trace flowing end to end — so every later
phase is "add real spans to a pipeline that's already proven to work," not "debug
connectivity while also debugging instrumentation logic."

## Why this is its own phase

Separating plumbing from application logic means a connectivity problem (wrong key, wrong
host, network) and an instrumentation-logic problem (wrong attribute, missing span) never
get debugged at the same time. This phase's only job is proving the plumbing works.

## Steps

1. Create a Langfuse Cloud account/project (see references below for the exact console
   flow — it changes over time, don't rely on this doc for click-by-click UI steps).
2. Get the public/secret API keys, set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` in
   `.env`. Update `.env.example`'s placeholder comments to describe Cloud, not
   self-hosted (`LANGFUSE_HOST` should point at Langfuse's hosted endpoint, not
   `localhost:3000` — see `PHASE_C_OBSERVABILITY_DESIGN.md` §2.3 for why).
3. Decide whether to keep `OTEL_EXPORTER_OTLP_ENDPOINT` in `.env.example` at all (design
   doc's recommendation: drop it from the required set, keep it documented as an optional
   path for a future local-collector setup).
4. `uv add langfuse` (check current major version — v4 shipped March 2026; pin what
   `uv add` resolves to, don't hand-pick an older version without reason).
5. Write one standalone script (not part of the app yet — a throwaway, e.g.
   `scripts/otel_smoke_test.py`, deleted once it passes) that initializes the Langfuse
   client and creates one span with one attribute. Run it, confirm the trace appears in
   the Langfuse Cloud UI.
6. Only once step 5 is visually confirmed in the UI, move to C.2.

## Core concepts to understand before writing any code

You don't need to be an OTel expert, but these specific concepts are load-bearing for
every later phase:

- **Trace vs. span vs. attribute vs. observation.** A trace is one end-to-end operation;
  spans are its nested steps; attributes are key-value metadata on a span; "observation"
  is Langfuse's term for a span, generation, or event specifically within its data model.
  Get this vocabulary straight now — every later phase file uses it without re-explaining.
- **Context propagation.** How a child span automatically knows which trace/parent span
  it belongs to, without you manually passing IDs around. This is what makes nested spans
  (C.3's arm spans inside a retrieve span) work without plumbing IDs through every
  function signature.
- **The `@observe()` decorator vs. `start_as_current_observation()` context manager.**
  Langfuse offers both; understand when each is idiomatic — this directly decides how
  `generate()`/`embed()` get wrapped in C.2.

## Study references

- [OpenTelemetry — Getting Started by Example (Python)](https://opentelemetry.io/docs/languages/python/getting-started/) —
  the official walkthrough of the trace/span/tracer vocabulary above, with runnable code.
  Read this first if any of those terms are unfamiliar.
- [OpenTelemetry — Instrumentation (Python)](https://opentelemetry.io/docs/languages/python/instrumentation/) —
  the reference page for `trace.get_tracer()` / `start_as_current_span()`, the primitives
  everything else builds on.
- [Langfuse — OpenTelemetry (OTEL) for LLM Observability](https://langfuse.com/integrations/native/opentelemetry) —
  how Langfuse's SDK relates to raw OTel (§2.2 of the design doc summarizes this, this is
  the primary source it's summarized from).
- [Langfuse — OTEL-based Python SDK v3](https://langfuse.com/changelog/2025-05-23-otel-based-python-sdk) —
  the changelog announcing the architecture this whole phase is built on; useful for
  understanding *why* the SDK looks the way it does now, not just how to call it.
- [Langfuse — Instrument your application with the Langfuse SDKs](https://langfuse.com/docs/sdk/python/decorators) —
  the `@observe()` decorator and `start_as_current_observation()` reference; read this
  fully before C.2, since that's where you'll actually decide which pattern to use for
  `generate()`/`embed()`.

## Acceptance for this phase

One manually-triggered script produces one visible trace in the Langfuse Cloud UI, with
correct project/keys, no errors. Nothing in `src/job_radar/` has changed yet.
