# Phase C Implementation Plan — Overview

Companion to [`docs/plans/PHASE_C_OBSERVABILITY_DESIGN.md`](../PHASE_C_OBSERVABILITY_DESIGN.md)
(read that first — this is the locked, buildable breakdown of its §7 build order, one
file per phase, each with its own study references per your request). Same
design→implementation-plan pattern this project already uses (`M3_FIT_DESIGN.md` →
`M3_IMPLEMENTATION_PLAN.md`).

## The five phases

| # | File | What it delivers | Depends on |
|---|---|---|---|
| C.1 | [`01-setup-and-foundations.md`](01-setup-and-foundations.md) | Langfuse Cloud project, env config, a trivial trace flowing end to end, core concepts understood | Nothing |
| C.2 | [`02-core-instrumentation-and-redaction.md`](02-core-instrumentation-and-redaction.md) | `generate()`/`embed()` instrumented — covers all 4 current LLM call sites at once — with the PII redaction boundary built in from the start, not bolted on after | C.1 |
| C.3 | [`03-retrieval-spans.md`](03-retrieval-spans.md) | `search()`/`build_run()` instrumented — what was retrieved, from which arm, at what weight | C.2 |
| C.4 | [`04-cost-tracking.md`](04-cost-tracking.md) | Custom model price definitions for local Ollama models; token counts verified flowing through | C.2 |
| C.5 | [`05-validation-and-acceptance.md`](05-validation-and-acceptance.md) | Proof it actually works: reproduce a known bug, confirm it's visible in a trace without writing a throwaway script | C.1-C.4 |

C.3 and C.4 don't depend on each other — do them in either order, or in parallel if
you're splitting sessions again.

## Why one file per phase, with study references in each

Per the design doc's own framing: this phase exists because tracing turns "write a
special investigation" into "read a trace." The reverse is also true for *building* it —
if you implement this without understanding what a span/trace/observation actually is,
or how OTel context propagation works, you'll be debugging the debugging tool. Each
phase file below front-loads exactly what you need to understand *before* touching code
for that phase, not a generic reading list — every reference is there because a specific
decision or piece of code in that phase depends on understanding it.

## Before starting C.1

- Decide: Langfuse Cloud account under your own login, region (US or EU — data residency,
  doesn't matter functionally for a solo project, pick whichever is geographically closer).
- Note the free Hobby tier: 50,000 units/month, no card required, 30-day retention. One
  important sizing detail for later: **a "unit" is every trace + every observation +
  every score** — a single `job-radar-fit` run against 100 jobs, once C.2 is live, is
  roughly 1 trace + ~100+ generation observations in that one command. Comfortably inside
  50k/month for normal solo iteration, but worth knowing before running large sweeps
  repeatedly in a short window.
