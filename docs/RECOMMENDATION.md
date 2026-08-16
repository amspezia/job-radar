# Job Radar — How I'd Actually Proceed

> Asked for explicitly: my honest opinion, ignoring existing docs/plans/roadmaps as authority.
> This draws on [ASSESSMENT.md](ASSESSMENT.md) (what's actually solid vs. shaky) rather than on
> `DESIGN.md`'s phase ordering or [STATUS.md](STATUS.md)'s "next steps," and it disagrees with
> both in sequencing — not in destination. The destination (LangGraph orchestration, real agent
> work, Langfuse/OTel observability) is still right. The order I'd build it in has changed,
> because the assessment surfaced problems that were invisible until read at the component level.

## The one-sentence version

**Fix the gaps between what this system claims about itself and what it actually does, before
adding a new layer (agents) that will make those gaps harder to find, not easier.**

## Why this ordering, not the roadmap's

`DESIGN.md`'s phases and [STATUS.md](STATUS.md)'s "M4+M7 next" are both organized around
*building forward*: finish Phase 1's remaining milestones, then move to Phase 2. That's a
reasonable way to sequence *new capability*. It's the wrong frame for what [ASSESSMENT.md](ASSESSMENT.md)
actually found, which is: the CI regression gate has been silently skipping, not passing, on
every commit; every retrieval-tuning conclusion in the codebase rests on a sample size of one
query topic; the fit-score weights were never checked against the judgment they claim to model;
and the most upstream artifact in the entire system (parsed CV → profile) has no quality
measurement at all. None of that is "unbuilt Phase 1 milestone." All of it is **an unmeasured, and
in one case falsely-believed-measured, foundation** — and the Anthropic engineering lesson that
was worth citing in [STATUS.md](STATUS.md) §5.5 applies directly here: *full production tracing
is what makes multi-step failures diagnosable at all.* Adding a LangGraph layer, an Interpreter
node, and eventually Requirements/Drafting/Critic agents on top of a foundation where you
currently cannot tell whether a bad fit score came from retrieval, profile parsing, or the LLM
judgment — because none of those are separately measured — means every new failure in the agent
layer will be indistinguishable from a pre-existing latent one. That's a worse position to debug
from, not a better one. Fix attribution first.

This is also, honestly, the more interesting engineering problem of the two right now. The
retrieval/fusion architecture is already good; tuning it further without fixing the n=1 problem
is polishing a number that isn't a valid comparison yet. The measurement and calibration gaps are
where the real unresolved technical risk in this project currently sits.

## The plan

### Phase A — Make the existing reliability claims true (days, not weeks)

These are all small, and every one of them closes a gap between what a doc claims and what the
code does — the exact category of bug this project's thesis exists to prevent.

1. **Fix the CI eval gate so it actually executes.** Seed a minimal fixture (small profile + the
   handful of jobs the golden qrels reference) into the CI Postgres before
   `test_ndcg_meets_golden_threshold` runs, rather than depending on a full dev-database reflect
   that CI never has. This is the highest-leverage single fix in this document — it's the
   difference between the regression gate being real and being theater.
2. **Wire `dense_query_cache` into production HyDE generation.** The invalidation trigger
   (CV reload) is already correct in `profile/loader.py`. Read it in `fit/pipeline.py` before
   regenerating; only resynthesize when it's empty. This removes 3 LLM calls from the hot path of
   every `job-radar-fit` invocation where the profile hasn't changed since the last run — a real
   latency and cost win, using a mechanism that's already half-built.
3. **Add extraction-null-rate and duplicate-rate to `quality/metrics.py`.** Both failure modes
   are real (see [ASSESSMENT.md](ASSESSMENT.md)'s ingestion section) and currently invisible.
   Small addition to a module that already has the right shape for it.
4. **Add basic retry-with-backoff to `generate()`/`embed()`.** A few lines each. Transient Ollama
   failures under concurrent load are currently permanent losses; this is the cheapest possible
   reliability win available in the codebase.

### Phase B — Fix what makes six months of tuning claims unverifiable

This is the part I'd actually spend the bulk of near-term effort on, ahead of any new feature
work, because it's what makes every future retrieval or fit change trustworthy going forward
instead of another anecdote.

5. **Build the synthetic multi-persona eval.** `docs/plans/SYNTHETIC_EVAL_DESIGN.md` already
   designed this correctly (5 personas, InPars-style query generation) — it just isn't built.
   This is the actual fix for the n=1 problem, and it's the precondition for trusting any further
   RRF-weight, field-boost, or arm-inclusion tuning. Do this before touching those knobs again.
6. **Build a small fit-score calibration pass**, separate from retrieval relevance labeling: take
   a sample of `score_fit` outputs, get genuine "would I apply" human judgments on them (the
   signal M6's own docs say is different from retrieval relevance and was never collected), and
   check whether the `0.50/0.15/0.25/0.10` weights and `80/60/40` bands actually track it. Adjust
   the constants if they don't — they're config, not doctrine.
7. **Give CV parsing *any* eval.** Doesn't need to be elaborate — even a dozen hand-annotated
   reference CVs checked against `parse_cv`'s output would surface whether the most-trusted,
   least-checked artifact in the system is actually reliable. Right now there's a wide gap
   between how much this output is trusted downstream and how much it's been checked.

### Phase C — Observability (M7), now that there's something real to observe

8. Wire Langfuse/OTel through the four existing LLM call sites (`profile/parse.py`,
   `ingest/extract.py`, `fit/pipeline.py`'s HyDE synthesis, `fit/analyze.py`). This was already
   next in [STATUS.md](STATUS.md), and the reasoning holds — I'm just moving it after Phase A/B
   instead of concurrent with them, because tracing a system whose measurement gaps you haven't
   found yet just gives you well-instrumented visibility into numbers you can't trust.

### Phase D — Then the graph, then the deliberate multi-agent work

9. **M4** — the small `interpret → retrieve → fit` LangGraph graph, as already scoped earlier in
   this conversation. No change to that design; it's still right-sized.
10. **The two agents we scoped as genuinely justified** — the Company Intelligence Agent and the
    Drafting↔Critic evaluator-optimizer loop (see the earlier discussion in this conversation).
    Building these now, on a measured and traced foundation, means a bad output is attributable
    to a specific step instead of adding to an already-unattributable pile.

### Phase E — Whenever it's actually needed, not before

11. Provider abstraction for the paid-API swap. Nothing forces this until the local-vs-API cost/
    quality comparison `DESIGN.md` calls a reported deliverable actually needs to run. Don't
    build it speculatively — but don't forget it's fully unbuilt today either, since `generate()`
    is Ollama-specific down to the payload shape.

## What I'm deliberately *not* recommending

- **Not** a rewrite of anything in Foundations, the adapter contract, retrieval's architecture,
  or fit's mechanics — all of that is genuinely solid and the assessment says so plainly. The
  problem in this codebase is almost entirely *unmeasured claims*, not *bad engineering*.
- **Not** building the full Phase 2 agent roster (Requirements/Drafting/Critic/Submission
  Handler) as originally scoped in `DESIGN.md` §8. The reframing from earlier in this
  conversation still holds: most of those aren't real agents, and building all of them now would
  add surface area on top of the same unmeasured foundation Phase D is designed to avoid doing.
- **Not** touching ParadeDB, HyDE, or the BM25/RRF architecture itself. The concern is the
  *tuning* built on top, not the design under it.

## The honest tradeoff

This ordering delays new user-facing capability (the agent work, which is also the part you said
you want for self-development) by however long Phases A-C take — realistically more like 1-2
focused weeks than months, since most of Phase A is small and Phase B's heaviest item (synthetic
eval) is already designed, not still being figured out. If the priority is "get to LangGraph/
Langfuse/multi-agent learning as fast as possible" over "make sure the foundation underneath it
is trustworthy first," that's a legitimate call to make differently than I'm recommending — but
I'd make it consciously, knowing Phase A's CI-gate fix in particular is the kind of thing that's
cheap now and much more annoying to discover six months into agent-building, when a bad Drafting
output could be coming from any of five unmeasured upstream layers instead of one.
