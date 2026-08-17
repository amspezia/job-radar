# C.5 — Validation & Acceptance

**Goal:** prove the actual point of this whole phase, not just that traces exist. Per
`PHASE_C_OBSERVABILITY_DESIGN.md`'s own framing: the acceptance bar for Phase C is
*"reproduce a known bug and confirm it's visible in a trace without writing a throwaway
script to find it"* — not "the dashboard has data in it."

## Why this is a distinct phase, not just "testing"

C.1-C.4 each have their own local acceptance checks already (see each file's last
section). This phase is different: it's a deliberate, end-to-end exercise using a bug
this session already found and fixed once — so you have a known-correct answer to check
your trace against, rather than trying to validate observability by looking for a bug you
don't yet know the shape of.

## Steps

1. **Pick one already-diagnosed bug to reproduce.** The HyDE partial-generation-failure
   bug (fixed in `fit/pipeline.py::build_hyde_embedding`, documented in
   `eval/results/RETRIEVAL_ANALYSIS.md` finding #1) is the best candidate — it's fully
   understood, has a clear before/after, and touches both C.2 (generation retries) and
   C.3 (its downstream effect on retrieval).
2. Temporarily reintroduce the bug's *symptom*, not the code — e.g., monkeypatch
   `_generate_hyde_posting` to fail 2 of 3 times deterministically for one test run
   (don't actually revert the real fix in `pipeline.py`; that's the thing being verified
   as fixed).
3. Run `job-radar-fit` against a small batch with this forced-failure condition active.
4. **Without looking at logs or writing a script**, open the resulting trace in Langfuse
   and answer: how many HyDE generation attempts happened, how many succeeded, and does
   the `build_hyde_embedding` span/its children make the partial-failure state visible?
   If you have to fall back to reading raw logs to answer this, C.2's instrumentation is
   incomplete — go back and add whatever attribute was missing.
5. Do the same exercise for the RRF-weight-mismatch bug (`RETRIEVAL_ANALYSIS.md` finding
   #4): run the same query once against a mis-set weight and once against the correct
   one, and confirm the `fuse` span's `weights` attribute (from C.3) is what actually
   shows the difference between the two traces — this validates C.3 specifically.
6. Write a short note (a few paragraphs, doesn't need its own doc) capturing what you
   found for both exercises — this becomes the evidence that Phase C actually achieved
   its goal, referenceable later the same way this session's earlier investigations
   became `RETRIEVAL_ANALYSIS.md`/`PARSING_ANALYSIS.md`.

## Core concepts to understand before this phase

Nothing new — this phase deliberately doesn't introduce new observability concepts. Its
job is exercising everything C.1-C.4 already taught, under a condition where you already
know the right answer. If something here is confusing, that's a signal to go back to the
relevant earlier phase's references, not to look for new material.

## Study references

- Re-read [`eval/results/RETRIEVAL_ANALYSIS.md`](../../../eval/results/RETRIEVAL_ANALYSIS.md)
  findings #1 and #4 in full before starting — you need the exact original symptom and
  root cause fresh in mind to judge whether the trace actually surfaces it, not a vague
  memory of "something about HyDE caching."
- Re-read [`tests/test_hyde_cache.py`](../../../tests/test_hyde_cache.py) — the existing
  test suite for the exact behavior you're temporarily un-fixing in step 2; useful both
  as a reminder of the precise mechanics and as a model for how to monkeypatch
  `_generate_hyde_posting` cleanly without touching the real fix.

## Acceptance for this phase — and for Phase C as a whole

Both reproductions in steps 4 and 5 are answerable from the Langfuse UI alone, with no
log-reading or script-writing. If either isn't, Phase C isn't actually done yet, even if
every prior phase's individual acceptance check passed — this is the integration test for
the whole phase, and it's allowed to send you back to C.2 or C.3 to close a gap.
