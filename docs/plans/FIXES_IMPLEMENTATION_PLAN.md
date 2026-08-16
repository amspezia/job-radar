# Fixes Implementation Plan

> Locked, buildable plan for every item [ASSESSMENT.md](../ASSESSMENT.md) flagged as needing a
> fix, sequenced per [RECOMMENDATION.md](../RECOMMENDATION.md)'s Phase A (cheap, mechanical) and
> Phase B (measurement-building). Phase C (observability), D (LangGraph/agents), and E (provider
> abstraction) are explicitly out of scope here — see "Out of scope" at the end for why each is
> deferred rather than silently dropped.

## Ordering and dependencies

```
Phase A (independent, do in any order, ~days each)
  A1 CI eval gate fixture        — highest leverage, do first
  A2 HyDE dense_query_cache wire-up
  A3 quality/metrics.py additions
  A4 generate()/embed() retry+backoff
  A5 dedup edge-case (decision point — see A5)

Phase B (B1 unblocks B3's fixtures; B2 is independent and human-time-bound)
  B1 Synthetic multi-persona eval  ──┐
                                      ├─▶ B3 CV-parsing quality eval (reuses B1's personas)
  B2 Fit-score calibration eval    ──┘   (independent of B1, can run in parallel)
```

---

## A1 — Make the CI eval gate actually execute

**Problem:** `tests/test_eval_gate.py::test_ndcg_meets_golden_threshold` skips every run because
CI's Postgres has no seeded `Profile`/`Job` rows matching `eval/golden/qrels.json`'s
`profile_id` (122 labeled jobs referenced). The golden files are committed; the fixture data they
depend on is not.

**Approach:**
1. Write a one-off export script, `scripts/export_golden_fixture.py`, run manually against the
   dev DB: select the `Profile` row for `qrels.json`'s `profile_id` and every `Job` row whose id
   appears in `qrels.json["labels"]`, serialize both (including `embedding`/`cv_embedding` as
   plain float lists) to `eval/golden/fixture.json`. Commit the output, not the script's DB
   connection — the script is a dev tool, the JSON is the artifact.
2. Add a `pytest_asyncio` fixture scoped to `test_eval_gate.py` (or a `conftest.py` addition) that
   loads `fixture.json` and upserts the rows into the test DB before
   `test_ndcg_meets_golden_threshold` runs — `INSERT ... ON CONFLICT DO NOTHING` on `id`, so it's
   safe to run against a DB that already has the data (local dev) or an empty one (CI).
3. Remove the `profile is None: pytest.skip(...)` branch's silent pass-through once the fixture
   load guarantees the row exists — keep the `_golden_ready()` skip (that one's legitimate: no
   golden files committed yet) but the "no profile in DB" skip should become impossible, not
   silently tolerated.
4. Document the update procedure in a comment at the top of the test file: whenever
   `eval/golden/{qrels,result}.json` are intentionally regenerated (a real, reviewed retrieval
   change), `fixture.json` must be regenerated in the same commit via the export script, or the
   gate will compare new code against stale data.

**Files:** `scripts/export_golden_fixture.py` (new), `eval/golden/fixture.json` (new, committed),
`tests/test_eval_gate.py` or `tests/conftest.py` (fixture loader), `.github/workflows/ci.yml`
(no change needed — CI already provisions Postgres and runs pytest).

**Acceptance:** `uv run pytest tests/test_eval_gate.py -v` against a *freshly migrated, empty*
local DB (simulating CI) executes the nDCG assertion — not skips it. Deliberately breaking
something upstream of retrieval (e.g., temporarily zeroing a BM25 boost) makes this test fail.

**Note on size:** ~122 jobs × 768-dim float embeddings is roughly 1-2 MB as JSON — acceptable to
commit, similar in spirit to already committing `qrels.json`/`result.json`.

---

## A2 — Wire `dense_query_cache` into production HyDE

**Problem:** `fit/pipeline.py::build_hyde_embedding` regenerates 3 fresh LLM-synthesized
postings on every run. `Profile.dense_query_cache` exists and is correctly invalidated
(`None`) on CV reload in `profile/loader.py`, but nothing in production ever reads or writes it —
the only reader today is `tests/test_eval_gate.py`.

**Decision to lock before building:** the 3-sample averaging exists specifically to reduce
single-draw variance. Caching only one posting text would quietly lose that property on every
cache hit. **Cache all 3 texts**, JSON-encoded, in the existing `Text` column — not just one.

**Approach:**
1. `build_hyde_embedding(profile, session)` — the `session` parameter is already threaded
   through but currently prefixed `_session` (unused). Start using it.
2. On entry: if `profile.dense_query_cache` is set, `json.loads()` it into the 3 texts, embed
   each (`task="document"`), average, return — **skip generation entirely**.
3. On a cache miss (`None`): generate as today (3 concurrent `_generate_hyde_posting` calls),
   and — for whichever succeeded (existing code already tolerates partial failure down to 1
   valid sample) — `json.dumps()` the valid texts into `profile.dense_query_cache`, commit via
   the passed session, *then* embed/average as today.
4. No change needed to the invalidation side — `profile/loader.py` already sets it to `None` on
   every reload, which is the correct trigger.

**Files:** `src/job_radar/fit/pipeline.py` (`build_hyde_embedding`), no schema change (column
already exists).

**Acceptance:** two consecutive `job-radar-fit` runs against the same profile — the second makes
zero calls to `generate()` for HyDE (verify via a debug log line or a test with a mocked
`generate` asserting call count). A CV reload in between causes the next run to regenerate.

---

## A3 — Extraction-failure and duplicate visibility in `quality/`

**Problem:** `quality/metrics.py` doesn't report the null-extraction rate (jobs where BM25 has
near-zero lexical signal because `requirements`/`responsibilities` extraction failed) or a
duplicate-rate proxy for the dedup gap ASSESSMENT.md's ingestion section describes.

**Approach — extraction-null rate:**
1. Add `requirements: str | None` and `responsibilities: str | None` to the `JobRow` dataclass in
   `quality/metrics.py`.
2. Add both columns to the `select(...)` in `quality/cli.py::_load_rows`.
3. Add `extraction_null_pct` to `SourceQuality` and `quality_for()`: percentage of rows where
   *both* fields are `None` (matches the actual degraded-search condition — either one present
   still gives BM25 some signal).
4. Add a row to `quality/cli.py::_DISPLAY`.

**Approach — duplicate-rate proxy:**
1. New function in `metrics.py`, `duplicate_rate(rows) -> float`: group by
   `(source-agnostic) normalized(company, title)`; within each group with >1 row, count it as a
   "likely duplicate cluster" if the rows have *different* `content_hash` values (i.e., dedup's
   hash didn't catch them) — this directly measures the cross-source-location-string-mismatch
   gap ASSESSMENT.md flags, not just raw row counts. Report as a corpus-wide percentage (not
   per-source, since the interesting case is cross-source).
2. Surface it as a second summary line under the main table in `quality/cli.py`'s output (it's a
   corpus-wide stat, not a per-source column, so it doesn't fit `_DISPLAY`'s shape).

**Files:** `src/job_radar/quality/metrics.py`, `src/job_radar/quality/cli.py`.

**Acceptance:** `job-radar-assess` output includes both new numbers; a synthetic test in
`tests/test_quality_metrics.py` (existing test file, extend it) constructs rows with known
null-extraction and duplicate-cluster counts and asserts the computed percentages.

---

## A4 — Retry/backoff for `generate()` and `embed()`

**Problem:** both adapters make one `httpx` call and propagate any failure immediately. Callers
degrade gracefully per-item, but a transient failure is a permanent loss, never a retry.

**Approach:** hand-rolled retry, consistent with the codebase's existing style — there's already
working prior art in the same repo (`adapters/sources/himalayas.py::_get_page`'s
`_RATE_LIMIT_RETRIES`/`_RATE_LIMIT_BACKOFF` pattern). Mirror it:
1. In both `generation.py` and `embeddings.py`, wrap the `httpx` call in a small retry loop: 3
   attempts, exponential backoff (e.g. `1s, 2s, 4s`).
2. Retry only on transient failures — `httpx.TimeoutException`, `httpx.ConnectError`, and
   `httpx.HTTPStatusError` with a 5xx status. **Do not** retry 4xx (a malformed request retrying
   won't fix) or `TruncatedGeneration` (a context-window problem, not a transient one).
3. Log a warning on each retry attempt (attempt number, backoff delay) so retries are visible in
   output, not silent.

**Files:** `src/job_radar/adapters/generation.py`, `src/job_radar/adapters/embeddings.py`.

**Acceptance:** a test with a mocked `httpx.AsyncClient` that fails twice with a 503 then
succeeds on the 3rd call returns successfully; a test with a 400 response does not retry and
raises immediately.

---

## A5 — Dedup edge case (decision point, not auto-approved)

ASSESSMENT.md's summary table lists this as "fix," but RECOMMENDATION.md's Phase A scoped it down
to *monitoring only* (A3, above) rather than changing the hashing algorithm itself, on the
reasoning that it's real but not urgent. Two real options if you want the algorithm itself
changed, not just measured — **flagging both rather than picking one silently**, since this is a
judgment call about corpus behavior, not a mechanical fix:

- **(a) Loosen the hash:** drop `location` from `content_hash` entirely, keying only on
  `company|title`. Fixes the cross-source miss ASSESSMENT.md describes; reintroduces the
  reopened-role collision risk (same title, same company, genuinely different posting) more
  often, since location was the one dimension that could disambiguate two same-titled postings.
- **(b) Add a secondary fuzzy pass:** keep the exact hash as the identity key, but add a
  non-blocking post-ingest check (or a `quality/` report, not a gate) that flags likely
  cross-source duplicates — same `company` + high title-string similarity + differing
  `content_hash` — for manual review, without changing what gets inserted.

**Recommendation if forced to choose:** (b). It doesn't trade one false-negative class for
another false-positive class the way (a) does, and it's additive (visibility only) rather than
changing insert behavior for a corpus that's already been ingested under the current hash. Treat
as optional for this pass — A3 already gives you the *rate*; this would tell you *which* rows.

---

## B1 — Synthetic multi-persona eval

**Problem:** every retrieval-tuning conclusion in the codebase (RRF weights, BM25 boosts, `k`,
the CV-arm ablation) is drawn from one query topic. This is the fix, and it's already fully
designed in `docs/plans/SYNTHETIC_EVAL_DESIGN.md` — this section is that doc's Phase A, made
buildable, not a new design.

**Approach (mirrors the existing doc's design closely):**
1. `eval/personas/*.json` — 5 fixed developer personas as committed fixtures, no PII:
   `alice-rust-senior`, `bob-python-ml-mid`, `carol-ts-fullstack-mid`, `dave-go-platform-staff`,
   `eve-data-junior`. Each defines the *ground-truth* fields a real profile would have
   (`target_titles`, `tech_stack`, `domains`, `seniority`, `work_history` sketch) — these are
   authored by hand, not generated, since they're the ground truth everything else validates
   against.
2. `eval/gen_personas.py` — for each persona JSON, synthesize a plausible `cv_text` from the
   ground-truth fields via `generate()`, then `embed()` it, and materialize a full `Profile`-shape
   row (not yet inserted into the live DB — held as a fixture object).
3. `eval/gen_synthetic_jobs.py` — for each persona, generate 20 synthetic postings via
   `generate()`: 5 per grade tier (0–3), where the grade is **defined by construction** (the
   prompt for a grade-3 job explicitly targets the persona's exact stack/seniority/domain; a
   grade-0 job is deliberately off-stack/off-seniority) — never LLM-judged after the fact, which
   is what keeps this immune to the labeling-circularity risk ASSESSMENT.md flags for
   `--auto-prelabel`.
4. `eval/inject_synthetic.py` — upserts personas + synthetic jobs into the **live** DB, every job
   tagged `source="synthetic"` so it's identifiable and excludable from production ranking if
   ever needed. Idempotent (safe to re-run). `--teardown` flag deletes everything tagged
   `synthetic` for cleanup.
5. `just eval-synthetic` (new justfile target) — runs `eval/run.py`-equivalent logic per persona
   (5 separate query topics now, not 1), reports nDCG@10 + BPref **per persona**, not just
   averaged — stratification matters because domain heterogeneity across personas can hide a
   regression that only hits one persona's slice.

**Files:** `eval/personas/*.json` (new), `eval/gen_personas.py` (new), `eval/gen_synthetic_jobs.py`
(new), `eval/inject_synthetic.py` (new), `eval/run_synthetic.py` or an extension of `eval/run.py`
(new), `justfile` (new target).

**Acceptance:** `just eval-synthetic` produces 5 independent per-persona nDCG@10/BPref numbers
from one run. `ranx.compare()`'s significance test in `eval/run.py`/`sweep.py` — currently
suppressed because n=1 — activates and produces a real result once qrels span ≥2 topics (5, once
this lands). Any future RRF-weight or field-boost tuning is re-validated against this multi-topic
set before being called "confirmed," not just the single existing profile.

---

## B2 — Fit-score calibration eval

**Problem:** `score_fit`'s weights (`0.50/0.15/0.25/0.10`) and verdict bands (`80/60/40`) were
never checked against "would I apply" judgment — only against retrieval relevance, a different
signal M6's own docs explicitly distinguish.

**Approach:**
1. Extend `EvalLabel` with a `label_type` column (default `"relevance"` for all existing rows via
   migration — no behavior change to anything reading the table today), and a second value
   `"apply_intent"` for the new signal.
2. New CLI, `eval/label_intent.py` → `job-radar-eval-label-intent`: shows a sample of jobs
   already scored by `score_fit` (score + verdict visible) and asks a direct yes/no/maybe "would
   you actually apply to this" question — deliberately *after* seeing the system's own
   score/verdict is fine here (unlike relevance labeling, where blind labeling avoids anchoring
   bias) because the goal is specifically to check whether the *verdict bands* correspond to real
   decisions, not to build an independent ranking ground truth.
3. New analysis script, `eval/calibrate_fit.py`: joins `apply_intent` labels against `score_fit`
   outputs, reports (a) a confusion-style breakdown — for each verdict band (strong/moderate/
   weak/none), what fraction of jobs in that band got a "yes"; (b) whether bands are monotonic
   (does "strong" have a higher yes-rate than "moderate," etc. — if not, the bands are
   miscalibrated, not just imprecise); (c) a suggested band adjustment if a clear
   miscalibration shows up (e.g., "moderate" behaving like "weak").
4. Constants (`_WEIGHTS`, `_BANDS` in `fit/score.py`) are adjusted only after this report exists
   and shows a specific, reasoned miscalibration — not spot-guessed.

**Files:** new Alembic migration (add `label_type` to `eval_labels`), `eval/label_intent.py`
(new), `eval/calibrate_fit.py` (new), `src/job_radar/db/models.py` (`EvalLabel.label_type`).

**Human time note:** unlike the other items in this plan, this one is bottlenecked on labeling
time, not engineering time — budget for actually sitting down and judging 30-50 jobs honestly
("would I apply") once the tooling exists, the same way relevance labeling already required.

**Acceptance:** `eval/calibrate_fit.py` produces a report with a clear per-band yes-rate; bands
are either confirmed monotonic-and-reasonable or a specific adjustment is proposed with the data
behind it.

---

## B3 — CV-parsing quality eval

**Problem:** `parse_cv`'s output (`target_titles`, `tech_stack`, `domains`, `seniority`) is the
most upstream, most-trusted, least-checked artifact in the system — nothing validates it.

**Approach — deliberately reuses B1's fixtures instead of authoring new ones:**
1. B1's 5 synthetic personas already pair a `cv_text` with hand-authored ground-truth fields
   (`target_titles`, `tech_stack`, `domains`, `seniority`) — that's exactly the (input, expected
   output) pair a parsing eval needs, and they're already non-PII by construction.
2. New script, `eval/eval_profile_parsing.py`: for each persona, run `parse_cv(persona.cv_text)`,
   compare the result against the persona's ground-truth fields:
   - `seniority` — exact match rate (it's a 6-value categorical).
   - `tech_stack` / `domains` — set recall (did the expected items get extracted) and precision
     (did anything ungrounded get invented, given the prompt explicitly forbids that) against
     the persona's declared ground truth.
   - `target_titles` — loose match (substring/synonym-tolerant), since phrasing legitimately
     varies.
3. Report per-field accuracy across the 5 personas. Small enough that it doesn't need TREC-style
   machinery — this is a straightforward extraction-accuracy check, not a ranking problem.

**Sequencing note:** if B1 hasn't landed yet, this can still run standalone against 3-5
hand-picked reference CVs (synthetic, non-PII, authored the same way B1's personas are) — but
building B1 first and reusing its fixtures avoids authoring two separate synthetic-CV fixture
sets for what's fundamentally the same need.

**Files:** `eval/eval_profile_parsing.py` (new); depends on `eval/personas/*.json` from B1.

**Acceptance:** a report showing per-field accuracy across all 5 personas; any field scoring
poorly (e.g., `domains` recall <70%) is a concrete, evidenced argument for revising
`profile/parse.py`'s prompt, not a guess.

---

## Out of scope for this plan

- **Provider abstraction** (paid-API swap capability) — ASSESSMENT.md flags it, but
  RECOMMENDATION.md's Phase E is explicit: don't build it until the local-vs-API comparison
  actually needs to run. Building an unused abstraction now is exactly the kind of speculative
  work this project's own CLAUDE.md argues against.
- **Observability (Langfuse/OTel)** — Phase C in RECOMMENDATION.md, comes after this plan, not
  part of it. Tracing a system whose measurement gaps haven't been fixed yet just gives clean
  visibility into numbers that still can't be trusted.
- **LangGraph graph / agent work** — Phase D, explicitly sequenced after all of the above.
