# Job Radar — Honest Component Assessment

> This is a critical read of the system as it actually runs, not as any doc says it should run.
> Every claim below was checked against the code directly (see [COMPONENTS.md](COMPONENTS.md)
> for the descriptive pass this builds on) — several were checked *again* here specifically to
> find out whether they hold up under pressure, including three I hadn't verified before writing
> this: whether the CI eval gate actually executes, whether all 6 ingest adapters handle "remote"
> consistently, and what `eval/metrics.py`'s `bpref` actually computes. Verdicts are blunt on
> purpose — that's what was asked for.

---

## The headline finding

**The CI regression gate this project's whole reliability story rests on does not currently run.**
`tests/test_eval_gate.py::test_ndcg_meets_golden_threshold` is the mechanism `docs/EVAL.md` and
`DESIGN.md` §12 both describe as the safety net: *"a meaningful drop in nDCG before a merge is
caught here, not in production."* The golden files (`eval/golden/qrels.json`, `result.json`) are
committed. The `skipif` that guards them (`_golden_ready()`) passes. But the test body then does
`profile = (await session.execute(select(Profile).where(Profile.id == profile_id))).scalars().first()`
— and CI's Postgres service (`.github/workflows/ci.yml`) starts empty, runs migrations, and
never seeds a `Profile` row. `profile is None` → `pytest.skip(...)` → the test never actually
executes its assertion. Every commit, including this one, has been merging with this gate
silently green-skipping, not green-passing. This is exactly the gap the project's own thesis
(reliability, not retrieval, is the centerpiece) exists to catch, and it slipped past because a
skip and a pass render identically in a CI summary unless you read the log.

This isn't a criticism of the *design* of the gate — the mechanism is right. It's a fixture gap:
nothing seeds the golden profile/job rows CI needs. Cheap to fix, high-leverage to fix, and it
should happen before anything else in this document.

---

## Foundations (config, persistence)

**Verdict: sound.**

- pydantic-settings + `.env` is the right amount of ceremony for this scope.
- SQLAlchemy 2.0 async + Postgres/pgvector/ParadeDB avoids standing up a second search stack
  (Elasticsearch/OpenSearch) for a single-user system — the right call given the scale, though
  it's a real bet on a comparatively young extension (ParadeDB) with a smaller community than
  the alternatives; worth knowing that's a deliberate operational-risk tradeoff, not a free win.
- The five-column `FitJudgmentCache` unique key (`profile_id, job_id, content_hash, model,
  prompt_version`) is genuinely well-designed — it's the rare cache whose invalidation policy is
  fully legible from its schema alone.

**Nothing here needs rethinking.** The one thing worth naming: `Profile` is fetched everywhere
with `.scalars().first()` — there is no `user_id`, it's a hardcoded singleton. Correct for the
current single-user scope, but it means "one candidate" is load-bearing in the schema itself, not
just a config value — extending to multi-user later is a real migration, not a flag flip.

---

## Generation & embedding adapters

**Verdict: the pattern is right; the implementation has two real gaps.**

**What's right:** funneling every LLM/embedding call through exactly two functions
(`generate()`, `embed()`) is a genuinely disciplined seam — I did not find a single call site
that bypasses it. Schema-constrained output everywhere, truncation detected explicitly rather
than surfacing as a confusing parse error, PII kept out of logs. This is better hygiene than most
production codebases I'd compare it to, not just personal projects.

**What's not:**

1. **No retry/backoff at the adapter level.** `generate()` and `embed()` do a single
   `httpx` call and propagate any failure immediately. Callers catch per-item (ingest, fit) so
   one bad job doesn't kill a batch, but a transient Ollama hiccup under the ~12-20-way
   concurrency this system runs at is *lost work*, not *retried work*, every time. For a system
   whose stated thesis is reliability, the most basic reliability primitive — retry a transient
   failure — isn't there yet.
2. **There is no abstraction for the paid-API swap `DESIGN.md`'s own tech stack table promises**
   (*"Local (Qwen/Llama) for dev; paid API for final quality pass"*). `generate()`'s payload is
   Ollama's `/api/chat` shape, hardcoded — `{"model", "messages", "stream", "format", "options"}`.
   There is zero code today that could point at a different provider without rewriting this
   function. That's fine as a "haven't gotten there yet," but it means the local-vs-API cost/
   quality comparison `DESIGN.md` §13 calls a "reported deliverable" currently has no path to
   existing at all — not started, not stubbed, not abstracted for.

---

## Source adapters & ingestion

**Verdict: sound engineering, two real correctness edge cases worth knowing about.**

Checked all 6 adapters directly this pass (previously I'd only read 2). All handle "remote"
consistently and deliberately — each either filters to remote-only at `fetch()` time
(Greenhouse, GetOnBoard) or asserts it structurally because the source itself is remote-only
(Himalayas, Lever's `workplaceType=="remote"` filter) or passes through the source's own flag
(Remotive, Arbeitnow). No inconsistency there. Per-source salary normalization is genuinely
careful — Himalayas drops non-annual figures rather than mixing periods, GetOnBoard correctly
annualizes monthly-USD, Lever keeps only `per-year-salary` — all real "don't fabricate a
comparable number from an incomparable one" discipline.

**Two things worth being honest about:**

1. **Dedup is both too tight and too loose, in different dimensions.** `content_hash` is
   `sha256(company|title|location)`, normalized and lowercased. Too tight: a company that
   genuinely reopens the same title in the same location for a different team collapses into one
   row. Too loose: the same role listed on an aggregator as `"Remote"` and on the company's own
   Greenhouse board as `"Remote - US"` will **not** collapse, because the location strings differ
   — meaning cross-source dedup, the exact thing `content_hash` exists for, misses a real class
   of duplicates it was built to catch. Neither is catastrophic, but both are real and neither is
   currently measured (quality/ reports nothing about duplicate rate).
2. **Extraction failures degrade lexical search silently.** When `ingest/extract.py` fails,
   `requirements`/`responsibilities` land as `NULL`, and BM25 only indexes those two fields plus
   `title` — so a job with failed extraction is nearly invisible to keyword search (title-only
   signal) even though the embedding still falls back to the full description and stays
   vector-searchable. This is a reasonable degradation, not a crash — but `quality/metrics.py`
   doesn't report the null-extraction rate, so nobody would notice if this were happening to,
   say, 15% of one source's postings.
3. **Board-token discovery depends on an external, community-maintained GitHub JSON file**
   (`outscal/OpenJobs`). The fallback-to-cache behavior on failure is correct engineering, but if
   that upstream repo goes stale or disappears, the fallback cache goes stale right alongside it,
   silently, with only a log warning to notice by.

None of this needs a rewrite. It needs the failure modes to show up in `quality/metrics.py`,
which already has the right shape to carry them.

---

## Hybrid retrieval (BM25 + HyDE + RRF)

**Verdict: the architecture is sound and genuinely well-reasoned. The *tuning* built on top of
it is not validated to the standard the eval harness's own rigor implies.**

**What's right, unambiguously:** BM25-via-ParadeDB over hand-rolled scoring or staying on
`ts_rank` was the correct call, for the reasons the project's own docs already argue well.
HyDE for the dense arm is a legitimate, literature-grounded technique for exactly this
query/document asymmetry problem, and averaging 3 samples to cut single-draw variance is a nice
touch. Moving seniority and geo eligibility to deterministic code instead of LLM judgment — with
two independently-maintained but *provably mirrored* implementations (SQL `~*`/`\y` for the
retrieval prefilter, Python `re`/`\b` for scoring) — is exactly the right instinct for a
reliability-focused system.

**What undermines confidence in the tuning specifically:**

1. **Every tuning conclusion in this codebase is drawn from one query topic.** One profile,
   fetched via `.scalars().first()` throughout. The RRF weight `[2.0, 1.0]` hardcoded into
   production, the BM25 field boosts (`5/3/1`), the choice of `k=60`, the decision to drop the CV
   arm because it *"consistently degrades all metrics"* — every one of these is a conclusion
   drawn from a sweep over one person's labeled job search. The eval code is honest about this
   in one place: `ranx.compare()`'s significance test is explicitly suppressed because *"with a
   single profile → single topic (n=1), the test is undefined."* That line is correct and it
   should be read as undermining every *other* tuning claim in the codebase that doesn't carry
   the same caveat, not just the one function it's attached to. "Consistently degrades" is not a
   claim n=1 can support, regardless of how the ablation came out.
2. **HyDE — the single most expensive step in a production query (3 LLM generations) — is not
   cached, despite the schema already half-building the mechanism to cache it.**
   `Profile.dense_query_cache` exists, and `profile/loader.py` correctly invalidates it (sets it
   to `None`) on every CV reload — exactly the right trigger. But production `fit/pipeline.py`
   never reads it; it regenerates 3 fresh HyDE postings on *every single run* regardless of
   whether the profile changed since the last one. The only place that column is read at all is
   `tests/test_eval_gate.py`, to avoid a live LLM call in CI. This is a genuinely strange state —
   the caching instinct that produced `FitJudgmentCache` and the ingest embedding cache stopped
   one component short of the most expensive call in the whole retrieval path.

**Bottom line:** don't distrust the architecture. Distrust any specific number that came out of
tuning it, until it's validated on more than one topic.

---

## Fit analysis

**Verdict: the best-designed component in the codebase, sitting on top of one unvalidated
assumption.**

**What's right:** separating grounded LLM classification from deterministic scoring
(`score.py` is pure arithmetic, zero LLM involvement) is exactly the discipline this kind of
system needs, and it's applied consistently — non-compensatory knockout gates for region/
seniority computed in code against structured data, never asked of the model; evidence-before-
classification field ordering with a measured (not assumed) justification; caching that
correctly excludes the derived score specifically *because* it depends on runtime overrides the
judgment doesn't. I don't have a real complaint about the mechanics here.

**What's unvalidated:** the scoring weights (`required 0.50, preferred 0.15, seniority 0.25,
domain 0.10`) and verdict bands (`≥80 strong / ≥60 moderate / ≥40 weak`) are design-time
constants. The code comment says they're *"calibrated against human labels in M6"* — but M6's
labels grade **retrieval relevance** ("should this appear in results"), which `docs/plans/
M6_EVAL_IMPLEMENTATION_PLAN.md` itself explicitly and correctly distinguishes from **application
intent** ("would I apply") as a *different signal*. Nothing in the current eval harness measures
whether a job `score_fit` calls "strong" (≥80) is actually a job a human would call a strong fit.
The weights and bands are reasonable-*looking* numbers that have never been checked against the
judgment they claim to model.

**Smaller, real gap:** the CLI can't distinguish "skipped, insufficient input" from "attempted,
LLM call failed" — both render as `score = "—"`, `verdict = "none"`. Not a design flaw, just a
missing bit of surfaced information that would matter the first time someone's debugging why a
specific job scored nothing.

---

## Data quality module

**Verdict: sound, useful, and the right lightweight shape for what it is — but it's watching
the wrong things given what actually breaks upstream.**

The centroid-similarity relevance check (embed 10 anchor phrases, compare in-Postgres via
pgvector, report the raw number rather than a hard pass/fail) is a well-judged "cheap diagnostic,
not a gate" design. But it doesn't report the two failure modes that would most directly explain
a bad search result: extraction-null rate (feeding the retrieval-arm silent-degradation issue
above) and cross-source duplicate rate. Both are cheap to add to `metrics.py`'s existing shape —
this module doesn't need new capability, it needs to look at two more columns it already has
access to.

---

## Evaluation harness

**Verdict: the most sophisticated component in the codebase by a wide margin, and also the one
most likely to be giving false confidence, for a structural reason rather than a bug.**

The metric implementations (`ndcg`, `bpref`, `mrr`, precision/recall@k) are correct, cited, and
cross-validated against `ranx` — I checked `bpref`'s implementation against Buckley & Voorhees
2004's formula and it matches. The OAT sweep design, the union-pool labeling to reduce pool bias,
the golden-config CI gate concept — all TREC-grade thinking, genuinely more rigorous than most
retrieval work I'd expect to see outside a research team.

**But the entire apparatus is currently validating tuning decisions against a sample size of
one query topic**, which is the retrieval section's finding restated at the level it actually
matters: this isn't a retrieval-tuning footnote, it's an evaluation-methodology problem that
makes every number the harness has produced so far a description of one person's search, not a
generalizable finding. `docs/plans/SYNTHETIC_EVAL_DESIGN.md` (5 synthetic personas + InPars-style
query generation) is the correct fix for exactly this — it's designed, cited, and not built.

**One more structural risk worth naming:** `eval/label.py --auto-prelabel` and `--fully-auto`
seed or directly assign relevance grades using `analyze_fit` — the same fit-analysis system whose
retrieval arm shares HyDE generation with the thing being measured. The docs correctly warn LLM
pre-labels are wrong 15-25% of the time on marginal cases and say to always review — but that
safeguard is a documented convention, not something the code enforces. `--fully-auto` exists as a
flag that produces a "labeled" set with zero human judgment in it, and nothing stops that set
from becoming the golden qrels. Worth being deliberate that this flag is for pre-labeling speed,
never for producing ground truth used as a regression baseline.

---

## Profile / CV analyzer

**Verdict: reasonable mechanics, unmeasured accuracy — and it's the most upstream artifact in
the entire system.**

The pdfplumber choice with `x_tolerance=1` is a real, specific, tested fix (glyph-merging), not a
default left alone. The PII discipline in the schema itself (`full_name`/`email` flagged
PII-local-only) and in the loader's logging (sizes/counts only) is consistent with the rest of
the codebase.

**What's missing:** every other measurable component in this system has an eval harness — fit
has (partial) calibration intent, retrieval has an extensive one. CV parsing has none. And it's
the most upstream: `target_titles`, `tech_stack`, `domains`, and `seniority` all flow from one
LLM call's output into the lexical query, the HyDE prompt, the fit prompt, and the domain-
relevance scoring dimension. If `parse_cv` under- or mis-extracts a skill, everything downstream
degrades silently and consistently in the same direction — and nothing in the system would show
you that the root cause was profile parsing rather than retrieval or fit. There's no eval, no
even a documented manual-spot-check habit, for the single call whose output every other component
trusts unconditionally.

---

## CI / testing

**Verdict: real gap beyond the headline finding.** Legitimately good elsewhere — ruff/format/
pytest/gitleaks all run on every push, migrations run against a real ParadeDB container (not
sqlite or a mock), `test_embeddings.py` mocks `httpx` properly rather than skipping. But the one
test built specifically to be this project's regression safety net is currently inert (see
top of this document), and that's the kind of gap that's invisible until you go looking for it —
which is itself worth internalizing: a green CI badge on this repo does not currently mean what
`docs/EVAL.md` says it means.

---

## Summary table

| Component | Architecture/premises | Current validation | Rethink or fix? |
|---|---|---|---|
| Foundations | Sound | N/A | No |
| Generation/embedding adapters | Sound pattern | N/A | Fix: retry/backoff, provider abstraction |
| Source adapters/ingestion | Sound | Partial | Fix: dedup edge cases, extraction-failure visibility |
| Retrieval architecture | Sound | — | No |
| Retrieval tuning (weights/boosts/k/CV-arm removal) | — | **Unvalidated (n=1)** | Fix before trusting further |
| Fit analysis mechanics | Sound | — | No |
| Fit score weights/bands | — | **Unvalidated** | Needs its own calibration eval |
| Data quality | Sound, incomplete coverage | — | Extend, don't rewrite |
| Eval harness engineering | Sound, sophisticated | — | No |
| Eval harness evidentiary base | — | **n=1, structurally** | Needs synthetic multi-persona data |
| Profile/CV parsing | Reasonable | **None** | Needs any eval at all |
| CI eval regression gate | Sound design | **Currently inert** | Fix immediately — cheap, high-leverage |
