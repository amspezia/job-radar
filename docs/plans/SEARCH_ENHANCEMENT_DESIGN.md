# Search Engine Enhancement — Design & Plan

> Status: **partially implemented.** Assesses the *current* hybrid-search retrieval (M0–M2),
> identifies where it under-uses the profile, and lays out a phased enhancement.
> **E1, E2, E5, E6 are now built** (CV-embedding RRF arm, profile-enriched query,
> preference filters, edge guards). **E3 (FTS `'english'` migration) and E4 (nomic prefixes)
> are deferred.** Agreed next step before any new milestone: a **thorough eval + parameter
> tuning** pass (see §7 E7) to measure these changes and tune fusion before continuing.

---

## 0. Where this sits

M0–M2 built the machine that **finds** jobs: ingest → embed → store (Postgres + pgvector) →
**hybrid search** (FTS + vector + RRF). M3 added the layer that **judges** a job against the
profile. This document is about making the **find** step actually use the candidate we
already know about — today retrieval is driven by a 2-word title string while a rich CV
representation sits unused.

Scope: `retrieval/` (`search`, `fts`, `vector`, `fusion`), the query construction in
`fit/pipeline.py`, the `adapters/embeddings.py` client, and the `embed()` call sites at
ingest (`ingest/pipeline.py`) and CV load (`profile/loader.py`). Out of scope: fit scoring
(covered by M3 docs), ingestion sources, the geo filter (already reworked).

---

## 1. Current state (what actually happens)

Call chain: `job-radar-fit [query]` → `run_fit_pipeline` → `search` → FTS arm + vector arm
→ `reciprocal_rank_fusion` → hydrate `Job` rows.

```
query = caller arg  OR  " ".join(profile.target_titles)      # e.g. "Backend Engineer"
geo   = build_geo_filter(profile.location_rules.allowed_keywords)

fts_results    = websearch_to_tsquery('simple', query) over Job.search_vector   # title^A + desc^B
vector_results = cosine( embed(query) , Job.embedding )                         # query string embedded fresh
fused          = RRF([fts_results, vector_results], k=60, pool=50)              # rank-only fusion
```

### What the profile contributes to retrieval today

| Profile field | Used? | How |
|---|---|---|
| `target_titles` | ✅ **only ranking signal** | the entire query string (FTS text *and* the embedded vector) |
| `location_rules.allowed_keywords` | ✅ | geo prefilter on both arms |
| `tech_stack` | ❌ | never enters the query |
| `domains` | ❌ | unused |
| `work_history` | ❌ | unused |
| `cv_text` | ❌ | unused for retrieval |
| **`cv_embedding`** | ❌ | **computed + stored in `loader.py`, never read anywhere** |
| `seniority` | ❌ | unused |
| `salary_floor` / `currency` | ❌ | unused (no salary filter) |
| `remote_required` | ❌ | unused (no remote filter) |

---

## 2. Defects this design addresses

1. **`cv_embedding` is dead data; the vector arm embeds a 2-word string.** The loader builds
   a 768-dim embedding of the whole CV and stores it — then retrieval re-embeds
   `"Backend Engineer"` instead. A 2-word vector is a near-centroid: almost every backend
   posting is equidistant, so the semantic arm has little discriminative power. **Highest
   leverage, lowest risk to fix** (the vector already exists, no re-embedding needed).

2. **Retrieval is blind to stack, domain, and seniority.** Because the query is *only*
   `target_titles`, two candidates with identical titles but opposite stacks (Python vs
   Java) retrieve the *same* jobs. None of `tech_stack` / `domains` / `seniority` text ever
   enters the query; that signal only re-appears later at fit-scoring.

3. **FTS narrows recall.** `websearch_to_tsquery` is **implicit-AND**, so
   `"Backend Engineer"` requires *both* lexemes — a *"Backend Developer"* posting that never
   says "engineer" drops out of the FTS arm. `'simple'` config does **no stemming**
   (`engineer` ≠ `engineering`, `developer` ≠ `developers`). Multi-title queries get *worse*
   here, not better (more AND-ed terms → fewer matches).

4. **No nomic task prefixes.** `nomic-embed-text` is trained for **asymmetric** retrieval
   with `search_query:` / `search_document:` prefixes. Every `embed()` call (jobs at ingest,
   the query, the CV) passes raw text. It is at least *consistent* (nothing prefixed), so
   it's a quality opportunity rather than a bug — but it leaves measurable accuracy on the
   table.

5. **Missing guards / unused preferences.** Empty `target_titles` → query `""` →
   `websearch_to_tsquery('simple','')` matches nothing and `embed("")` is degenerate, with
   no guard. `remote_required` and `salary_floor` are never applied as filters even when set.

---

## 3. Enhancement design (lever by lever)

### E1 — Use `cv_embedding` for the vector arm  ⭐ headline
**What:** rank `Job.embedding` against the stored `profile.cv_embedding` instead of
`embed(target_titles)`.
**Why:** the CV vector encodes the actual candidate (stack, domains, history); it's already
computed and free to use.
**Choices:**
- (a) **Replace** the query-string vector with `cv_embedding`.
- (b) **Dual vector arms**: keep a title/query vector *and* add a CV vector as a third RRF
  ranking — lets a caller's ad-hoc query still steer results while the CV anchors relevance.
- (c) Blend (average) the two vectors into one — *rejected*: averaging dilutes both signals
  and is harder to reason about than RRF over separate arms.
**Recommendation:** **(b)** — RRF already fuses arbitrary arms cleanly. When the caller
passes an explicit query, both the query-vector and FTS arms reflect it; the CV-vector arm
keeps results grounded in the candidate. When there's no query, the CV-vector arm carries
ranking. Falls back gracefully when `cv_embedding` is NULL (drop that arm).

### E2 — Profile-enriched query text
**What:** when no explicit caller query, build the FTS/query text from
`target_titles + tech_stack + domains` (and optionally salient `work_history` terms) rather
than titles alone.
**Why:** makes the lexical arm candidate-aware so a Python role outranks a Java one.
**Choice:** keep this server-side in `default_query`/a new `build_query()` helper; don't make
the user type it.
**Recommendation:** do it, but keep title terms weighted/ordered first (they're the strongest
intent signal), and combine with E3's OR semantics so added terms broaden rather than
AND-narrow.

### E3 — FTS recall: stemming + OR semantics
**What:** switch `'simple'` → `'english'` (stemming) and move multi-term/title queries from
implicit-AND to OR.
**Why:** `'english'` generalizes `developer`/`developers`; OR makes "any of these titles"
work (today more titles = fewer hits).
**Choices:**
- Keep `websearch_to_tsquery` and rely on the literal `OR` operator (works today), **or**
- Build the tsquery explicitly (`to_tsquery` with `|` between title groups) for full control.
**Recommendation:** explicit tsquery construction for titles (OR between titles, AND within a
title phrase), `'english'` config. **Caveat:** `search_vector` is a **generated column** using
`'simple'` ([models.py](../../src/job_radar/db/models.py)); changing the *query* config to
`'english'` without changing the *indexed* config breaks matching. So E3 requires a migration
that regenerates `search_vector` (and its GIN index) under `'english'`. Treat as its own
phase.

### E4 — nomic task prefixes (asymmetric retrieval)
**What:** prefix `search_document:` at ingest + CV load, `search_query:` at query time.
**Why:** it's how nomic was trained; improves query↔document matching.
**Cost / caveat:** query and documents must live in the **same** space, so this requires
**re-embedding the entire `jobs` table and the CV** (a backfill). Doing it half-way (prefix
queries but not stored docs) would *degrade* results. **Migration, not a code tweak.**
**Recommendation:** worthwhile but sequence it last, behind a one-shot backfill script, and
verify on the eval set before/after.

### E5 — Structured preference filters
**What:** optionally apply `remote_required` and `salary_floor` as WHERE clauses alongside
the geo filter.
**Why:** symmetry with how location is already enforced; honors preferences that are silently
ignored today.
**Recommendation:** low priority, additive. Gate behind the profile values being set; be
careful with `salary_min`/`salary_max` NULLs (most postings omit salary — filtering on it
would drop almost everything, so default to *not* excluding NULL-salary jobs).

### E6 — Edge guards
**What:** guard empty query text (skip FTS arm / skip embed when query is blank), and the
empty-`cv_embedding` case for E1.
**Recommendation:** cheap correctness; bundle with E1/E2.

### (Future) E7 — Tunable fusion + eval
RRF is rank-only with equal arm weights today. Once there are ≥3 arms (FTS, query-vector,
CV-vector), arm weighting and `k` become real knobs — but they should be tuned against the
**M6 labeled eval set**, not guessed. Out of scope here beyond noting the hook.

---

## 4. Proposed retrieval shape (after E1–E3)

```
query_text  = caller arg OR build_query(profile)        # titles (OR'd) + tech_stack + domains
query_vec   = embed(query_text)
cv_vec      = profile.cv_embedding                       # may be None

arms = [
    search_fts(query_text, 'english', OR-semantics, geo),
    search_vector(query_vec, geo),
    search_vector(cv_vec, geo) if cv_vec is not None else None,
]
fused = RRF([a for a in arms if a], k=60, pool=50)
```

---

## 5. Locked vs. open decisions

**Proposed-locked (pending your nod):**
| # | Decision |
|---|---|
| 1 | E1 ships first; CV vector added as a **separate RRF arm**, not a blend. |
| 2 | Server-side `build_query()` enriches with `tech_stack` + `domains`; user is never required to type them. |
| 3 | FTS recall fix (`'english'` + OR) is gated behind a `search_vector` regeneration migration. |
| 4 | nomic prefixes (E4) ship **last**, behind a full re-embed backfill, verified on eval. |

**Open questions** — see §7.

---

## 6. Phased build (with verification gates)

- **Phase 1 — E1 + E6 (no migration). ✅ done.** `cv_embedding` wired into `search` as a
  separate RRF arm (`retrieval/search.py`); blank-query / NULL-vector guards added. Covered by
  `tests/test_search.py` (CV-arm-only path, blank-query short-circuit).
- **Phase 2 — E2. ✅ done.** `build_query(profile)` merges titles + tech_stack + domains
  (`fit/pipeline.py`); covered by `tests/test_filters.py`.
- **Phase 3 — E3 (migration). ⏸ deferred.** Regenerate `search_vector` + GIN index under
  `'english'` and switch to OR semantics. *Verify:* `"Backend Developer"`-titled postings
  appear for a "Backend Engineer" profile; stemming hits (`developers`).
- **Phase 4 — E5 + filters. ✅ done.** `build_profile_filter` combines geo + remote + salary
  (`retrieval/filters.py`); NULL-salary and other-currency rows are retained. Covered by
  `tests/test_filters.py`.
- **Phase 5 — E4 (migration + backfill). ⏸ deferred.** nomic prefixes + full re-embed.

**Next (agreed before any new milestone): eval + tuning.** Stand up a labeled/golden query
set and retrieval metrics (recall@k, MRR/nDCG) so the E1/E2/E5 changes can be *measured*, and
the fusion knobs (per-arm weights now that there are three arms, RRF `k`, `pool`) tuned
against it rather than guessed — then revisit whether E3/E4 earn their migrations.

---

## 7. Open questions

1. **CV vector: add or replace?** Keep the query-string vector arm *and* add the CV arm (E1b,
   my rec), or replace the query vector with the CV vector outright? Replacing is simpler but
   loses ad-hoc query steering.
2. **Query enrichment breadth:** titles + tech_stack + domains only, or also mine
   `work_history` highlights? More terms = more recall but more drift.
3. **Is the `'english'` migration in appetite now?** It's the only change touching the indexed
   column (regenerate `search_vector` + reindex). If not, we keep `'simple'` and only fix the
   AND→OR semantics (still a real win, no migration).
4. **nomic re-embed cost:** how many jobs are in the table (re-embedding all of them is the
   gate on E4)? Acceptable to run as a one-shot backfill, or defer E4 entirely?
5. **Preference filters & NULLs:** confirm the rule "never exclude a job just because its
   salary is NULL" — i.e. `salary_floor` only filters rows that *have* a salary below it.
6. **Eval coverage:** should Phases 3 and 5 block on the M6 labeled set existing, so we can
   prove the migrations are net-positive rather than eyeballing?
```
