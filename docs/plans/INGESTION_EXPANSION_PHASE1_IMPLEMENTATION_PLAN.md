# Ingestion Expansion, Phase 1 — Implementation Plan

Locks down `INGESTION_EXPANSION_DESIGN.md` §5 Phase 1: three new ATS adapters (Ashby, Workable,
SmartRecruiters) reusing the existing `discovery.py` mechanism, plus the `content_hash`
location-dedup fix. Explicitly **not** in scope: Phase 2–4 (new aggregators, Workday, Adzuna), or
any backfill/cleanup of already-duplicated rows already in the DB (see §3 for why).

Every endpoint shape below was hit live tonight (2026-08-21), including the harder cases the
design doc didn't drill into — Workable and SmartRecruiters both turned out to need one more
request than Greenhouse's pattern, caught here rather than mid-implementation.

## 1. New adapters

All three follow `greenhouse.py`'s exact shape: `discovery.get_tokens()` for live board tokens,
one `fetch()` per token, one `map()` per raw posting. Each is a new, independent file — no shared
adapter base beyond what `SourceAdapter`/`discovery.py` already provide.

### 1.1 `src/job_radar/adapters/sources/ashby.py`

The cleanest of the three — one list call per company, full description included, no N+1.

- `link_regex = re.compile(r"jobs\.ashbyhq\.com/([a-zA-Z0-9_-]+)")`
- `board_url = "https://api.ashbyhq.com/posting-api/job-board/{token}"` — same URL for discovery
  verification and the real fetch (unlike Workable/SmartRecruiters below, there's no lighter vs.
  fuller variant).
- `has_jobs = lambda payload: bool(payload.get("jobs"))`
- `_TOKENS_CACHE = Path("data/ashby_tokens.json")`

Real response shape (`{"jobs": [...], "apiVersion": "1"}`), one job object:
```
id, title, department, team, employmentType, location, secondaryLocations, publishedAt,
isListed, isRemote, workplaceType, address, jobUrl, applyUrl, descriptionHtml, descriptionPlain
```

`map()` field-by-field:

| `NormalizedJob` field | Source | Notes |
|---|---|---|
| `source_id` | `id` | already a string |
| `url` | `jobUrl` | |
| `title` | `title` | |
| `company` | *(not in the job object — pass the board `token` through, title-cased, or fetch it once per-adapter-run and pass as a constructor arg to `map()`; Greenhouse gets `company_name` for free from its payload, Ashby doesn't)* — see open question below |
| `description` | `descriptionPlain` | already plain text, no HTML stripping needed (unlike Greenhouse) — **do not** run `html_to_text` on `descriptionPlain`, only use `descriptionHtml` if plain is ever empty |
| `salary_min/max/currency` | — | not present in this payload; leave `None` like `arbeitnow.py` does |
| `location` | `location` | plain string already, no nested `.name` unwrap needed |
| `job_type` | `employmentType` | e.g. `"FullTime"` — pass through as-is, same as other adapters do with their native vocab |
| `remote` | `isRemote` | **filter `fetch()` to `isRemote is True` only**, mirroring Greenhouse's `_remote_jobs` |
| `published_at` | `publishedAt` | ISO 8601 with offset, `datetime.fromisoformat` works directly like Greenhouse's `first_published` |

**Resolved**: checked three real boards tonight (`hightouch`, `haydenai`, `opensea`) — none
expose a company name field at any level. Confirmed Ashby-specific: Greenhouse/Lever/GetOnBoard
all get a real company name directly from their own payloads (`company_name`, etc.), so this is
not a gap in the shared mechanism, just an Ashby quirk. Decision: **title-case the token**
(`"hightouch"` → `"Hightouch"`) as `company` — lossy for multi-word/stylized names (won't recover
`"OpenSea"`'s internal capitalization, will render as `"Opensea"`), but keeps `discovery.py`'s
shared return shape (`list[str]`) untouched for the three existing adapters that don't need this.
Explicitly a known, accepted cosmetic limitation for Phase 1 — not worth threading real company
names through a shared function three other callers don't need it for. Revisit only if the
lossy name ever actually confuses a user reading fit results, not preemptively.

### 1.2 `src/job_radar/adapters/sources/workable.py`

List endpoint's default response has **no per-job description** — confirmed live. The fix is a
query param, not a second endpoint: `?details=true` on the same list call returns every job with
a `description` field included, so this stays a single call per company, not N+1.

- `link_regex = re.compile(r"(?:apply|jobs)\.workable\.com/([a-zA-Z0-9_-]+)")` — the dataset has
  both `apply.workable.com` (229 companies) and `jobs.workable.com` (22) domains for the same ATS.
- Discovery verification: `board_url = "https://apply.workable.com/api/v1/widget/accounts/{token}"`
  (no `details=true` needed just to confirm the board is alive with jobs — keep the verify request
  cheap).
- Real fetch: a **second, separate URL constant** for the actual data pull —
  `_JOBS_URL = "https://apply.workable.com/api/v1/widget/accounts/{token}?details=true"`. This is
  the first adapter where the verify URL and the fetch URL genuinely differ; `greenhouse.py`'s
  `_BOARD_URL` happens to serve both roles via a `content=true` param passed at call time instead
  of baked into the URL — consider matching that pattern here (`params={"details": "true"}` on the
  `client.get()` call in `fetch()`) rather than a second URL constant, for consistency with
  `greenhouse.py`.
- `has_jobs = lambda payload: bool(payload.get("jobs"))`
- `_TOKENS_CACHE = Path("data/workable_tokens.json")`

Real per-job shape with `details=true`:
```
title, shortcode, code, employment_type, telecommuting, department, url, shortlink,
application_url, published_on, created_at, country, city, state, education, experience,
function, industry, locations, description
```

`map()` field-by-field:

| `NormalizedJob` field | Source | Notes |
|---|---|---|
| `source_id` | `shortcode` | |
| `url` | `url` (the `apply.workable.com/j/{shortcode}` form) | |
| `title` | `title` | |
| `company` | account-level `name` field (present in the same response, sibling to `jobs`, unlike Ashby) | thread the outer `name` through, same call, no extra request |
| `description` | `description` | HTML — run through `html_to_text` like Greenhouse does |
| `salary_min/max/currency` | — | not present; `None` |
| `location` | **`locations` array**, joined — see the resolved analysis below, not the flat `city`/`state`/`country` fields |
| `job_type` | `employment_type` | e.g. `"Full-time"` |
| `remote` | `telecommuting` | **filter `fetch()` to `telecommuting is True` only** |
| `published_at` | `published_on` | **date only, no time** (`"2026-05-05"`, not a full ISO timestamp) — `datetime.fromisoformat` still parses a bare date string fine in 3.12, but confirm the resulting `datetime` is naive (no tzinfo) and check whether `Job.published_at`'s column/comparisons elsewhere assume tz-aware values (`retrieval/filters.py`'s `coalesce(...) >= cutoff` compares against `datetime.now(UTC)` — a naive `published_at` would raise or compare incorrectly against an aware cutoff). If so, attach `tzinfo=UTC` explicitly rather than trusting `fromisoformat` on a bare date. |

**Resolved — which location representation to use.** Pulled all 40 Workable tokens in the
dataset (38 live, 2 dead 404s that `discovery.py`'s verify step already filters out) and analyzed
**416 real jobs**:

| Field | Populated |
|---|---|
| flat `country` | 416/416 (100%) |
| flat `city` | 349/416 (83.9%) |
| flat `state` | 350/416 (84.1%) |
| `locations` array | 416/416 (100%), never empty |

Flat fields and `locations[0]` **agreed in 416/416 cases — zero disagreements**, so the risk of
picking the "wrong" one is nil for single-location postings. The deciding factor is the 2 jobs
(0.5%) with **multiple** entries in `locations`, both of them remote multi-country postings:

```
zipdev | Jr Fullstack Developer   flat country='Mexico'  locations=[Mexico, Costa Rica, El Salvador, Nicaragua]
zipdev | Senior SRE               flat country='Brazil'  locations=[Brazil, Honduras, Uruguay]
```

The flat fields silently truncate to the *first* country; `locations` has the real eligible set.
**Decision: read `locations`**, joining each entry's `city`/`region`/`country` into one string.
Worth noting this is the same "one role, many eligible countries" shape that causes the §3 dedup
problem — except Workable models it correctly as one posting with N locations rather than N
duplicate rows, and reading `locations` preserves that instead of throwing the extra countries
away.

Incidental yield datapoint from the same pull: **82/416 (19.7%)** of Workable postings are
`telecommuting=True`, i.e. survive the remote filter.

### 1.3 `src/job_radar/adapters/sources/smartrecruiters.py`

The heaviest of the three: the list endpoint (`/postings`) never includes the job ad body under
any query param tested (`?fields=jobAd` does not work) — the only way to get a description is one
more GET per posting, to `/postings/{id}` (its own URL is handed to you as the `ref` field on each
list item, so no URL-building needed). This is the one adapter in Phase 1 with an N+1 fetch
pattern; every other adapter (existing and new) fetches a whole company's board in one call.

- `link_regex = re.compile(r"(?:jobs|careers)\.smartrecruiters\.com/([a-zA-Z0-9_-]+)")` — dataset
  has both `jobs.smartrecruiters.com` (168) and `careers.smartrecruiters.com` (136) domains.
- `_LIST_URL = "https://api.smartrecruiters.com/v1/companies/{token}/postings"` — used both for
  discovery verification and as `fetch()`'s first call per company. Paginated
  (`offset`/`limit`/`totalFound`), same shape as Greenhouse's implicit assumption that one call
  gets everything — **this one doesn't**; page through it (see open question below on limit
  size/max pages).
- `has_jobs = lambda payload: bool(payload.get("content"))`
- `_TOKENS_CACHE = Path("data/smartrecruiters_tokens.json")`

**Resolved — pagination parameters, measured across all 108 tokens in the dataset.**

| Parameter | Measured behavior |
|---|---|
| default `limit` | **100** (echoed back on every board) |
| max `limit` | **hard-capped at 100 server-side** — requesting 200/500/1000 all return exactly 100. Do not bother sending a larger value. |
| `offset` | works cleanly to the end; on the 4784-posting board, `offset=4700` → 84 rows, `offset=4800` → 0 rows. Page until `content` is empty or `offset >= totalFound`. |
| server-side remote filter | **does not exist.** `?remote=true` and `?location.remote=true` are silently ignored (still return the full 4784). `?q=remote` is a free-text search (returns 53) — unreliable as a remote filter, do not use it. |

**The N+1 cost is much smaller than feared, because `location.remote` is present on 100% of
*list* items** — so the remote filter runs before any detail fetch, not after. Measured by paging
every live board:

```
94 live boards (14 of 108 have zero postings)
15,669 total postings
 1,623 remote (10.4%)  ← the only ones needing a detail call
   234 list calls + 1,623 detail calls = 1,857 HTTP requests
        (vs 15,903 if detail-fetching everything — an 8.6x reduction)
```

Detail-fetch latency, measured: **1.04s/call sequential**, but **0.119s/call effective at 10
concurrent** — so all 1,623 details is roughly **3 minutes** of HTTP, not hours. Sequential
fetching would be ~28 minutes, so this adapter should fetch details concurrently (bounded by a
semaphore, the same shape `ingest/pipeline.py` and `discovery.py` already use — `discovery.py`
uses `_CONCURRENCY = 15`).

**Decision on a `_MAX_PAGES` cap: do not add one.** The remote postings are scattered throughout
big boards rather than concentrated on the first pages (BoschGroup has 4,784 postings but only 72
remote, spread across all 48 pages), so a low page cap would silently drop most of the remote
yield from exactly the largest employers, while saving only a few hundred cheap list calls. The
`location.remote` filter is the real bound, and it is already an 8.6x cut. Page fully.

`fetch()` shape, concretely:
```
for token in tokens:
    postings = <page _LIST_URL by offset, limit=100, until exhausted>
    remote   = [p for p in postings if (p.get("location") or {}).get("remote")]   # filter FIRST
    details  = <concurrently GET p["ref"] for p in remote, semaphore-bounded>
    jobs.extend(<merge each list item with its detail response>)
```

List-level fields available without the detail call: `id, name, uuid, jobAdId, defaultJobAd,
refNumber, company, releasedDate, location (includes an explicit remote: bool!), industry,
department, function, typeOfEmployment, experienceLevel, customField, visibility, ref, language`.

Detail-level fields (from `/postings/{id}`), additional to the above: `jobId, postingUrl,
applyUrl, jobAd.sections.{companyDescription,jobDescription,qualifications,
additionalInformation}.text` (each an HTML string, some possibly empty), `active`.

`map()` field-by-field (operates on the **merged** list+detail dict — decide in code whether
`fetch()` merges them into one dict before calling `map()`, or `map()` takes both and merges
itself; the former keeps `map()`'s signature consistent with every other adapter's
`map(self, raw: dict)`):

| `NormalizedJob` field | Source | Notes |
|---|---|---|
| `source_id` | `id` | |
| `url` | `postingUrl` | present only in the detail response |
| `title` | `name` | |
| `company` | `company.name` | present at list level already |
| `description` | join `jobAd.sections.jobDescription.text` + `qualifications.text` (skip `companyDescription`/`additionalInformation` — boilerplate, matches this codebase's stated preference for structured requirement-bearing text over company copy) through `html_to_text` | |
| `salary_min/max/currency` | — | not present in any field seen; `None` |
| `location` | `location.fullLocation` (already a formatted string, e.g. `"Kolkata, WB, India"`) | |
| `job_type` | `typeOfEmployment.label` | e.g. `"Full-time"` |
| `remote` | `location.remote` | **explicit boolean, unlike Ashby/Workable's flatter fields** — filter `fetch()` to `location.get("remote") is True` |
| `published_at` | `releasedDate` | full ISO timestamp with `Z` suffix — `datetime.fromisoformat` needs the `Z` normalized to `+00:00` first in pre-3.11 style parsers, but Python 3.12's `fromisoformat` accepts `Z` directly; confirm with a real value before assuming |

**Detail-payload completeness, verified on 40 real remote postings**: 0/40 were missing *both*
`jobDescription` and `qualifications` (so the §description join never produces an empty string in
practice); `jobDescription` text length ran min 606 / median 2858 / max 6752 chars. `postingUrl`
and `releasedDate` were present on 40/40. One extra section name shows up beyond the four the
design doc listed — `videos` — so treat `jobAd.sections` as an open-ended dict and read the two
keys wanted by name rather than assuming a fixed set.

**Volume note worth flagging before this ships**: 1,623 new remote postings is a ~14% increase on
the entire current 11,441-row corpus, all arriving in one ingestion run, each needing an LLM
`extract_fields` call plus an embed. At the ~0.6-0.7s/posting warm rate measured earlier (bounded
by `_MAX_CONCURRENT_INGEST = 20`), expect the first `job-radar-ingest` run after this adapter
lands to take substantially longer than usual. This is the same "silently slow, looks frozen"
shape already noted in `docs/STATUS.md` §4 — worth having the per-source progress logging in
place before running it, or at minimum expecting the wait.

### 1.4 Registration

Add all three to `src/job_radar/ingest/runner.py::ENABLED_ADAPTERS`, alongside the existing six.
No changes needed to `run_ingestion`/`run_all_ingestion` — per-source failure isolation is already
in place there.

## 2. `discovery.py` — no changes needed

Confirmed: `discovery.py::get_tokens()`/`_discover()`/`_verify()` are already fully generic over
`link_regex`/`board_url`/`has_jobs`/`cache_path` — every one of the three new adapters plugs into
the exact same functions Greenhouse/Lever/GetOnBoard already use, with zero modification to
`discovery.py` itself. `tests/test_discovery.py`'s existing coverage (token parsing, caching,
fallback-on-failure) already exercises this generically and needs no changes either — it's tested
against fake regexes/predicates, not real ATS specifics.

## 3. The `content_hash` dedup fix

### 3.1 The code change

`src/job_radar/ingest/dedup.py`:

```python
def content_hash(job: NormalizedJob) -> str:
    """Stable identity hash for cross-source dedup: company + title (+ location
    for non-remote postings only). Remote-only sources that repost the identical
    role once per eligible country (confirmed: Mindrift, CapsLock, Bjak,
    Bluelight Consulting via Himalayas, up to 27 rows for one real posting) would
    otherwise never dedupe against each other, since each country string is a
    different "location". A non-remote posting's location is still part of its
    identity — "Software Engineer @ Google, NYC" and "...@ Google, London" are
    genuinely different reqs.
    """
    fields = [job.company, job.title]
    if not job.remote:
        fields.append(job.location or "none")
    normalized = [" ".join(s.split()).lower() for s in fields]
    return hashlib.sha256("|".join(normalized).encode("utf-8")).hexdigest()
```

### 3.2 What this does *not* fix retroactively — and why that's the right call for Phase 1

`content_hash` has a DB-level `unique=True` constraint (`db/models.py`). The ~17-27 already-
duplicated rows per affected posting were inserted under the *old* hash formula (location
included), so they already hold 17-27 distinct hash values in the DB. Changing the formula alone
does not make those rows collide or get cleaned up — a real backfill would mean identifying
existing duplicate groups under the *new* formula and deleting all but one row per group.

That's deliberately **out of scope for Phase 1**:
- Deleting a `Job` row that has dependent `fit_judgments`/`eval_labels` rows (both FK-reference
  `jobs.id` with no `ondelete` cascade configured) requires either cascading the delete manually
  or reassigning those dependents to the surviving row first. `eval_labels` in particular holds
  human-labeled ground truth for the eval harness — deleting the wrong row in a duplicate group
  could silently destroy a label with no easy way to notice.
- It's genuinely optional, not urgent: the already-duplicated rows are weeks old (`collected_at`
  in the 2026-08-06 to 2026-08-14 range as of this writing) and will age out of any
  `max_age_days`-filtered candidate pool (the default `job-radar-fit` run uses `max_age_days=30`)
  within the next couple of weeks on their own, with zero code required.
- **One transitional caveat to know about, not to fix**: the very next ingestion run after this
  change ships, for each already-duplicated posting, will compute a *new*-formula hash that
  doesn't match any of the old-formula hashes already in `existing_hashes` — so one more row (in
  the new hash format) gets inserted before the fix actually starts preventing further
  duplicates for that posting. This is expected, bounded (one extra row, once, per already-
  affected posting), and not worth engineering around.

If a manual cleanup is ever wanted later, it's a separate, explicitly-scoped piece of work (decide
which row in a duplicate group to keep — likely the one with an `eval_labels` row if any, else
earliest `collected_at` — and cascade-delete the rest's `fit_judgments`/`eval_labels` first) — not
part of this plan.

### 3.3 Existing test that will break, and how to fix it

`tests/test_dedup.py::TestContentHash::test_different_identity_fields_change_the_hash` is
parametrized with `{"location": "Remote - US only"}` as a case that should change the hash — but
`_job()`'s default fixture already sets `remote=True`. Under the new formula, a `remote=True`
job's `location` no longer participates in the hash at all, so this specific parametrized case
will start failing (hash stays the same, test expects it to differ).

Fix: split it into two explicit cases instead of one shared parametrize entry —
- a **non-remote** job (`_job(remote=False)`) where changing `location` still changes the hash
  (extend `_job()`'s helper to accept a `remote` kwarg, defaulting to `True` as it does today, to
  avoid touching every other existing call site), and
- a new **remote** job case (`_job(remote=True)`, the existing default) asserting `location`
  changes do *not* change the hash — the actual new behavior being introduced, and worth its own
  explicit test rather than only being implied by the fix to the other one.

## 4. Testing plan for the new adapters

Mirror `tests/test_greenhouse.py` + `tests/fixtures/greenhouse_jobs.json` exactly, once per new
adapter:

- `tests/fixtures/ashby_jobs.json`, `tests/test_ashby.py` — cover: a remote job maps correctly
  end-to-end; a non-remote job is excluded by `fetch()`'s filter (test this at the `fetch()`
  level with a monkeypatched HTTP client, the way `test_search.py`/`test_pipeline.py` fake their
  I/O, not by calling the real API in tests); `descriptionPlain` passes through without
  HTML-stripping since it's already plain text (this is a real behavioral difference from
  Greenhouse worth its own assertion, not an oversight if a reviewer notices no `html_to_text`
  call).
- `tests/fixtures/workable_jobs.json`, `tests/test_workable.py` — cover: `details=true` field
  presence assumed in `map()` (i.e. the fixture should look like the *detailed* response, not the
  bare list); `telecommuting=False` job excluded by `fetch()`; `published_on`'s bare-date parsing
  produces a comparable (tz-aware or coalesce-compatible — see §1.2) `datetime`.
- `tests/fixtures/smartrecruiters_jobs.json`, `tests/test_smartrecruiters.py` — cover: `map()`
  operating on a merged list+detail dict (construct the fixture as already-merged, matching
  whatever shape `fetch()` is decided to produce); `location.remote=False` excluded; the
  `jobDescription`+`qualifications` join renders sensibly without either section being empty
  crashing the join (test the case where `qualifications.text` is `""`, seen in the real
  Xiaomi India payload's `additionalInformation` field tonight — confirm which sections are
  actually sometimes empty in practice, not just assume).
- `tests/test_dedup.py` — apply §3.3's fix.

No new tests needed for `discovery.py`, `runner.py`, or `ingest/pipeline.py` — nothing about their
behavior changes; the new adapters exercise existing, already-tested generic paths.

## 5. Rollout / verification steps, in order

1. Implement `content_hash` fix + its test split (§3) first, independently — smallest, most
   isolated change, easiest to verify in isolation (`uv run pytest tests/test_dedup.py`).
2. Implement `ashby.py` — simplest of the three *and* the highest-yield (§5b: 811 remote
   postings for 43 calls, no N+1). End to end with its tests, verified against the real API
   (`uv run python -c "..."` pulling a real token's board, the same manual verification style used
   throughout this research phase) before moving on. If Phase 1 ever needs to ship in pieces, this
   one alone delivers most of the value.
3. Implement `workable.py`, resolving the `company` field's source (confirmed available at the
   account level, unlike Ashby) and the `published_on` tz question concretely in code, not left
   as an open question.
4. Implement `smartrecruiters.py` last — the N+1 pattern and pagination cap are the most involved
   part of Phase 1; get the other two working and merged (conceptually — nothing here implies an
   actual git commit strategy) first so this one bug's-worth of complexity is isolated.
5. Register all three in `runner.py`, run `uv run job-radar-ingest`, confirm all three enabled
   adapters appear in the log with `Finished ingestion for source=...` and no tracebacks.
6. Query per-source `Job` counts (same pattern used throughout this research:
   `select(Job.source, func.count(Job.id)).group_by(Job.source)`) to confirm nonzero rows landed
   for `ashby`/`workable`/`smartrecruiters`.
7. Full suite (`uv run pytest`) + lint (`uv run ruff check .` / `ruff format --check .`) clean
   before considering Phase 1 done.

## 5b. Correction to the design doc's coverage numbers, and measured yield

`INGESTION_EXPANSION_DESIGN.md` §2 lists per-ATS counts (e.g. "234 ashby", "229+22 workable",
"168+136 smartrecruiters") and §5 summarizes Phase 1 as "~700 more companies' worth of coverage."
Those were **raw `ats_links` counts across the whole dataset** — but `discovery.py` filters to
`industry_category == "tech"` (the dataset is ~69% gaming studios) *and* dedupes tokens. The real
numbers `discovery.py` will actually see:

| ATS | Raw links (all industries) | Unique **tech** tokens |
|---|---|---|
| greenhouse *(already built)* | 773 | **325** |
| lever *(already built)* | 336 | **154** |
| smartrecruiters | 304 | **108** |
| ashby | 234 | **43** |
| workable | 251 | **40** |

So Phase 1 adds **191 boards, not ~700 companies**. The design doc's framing overstated it by
~3.7x. Correct that doc when this plan is executed.

**Board count is the wrong metric anyway — measured remote yield is what matters**, and Ashby
inverts the ranking entirely despite having the fewest boards:

| Adapter | Live boards | Total postings | **Remote** | Remote rate | HTTP cost |
|---|---|---|---|---|---|
| SmartRecruiters | 94 of 108 | 15,669 | **1,623** | 10.4% | 1,857 calls (list + filtered detail) |
| Ashby | 39 of 43 | 1,378 | **811** | **58.9%** | 43 calls (one per board, no detail) |
| Workable | 38 of 40 | 416 | **82** | 19.7% | 40 calls (one per board, `details=true`) |
| **total** | **171** | **17,463** | **2,516** | | **~1,940 calls** |

**2,516 new remote postings against a current corpus of 11,441 — a ~22% increase**, now a measured
figure rather than an estimate.

This changes the recommended build order in §5. **Ashby should be built first** on merit, not just
because it is structurally simplest: 811 remote postings for 43 HTTP calls and no N+1 is by far
the best return in Phase 1, and it is the only one of the three where a majority of postings are
remote. SmartRecruiters is still worth building (it is the single largest absolute contributor)
but it costs ~1,857 calls and the most code for 2x Ashby's yield. Workable is the weakest of the
three by every measure — 40 boards, 82 remote postings — and would be a defensible cut if Phase 1
needs narrowing, though it is cheap enough (one call per board, no N+1) that dropping it saves
little.

## 6. Open questions — all resolved

All three questions this plan originally carried have been answered empirically against the live
APIs (2026-08-21), not left for implementation time:

- **Ashby's `company` field** (§1.1) — checked 3 real boards; no company name is exposed at any
  level. Decision: title-case the token, accept the cosmetic loss, leave `discovery.py` alone.
- **Workable's location representation** (§1.2) — analyzed 416 real jobs across 38 live boards.
  Flat fields and `locations[0]` never disagree, but flat fields truncate multi-country remote
  postings. Decision: read the `locations` array.
- **SmartRecruiters pagination + N+1 cost** (§1.3) — paged all 94 live boards (15,669 postings).
  `limit` is hard-capped at 100, `offset` pages cleanly, no server-side remote filter exists, but
  `location.remote` is on every *list* item so the filter runs pre-detail: 1,857 total HTTP calls
  instead of 15,903, ~3 min at 10-way concurrency. Decision: no page cap, concurrent detail
  fetches.

Nothing is blocked on further research. See §5b for the full measured yield table across all
three adapters (including Ashby, scanned after the fact — 811 remote postings from 43 boards, the
best yield-per-call in Phase 1).
