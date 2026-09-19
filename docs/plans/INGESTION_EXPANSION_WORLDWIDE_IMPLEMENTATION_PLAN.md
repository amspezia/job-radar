# Ingestion Expansion, Worldwide-Remote Track — Implementation Plan

Locks down the next five ingestion steps (seven sources) after `INGESTION_EXPANSION_DESIGN.md`,
`..._PHASE1_...` (built) and `..._PHASE2_...` (not built). Every endpoint, payload shape and number
below was hit live on 2026-09-18; the eligibility rules in §1.1 were prototyped and validated against
1,795 real location strings and 120 real descriptions, and §12 carries the resulting reference
implementation. Counts from sampled boards are planning figures (±40%), not results.

**What this track optimises for.** Remote engineering roles at *foreign employers* whose posting is
open to worldwide / LATAM / Brazil-based candidates — not roles hiring inside one country. That
changes which sources are worth building, and it is why this plan re-orders the earlier phases:

| Earlier plan item | Status here |
|---|---|
| Phase 2: RemoteOK | Stays dropped. Re-checked 2026-09-18: 99/99 descriptions still carry the anti-bot canary phrase and are truncated; only 99 items. |
| Phase 2: The Muse | Deferred. 41 of 100 sampled roles are a bare "Flexible / Remote" with no eligibility signal; category tags are noisy. |
| Phase 2: WeWorkRemotely | Moved into Step 3, with different feeds (see §4.1). |
| Phase 3: Workday | Dropped for this track: US-enterprise tenants, remote not detectable from the list call. |
| Phase 4: Adzuna | Dropped: per-country domestic listings, truncated descriptions, call budget. |
| Not in any plan: Gupy, Brazilian GitHub job boards, Recrutei | Rejected: Brazil-domestic employers. |

Explicitly **not** in scope: a profile-driven ingest gate (§1.2), per-source CLI flags, retrieval or
fit changes, any backfill of already-ingested rows.

## 0. Baseline and success metric

Rows in the DB on 2026-09-18, and how many pass the real profile's retrieval filter
(`build_profile_filter(profile, max_age_days=30)`: remote + geo + freshness; seniority not applied):

| Source | Total | Published/collected ≤30d | Pass profile filter |
|---|---|---|---|
| himalayas | 6,164 | 1,059 | 107 |
| greenhouse | 5,549 | 725 | 46 |
| lever | 1,524 | 145 | 12 |
| smartrecruiters | 1,213 | 268 | 13 |
| ashby | 896 | 202 | 12 |
| arbeitnow | 577 | 150 | 14 |
| getonboard | 338 | 67 | 62 |
| remotive | 68 | 7 | 2 |
| workable | 60 | 18 | 3 |
| **Total** | 16,389 | | **271** |

**The success metric is that last column** — the *eligible pool* — not the row count. 16.4k rows
yield 271 usable ones today (1.7%). §8 has the query to re-measure it after each step.

## 1. Cross-cutting decisions

### 1.1 Eligibility policy and the shared gate

New module `src/job_radar/adapters/sources/eligibility.py` — pure functions, no I/O, no settings
import (adapters do not read config today). It owns the policy so five adapters cannot drift.
**§12 holds the validated reference implementation**; copy it rather than re-deriving the regexes,
which were tuned against real data and contain non-obvious fixes.

A posting's location is split into parts and each part is judged:

| Class | Rule | Real strings from the samples |
|---|---|---|
| `EXPLICIT` | any part names Brazil/Brasil, LATAM, Latin America, South America, worldwide, global, international, a *standalone* "Anywhere"/"anywhere in the world", or "Americas" with no US/Canada/North America anywhere in the string | `Remote - LATAM`, `Brasil (Remote)`, `Brazil/Remote`, `Global Remote`, `Remote • Anywhere`, `Remote International`, `Remote (North/South America)`, `BR-Brazil-Remote`, `Remote – Americas`, `LATAM ; Eastern Europe` |
| `BARE` | every part is just remote/fully remote/remote first/distributed/WFH/telecommute, or the location is empty | `Remote`, `Remote First`, `` |
| `RESTRICTED` | anything else | `Remote - US`, `Remote (Canada)`, `United States`, `US, Remote`, `Europe`, `Hamburg or Berlin`, `India Remote` |

Splitting rules that the validation exposed as load-bearing:

- **Normalise separators before matching**: en/em dash → `-`, bullet and middot → `;`. Boards write
  `Remote – Americas`, `Remote — Americas` and `Remote • Anywhere` interchangeably, and normalising
  once means no pattern has to carry those characters (§12 has the lint reason too).
- **Split on `; | /` and newlines, and on a comma only when it is *not* inside parentheses.**
  `Remote (US, Canada)` must stay one part, or the comma split would leave a bare `Canada)` and lose
  the pairing. `Boston, Massachusetts, United States` still splits (no parens) and is restricted
  either way.
- **`Americas` is judged against the whole string, not its own part.** `Americas (USA or Canada)`
  (8 of 34 loose-regex "explicit" hits in one Ashby sample) is restricted; a standalone
  `Remote – Americas` is explicit. Checking only the part would let a comma split defeat the rule.
- **"Anywhere" counts only as a whole part** (or in "anywhere in the world"), so
  `Europe - Anywhere ; Greece ; Belgium` and `Anywhere in the US` stay restricted.
- **Any eligible part wins.** `Remote - California, …, Texas ; Remote - Uruguay ; LATAM - Brazil ;
  Remote - Mexico` is explicit.

Module surface:

| Function | Purpose |
|---|---|
| `classify(*location_parts) -> LocationClass` | The table above. `LocationClass` is a `StrEnum` (`explicit`/`bare`/`restricted`) so it logs and compares cleanly. Accepts several strings so a caller can pass a primary location plus secondaries without pre-joining. |
| `gate_listing(title, *location_parts) -> LocationClass \| None` | `None` unless `is_engineering_title(title)` and the class is not `RESTRICTED`; otherwise the class. Cheap; runs before any description is fetched. |
| `passes_description(cls, description) -> bool` | `False` if the text is Portuguese (proxy for a Brazil-domestic employer), whatever the class. Then `EXPLICIT` passes; `BARE` fails only if the text states US-only/US-authorisation without a worldwide override. |
| `is_engineering_title(title) -> bool` | Engineering keyword minus a negative list (sales, account executive, recruiter, marketing, customer success/support, product manager/owner/designer, artist, game design, annotation, clinical, legal). |
| `is_excluded_employer(name) -> bool` | Substring match against `EXCLUDED_EMPLOYERS` on a whitespace-normalised, case-folded name (§1.3). |
| `is_portuguese(text) -> bool` | ≥6 hits from an unambiguous pt-BR stopword set. |

**Why the description can only *disqualify*.** Measured on 120 real bare-"Remote" Greenhouse
engineering postings with the §12 regexes: the US-only pattern drops 11 (9%), and reading all 11
confirms every one is genuinely US-restricted (visa sponsorship, US citizens, DoD clearance,
"Remote (U.S.)"). A *positive* worldwide regex, by contrast, hit 6 of 120 with an earlier looser
pattern and at most 2 were real — the rest were company boilerplate ("customers worldwide",
"distributed team", "across the globe"). So a bare "Remote" is overwhelmingly *unclear*, not
worldwide: the gate keeps unclear postings (dropping them would discard most of the pool) but never
treats description prose as positive evidence. `_US_OVERRIDE` exists only to stop a genuine
"we hire anywhere in the world" sentence from being overridden by a stray authorisation clause.

**The Portuguese trap, and why the stopword list looks the way it does.** A first attempt included
`com` and `para`. `\bcom\b` matches the `com` in every `.com` URL once `html_to_text` has flattened
the markup — 65 hits across the 120 English descriptions, enough to flag one on its own. The final
list uses unambiguous markers only: **0 false positives on those 120 English descriptions, and 7/7
on real pt-BR postings** pulled from the Workable feed.

### 1.2 The gate is a static ingest policy, and must stay in sync with the profile

The retrieval geo filter (`retrieval/geo.py`, driven by `profile.location_rules.allowed_keywords`)
stays authoritative. The ingest gate exists to bound LLM extraction cost (§1.4), so it is a
module constant, not profile-driven — the profile lives in Postgres and adapters have no session.

The profile's keyword set is runtime data in Postgres and is not reproduced here. As of 2026-09-18 it
covers every `EXPLICIT` token in §1.1 **except three**: `americas`, `south america` and
`international`. A posting the gate keeps for one of those is ingested and then dropped by the geo
filter. Either add them to the profile's `allowed_keywords` (a data change, §8 step 0) or accept the
loss (~10 rows in the samples). Keep the gate's token set and the profile in sync by hand; a comment
in `eligibility.py` says so.

The eval personas in `eval/personas/` have their own `allowed_keywords` (US, EU, Canada) and run
against static fixtures, so this ingest policy does not affect the golden eval.

### 1.3 Excluded employers

`eligibility.EXCLUDED_EMPLOYERS` — a constant, edited in code, **substring-matched** against a
normalised name. Equality matching was tried and fails on real data: the Workable feed carries
`Mindrift - Data annotation`, which an exact match misses while letting 27 gig postings through.

Seeded with the recruiting marketplaces and annotation/staffing shops that dominate the measured
Workable feed: `huzzle, pavago, uptalent, hire overseas, workana, mindrift` — 116 of 287 engineering
postings. Outsourcing/consulting shops that also appear (Dev.Pro 14, Jalasoft 9, Capgemini 7, Stack
Builders 5) and Toptal (WeWorkRemotely) are **not** seeded — they are the owner's call (§10).

### 1.4 Throughput budget

Extraction + embedding runs at about 42 postings/min (`ingest/pipeline.py`,
`_MAX_CONCURRENT_INGEST = 8`). Fetching is cheap; the LLM is the bottleneck. Estimated *new* rows on
the first run after each step (later runs are far smaller — dedup collapses everything already seen):

| Step | New rows (first run) | Extraction time |
|---|---|---|
| 1 Himalayas | a few hundred (≈170 unique per busy query × 15 queries, heavily overlapping the 6,164 already stored) | minutes |
| 2 Greenhouse + Ashby seeds | ~1,600 (range 1.1–2.2k; from ~12k engineering-remote → gate) | ~40 min |
| 3 WeWorkRemotely + Jobicy | 59 + 149 | ~5 min |
| 4 Workable | 171 | ~4 min |
| 5 HN + Working Nomads | ~60–80/month + (only if §6.2 passes) | negligible |

Without the gate Step 2 alone is ~12k postings ≈ 5 hours of extraction (and ~65k if the remote filter
were also dropped). That is why the gate ships *with* the seeds in the same step, never after.

### 1.5 Registration and dedup

`runner.py::ENABLED_ADAPTERS` gains `WeWorkRemotelyAdapter`, `JobicyAdapter`, `HackerNewsAdapter` and
(conditionally) `WorkingNomadsAdapter`; `WorkableAdapter` keeps its slot and name (its internals are
replaced). No change to `run_all_ingestion` — per-source failure isolation and fetch/LLM overlap
already exist. Put the two heavy board adapters (`greenhouse`, `ashby`) **non-adjacent** in the list
so a ~10-minute fetch overlaps an LLM phase rather than another long fetch.

`content_hash` is company + title for remote postings (location excluded), so the same role re-listed
by one source under several countries collapses to one row, and the URL check catches an edited
posting. Cross-source collapse is *partial*, not guaranteed: it only happens when both sources spell
the company identically, which HN (`Snout`) and a board (`Snout Inc`) often do not. Expect some
cross-source duplicates; that is the existing `content_hash` contract, not a new problem.

`source_type` is `"aggregator"` for every new source; Workable moves from `"board"` because it is
now a cross-company feed.

### 1.6 Compliance

| Source | Obligation | How this plan honours it |
|---|---|---|
| Himalayas | Visible link back and "sourced from Himalayas" when displaying (already tracked). | `source` and `url` stored; UI attribution lands with M5. |
| Jobicy | Credit Jobicy, keep the canonical Jobicy URL, do not poll more than once an hour. | `url` = Jobicy's own URL; §4.2 makes one request per industry per ingest run; do not re-run `job-radar-ingest` within the hour. |
| Feashliaa token lists | CC BY-NC 4.0 (data), MIT (code). Non-commercial use with attribution. | Fetched at runtime, cached under gitignored `data/`, never committed. Credit in the README. This project is non-commercial; revisit if that changes. |
| Workable `jobs.workable.com` | robots.txt allows `/api`; `Content-Signal: search=yes, ai-input=yes, ai-train=no`. | Used for search/extraction only; nothing here trains a model on it. |
| WeWorkRemotely | Public RSS; robots.txt allows all but admin/account paths. | Standard `USER_AGENT`, five feeds. |
| Working Nomads | No terms found; `/api/exposed_jobs/` is deliberately public, the search endpoint is the site's own front-end API. | Step 5 go/no-go (§6.2). |
| HN | Algolia HN API is public and keyless. | Three requests per run. |

### 1.7 Rate limiting and politeness (new code, needed by Step 2)

Step 2 issues ~11.5k HTTP requests per run — twenty times anything this codebase does today — so the
failure mode changes. `adapters/retry.py::with_retry` currently treats only timeouts, connect errors
and 5xx as transient; **429 is not retried**, and `_verify`/`_board` swallow `httpx.HTTPError`
entirely, so a rate-limit storm would silently return zero postings and look like an empty corpus.

**Measured first, to avoid building for a problem that does not exist:** 900 sustained Ashby board
requests at 20-way concurrency (~19 req/s) and 300 Greenhouse requests (~15 req/s) returned **zero
429s and no `Retry-After` headers**. So do not add per-host throttling or elaborate backoff.

Do make these two small changes, because the cost of being wrong is a silent empty source:

1. `retry.py::_is_transient` also returns `True` for status 429, and `with_retry` honours a
   `Retry-After` header when present (falling back to the existing exponential backoff). This is a
   strict improvement for every existing caller and keeps one retry policy in the codebase instead
   of a second bespoke one (`himalayas.py` already hand-rolls its own 429 loop; leave it alone in
   this plan, but do not copy it).
2. Board fetches count 429s and log **one warning per source per run** if any occurred
   (`source=ashby rate-limited on N/M boards`), so the condition is visible without one line per
   board.

Keep `_BOARD_CONCURRENCY = 20` for both ATS, matching the existing Greenhouse/Lever constant and the
measurement above. Feashliaa's own scraper uses 5 workers for Ashby; that is a different client
profile (no `Retry-After` was observed here at 20), so revisit only if a real run logs 429s.

## 2. Step 1 — Himalayas: filter server-side for Brazil-eligible roles

**Finding.** [himalayas.py](../../src/job_radar/adapters/sources/himalayas.py) queries 15 role
phrases with no location filter and takes the top 120 "relevant" results of thousands (unfiltered
totals run from ~1k to the API's 5,000 cap), so most fetched roles are region-restricted. The search
API (documented in the live OpenAPI spec at `himalayas.app/docs/openapi.json`) has `country`
(ISO alpha-2, names and slugs) and `sort` (`relevant`/`recent`/…):

- `country=BR` returns Brazil-listed *and* worldwide roles. `worldwide=true` is a strict subset of it
  (verified per-guid on a full page-through of one query: 35 of 35 present, 0 missing), so
  `country=BR` alone is enough. `exclude_worldwide` stays unused.
- Per-query totals with `country=BR` sum to 4,334 across the 15 queries (heavily overlapping); the
  largest is `software engineer` at 935.
- A result's `locationRestrictions` is empty (worldwide) or lists Brazil in every sampled row.
- **Pages are short but not empty**: with `country=BR&sort=recent`, `software engineer` returned
  20/18/14/19/19/15/15/16/18/19 items on pages 1–10 (171 unique), i.e. the server filters *after*
  paging. `totalCount` also drifts between requests (935 → 937). The existing break condition
  (`not batch or payload["offset"] + len(batch) >= payload["totalCount"]`) is correct for this: it
  stops on an empty page, never on a short one, and small queries terminate properly
  (`data scientist`: 62 total, exhausted on page 4).

**Changes.**

| Change | Detail |
|---|---|
| Params | `_get_page` sends `params={"q": query, "country": _COUNTRY, "sort": "recent", "page": page}`, with `_COUNTRY = "BR"` and a comment on why `worldwide` is not also sent (it is a subset). |
| Depth | `_MAX_PAGES_PER_QUERY` 6 → 10 (~170 newest per query, given short pages). Worst case 132 requests per run across the 15 queries; the existing 429 backoff already covers rate limiting. |
| Queries | Unchanged. |
| `map()` | Unchanged. Worldwide roles keep `location=None`, which the geo filter treats as no-signal; Brazil-listed roles keep "Brazil". |

No gate needed: the server already applied eligibility. No new file, no new dependency.

**Tests** (`tests/test_himalayas.py`): the existing `_FakeClient.get(url, params)` already records
`params`; assert `_get_page` sends `country="BR"` and `sort="recent"`. Add a short-page test —
a page of 14 with `offset=40, totalCount=935` must **not** end the loop — since that is the
behaviour this step depends on. The existing 429-retry test is unchanged.

**Done when:** a live one-query run returns only rows whose `locationRestrictions` is empty or
contains Brazil, and the eligible pool (§8) does not decrease.

## 3. Step 2 — Feashliaa token lists + the eligibility gate (Greenhouse, Ashby)

The single biggest lever. `discovery.py` finds 207 Greenhouse / 72 Lever / 37 Ashby live boards via
the OpenJobs dataset. `Feashliaa/job-board-aggregator` (updated daily, ~95k slugs harvested from
Common Crawl) publishes flat JSON arrays of slugs under `data/{ats}_companies.json`:

| ATS | Slugs listed | Live in samples | Avg payload per live board |
|---|---|---|---|
| greenhouse | 8,333 | 55–61% | 22 KB list-only (284 KB with `content=true`) |
| ashby | 3,161 | 70–76% | 225 KB (descriptions always included) |
| lever | 4,368 | 34–37% | 348 KB |

All slugs are lowercase and match `[a-z0-9_-]+` (verified: 0 violations, 0 uppercase across all three
lists; 9 all-digit Greenhouse slugs, which are valid boards). The curated remoteintech list (553
worldwide/Americas remote-friendly companies) resolved 135 boards by slug guess; 118 of those (87%)
were already in these lists, so the lists subsume that seed and no separate ingestion of it is planned.

### 3.1 What the gate leaves

Measured with the §12 regexes over engineering-titled remote postings from random boards
(Greenhouse n=900 boards → 627 postings; Ashby n=600 → 1,168; Lever n=250 → 110), then scaled to the
full lists:

| | Greenhouse | Ashby | Lever |
|---|---|---|---|
| Engineering + remote (sampled → scaled) | 627 → ~5,800 | 1,168 → ~6,150 | 110 → ~1,900 |
| `EXPLICIT` | 6 → ~56 | 25 → ~132 | 0 → 0 |
| `BARE` | 137 → ~1,270 | 53 → ~280 | 8 → ~140 |
| `RESTRICTED` | 484 → ~4,480 | 1,090 → ~5,750 | 102 → ~1,760 |

Applying the 9% US-only description drop to `BARE` leaves roughly **1,600 postings worth ingesting
(~190 explicit, ~1,410 unclear)**. Treat the range as 1.1–2.2k: sampling variance is high because a
few very large boards dominate the remote counts, and a 500-board Greenhouse sample gave ~3.4k
engineering-remote where the 900-board one gave ~5.8k.

**Lever is left alone** (no seeds, no gate): of 110 sampled engineering-remote postings, 102 were
`RESTRICTED`, 8 `BARE` and **none** `EXPLICIT`, so ~1.5 GB of downloads per run would buy a few
hundred unclear postings and nothing explicit. Revisit only if Lever's location data changes.

### 3.2 Changes

**`discovery.py`** — add one function, leave `get_tokens` untouched:

```python
async def get_seed_tokens(client, *, ats: str, cache_path: Path) -> list[str]
```

Fetches `https://raw.githubusercontent.com/Feashliaa/job-board-aggregator/main/data/{ats}_companies.json`,
keeps entries matching `[a-z0-9_-]+`, dedupes and sorts, caches to `cache_path` on success, and falls
back to the cache on an HTTP error or an empty list (same contract and log wording as `get_tokens`).
Seeds are **not** liveness-verified: verifying would download every board twice (~1.3 GB across the
two ATS), while a dead board's 404 is near-free when it is fetched directly. Cache files:
`data/greenhouse_seed_tokens.json`, `data/ashby_seed_tokens.json`. On the very first run with the
dataset unreachable there is no cache, `get_seed_tokens` returns `[]`, and the adapter still works
from its verified OpenJobs tokens — the seed list is strictly additive.

**`greenhouse.py`** — token set becomes `get_tokens(...)` (OpenJobs, verified) ∪ `get_seed_tokens(...)`,
deduped; `_board` becomes two-phase so only gate survivors download a description. **Verified on
three live boards:** the bare list carries `id, title, location, metadata, company_name,
absolute_url, first_published, updated_at` — everything the gate and `map()` need except `content`.

1. `GET /boards/{token}/jobs` **without** `content=true` (22 KB vs 284 KB).
2. Keep `_remote_jobs` (`"remote"` in the location name) as the first filter, then
   `gate_listing(title, cls._location(raw))` — reusing the existing `_location`, so the
   "target countries for posting" metadata (present on the bare list) is judged too.
3. For survivors only, `GET /boards/{token}/jobs/{id}`, bounded by the same semaphore. Verified to
   return `content` plus every list field, so **the detail dict replaces the list item and `map()`
   is unchanged**.
4. `passes_description(cls, html_to_text(html.unescape(content)))` — the detail `content` is
   HTML-**escaped** (verified: `&lt;p&gt;`, not `<p>`), exactly like the `content=true` payload
   `map()` already unescapes. Unescaping before the check is required or the regexes see entities.
   Drop the posting on failure.

A failed detail request drops that posting only (debug log), never the board. A **404** on a board is
logged at debug, not warning — with ~3–4k dead seeds, a warning per board would bury the log; non-404
errors stay warnings. `fetch()` logs one INFO line per source: boards attempted, boards live,
postings kept, and 429 count (§1.7).

**`ashby.py`** — same token union. Two changes beyond the gate:

- `fetch()` is currently a sequential `for token in tokens` loop. At ~3.2k boards that is far too
  slow, so it becomes the Greenhouse pattern: a `_board` classmethod under
  `asyncio.Semaphore(_BOARD_CONCURRENCY = 20)`, gathered.
- The gate judges `location` plus every `secondaryLocations[].location`, and `map()` stores those
  parts joined (`", "`, order-preserving dedupe) as `location`. Today only the primary location is
  stored, so a posting eligible via a secondary "Brazil" would pass the gate and then be dropped by
  the retrieval geo filter. Real example that must survive end to end:
  `Brazil - Remote ; Remote - India ; Remote - UK ; Remote - Poland ; Remote - Canada`.
  `descriptionPlain` is already in the list payload, so no second request is needed.

`isRemote`/`isListed` filtering and the title-cased token as `company` are unchanged.

### 3.3 Cost per run

| | Requests | Bytes | Wall clock at 20-way |
|---|---|---|---|
| Greenhouse list | 8,333 (~5,100 live) | ~110 MB | ~9 min (measured 14.7 req/s) |
| Greenhouse detail | ~1,500 | ~30 MB | ~1 min |
| Ashby | 3,161 (~2,400 live) | ~540 MB | ~3 min (measured 18.8 req/s) |

Ashby's ~540 MB per run is the price of an API with no list-only variant; it is bandwidth, not
memory — postings are gated inside `_board`, so only survivors are held. If that becomes a problem
the lever is fewer boards (a curated seed subset), not a lighter request.

### 3.4 Tests

- `tests/test_eligibility.py` (new): parametrise `classify` over the §12.1 case table (all 42 cases,
  which the reference implementation passes), especially `Americas (USA or Canada)` → restricted,
  `Remote – Americas` → explicit, `Anywhere in the US` → restricted, `Remote (US, Canada)` →
  restricted (the paren-comma rule), the em-dash/bullet variants of an eligible string → explicit
  (the normalisation rule), and the multi-part string above → explicit. Then `gate_listing`
  (non-engineering title → `None`; `RESTRICTED` → `None`); `passes_description` (a US-only `BARE`
  fails, the same text as `EXPLICIT` passes, real-shaped pt-BR text fails, a description containing
  only "customers worldwide" does **not** rescue a US-only posting, and a text full of `.com` URLs is
  **not** flagged Portuguese — the regression that this list was designed around);
  `is_engineering_title` (`Sales Engineer` and `Senior Product Manager` excluded, `Staff Backend
  Engineer` kept); `is_excluded_employer` (`Mindrift - Data annotation` → `True`, `Dev.Pro` → `False`).
- `tests/test_discovery.py`: `get_seed_tokens` returns/caches; drops entries violating the slug
  pattern; falls back to the cache on `httpx.HTTPError`; falls back on an empty list.
- `tests/test_retry.py`: 429 is retried; a `Retry-After` header is honoured over the default backoff;
  4xx other than 429 still raises immediately.
- `tests/test_greenhouse.py`: `_board` requests a detail **only** for gate survivors (fake client
  records requested URLs); an escaped-`content` detail whose text is US-only drops that `BARE`
  posting; a board 404 returns `[]`; one failed detail drops one posting, not the board. The existing
  `test_board_returns_only_remote_jobs` is rewritten for the two-phase shape.
- `tests/test_ashby.py`: a posting eligible only through `secondaryLocations` passes the gate and its
  stored `location` contains the secondary; `_board` returns `[]` on HTTP error.

**Done when:** a live single-adapter run of each of Greenhouse and Ashby logs the boards/kept line,
inserts rows, no inserted row classifies as `RESTRICTED`, and the eligible pool (§8) rises.

## 4. Step 3 — WeWorkRemotely and Jobicy

Both are worldwide-first boards with no discovery step.

### 4.1 `weworkremotely.py`

**Feed choice (differs from the Phase 2 plan).** `remote-programming-jobs.rss` has only 25 items and
fewer fields. The category feeds carry `region`, `country`, `state`, `skills`, `type`, `expires_at`.
Verified live, one request each:

| Feed (`https://weworkremotely.com/categories/{slug}.rss`) | Items | Regions |
|---|---|---|
| `remote-programming-jobs` | 25 | all "Anywhere in the World" |
| `remote-full-stack-programming-jobs` | 78 | 74 worldwide, 3 North America Only, 1 USA Only |
| `remote-back-end-programming-jobs` | 6 | all worldwide |
| `remote-devops-sysadmin-jobs` | 18 | all worldwide |
| `remote-front-end-programming-jobs` | 0 | valid feed, currently empty — keep it, it will refill |

`remote-data-science-jobs` and `remote-all-other-jobs` return a **301 that redirects to themselves**
(an infinite loop, confirmed with `curl -L`) — do not use them. The all-jobs `remote-jobs.rss`
carries only ~10 items per category, so the five feeds above dominate it for engineering.

Measured end to end with the §12 filters: **101 unique items → 97 worldwide → 59 engineering.**

Fetch each feed with the standard `USER_AGENT`, parse with `xml.etree.ElementTree` (no new
dependency), dedupe across feeds by `guid`, keep items whose `region` is exactly
`"Anywhere in the World"`, and apply `is_engineering_title` to the title half — the programming feeds
still carry product, design and support roles. **Do not treat `region` as an enum:** besides the
three canonical values it returns raw city names (`Jabalpur`, `Manama` observed in sibling feeds), so
exact-match the one value you want rather than excluding a known-bad list.

| `NormalizedJob` | Source | Notes |
|---|---|---|
| `source_id` / `url` | `guid` / `link` | Both are the canonical job URL. |
| `title`, `company` | `title` split on the **first** `": "` | 86/86 sampled titles had it. No `": "` → whole string is the title, `company=""` (store unknown, never fabricate — the `normalize.parse_salary` rule). |
| `description` | `description` through `html_to_text` | |
| `job_type` | `type` (`Full-Time`, `Contract`) | **Not** `category` — that is a role domain. |
| `location` | `None` | `region` is constant after filtering; `country` is the company's base, not eligibility. `None` passes the geo filter as no-signal, like Himalayas worldwide roles. |
| `remote` | `True` | |
| `published_at` | `email.utils.parsedate_to_datetime(pubDate)` | RFC 822, tz-aware; the first adapter that cannot use `fromisoformat`. |
| salary | `None` | Not in the feed. |

A live snapshot with no pagination, so value accrues across daily runs rather than one large pull.
`expires_at` is available but unmapped (`NormalizedJob` has no field for it); freshness already comes
from `published_at`.

### 4.2 `jobicy.py`

`GET https://jobicy.com/api/v2/remote-jobs?count=200&geo=brazil&industry={slug}` — no key.
Verified: `geo=brazil` is a superset of `geo=anywhere` and returns Anywhere, LATAM, Brazil and
multi-region-including-Brazil roles (`geo=latam` would also return Mexico-only roles, so it is not
used). Industry slugs come from the live `?get=industries` taxonomy; note `admin` is
"DevOps & Infrastructure", and `dev` is a deprecated alias for `engineering`:

```python
_INDUSTRIES = ["engineering", "admin", "data-science", "qa-testing", "cybersecurity"]
```

Union by `id`: **149 unique postings** (engineering 102, admin 25, qa-testing 22, data-science 11,
cybersecurity 11), 29 with a salary (28 USD, 24 yearly). One request per industry, five per run.

**Hard limits, verified — the adapter must not try to page.** `count` caps at 200 server-side
(`count=500` echoes `count: 200` back), and there is **no pagination**: `offset` and `page` return
**HTTP 400**, so sending an unknown parameter breaks the call rather than being ignored. The feed is
"latest N", so 200 per industry is the ceiling and repeat runs pick up what is new. If an industry
ever returns exactly 200, log it — that is the signal the ceiling is binding and the industry needs
splitting by `tag`.

| `NormalizedJob` | Source | Notes |
|---|---|---|
| `source_id` / `url` | `str(id)` / `url` | Keep Jobicy's canonical URL (fair-use term). |
| `title`, `company` | `jobTitle`, `companyName` | |
| `description` | `jobDescription` through `html_to_text` | HTML; `jobExcerpt` is a 55-word teaser, not a substitute. |
| salary | `salaryMin`/`salaryMax`/`salaryCurrency` **only when `salaryPeriod == "yearly"`** | The Himalayas `_salary` rule: hourly/monthly are not comparable in period-less columns. |
| `location` | `jobGeo` with whitespace collapsed (`"EMEA,  LATAM"` → `"EMEA, LATAM"`); `None` when it is `"Anywhere"` | Matches the Himalayas convention and cannot be lost by a profile-keyword change. |
| `job_type` | first of `jobType[]` | Nullable; values are `Full-Time`/`Contract`/`Internship`/`Part-Time`. |
| `remote` / `published_at` | `True` / `fromisoformat(pubDate)` | `pubDate` is ISO with a `+00:00` offset. |

No engineering-title filter: the industry slugs already scope this, and 149 rows is cheap.

### 4.3 Tests

`tests/test_weworkremotely.py` — a dict fixture of parsed items plus one canned-XML test that
`fetch()` parses XML, dedupes across two feeds by `guid`, and drops a `North America Only` item and a
raw-city `region`: title split (with `": "`, without, and with a second colon), `type` → `job_type`
while `category` is never written to it, RFC 822 date is tz-aware, non-engineering title dropped.
`tests/test_jobicy.py` — fixture of real-shaped items: yearly salary kept, hourly/monthly dropped,
`"Anywhere"` → `None`, double-space collapse, cross-industry dedupe by `id`.

## 5. Step 4 — Workable global feed (replaces the token-based adapter)

`jobs.workable.com/api/v1/jobs` is the public job-marketplace backend: 170,109 jobs, 49,375 with
`workplace=remote`, 1,103 of those located in Brazil. It is cross-company, so no token discovery is
needed. **Workable has no worldwide flag** — each job carries the country it is located in — so
remote roles located in other countries are country-restricted and are not fetched.

The existing [workable.py](../../src/job_radar/adapters/sources/workable.py) (40 tokens → 60 rows in
the DB) is rewritten in place; its `discovery` import, `_LINK_RE`, `_TOKENS_CACHE`, `_has_jobs`,
`_remote_jobs` and `_location` are deleted. `data/workable_tokens.json` is stale and can be removed
locally (it is gitignored).

**Fetch: page the whole filter, no query fan-out.** Verified by exhausting it live:
`workplace=remote&location=Brazil` with **no `query`** reports `totalSize: 1103` and pages cleanly to
**1,103 unique ids in 56 requests** (page size is fixed at 20; follow `nextPageToken` until absent).
An earlier design fanned out over 12 query phrases and found only ~300 unique postings for ~40
requests, so the single page-through is both cheaper per posting and complete. Guard the loop with
`_MAX_PAGES = 100` so a token that never clears cannot spin forever.

Then filter locally: `is_engineering_title` (**1,103 → 287**; it removes 3D artists, game designers,
copywriters, recruiters, virtual assistants) and `is_excluded_employer` (**287 → 171**). Language is
almost all English (1,095 `en`, 7 `pt`, 1 `fr`); the 7 Portuguese ones are BigDataCorp
"banco de talentos" pipelines, which `passes_description`'s Portuguese check removes — so run
`passes_description(LocationClass.EXPLICIT, …)` on the assembled description too (the location always
names Brazil, so the class is `EXPLICIT` and only the Portuguese rule can fire).

| `NormalizedJob` | Source | Notes |
|---|---|---|
| `source_id` / `url` | `id` (a UUID string) / `url` | The `jobs.workable.com/view/…` URL. |
| `title` / `company` | `title` / `company.title` | Note `company.title`, not `company.name`. |
| `description` | `html_to_text(description + " " + requirementsSection)` | `benefitsSection` skipped, matching the SmartRecruiters preference for requirement-bearing text. Measured: `description` present on 245/245 (~1.2–5.8k chars), `requirementsSection` on only 171/245 — so treat it as optional, not guaranteed. |
| `location` | `locations[]` minus the literal `"TELECOMMUTE"` sentinel, joined `", "` (→ `São Paulo, São Paulo, Brazil`) | Contains "Brazil", so the geo filter matches. |
| `job_type` | `employmentType` or `None` | **Empty on 86/245** — commonly `""`, so normalise falsy to `None`. |
| `remote` | `True` (`workplace == "remote"` is a query filter) | |
| `published_at` | `fromisoformat(created)` | ISO with `Z`, accepted directly on Python 3.12. |
| salary | `None` | Not exposed on this endpoint. |

`source="workable"`, `source_type="aggregator"`. Old widget-adapter rows used a different `source_id`
and URL; `content_hash` (company + title, remote) still collapses the re-listings.

**Risks.** Undocumented front-end API, so shape drift is possible — per-source failure isolation
contains it, and the §8 row-count check catches a silent empty result. The Brazil-tagged tail is
agency-heavy, hence the blocklist.

**Tests.** `tests/fixtures/workable_jobs.json` and `tests/test_workable.py` are rewritten to the new
shape: `map()` on a real-shaped item; `TELECOMMUTE` stripped from `location`; a missing
`requirementsSection` still yields a description; empty `employmentType` → `None`; `created` parses
tz-aware; `fetch()` (fake client) follows `nextPageToken` to exhaustion, dedupes by `id` across
pages, stops at `_MAX_PAGES`, and drops an excluded employer, a non-engineering title and a
Portuguese posting.

## 6. Step 5 — Hacker News "Who is hiring" and Working Nomads

### 6.1 `hackernews.py`

Free-text but consistently shaped: in the September 2026 thread **240 of 262** top-level comments have
a `Company | Role | Location | …` header line. Across three months, 113–143 comments per month say
REMOTE; 20–25 read as worldwide, 41–59 as US-only, the rest unclear.

**Fetch.** `GET https://hn.algolia.com/api/v1/search_by_date?tags=story,author_whoishiring` → take the
two most recent stories whose title contains "who is hiring" (never "who wants to be hired"; the
prior month's thread still holds live postings) → `GET /api/v1/items/{id}` for each. Every direct
child of the story is one posting; nested replies are ignored. Three requests per run, idempotent
(URL dedup collapses re-fetches).

**Header parse** (first line only, segments split on `|`):

| Field | Rule |
|---|---|
| `company` | First segment, with any URL and trailing parenthetical stripped (`Snout https://snout.com/` → `Snout`). Empty after stripping → skip. |
| `title` | First *other* segment for which `is_engineering_title` is true. **No such segment → skip the comment** (debug counter); never invent a title. |
| location parts | The remaining segments except the salary one; passed to the gate and stored as `location`. |
| gate | Skip unless some part mentions remote. Then `gate_listing(title, *parts)`, then `passes_description(cls, header + first 1,000 chars of the body)`. Bare `REMOTE` stays; `Remote (Europe)` and `Remote US or Ontario, Canada` drop. |
| salary | `parse_salary` on the **first segment containing a currency symbol** only, so `(10–40 hrs/wk)` and headcounts are never parsed. Known quirk, accepted: `$150 - 210K USD` yields `min=None, max=210000`, because `normalize._MIN_PLAUSIBLE_SALARY` correctly rejects a bare `150`. `€75k–110k` → `(75000, 110000, EUR)`. |
| `url` / `source_id` | `https://news.ycombinator.com/item?id={id}` / `str(id)`. |
| `published_at` | `created_at` (`…Z`, `fromisoformat` on 3.12). |
| `description` | Comment `text` through `html_to_text`. |
| `job_type` | `None`. `remote=True`. |

The ~8% of comments with no pipes are skipped by design; an LLM fallback parse is a later option, not
part of this plan. Expect ~60–80 rows on the first run and a trickle after.

### 6.2 `workingnomads.py` — go/no-go before building

Two endpoints. `GET /api/exposed_jobs/` is deliberately public but returns only the latest 49. The
site's own search, `POST /jobsapi/_search` (an exposed Elasticsearch query endpoint), reports 1,374
development jobs with structured fields:

- The 200 newest dev jobs span four days (~50/day), but only ~6% (34 of 600 sampled) list worldwide,
  Latin America or Brazil; the rest are per-country rows (`['Spain']`, `['Ireland']`). The eligible
  ones are long country lists that include Brazil, e.g. Kraken's
  `['Argentina', 'Brazil', 'Bulgaria', …]`.
- `salary_range` is a display string (`$210k-$290k per year`, `£75k per year`,
  `$12.0k-$15.0k per month`), present on 52% of sampled dev jobs — the best pay visibility of any
  source here. `annual_salary_usd` is the USD-normalised **upper** bound, so it is not `salary_min`.
- Latency is erratic: 2 s for 30 hits, 57–137 s for 200. Use `size=50`, at most 4 pages,
  `timeout=120` — well above the 30 s every other adapter uses, and the reason this is last.

**Go/no-go check first (~30 minutes, before writing the adapter):** pull the newest 200 dev jobs,
keep the eligible ones, and measure how many `apply_url`s already point at a Greenhouse/Ashby/Lever
board reachable from Step 2 (the first eligible hit in the sample was a Kraken role linking to
Ashby). If more than ~70% are already reachable, **skip Working Nomads** — it would add an
unofficial-endpoint dependency, a 120 s timeout and POST-shaped fetch code for almost no new
postings. Record the decision in `docs/STATUS.md` either way.

If it goes ahead: `match category_name=Development`, `sort pub_date desc`; keep a job when any
`locations[]` entry classifies `EXPLICIT`; `salary_range` through `parse_salary` **only when it
contains "per year"**; `url` = `apply_url`; `source_id = str(id)`; `company`; `title`; `description`
through `html_to_text`; `location` = `", ".join(locations)`; `remote=True`;
`published_at = fromisoformat(pub_date)`; skip rows where `expired` is true.

### 6.3 Tests

`tests/test_hackernews.py` — synthetic headers in the observed shapes (invented company names):
company/URL stripping, title = first engineering segment, no engineering segment → skipped,
`Remote (Europe)` dropped, bare `REMOTE` kept, a US-only body drops a bare posting, salary read only
from the currency segment (and `(10–40 hrs/wk)` never parsed), and `fetch()` (fake client) picks the
two newest "who is hiring" stories, ignores "who wants to be hired", and ignores nested replies.
`tests/test_workingnomads.py` (only if the go/no-go passes) — explicit-location keep, per-country
row dropped, `salary_range` "per month" ignored, `expired` skipped.

## 7. Testing and quality gates (all steps)

- Every commit stays lint-clean and CI-green: `uv run ruff check .`, `ruff format --check .`,
  `uv run pytest`.
- **No new dependencies.** Everything uses `httpx`, `beautifulsoup4` (via `html_to_text`) and the
  standard library (`xml.etree.ElementTree`, `email.utils`, `enum.StrEnum`, `re`, `html`).
- Fixtures follow the existing pattern: JSON under `tests/fixtures/`, fake clients exposing
  `async get(url, params=None)` (and `async post(...)` for Working Nomads); no test touches the
  network or the DB. Fixtures carry invented company names, never real PII.
- Suggested commit boundaries (one per step; single-line subjects per the repo's commit style):
  1. `feat: filter himalayas searches to Brazil-eligible roles`
  2. `feat: gate ingestion on location eligibility and seed greenhouse/ashby board tokens`
  3. `feat: add weworkremotely and jobicy sources`
  4. `feat: replace workable board adapter with global remote feed`
  5. `feat: add hacker news who-is-hiring source`

  The `retry.py` 429 change (§1.7) can lead commit 2 or land as its own
  `fix: retry rate-limited requests and honour retry-after`.

## 8. Rollout and verification, in order

There is no per-source CLI flag; `job-radar-ingest` runs every adapter. To exercise one adapter, use
a throwaway script (not committed) that opens `async_session_factory()`, builds the adapter and calls
`ingest.pipeline.run_ingestion(adapter, session, ingested_via="manual")`.

**Step 0 — preflight (data, not code).**
1. Record the baseline (§0) with the eligible-pool query below.
2. Decide on `americas` / `south america` / `international` in the real profile's
   `location_rules.allowed_keywords` (§1.2).
3. Have Ollama running and warm; the first Step 2 run is ~40 min of extraction after ~13 min of fetch.

**Eligible-pool query** (read-only; reuse after every step):

```python
f = build_profile_filter(profile, max_age_days=30)
select(Job.source, func.count(Job.id)).where(f).group_by(Job.source)
```

**Per step:** implement with tests → live-verify the adapter against the real API (fetch a real
sample; assert the params and the gate behave) → single-adapter ingest → compare the eligible pool to
the baseline → spot-check ~10 inserted rows (`title`, `company`, `location`, `description` — the
check that caught RemoteOK in Phase 2; do not skip it) → `uv run job-radar-assess` for the new
source's salary/location/HTML-leak rates → full `pytest` + `ruff`.

Step-specific assertions:

| Step | Live check |
|---|---|
| 1 | Every fetched row has empty `locationRestrictions` or lists Brazil; a short page did not end the loop early (compare unique count against ~170 for `software engineer`). |
| 2 | Log shows boards attempted / live / kept, and kept is ~1–2k across both ATS, not tens of thousands. No inserted row classifies `RESTRICTED`. No per-board 404 warnings. Zero 429s (else revisit §1.7). Run Greenhouse alone first and time it. |
| 3 | WWR: no `North America Only` / `USA Only` row. Jobicy: every row's `jobGeo` is Anywhere/LATAM/Brazil/multi-region-with-Brazil; no industry returned exactly 200; at most one run per hour. |
| 4 | 56-ish page requests, ~1.1k fetched, ~170 inserted. `location` ends in "Brazil" on every row and contains no `TELECOMMUTE`; no excluded employer present; no Portuguese description. |
| 5 | HN: every row has a non-empty company and an engineering title; no `Remote (Europe)` posting present. |

Success is judged on the eligible pool after all five steps, against the 271 baseline — with the
caveat that the seeded Greenhouse/Ashby rows carry the bare-"Remote" *unclear* cohort, so the pool
counts postings worth reading, not postings guaranteed open to you.

## 9. Docs housekeeping (alongside the code, not separate work)

- `docs/STATUS.md` §1.1 sources row and §4: list the new sources; the ingestion-expansion item is no
  longer "proposal". Record the §6.2 go/no-go outcome.
- `INGESTION_EXPANSION_DESIGN.md`: correct §3 (RemoteOK as above), note the Workable global feed
  supersedes the token adapter, and mark Phases 3–4 as dropped for this track.
- `INGESTION_EXPANSION_PHASE2_IMPLEMENTATION_PLAN.md` (untracked): note that WeWorkRemotely moved
  here (with different feeds) and The Muse is deferred.
- `README.md`: when it gets real content, add a "Data sources" credit for Himalayas, Jobicy and the
  Feashliaa token lists (CC BY-NC 4.0).

## 10. Open questions for the owner

1. **Brazil-only postings from foreign employers** (e.g. `Remote - Brazil` at a US company) are kept
   by default: explicit Brazil eligibility at a non-Brazilian employer is the target, and pay can be
   judged from the salary field. The Portuguese check and the blocklist remove domestic employers. If
   these turn out to be local-payroll BRL offers, add a `BRAZIL_ONLY` class and drop it in one place.
2. **Blocklist scope.** Seeded with six marketplaces/annotation shops. Outsourcers (Dev.Pro,
   Jalasoft, Capgemini, Stack Builders, Intellectsoft) and Toptal are kept unless you say otherwise —
   together they are ~40 of Workable's 171.
3. **Lever** stays untouched (§3.1, zero explicit in 110 sampled). Say so if you want it gated anyway.
4. **Profile keywords** `americas` / `south america` / `international` (§1.2): add or accept the loss.
5. **Static vs profile-driven gate.** Deferred; it needs adapters (or the runner) to see the profile.

## 11. Explicitly not in this plan

`SmartRecruiters` and `Lever` changes; Ashby `includeCompensation=true` (adds a salary field — worth a
follow-up for a pay-focused search); Torre and the web3 RSS boards (measured only shallowly); The
Muse; Workday; BambooHR; iCIMS; Paylocity; a `--source` flag on `job-radar-ingest`; description-level
*positive* worldwide detection (§1.1 explains why it is unreliable); retiring `himalayas.py`'s
bespoke 429 loop in favour of `with_retry`; cleaning up rows ingested before the gate.

## 12. Validated reference implementation of `eligibility.py`

Prototyped and measured on 2026-09-18 against 1,795 real location strings (Greenhouse n=900 boards,
Ashby n=600), 120 real bare-"Remote" Greenhouse descriptions, 7 real pt-BR descriptions, 1,103 real
Workable titles and 101 real WWR items. Results: **42/42 unit cases pass; 0 Portuguese false
positives on English text; 7/7 real pt-BR detected; 11/120 (9%) bare postings dropped as US-only,
all 11 correct on inspection.** It also passes `ruff check` and `ruff format --check` under this
repo's config unmodified — which is why separators are normalised through `_NORMALISE` rather than
written as literals: an en dash inside a pattern trips `RUF001` (ambiguous-unicode), and the repo
carries no `noqa` and no ignore list. Normalising also turned out to be the better design, since it
catches em dash, bullet and middot variants anywhere in the string rather than only where a pattern
happened to list them.

Port this as-is (it is written to the repo's style — module-level compiled regexes, a docstring per
non-obvious rule); do not re-derive the patterns.

```python
import re
from enum import StrEnum


class LocationClass(StrEnum):
    EXPLICIT = "explicit"
    BARE = "bare"
    RESTRICTED = "restricted"


# Boards spell dashes and separators inconsistently ("Remote \u2013 Americas",
# "Remote \u2022 Anywhere"), so normalise once on the way in: en/em dash become
# "-", bullet and middot become ";". Every pattern below then stays plain ASCII,
# which also keeps them free of characters a reader can confuse with a hyphen.
_NORMALISE = {0x2013: "-", 0x2014: "-", 0x2022: ";", 0x00B7: ";"}

# Split on the separators real boards use, plus a comma only when it is NOT
# inside parentheses: "Remote (US, Canada)" has to stay one part, or the split
# leaves a bare "Canada)" and the pairing is lost.
_PART_SPLIT = re.compile(r"[;|/\n]|,(?![^(]*\))")
_ELIGIBLE = re.compile(
    r"\b(brazil|brasil|latam|latin america|south america|worldwide|world-wide"
    r"|global|globally|international)\b",
    re.I,
)
# "Anywhere" only counts as a whole part: "Anywhere in the US" and
# "Europe - Anywhere" are restrictions, not worldwide eligibility.
_ANYWHERE = re.compile(r"^(remote\W*)?anywhere(\W*remote)?$|anywhere in the world", re.I)
_AMERICAS = re.compile(r"\bamericas\b", re.I)
_NORTH_ONLY = re.compile(r"\b(us|u\.s\.|usa|united states|canada|north america)\b", re.I)
_BARE = re.compile(
    r"^(fully\s+|100%\s+)?remote(\s*-?\s*(first|global|anywhere))?$|^distributed$|"
    r"^work from home$|^wfh$|^telecommute$|^$",
    re.I,
)


def _parts(*location_parts: str | None) -> list[str]:
    out: list[str] = []
    for raw in location_parts:
        for part in _PART_SPLIT.split((raw or "").translate(_NORMALISE)):
            cleaned = " ".join(part.split()).strip(" \t-,")
            if cleaned and cleaned not in out:
                out.append(cleaned)
    return out


def classify(*location_parts: str | None) -> LocationClass:
    """Judge a posting's eligibility for a worldwide/LATAM/Brazil candidate.

    Any eligible part wins: a role open in twelve US states AND Brazil is
    eligible. "Americas" is judged against the whole string rather than its own
    part, because "Americas (USA or Canada)" is a North-America-only role and a
    comma split would otherwise strand the qualifier.
    """
    parts = _parts(*location_parts)
    whole = " ; ".join(parts)
    for part in parts:
        if _ELIGIBLE.search(part) or _ANYWHERE.match(part):
            return LocationClass.EXPLICIT
        if _AMERICAS.search(part) and not _NORTH_ONLY.search(whole):
            return LocationClass.EXPLICIT
    if not parts or all(_BARE.match(p) for p in parts):
        return LocationClass.BARE
    return LocationClass.RESTRICTED


_US_ONLY = re.compile(
    r"\b(us|u\.s\.|usa|united states)[\s-]?only\b"
    r"|(must|need to|required to|should|have to)\s+(be\s+)?"
    r"(located|reside|residing|live|living|based)\s+(in|within)\s+(the\s+)?"
    r"(us|u\.s\.|usa|united states)\b"
    r"|(authori[sz]ed|eligible|legal right)\s+to\s+work\s+(legally\s+)?"
    r"(for any employer\s+)?in\s+(the\s+)?(us|u\.s\.|usa|united states)\b"
    r"|\b(us|u\.s\.)\s+(citizens?|residents?)\b"
    r"|\bcitizenship\s+(is\s+)?required\b"
    r"|remote\s*\(\s*(us|u\.s\.|usa|united states)[^)]*\)"
    r"|remote\s+within\s+the\s+(us|united states)\b"
    r"|located\s+(and eligible to work\s+)?(in|within)\s+the\s+(us|united states)\b",
    re.I,
)
# Only ever rescues a posting the US-only patterns matched — never promotes a
# BARE posting on its own. Company boilerplate ("customers worldwide",
# "distributed team") is deliberately NOT here: on 120 real descriptions a
# positive-worldwide regex was wrong more often than right.
_US_OVERRIDE = re.compile(
    r"(anywhere in the world|worldwide|any country|from anywhere)"
    r"|(open to|hiring|hire|accept\w*|welcome\w*)[^.]{0,60}"
    r"\b(latam|latin america|brazil|internationally|international applicants)\b"
    r"|remote\s*-\s*(latin america|latam|brazil)",
    re.I,
)
# Unambiguous Portuguese markers only. "com" and "para" are deliberately absent:
# \bcom\b matches the "com" inside every ".com" URL once html_to_text has
# flattened the markup (65 hits across 120 real English descriptions — enough to
# trip the threshold on its own).
_PORTUGUESE = re.compile(
    r"\b(você|voce|vaga|vagas|nossa|nosso|somos|sobre a|não|nao|será|sera|"
    r"conhecimento|experiência|trabalho|equipe|desenvolvedor|desenvolvedora|"
    r"pessoa|atividades|requisitos|benefícios|beneficios|também|desejável|"
    r"nós|salário|vale)\b",
    re.I,
)
_PT_MIN_HITS = 6


def is_portuguese(text: str) -> bool:
    """A pt-BR posting means a Brazil-domestic employer paying in BRL."""
    return len(_PORTUGUESE.findall(text or "")) >= _PT_MIN_HITS


def passes_description(cls: LocationClass, description: str) -> bool:
    """Second gate stage: the description may only DISQUALIFY a posting.

    A bare "Remote" is unclear, not worldwide, so it is kept unless the text
    says otherwise. See the plan's §1.1 for the measurement behind that.
    """
    text = (description or "").translate(_NORMALISE)
    if is_portuguese(text):
        return False
    if cls is LocationClass.EXPLICIT:
        return True
    return not (_US_ONLY.search(text) and not _US_OVERRIDE.search(text))


# Over-inclusive on purpose: a non-engineering posting that slips through costs
# one extraction call and then ranks low, while a dropped one is invisible. The
# negative list removes the roles that carry an engineering word without being
# engineering ("Sales Engineer", "Technical Recruiter", "AI Product Manager").
_ENGINEERING_TITLE = re.compile(
    r"engineer|engineering|developer|programmer|software|devops|\bsre\b|"
    r"reliability|architect|backend|back-end|frontend|front-end|full[\s-]?stack|"
    r"machine learning|\bml\b|\bai\b|data scien|data engineer|analytics engineer|"
    r"\bqa\b|\bsdet\b|test automation|platform|infrastructure|cloud|security engineer|"
    r"mobile|\bios\b|android|tech(nical)? lead|staff engineer|\bcto\b",
    re.I,
)
_NOT_ENGINEERING_TITLE = re.compile(
    r"sales|account (executive|manager)|business development|recruit|talent acquisition|"
    r"marketing|customer (success|support|experience)|solutions? (consultant|specialist)|"
    r"accountant|bookkeep|payroll|paralegal|attorney|nurse|physician|teacher|"
    r"copywriter|content writer|social media|graphic design|merchandis|technician|"
    r"product manager|product owner|product marketing|product design|"
    r"\bartist\b|game design|annotation|\btutor\b",
    re.I,
)


def is_engineering_title(title: str) -> bool:
    text = title or ""
    return bool(_ENGINEERING_TITLE.search(text)) and not _NOT_ENGINEERING_TITLE.search(text)


def gate_listing(title: str, *location_parts: str | None) -> LocationClass | None:
    """Pre-description gate: the class to carry forward, or None to drop.

    Runs on list-endpoint data only, so it costs nothing per posting — this is
    what keeps a ~12k-posting fetch down to a ~1.6k extraction batch.
    """
    if not is_engineering_title(title):
        return None
    cls = classify(*location_parts)
    return None if cls is LocationClass.RESTRICTED else cls


# Recruiting marketplaces and annotation/staffing shops: the posting is a
# pipeline, not a role at the employer. Substring-matched on a normalised name —
# real feeds carry suffixed variants ("Mindrift - Data annotation") that an
# equality check misses, which hid 27 gig postings in the measured Workable feed.
EXCLUDED_EMPLOYERS = (
    "huzzle",
    "pavago",
    "uptalent",
    "hire overseas",
    "workana",
    "mindrift",
)


def is_excluded_employer(company: str) -> bool:
    name = " ".join((company or "").split()).casefold()
    return any(bad in name for bad in EXCLUDED_EMPLOYERS)
```

### 12.1 `classify` cases the tests must cover

All 42 pass against the implementation above (re-verified by extracting the code block and running
it, so the table and the code cannot drift apart silently).

| Expected | Locations |
|---|---|
| `EXPLICIT` | `Remote - LATAM` · `Remote, LATAM` · `Remote, LatAm` · `Brasil (Remote)` · `Brazil/Remote` · `Brazil - Remote` · `Brazil ; India` · `BR-Brazil-Remote` · `Global` · `Global Remote` · `Remote Global` · `Remote International` · `Remote • Anywhere` · `Americas` · `Remote – Americas ;` · `Remote (North/South America)` · `Kyiv, Ukraine ; Global` · `Global ; Colombia ; Argentina ; Philippines ; Mexico` · `LATAM ; Eastern Europe` · `Remote - Argentina; Brazil;Chile; Costa Rica` · `Remote - Romania, EMEA, Brazil, Poland, Romania` · `Remote - California, Colorado ; Remote - Uruguay ; LATAM - Brazil ; Remote - Mexico` |
| `BARE` | `Remote` · `Remote ` · `Remote First` · `Distributed` · `` (empty) |
| `RESTRICTED` | `Americas (USA or Canada)` · `Europe - Anywhere ; Greece ; Belgium` · `Anywhere in the US` · `Remote - Americas, USA` · `Remote - US` · `Remote (Canada)` · `Remote (United States \| Canada)` · `United States` · `US, Remote` · `US-CA-Menlo Park` · `Europe` · `Hamburg or Berlin` · `India Remote` · `Boston, Massachusetts, United States; Remote U.S.` · `Chicago or Remote*` |

## Appendix — how the numbers were measured

Live APIs on 2026-09-18, random samples with fixed seeds. Board sampling: Greenhouse 300/500/900/1,200
boards, Ashby 200/400/600/900, Lever 250 — for location classes, live rates and payload sizes. The
description regexes were tuned on 120 bare-"Remote" Greenhouse engineering postings fetched through
the per-job detail endpoint, reading **every** match rather than trusting the count. Rate limiting was
probed with a 900-request sustained Ashby burst and a 300-request Greenhouse burst at 20-way
concurrency. Workable, Jobicy, Himalayas, WeWorkRemotely, Working Nomads and HN were exhausted or
paged directly. Baseline pool counts came from the live DB through `build_profile_filter`.

Known weaknesses: board samples are clustered (a few very large boards dominate remote counts), which
is why §3.1 carries a 1.1–2.2k range rather than a point estimate; Ashby's explicit hits are small-n
(25 in 1,168); "engineering" is a title regex, so sales-engineer-style roles still leak; and the
absence of 429s in one burst is not a guarantee across a full 11.5k-request run, which is what §1.7's
retry change insures against. Re-measure before tuning any threshold.
