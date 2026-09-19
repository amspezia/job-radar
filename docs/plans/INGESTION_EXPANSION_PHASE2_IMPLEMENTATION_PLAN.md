# Ingestion Expansion, Phase 2 — Implementation Plan

Locks down `INGESTION_EXPANSION_DESIGN.md` §5 Phase 2: new `SourceAdapter`-shaped aggregators,
still no new mechanism beyond what `remotive.py`/`arbeitnow.py`/`himalayas.py` already
establish (`fetch()` + `map()`, registered in `runner.py`). Explicitly **not** in scope: Phase 3
(Workday), Phase 4 (Adzuna), or any retrieval/geo-filter changes (a separate, already-tracked
problem — see `INGESTION_EXPANSION_PHASE1_IMPLEMENTATION_PLAN.md`'s follow-up notes).

Every endpoint below was hit live tonight (2026-08-22), including RemoteOK, which the design
doc had only ToS/shape-checked, not content-checked — that check changes this phase's scope.

## 1. RemoteOK — dropped from Phase 2, not implemented

The design doc listed RemoteOK as a simple, `~100 postings/call` addition needing only an
attribution backlink. Pulling the live public `remoteok.com/api` response tonight tells a
different story:

- **100/100 sampled postings carry an anti-scraping canary phrase** baked directly into the
  `description` field: *"Please mention the word **WONDERFUL** and tag `RM...` when applying to
  show you read the job post completely."* Every posting has a different encoded tag. This is a
  bot-detection watermark, not incidental copy — ingesting it means every stored description in
  this source is contaminated with a trap phrase designed to catch exactly this kind of
  automated consumption.
- **The postings themselves are not tech jobs.** Titles pulled live: "Removalist Offsider"
  (a moving company, Albury AU), "Beauty merchandiser" (Shoppers Drug Mart, Kingston), "Store
  Manager Bunbury" (Retail FX), "Roupeiro" (a Brazilian costume-department role). These are
  local retail/hospitality postings, not remote dev roles.
- **The `tags` field is not usable as a relevance filter.** "Removalist Offsider" is tagged
  `golang`; "Beauty merchandiser" is tagged `dev`, `exec`. Of 100 postings, 57 carry at least one
  tech-sounding tag despite the title/company confirming they aren't tech postings at all —
  tag-based filtering would not recover a clean subset.
- Confirmed this isn't a User-Agent artifact: re-fetched with a standard browser UA, same result.

**Decision: do not implement `RemoteOKAdapter`.** Whatever the current state of RemoteOK's
public API is (a deliberately degraded feed for unauthenticated/scraping clients, or a genuine
change since the design doc's 2026-08-20 check), the live data fails this project's own
hygiene bar — CLAUDE.md's "more distinct, trustworthy, current postings matters more than raw
row count" is exactly the standard this source fails. Correct
`INGESTION_EXPANSION_DESIGN.md` §3's RemoteOK row to note this when this plan lands; the
attribution requirement noted there is now moot.

## 2. `TheMuseAdapter`

### 2.1 Verified endpoint and fetch-time filtering

`GET themuse.com/api/public/jobs`, params `category`, `location`, `page` (0-indexed). No auth
for this volume. Response: `{page, page_count, results: [...]}` — no `aggregations` facets
despite being listed in the top-level keys (empty dict live), so category/location values must
be tested directly rather than discovered from the API itself.

This directly answers the design doc's §6 open question ("fetch-time or post-fetch filtering?")
— **fetch-time, using both `category` and `location` together**, measured live:

| Query | `page_count` (×20/page) | Sample relevance |
|---|---|---|
| *(no filter)* | 5037 | the full firehose — not usable |
| `category=Software Engineering` | (not isolated alone; see combined row) | — |
| `category=Software Engineering&location=Flexible / Remote` | **61** (~1,220 jobs) | mostly relevant SWE roles, some mistagged noise (one "Lead Electrical Superintendent" categorized as Software Engineering) |
| `category=Data Science&location=Flexible / Remote` | **1** (~20 jobs) | small, cheap to add, some overlap with profile's data/ML keywords |
| `category=Data and Analytics&location=Flexible / Remote` | 53 (~1,060 jobs) | noisier — "Pricing and Revenue Planner", "Senior Value Engineer" — not adopted |
| `category=IT` / `category=Engineering` | 0 | not valid category strings on this API — do not use |

**Decision: query `category="Software Engineering"` and `category="Data Science"`**, both with
`location="Flexible / Remote"`, mirroring `himalayas.py`'s multi-query-list shape
(`_QUERIES`-style constant, here `_CATEGORIES`). Total volume ≈1,240 postings across ~62 page
fetches per run — far below the 5,037-page firehose, so **no `_MAX_PAGES` cap is needed** (same
reasoning Phase 1 landed on for SmartRecruiters: the filter itself is the bound, not an arbitrary
page ceiling). `location="Flexible / Remote"` is passed as a literal string; `httpx` percent-
encodes the space and `/` correctly when passed via `params=` (confirmed live — no manual
encoding needed).

Some mistagged noise (an Electrical Superintendent under "Software Engineering") will get
through; this is The Muse's own category tagging, not something `fetch()` can filter further
without discarding real results too. Accepted as-is, same spirit as Phase 1 accepting Ashby's
lossy company-name cosmetic limitation — not worth over-engineering a heuristic title filter for
one observed mistag.

Pagination loop shape (mirrors `himalayas.py`'s per-query break condition):
```
for category in _CATEGORIES:
    page = 0
    while True:
        payload = GET(..., category=category, location="Flexible / Remote", page=page)
        results = payload["results"]
        jobs.extend(results)
        page += 1
        if not results or page >= payload["page_count"]:
            break
```
No rate-limiting observed across ~10 live test calls tonight, unlike Himalayas (which needed
429-retry handling in Phase 1's predecessor work). Not proactively adding retry-backoff without
evidence — if the first real `job-radar-ingest` run after this ships logs 429s, add the same
`_RATE_LIMIT_RETRIES`/backoff shape `himalayas.py` already has; don't build it preemptively for a
problem not yet observed (contra Himalayas, which *was* observed rate-limiting during its own
build).

### 2.2 `map()` field-by-field

Real response shape, one result:
```
contents, name, type, publication_date, short_name, model_type, id, locations, categories,
levels, tags, refs, company
```

| `NormalizedJob` field | Source | Notes |
|---|---|---|
| `source_id` | `id` | int in the payload; cast to `str` |
| `url` | `refs.landing_page` | |
| `title` | `name` | |
| `company` | `company.name` | present directly, unlike Ashby's gap |
| `description` | `contents` through `html_to_text` | HTML, same as Greenhouse/Workable |
| `salary_min/max/currency` | — | not present in any sampled payload; `None` |
| `location` | join every `locations[].name` with `", "` | a hybrid posting can list both `"Flexible / Remote"` and `"New York, NY"` — preserve both rather than picking one, same reasoning as Workable's `locations` array decision in Phase 1 |
| `job_type` | — | no employment-type field in this payload (no "Full-time"/"Contract" equivalent found); leave `None` rather than guessing from `type` (which means "external vs. on-Muse apply flow", not employment type) |
| `remote` | `True`, unconditionally | `fetch()` only queries `location="Flexible / Remote"`, so every returned posting already matched that facet — no separate per-posting check needed |
| `published_at` | `publication_date` | ISO 8601 with `Z` suffix; Python 3.12's `fromisoformat` accepts `Z` directly (same note as Phase 1's SmartRecruiters `releasedDate`) |

`levels`/`tags`/`categories` are not mapped to any `NormalizedJob` field — seniority is derived
downstream by the fit pipeline's own extraction, not sourced from adapter metadata by any
existing adapter, so this isn't a gap specific to The Muse.

## 3. `WeWorkRemotelyAdapter`

### 3.1 Verified feed and why it's structurally different from every adapter so far

`GET weworkremotely.com/categories/remote-programming-jobs.rss` — real RSS/XML, not JSON.
Confirmed live: **25 items per fetch, always the current snapshot — no pagination, no way to
page further back.** This makes it a fundamentally different shape from every other Phase 1/2
source: there's no bulk historical pull available, only "whatever is live right now." Its value
compounds slowly across repeated runs (each run's dedup against `existing_hashes` collapses
whatever already landed), not from one large ingest. Worth setting expectations here: this
source adds at most ~25 candidate rows per run, likely fewer after the first run once overlap
sets in — nothing like Ashby's 811 or SmartRecruiters' 1,623.

One stale item was observed in the live feed (`pubDate: Mon, 13 May 2024...`, a "Storetasker"
evergreen freelance listing sitting alongside items from the last few weeks) — WWR appears to
keep a small number of always-on postings pinned in the feed. Not filtered out: `content_hash`
dedup and the fit pipeline's own `max_age_days` filtering are the right place to age this out,
not adapter-level logic guessing at which postings are "evergreen."

Parse with the stdlib `xml.etree.ElementTree` — confirmed sufficient for this feed's flat
`<item>` structure; no need for `BeautifulSoup` at the RSS layer (it's still used inside
`map()` for the HTML-embedded `<description>`, same as every other adapter's `html_to_text`
call).

Real `<item>` fields, confirmed live: `title, region, category, description, pubDate, guid,
link`. (A `<media:content>` logo URL is also present but not mapped — no `NormalizedJob` field
for it, same as every adapter ignoring logo URLs today.)

### 3.2 `fetch()` shape

```python
async def fetch(self) -> list[dict]:
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
        resp = await client.get(_RSS_URL)
        resp.raise_for_status()
    root = ET.fromstring(resp.text)
    items = [_item_to_dict(item) for item in root.findall(".//item")]
    return [i for i in items if (i.get("region") or "").strip().lower() == "anywhere in the world"]
```

The region filter is defensive rather than load-bearing — all 25 live items tonight were
`"Anywhere in the World"` (this specific feed URL is already the "remote-programming-jobs"
category), but explicit filtering matches every other adapter's pattern of asserting remote
status in `fetch()` rather than assuming the endpoint's name guarantees it forever.

### 3.3 `map()` field-by-field

The one real parsing wrinkle: **WWR's `title` is `"Company: Job Title"` as a single string** —
there is no separate company field anywhere in the item. Split on the *first* `": "` only
(`str.split(": ", 1)`), so a title containing a second colon (none observed live, but plausible)
doesn't break the split. Fallback when no `": "` is present at all: treat the whole string as
`title` and leave `company` as an empty string — matches this codebase's "store unknown, never
fabricate" rule (`normalize.py`'s `parse_salary` docstring) rather than inventing a placeholder
like `"Unknown"`.

| `NormalizedJob` field | Source | Notes |
|---|---|---|
| `source_id` | `guid` | RSS has no numeric ID; `guid` is the canonical job URL, stable and unique |
| `url` | `link` | |
| `title` | second half of `title.split(": ", 1)` | |
| `company` | first half of the same split | `""` when no `": "` present |
| `description` | `description` through `html_to_text` | HTML, embeds a logo `<img>` tag that `html_to_text` will render as empty/whitespace — harmless |
| `salary_min/max/currency` | — | not present; `None` |
| `location` | `None` | remote-only feed; no city/region beyond the already-filtered `"Anywhere in the World"`, which isn't a useful stored value (same call `himalayas.py` doesn't make — wait, Himalayas does store `locationRestrictions`; WWR's `region` here is constant across every row post-filter, so storing it would add zero information) |
| `job_type` | — | `category` (e.g. `"Full-Stack Programming"`) is a **role domain**, not an employment type — do not conflate with `job_type`, which every other adapter uses for Full-time/Contract-style values; leave `None` |
| `remote` | `True` | `fetch()` already filtered to `"Anywhere in the World"` |
| `published_at` | `pubDate`, parsed with `email.utils.parsedate_to_datetime` | **RFC 822 format** (`"Mon, 13 May 2024 03:14:30 +0000"`), not ISO 8601 — every other adapter in this codebase parses `published_at` with `datetime.fromisoformat`; this is the first one that can't, because RSS's date format simply isn't ISO. `parsedate_to_datetime` is stdlib (`email.utils`), already tz-aware on output, no extra dependency |

## 4. Registration

Add `TheMuseAdapter()` and `WeWorkRemotelyAdapter()` to `runner.py::ENABLED_ADAPTERS`, alongside
the existing nine. No `discovery.py` involvement (both are aggregator-mechanism sources, same
family as `remotive`/`arbeitnow`/`himalayas`) — no changes needed there.

## 5. Testing plan

Mirror the Phase 1 pattern (`tests/test_greenhouse.py`-style fixture + test pairing):

- `tests/fixtures/themuse_jobs.json`, `tests/test_themuse.py` — cover: a job with a single
  location maps correctly; a hybrid job with two `locations` entries joins both into one string;
  `contents` HTML is stripped; `publication_date`'s `Z`-suffixed ISO parses correctly;
  `source_id` is cast to `str` from the payload's int `id`.
- `tests/fixtures/weworkremotely_items.json`, `tests/test_weworkremotely.py` — this fixture
  holds already-parsed item dicts (the shape `fetch()` hands to `map()`), not raw XML, matching
  how `test_smartrecruiters.py` fixtures the merged list+detail shape rather than raw HTTP
  bodies. Cover: `"Company: Title"` splits correctly; a title with **no** `": "` falls back to
  empty `company` rather than raising; `pubDate`'s RFC 822 string parses to a tz-aware
  `datetime`; `category` is never written to `job_type`.
  A second, small test — `test_fetch_parses_real_shaped_rss_and_filters_region` — feeds a
  monkeypatched client a canned XML string (2-3 `<item>`s, one with a non-`"Anywhere in the
  World"` `<region>`) and asserts `fetch()` both parses XML into the right dicts and drops the
  non-remote item.
- No new tests needed for `discovery.py` or `runner.py` — neither adapter touches that
  mechanism, and `runner.py`'s per-source failure isolation is already generically tested.

## 6. Rollout / verification steps, in order

1. Implement `themuse.py` first — higher volume, same aggregator shape already proven three
   times over (`remotive`/`arbeitnow`/`himalayas`), lower structural risk than WWR's XML parsing.
   Verify end-to-end against the real API before moving on.
2. Implement `weworkremotely.py` — the RSS parsing and the `pubDate`/title-split edge cases are
   the only genuinely new code shapes in this phase; keep it isolated from TheMuse's change so a
   bug here doesn't block the higher-yield adapter.
3. Register both in `runner.py`.
4. Run `uv run job-radar-ingest`; confirm both sources log `Finished ingestion for source=...`
   with no tracebacks.
5. Query per-source `Job` counts (`select(Job.source, func.count(Job.id)).group_by(Job.source)`)
   to confirm nonzero rows for `themuse` and `weworkremotely`.
6. Spot-check a handful of real inserted rows per source the same way Phase 1's audit did
   (sample `title`/`company`/`location`/`description` directly from the DB) — this phase's whole
   motivation (RemoteOK) was a live-content check catching what a shape-only check missed, so
   don't skip repeating that check after real rows land.
7. Full suite (`uv run pytest`) + lint (`ruff check .` / `ruff format --check .`) clean before
   considering Phase 2 done.

## 7. Open questions — resolved

- **Fetch-time vs. post-fetch category filtering for The Muse** (design doc §6): resolved —
  fetch-time, via `category` + `location="Flexible / Remote"` together, cutting ~5,037 pages to
  ~62. 
- **Whether a `_MAX_PAGES` cap is needed for The Muse** (design doc §6): resolved — no, the
  category+location filter is already the bound (~1,240 jobs total), same reasoning Phase 1 used
  to skip a page cap for SmartRecruiters.
- **RemoteOK's viability** (not an open question in the design doc, but should have been):
  resolved by live content-check — not viable, dropped from this phase entirely.

Nothing here is blocked on further research. The one thing worth flagging for whoever picks this
up: The Muse's `location="Flexible / Remote"` facet is a string literal discovered by testing,
not documented in any public API reference found — if The Muse ever renames this facet value,
`fetch()` would silently start returning 0 results (an empty `page_count`, not an error), so the
rollout step 5 count check above is the thing that would actually catch that, not a crash.
