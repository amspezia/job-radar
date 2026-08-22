# Ingestion Expansion — Design

**Status:** proposal, not yet implemented. **Motivation:** the corpus is too narrow to actually
job-search with day to day — 6 sources, ~11.4k rows, and (per tonight's separate investigation)
a meaningful fraction of `himalayas` rows are the same freelance/gig posting re-listed per
country. More *distinct, trustworthy, current* postings matters more than raw row count.

Every number and endpoint below was verified live tonight (2026-08-20), not recalled from
training data — this codebase's own discipline (see `docs/STATUS.md`, `docs/ASSESSMENT.md`) is
"verify empirically, don't guess," and API shapes/ToS drift over time.

## 1. Current state

| Source | `source_type` | Mechanism | Rows |
|---|---|---|---|
| `greenhouse` | `board` | `discovery.py` finds live board tokens, `boards-api.greenhouse.io/v1/boards/{token}/jobs` | 4695 |
| `himalayas` | `aggregator` | 15 role-keyword searches × `himalayas.app/jobs/api/search`, paginated | 4559 |
| `lever` | `board` | `discovery.py` + `api.lever.co/v0/postings/{token}` | 1330 |
| `arbeitnow` | `aggregator` | `arbeitnow.com/api/job-board-api`, paginated | 429 |
| `getonboard` | `board` | `discovery.py` + GetOnBoard's JSON:API board endpoint | 268 |
| `remotive` | `aggregator` | `remotive.com/api/remote-jobs` | 60 |

Two mechanisms already exist:
- **`SourceAdapter` (`fetch()` + `map()`)** — one aggregator API, paginate, normalize. Used by
  `remotive`/`arbeitnow`/`himalayas`.
- **`discovery.py` + per-ATS adapter** — a shared helper discovers live company board tokens from
  a public dataset (see §2), verifies each resolves with real jobs, caches the list, then the
  adapter fetches every token's board. Used by `greenhouse`/`lever`/`getonboard`. This is the
  cheaper mechanism to extend: a new ATS adapter is a `link_regex` + `board_url` + `has_jobs`
  predicate + a `map()` — no new discovery logic.

## 2. What `discovery.py` actually discovers

`discovery.py` pulls company→ATS-link mappings from a public GitHub dataset
([`outscal/OpenJobs`](https://github.com/outscal/OpenJobs)), filters to `industry_category ==
"tech"` (2534 of 12144 companies — the dataset is ~69% gaming studios, already correctly
excluded), and regex-extracts board tokens per ATS. Pulled and analyzed the live dataset tonight;
link counts among the `tech`-tagged companies:

```
622+233   boards.greenhouse.io / job-boards.greenhouse.io   (greenhouse — implemented)
322+17    jobs.lever.co / jobs.eu.lever.co                  (lever — implemented)
234       jobs.ashbyhq.com                                  (ashby — NOT implemented)
229+22    apply.workable.com / jobs.workable.com             (workable — NOT implemented)
168+136   jobs.smartrecruiters.com / careers.smartrecruiters.com (smartrecruiters — NOT implemented)
183       *.wdN.myworkdayjobs.com                            (workday — NOT implemented)
110       jobs.jobvite.com                                   (jobvite — no public JSON found)
```

**Ashby, Workable, and SmartRecruiters are sitting right there, unused, in a mechanism already
built and paid for.** Each has a public, unauthenticated, JSON board API — verified live against
a real company token pulled from the same dataset:

| ATS | Verified endpoint | Auth | Shape |
|---|---|---|---|
| Ashby | `GET api.ashbyhq.com/posting-api/job-board/{token}` | none | `{jobs: [...], apiVersion}` — clean, includes `descriptionHtml`/`descriptionPlain` directly |
| Workable | `GET apply.workable.com/api/v1/widget/accounts/{token}` | none | `{name, description, jobs: [...]}` |
| SmartRecruiters | `GET api.smartrecruiters.com/v1/companies/{token}/postings` | none | `{offset, limit, totalFound, content: [...]}` — paginated like Greenhouse |

Workday works too, but is structurally heavier: `POST {tenant}.wd{N}.myworkdayjobs.com/wday/cxs/
{tenant}/{site}/jobs` with a JSON search body (`{"appliedFacets":{},"limit":N,"offset":0,
"searchText":""}`), and needs a **two-part** token (tenant *and* site slug, e.g.
`valeo` + `valeo_jobs`) that `discovery.py`'s single-token-into-URL-template design doesn't
support today. One real tenant tested (Valeo) returned **1088 open postings** — Workday is used
by large enterprises, so per-tenant volume is disproportionately high versus the boutique-startup
skew of Greenhouse/Lever/Ashby. Worth it, but it's a bigger lift than the other three.

Jobvite: 110 companies in the dataset, but the obvious guessed endpoint
(`jobs.jobvite.com/{token}/jobs.json`) returned an HTML page live, not JSON — no public API found
without further digging. Lowest priority of the ATS candidates given the smaller count and
unclear mechanism.

## 3. New aggregator candidates (same `SourceAdapter` pattern)

Tested live, unauthenticated unless noted:

| Source | Endpoint | Auth | Notes |
|---|---|---|---|
| **RemoteOK** | `GET remoteok.com/api` | none | ~100 postings/call. Response embeds its own ToS text: requires a followed backlink to RemoteOK crediting them as source — a real, checkable term, not boilerplate; must actually implement the attribution if this ships. |
| **The Muse** | `GET themuse.com/api/public/jobs?category=Software+Engineering` | none for light use | **100,743 total jobs**, paginated (5038 pages), filterable by category/level. By far the largest single source tested. Official public API, not scraping. |
| **WeWorkRemotely** | `GET weworkremotely.com/categories/remote-programming-jobs.rss` | none | RSS/XML, not JSON — needs an XML parser (`xml.etree` or reuse `BeautifulSoup`, already a dependency via `normalize.py`'s HTML handling) rather than `resp.json()`. Full HTML job body embedded in `<description>`. |
| **Adzuna** | `GET api.adzuna.com/v1/api/jobs/{country}/search/{page}` | `app_id`+`app_key`, free signup | Aggregates many boards across 12 countries — a different kind of breadth (cross-board dedup already done by Adzuna itself). Free tier: **1000 calls/month** (~33/day) — a real budget constraint, not generous; needs deliberate call-count management (e.g. one scheduled run/day, not ad hoc dev runs) if adopted. |

**Explicitly not recommended: LinkedIn, Indeed.** Checked current ToS for both — LinkedIn's
Service Terms and Indeed's Terms of Service both explicitly prohibit scraping/automated
collection of job postings; the *hiQ v. LinkedIn* litigation resolved the CFAA question but
LinkedIn still won on breach-of-contract grounds, so ToS prohibition remains the operative risk
regardless of the criminal-law question. Neither offers a public job-search API to individual
developers (LinkedIn's public APIs cover content/messaging, not job search). Given this project's
own CLAUDE.md hygiene bar, these are out of scope — not worth the legal/ethical exposure for a
portfolio project when Ashby/Workable/SmartRecruiters/The Muse alone dwarf what either would add.

Sources: [Adzuna developer docs](https://developer.adzuna.com/), [LinkedIn Service
Terms](https://www.linkedin.com/legal/l/service-terms), [Indeed Terms of
Service](https://www.indeed.com/legal).

## 4. A prerequisite this expansion makes worse if skipped

Separately discovered tonight (not part of this research, but directly relevant): `Job` identity
(`ingest/dedup.py::content_hash`) keys on `company + title + location`. Sources that post the
same remote role once per eligible country (Mindrift, CapsLock, Bjak, Bluelight Consulting — up
to 27 rows for one real posting, confirmed in the live DB) aren't caught by this, because each
country counts as a different identity. **Adding more sources without fixing this first
multiplies the pollution** — more sources means more per-country-reposting patterns feeding the
same broken identity check, further crowding out genuinely distinct postings from the retrieval
candidate pool. Recommend fixing dedup (exclude `location` from `content_hash` when
`job.remote is True`) *before or alongside* Phase 1 below, not after.

## 5. Recommended phasing

**Phase 1 — cheapest, reuses 100% of existing infrastructure.** Add `AshbyAdapter`,
`WorkableAdapter`, `SmartRecruitersAdapter`, each a `discovery.py`-based adapter following the
exact shape of `greenhouse.py`/`lever.py`/`getonboard.py` (new `link_regex`/`board_url`/
`has_jobs`/`map()`, register in `runner.py`'s `ENABLED_ADAPTERS`, own `_TOKENS_CACHE` path under
`data/`). No new dependencies, no new mechanism to design. ~700 more companies' worth of
candidate coverage across the three. Fix the dedup bug (§4) in the same pass.

**Phase 2 — new aggregator mechanism, still `SourceAdapter`-shaped.** Add `RemoteOKAdapter`
(simple JSON, remember the attribution requirement) and `TheMuseAdapter` (largest single volume
increase available; needs pagination logic and a category filter to stay relevant — "Software
Engineering" plus whatever adjacent categories the CV/profile keywords suggest, not the whole
100k-job firehose). `WeWorkRemotelyAdapter` if the RSS/XML parsing is worth it over the two JSON
sources.

**Phase 3 — optional, bigger lift.** `WorkdayAdapter`: needs `discovery.py` extended to support
two-part tokens (tenant + site) rather than a single substitution, and a POST-based fetch loop
with offset pagination instead of the GET-based pattern every current adapter uses. Justified by
volume (enterprise tenants run into the thousands of postings each) but is a real design task,
not a copy-paste of the existing ATS adapter shape.

**Phase 4 — optional, budget-constrained.** `AdzunaAdapter`: requires `.env.example` additions
(`ADZUNA_APP_ID`, `ADZUNA_APP_KEY`), and the 1000-call/month free tier means it can't be hit on
every dev-driven `job-radar-ingest` run the way the unauthenticated sources can — needs either a
call budget/rate limiter or restriction to the scheduled run only. Lower priority given
Phase 1+2 alone already substantially outweigh what Adzuna adds at this call volume.

## 6. Open questions before implementation starts

- Does The Muse's category/level filtering need to happen at fetch-time (narrower API calls,
  fewer irrelevant postings to embed/extract) or post-fetch? Given `ingest/pipeline.py`'s
  per-posting LLM extraction cost (confirmed tonight: ~0.6–0.7s/posting once Ollama is warm,
  but real at 100k-job scale), fetch-time filtering matters — pulling all 100k and discarding
  most after paying the extract+embed cost would be wasteful.
- Should `_MAX_PAGES`-style volume caps (already used by `arbeitnow.py`/`himalayas.py`) apply to
  The Muse given its 5038-page depth, to bound a single ingestion run's wall-clock time?
