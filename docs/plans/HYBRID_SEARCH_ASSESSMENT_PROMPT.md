# Prompt: Hybrid Search Deep Assessment

> This file is a prompt to be handed verbatim (or with light edits) to a capable AI
> (Claude Opus, GPT-4o, Gemini Ultra, etc.) for a deep, research-grounded assessment.
> It contains the full current implementation so the AI can reason from actual code.

---

## Your mission

You are a senior information-retrieval engineer with deep expertise in hybrid search,
dense retrieval, and production search systems. You are auditing the hybrid search engine
of **Job Radar**, a personal job-matching system. Your job is to:

1. **Understand the system deeply** from the code below — every design decision, every
   implicit assumption, every known gap.
2. **Compare it against state-of-the-art production hybrid search systems** — cite real
   published cases (papers, engineering blogs, open-source projects) and explain
   specifically how they differ from this implementation and why those differences matter.
3. **Propose a thorough, deterministic evaluation framework** — offline metrics, a
   labeling protocol, a parameter sweep design, and a concrete before/after decision
   rule so the team knows objectively when a change is an improvement.
4. **Propose concrete, prioritized changes** — not generic advice; specific changes to the
   code below with the expected directional effect on each metric and the rationale tied
   to a real reference.

Do not be diplomatic. Flag everything that is suboptimal, under-specified, or inconsistent.
This assessment will directly drive implementation decisions.

---

## Context: what Job Radar is

A personal job-search assistant. It ingests job postings from multiple boards (Lever,
Greenhouse, Himalayas, Remotive, etc.) into Postgres (~5,000–10,000 rows, growing). When
the user runs the fit pipeline, it retrieves the top-N most relevant postings for their
profile and then scores each one with an LLM. **Retrieval is the gate**: if a good job
is not retrieved, it is never scored. The retrieval budget is `limit=20` final results
from a `pool=50` per arm. Getting retrieval right matters more than anything else in the
system.

The candidate has a single stored `Profile` with: target titles, tech stack keywords,
domain keywords, seniority level, work history, and a precomputed 768-dim embedding of
their full CV text (`cv_embedding`). The embedding model is `nomic-embed-text` running
locally via Ollama.

---

## The full current implementation

### `retrieval/search.py` — orchestrator

```python
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from job_radar.adapters.embeddings import embed
from job_radar.db.models import Job
from job_radar.retrieval.fts import search_fts
from job_radar.retrieval.fusion import reciprocal_rank_fusion
from job_radar.retrieval.vector import search_vector


async def search(
    session: AsyncSession,
    query: str,
    *,
    limit: int = 20,
    pool: int = 50,
    extra_filter: ColumnElement[bool] | None = None,
    profile_embedding: list[float] | None = None,
) -> list[Job]:
    """Hybrid search fusing up to three rankers via RRF.

    Arms, each contributing only when it has signal:
    - lexical (FTS) over the query text,
    - semantic over the embedded query text,
    - semantic over a precomputed profile/CV embedding (candidate-anchored).

    The first two are skipped when the query is blank; the CV arm is skipped when
    no embedding is supplied. With no arms (blank query and no embedding) the
    result is empty rather than an unfiltered dump.
    """
    arms: list[list[tuple[UUID, float]]] = []

    if query and query.strip():
        arms.append(await search_fts(session, query, pool, extra_filter))
        arms.append(await search_vector(session, await embed(query), pool, extra_filter))

    if profile_embedding is not None:
        arms.append(await search_vector(session, profile_embedding, pool, extra_filter))

    fused = reciprocal_rank_fusion(arms, limit=limit)
    if not fused:
        return []

    ids = [job_id for job_id, _ in fused]
    rows = (await session.execute(select(Job).where(Job.id.in_(ids)))).scalars().all()
    by_id = {job.id: job for job in rows}

    return [by_id[job_id] for job_id in ids]
```

### `retrieval/fts.py` — lexical arm

```python
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from job_radar.db.models import Job

_FTS_CONFIG = "simple"


async def search_fts(
    session: AsyncSession,
    query_text: str,
    limit: int,
    extra_filter: ColumnElement[bool] | None = None,
) -> list[tuple[UUID, float]]:
    tsquery = func.websearch_to_tsquery(_FTS_CONFIG, query_text)
    rank = func.ts_rank(Job.search_vector, tsquery)
    stmt = (
        select(Job.id, rank)
        .where(Job.search_vector.bool_op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    if extra_filter is not None:
        stmt = stmt.where(extra_filter)
    rows = (await session.execute(stmt)).all()
    return [(job_id, float(score)) for job_id, score in rows]
```

### `retrieval/vector.py` — dense arm

```python
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from job_radar.db.models import Job


async def search_vector(
    session: AsyncSession,
    query_embedding: list[float],
    limit: int,
    extra_filter: ColumnElement[bool] | None = None,
) -> list[tuple[UUID, float]]:
    similarity = 1 - Job.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Job.id, similarity)
        .where(Job.embedding.is_not(None))
        .order_by(similarity.desc())
        .limit(limit)
    )
    if extra_filter is not None:
        stmt = stmt.where(extra_filter)
    rows = (await session.execute(stmt)).all()
    return [(job_id, float(score)) for job_id, score in rows]
```

### `retrieval/fusion.py` — RRF

```python
from collections import defaultdict
from uuid import UUID


def reciprocal_rank_fusion(
    rankings: list[list[tuple[UUID, float]]], *, k: int = 60, limit: int | None = None
) -> list[tuple[UUID, float]]:
    fusion_rank: defaultdict[UUID, float] = defaultdict(float)

    for ranking in rankings:
        for rank, (job_id, _) in enumerate(ranking, start=1):
            fusion_rank[job_id] += 1 / (k + rank)

    fused = sorted(fusion_rank.items(), key=lambda item: (-item[1], str(item[0])))
    return fused[:limit]
```

### `retrieval/filters.py` — pre-retrieval hard constraints (WHERE)

```python
from sqlalchemy import and_, or_
from sqlalchemy.sql.elements import ColumnElement

from job_radar.db.models import Job, Profile
from job_radar.retrieval.geo import build_geo_filter
from job_radar.retrieval.seniority import LADDER, allowed_levels


def build_profile_filter(
    profile: Profile, *, levels: list[str] | None = None
) -> ColumnElement[bool] | None:
    clauses: list[ColumnElement[bool]] = []

    keywords = (profile.location_rules or {}).get("allowed_keywords")
    if keywords:
        clauses.append(build_geo_filter(keywords))

    if profile.remote_required:
        clauses.append(Job.remote.is_(True))

    allowed = levels or allowed_levels(profile)
    if set(allowed) < set(LADDER):
        clauses.append(or_(Job.seniority.is_(None), Job.seniority.in_(allowed)))

    if profile.salary_floor is not None:
        keep = or_(Job.salary_max.is_(None), Job.salary_max >= profile.salary_floor)
        if profile.currency is not None:
            keep = or_(keep, Job.currency.is_distinct_from(profile.currency))
        clauses.append(keep)

    if not clauses:
        return None
    return and_(*clauses)
```

### `fit/pipeline.py` — caller that builds the query and passes the profile embedding

```python
def build_query(profile: Profile) -> str:
    """Build a retrieval query from the profile when the caller has none."""
    keywords = profile.domains_keywords or {}
    parts = [
        *(profile.target_titles or []),
        *keywords.get("tech_stack", []),
        *keywords.get("domains", []),
    ]
    return " ".join(parts)


async def run_fit_pipeline(session, query=None, *, limit=20, levels=None):
    profile = await _load_profile(session)
    query = query or build_query(profile)
    profile_filter = build_profile_filter(profile, levels=levels)
    jobs = await search(
        session,
        query,
        limit=limit,
        extra_filter=profile_filter,
        profile_embedding=profile.cv_embedding,
    )
    # ... fit scoring follows
```

### `adapters/embeddings.py` — embedding client

```python
async def embed(text: str) -> list[float]:
    payload = {"model": settings.embedding_model, "input": text}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url=f"{settings.ollama_base_url}/api/embed", json=payload)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]
```

### `db/models.py` — relevant schema fields

```python
class Job(Base):
    __tablename__ = "jobs"
    # ...
    title: Mapped[str]                  # String(255)
    description: Mapped[str]            # Text (full job description)
    seniority: Mapped[str | None]       # String(50), normalized at ingest from title
    location: Mapped[str | None]        # Text
    remote: Mapped[bool]
    salary_min: Mapped[int | None]
    salary_max: Mapped[int | None]
    currency: Mapped[str | None]
    embedding: Mapped[list[float] | None]  # Vector(768), embed(title + "\n" + description) at ingest
    search_vector: Mapped[str]  # TSVECTOR, GENERATED ALWAYS AS:
        # setweight(to_tsvector('simple', title), 'A') ||
        # setweight(to_tsvector('simple', description), 'B')

class Profile(Base):
    __tablename__ = "profile"
    # ...
    seniority: Mapped[str]
    target_titles: Mapped[dict]         # list of job title strings
    domains_keywords: Mapped[dict]      # {"tech_stack": [...], "domains": [...]}
    cv_text: Mapped[str]                # full CV text
    cv_embedding: Mapped[list[float] | None]  # Vector(768), embed(cv_text) at load time
    location_rules: Mapped[dict]        # {"allowed_keywords": ["brazil", "worldwide", ...]}
    seniority_rules: Mapped[dict | None]  # {"allowed_levels": ["senior", "staff"]}
    remote_required: Mapped[bool]
    salary_floor: Mapped[int | None]
```

**Indexes:**
- GIN index on `jobs.search_vector`
- HNSW index on `jobs.embedding` (cosine)
- HNSW index on `profile.cv_embedding` (cosine)

**Embedding model:** `nomic-embed-text` (768 dims) via Ollama, no task prefixes used
anywhere (`search_query:` / `search_document:` prefixes are a known gap).

---

## Known gaps (already identified, not yet addressed)

The team is aware of these — do not re-identify them as discoveries, but do factor them
into your proposal:

1. **No nomic task prefixes** (`search_query:` / `search_document:`) — every `embed()`
   call passes raw text. Fixing this requires re-embedding the entire corpus and the CV.
2. **FTS config is `'simple'`** (no stemming) — `engineer` ≠ `engineering`,
   `developer` ≠ `developers`. Fixing this requires regenerating the `search_vector`
   generated column and its GIN index.
3. **Implicit AND in FTS** — `websearch_to_tsquery('simple', "Backend Engineer Python")`
   requires all terms to appear; the OR semantics that would make tech-stack terms broaden
   rather than narrow retrieval are not yet implemented.
4. **All three RRF arms have equal implicit weight** — there is no per-arm weight; the
   choice of `k=60` and `pool=50` are untuned defaults.
5. **`pool` is per-arm** — the FTS arm and each vector arm independently return up to 50
   candidates; the union before RRF can be up to 150 items. The interaction between
   per-arm pool size and final `limit=20` has not been analyzed.

---

## What I need from you

### Part 1: Deep technical assessment

For each component — FTS arm, query-vector arm, CV-vector arm, fusion, query
construction, embedding usage, pre-retrieval filtering — answer:

- What does this implementation do well?
- What is wrong, suboptimal, or architecturally inconsistent with how this kind of
  system should work? Be specific about mechanism, not just outcome.
- What does the research/production literature say about this decision? Cite specific
  papers, engineering blog posts, or well-known open-source systems (e.g. Weaviate,
  Vespa, Qdrant, ElasticSearch, Milvus, LlamaIndex, LangChain retrieval, Cohere
  Rerank, ColBERT, BEIR benchmark results, the original RRF paper by Cormack et al.,
  the MTEB leaderboard, nomic-embed-text's own published evaluation, etc.).
  For each citation, explain what it actually says that is directly relevant to a
  decision made (or not made) in this code.

### Part 2: State-of-the-art production reference cases

Pick **3–5 real, named production hybrid search deployments** (not toy examples) and for
each one describe:

- Who built it and what it retrieves (job postings, e-commerce, docs, etc.)
- The exact fusion strategy they use (RRF, learned weights, re-ranking, etc.)
- How they handle the query-document asymmetry problem (task prefixes, separate
  encoders, cross-encoders, etc.)
- What they do that this system does not — and whether that gap matters at this scale
  (~5k–10k documents, single-user, personal assistant context).

The goal is to understand which production patterns are worth adopting and which are
overkill for this use case.

### Part 3: Evaluation framework (the core deliverable)

Design a **complete, deterministic offline evaluation framework** for this hybrid search
system. Be concrete enough that a developer can implement it in a day.

The framework must cover:

**3a. The labeled dataset (golden set)**
- What exactly gets labeled: (query, job_id, relevance_grade) triples?
  Binary or graded (0/1/2/3)? How many queries minimum to be statistically meaningful?
- How to construct queries that represent realistic usage: from the profile? manually
  written? both? How to avoid overfitting the eval set to the current system's biases
  (i.e. don't label only what the current system retrieves).
- A specific labeling rubric: given a query and a job posting, what makes it grade 2
  vs grade 1 vs grade 0? Make this concrete for the job-search domain (e.g. what does
  "relevant" mean when the query is "Backend Engineer Python Kafka fintech"?).
- Minimum viable set size: how many (query, job) pairs to label first to get a
  meaningful signal, and how to expand from there.

**3b. Metrics**
- Which metrics to compute and why each one: Recall@K (K=20, K=50), Precision@K,
  nDCG@K, MRR. For each: what does it measure, why does it matter specifically for a
  retrieval gate (where miss = never scored), and what an acceptable vs good vs great
  value looks like for a corpus of this size.
- How to handle the "not retrieved = unknown relevance" problem in offline eval
  (unjudged documents). Recommend a specific approach.
- Which single metric should be the primary decision criterion for "did this change help"
  and why.

**3c. Parameter sweep design**
For each of the following parameters, specify:
- The range of values to sweep
- Whether it requires re-embedding or re-indexing (i.e. the cost)
- The expected directional effect on the primary metric and why
- Whether parameters interact (must be swept jointly)

Parameters to cover:
- RRF `k` constant (currently 60)
- Per-arm `pool` size (currently 50)
- Number of arms (FTS only / vector only / FTS+query-vec / FTS+CV-vec / all three)
- Arm weighting (if moving beyond standard RRF to weighted RRF or score normalization)
- `ts_rank` vs `ts_rank_cd` (cover density) for the FTS arm
- nomic task prefix variants (once re-embedding is done)
- FTS config: `'simple'` vs `'english'` (once `search_vector` is regenerated)
- Query construction: titles-only vs titles+stack vs titles+stack+domains vs full CV text
- `limit` (final result count): 10, 20, 30, 50

**3d. Test harness design**
- The exact Python harness to run a sweep: inputs, what it calls, what it records.
  Include pseudocode or actual code.
- How to store results so they are comparable across runs (schema for a results table
  or file format).
- How to visualize the sweep results to find optima (what plots, what axes).
- How to avoid data leakage between the labeled set and parameter tuning.

**3e. Decision protocol**
- The exact rule for "this change is an improvement": e.g. "primary metric improves by
  ≥X% on the held-out set with p<0.05 by a paired t-test or Wilcoxon signed-rank test."
- How to handle a change that improves nDCG but hurts Recall (or vice versa).
- When to re-label vs when the existing labels are sufficient for a new experiment.

### Part 4: Prioritized change list

Given your assessment and the evaluation framework, list the changes you recommend in
priority order. For each change:

- The specific code change (file, function, what changes)
- The expected effect on the primary metric (directional, with reasoning)
- The cost (lines of code, whether it needs re-embedding/re-indexing/re-labeling)
- The reference that supports this recommendation
- Whether it must be done before or after the eval framework is in place

Flag clearly which changes are **pre-eval** (safe to apply now, low risk of regression)
vs **post-eval** (should be measured before/after with the framework).

---

## Constraints to respect

- The stack is **Postgres + pgvector**. No external vector database.
- The embedding model is **`nomic-embed-text`** via local Ollama. Swapping models is
  possible but is a full re-embed.
- The system is a **personal assistant** (~1 user, ~5k–10k docs). Production-scale
  infrastructure (distributed search, streaming index updates, GPU inference clusters)
  is out of scope. But algorithmic best practices are not.
- Python 3.12, async, SQLAlchemy. Any proposed code changes must fit this stack.
- The project prioritizes **correctness and observability** over raw throughput.
- The team has a `eval_labels` table in the DB already:
  `(id, job_id, label, labeled_by, notes)` — this is the starting point for the golden
  set, currently empty.

---

## Output format

Structure your response as:

1. **Executive summary** (5–10 bullets: the most important findings and the single most
   impactful change)
2. **Part 1: Component assessment** (one section per component)
3. **Part 2: Production reference cases** (one subsection per case)
4. **Part 3: Evaluation framework** (3a through 3e as subsections)
5. **Part 4: Prioritized change list** (numbered, most impactful first)

Be concrete. Cite sources by name (paper title + authors + year, or URL of engineering
blog post). Do not pad with generic retrieval theory — every paragraph should either
explain something specific about this code or support a specific recommendation.
