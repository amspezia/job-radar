"""TREC qrel/run adapters — bridges between the DB and the metric primitives.

Two public functions:
  load_qrels  — reads EVAL_LABEL rows → {job_id: grade (0-3)}
  build_run   — executes retrieval arms under a SearchConfig → {job_id: rrf_score}

SearchConfig is the single place that encodes an experiment's parameters:
k, pool, limit, weights, field_boosts, and which arms to activate.  The three
standard §12 configurations (hybrid / vector-only / keyword-only) are exported
as module-level constants.
"""

import asyncio
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import EvalLabel, Profile
from job_radar.retrieval.bm25 import search_bm25
from job_radar.retrieval.filters import build_profile_filter
from job_radar.retrieval.fusion import reciprocal_rank_fusion
from job_radar.retrieval.vector import search_vector

# Integer strings "0"-"3" are the canonical stored form (written by label.py).
# Verdict aliases are accepted for labels written by other tools.
_LABEL_TO_GRADE: dict[str, int] = {
    "0": 0,
    "none": 0,
    "1": 1,
    "weak": 1,
    "marginal": 1,
    "2": 2,
    "moderate": 2,
    "relevant": 2,
    "3": 3,
    "strong": 3,
}


@dataclass
class SearchConfig:
    """Parameters for one retrieval experiment.

    arms      — which rankers contribute; any subset of {"lexical", "hyde", "cv"}.
    k         — RRF smoothing constant (Cormack et al. default: 60).
    pool      — top-N each vector arm retrieves before fusion.
    bm25_pool — top-N the BM25 arm retrieves; defaults to pool when None.
                Set higher than pool to give BM25 more candidates without
                expanding the (GPU-bound) vector arm pools symmetrically.
    limit     — final result count after fusion.
    weights   — per-arm RRF weight multipliers; None → equal weights.
    field_boosts — BM25 per-field boost overrides; None → current defaults.
    """

    arms: list[str] = field(default_factory=lambda: ["lexical", "hyde", "cv"])
    k: int = 60
    pool: int = 50
    bm25_pool: int | None = None
    limit: int = 50
    weights: list[float] | None = None
    field_boosts: dict[str, int] | None = None


# Standard configurations matching production search (2-arm: lexical + HyDE).
# CV arm removed after ablation showed it consistently degrades all metrics
# (backward-looking resume embedding conflicts with forward-looking HyDE).
# pool=100 matches the production search() default.
HYBRID = SearchConfig(arms=["lexical", "hyde"], pool=100, limit=100)
HYBRID_PROD = SearchConfig(arms=["lexical", "hyde"], pool=100, limit=50)
HYDE_ONLY = SearchConfig(arms=["hyde"], pool=100, limit=100)
KEYWORD_ONLY = SearchConfig(arms=["lexical"], pool=100, limit=100)


async def load_qrels(session: AsyncSession, profile_id: UUID) -> dict[UUID, int]:
    """Return {job_id: grade} for this profile from EVAL_LABEL rows.

    Rows with unrecognised label strings are silently dropped rather than
    crashing — they represent labels written by a future version or by hand
    that don't map to the 0-3 scale yet.
    """
    rows = (
        await session.execute(
            select(EvalLabel.job_id, EvalLabel.label).where(
                EvalLabel.profile_id == profile_id
            )
        )
    ).all()
    result: dict[UUID, int] = {}
    for job_id, label in rows:
        grade = _LABEL_TO_GRADE.get(str(label).strip().lower())
        if grade is not None:
            result[job_id] = grade
    return result


async def build_run(
    session: AsyncSession,
    profile: Profile,
    config: SearchConfig,
    *,
    query: str = "",
    hyde_embedding: list[float] | None = None,
) -> dict[UUID, float]:
    """Execute the retrieval arms defined by `config`, return {job_id: rrf_score}.

    Calls search_bm25 / search_vector directly so k, pool, and field_boosts
    can be varied without touching the production search() signature.  The
    fused RRF scores are returned raw (not normalised) so callers can pass
    them directly to ranx without further transformation.

    Arms are skipped when their required input is absent:
      - "lexical" skipped when `query` is blank
      - "hyde"    skipped when `hyde_embedding` is None
      - "cv"      skipped when profile.cv_embedding is None
    With no active arms the result is an empty dict (no corpus dump).
    """
    # Apply the same hard filters as production search so the eval corpus matches
    # what the system actually serves (geo, seniority, remote, salary floor).
    _filter = build_profile_filter(profile)

    # Determine which arms are active and build coroutines for each.
    active_indices: list[int] = []
    coros = []
    bm25_pool = config.bm25_pool if config.bm25_pool is not None else config.pool
    for idx, arm in enumerate(config.arms):
        if arm == "lexical" and query.strip():
            active_indices.append(idx)
            coros.append(
                search_bm25(session, query, bm25_pool, _filter, field_boosts=config.field_boosts)
            )
        elif arm == "hyde" and hyde_embedding is not None:
            active_indices.append(idx)
            coros.append(search_vector(session, hyde_embedding, config.pool, _filter))
        elif arm == "cv" and profile.cv_embedding is not None:
            active_indices.append(idx)
            coros.append(search_vector(session, list(profile.cv_embedding), config.pool, _filter))

    if not coros:
        return {}

    arms: list[list[tuple[UUID, float]]] = list(await asyncio.gather(*coros))

    # Slice weights to match only the arms that were active.  If weights is
    # shorter than config.arms (misconfigured) fall back to equal weights.
    weights: list[float] | None = None
    if config.weights is not None and len(config.weights) == len(config.arms):
        weights = [config.weights[i] for i in active_indices]

    fused = reciprocal_rank_fusion(arms, k=config.k, limit=config.limit, weights=weights)
    return dict(fused)
