from collections import defaultdict
from uuid import UUID


def reciprocal_rank_fusion(
    rankings: list[list[tuple[UUID, float]]],
    *,
    k: int = 60,
    limit: int | None = None,
    weights: list[float] | None = None,
) -> list[tuple[UUID, float]]:
    """Fuse component rankings into one, using only rank position, not score.

    Each item contributes w * 1/(k+rank) per list it appears in (rank 1-based),
    where w defaults to 1.0 (standard RRF). Provide `weights` to down- or
    up-weight individual arms; its length must match `rankings`.

    Ties break deterministically on the id so output ordering is stable.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(f"weights length {len(weights)} != rankings length {len(rankings)}")

    fusion_rank: defaultdict[UUID, float] = defaultdict(float)

    for i, ranking in enumerate(rankings):
        w = weights[i] if weights is not None else 1.0
        for rank, (job_id, _) in enumerate(ranking, start=1):
            fusion_rank[job_id] += w / (k + rank)

    fused = sorted(fusion_rank.items(), key=lambda item: (-item[1], str(item[0])))
    return fused[:limit]
