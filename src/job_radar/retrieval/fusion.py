from collections import defaultdict
from uuid import UUID


def reciprocal_rank_fusion(
    rankings: list[list[tuple[UUID, float]]], *, k: int = 60, limit: int | None = None
) -> list[tuple[UUID, float]]:
    """Fuse component rankings into one, using only rank position, not score.

    Each item contributes 1 / (k + rank) per list it appears in (rank 1-based),
    summed across lists. Items ranking highly in multiple lists accumulate the
    most, which is the point of hybrid search. Ties break deterministically on
    the id so output ordering is stable across runs.
    """
    fusion_rank: defaultdict[UUID, float] = defaultdict(float)

    for ranking in rankings:
        for rank, (job_id, _) in enumerate(ranking, start=1):
            fusion_rank[job_id] += 1 / (k + rank)

    fused = sorted(fusion_rank.items(), key=lambda item: (-item[1], str(item[0])))
    return fused[:limit]
