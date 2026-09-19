"""Exact cosine ranking and tie-aware ranking comparison. Pure: numpy and stdlib only."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from uuid import UUID

import numpy as np

# How many offending positions a failure reason lists before it stops.
_MAX_REASONS = 5


def cosine_rank(
    matrix: np.ndarray, ids: Sequence[UUID], query: np.ndarray
) -> list[tuple[UUID, float]]:
    """Cosine of every row of `matrix` against `query`, best first, ties broken by ascending id.

    The id tie-break is mandatory, not cosmetic: 14% of production's stored vectors are
    byte-identical to another row (F6), so score order alone is not a deterministic ranking.
    Rows and the query need not be unit norm — the query is a mean of unit vectors — and a
    zero-norm row or query scores 0.0 rather than NaN.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    vec = np.asarray(query, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2-d, got shape {mat.shape}")
    if vec.ndim != 1:
        raise ValueError(f"query must be 1-d, got shape {vec.shape}")
    if mat.shape[0] != len(ids):
        raise ValueError(f"matrix has {mat.shape[0]} rows but {len(ids)} ids were given")
    if mat.shape[1] != vec.shape[0]:
        raise ValueError(f"matrix is {mat.shape[1]}-d but query is {vec.shape[0]}-d")

    denom = np.linalg.norm(mat, axis=1) * float(np.linalg.norm(vec))
    scores = np.divide(mat @ vec, denom, out=np.zeros(mat.shape[0]), where=denom > 0.0)
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    return [(ids[i], float(scores[i])) for i in order]


@dataclass(frozen=True)
class CompareResult:
    ok: bool
    reasons: list[str]


def compare_rankings(
    a: Sequence[tuple[UUID, float]],
    b: Sequence[tuple[UUID, float]],
    k: int,
    eps: float,
) -> CompareResult:
    """Tie-aware equality of two best-first rankings over their first `k` positions.

    Exact order can never be required: byte-identical vectors (F6) make the order inside a
    tie group arbitrary, and production has no secondary sort key. So `a` is partitioned into
    *eps-chains* (runs of consecutive scores no more than `eps` apart) and equality is asked
    of the chains, not of the positions: scores must agree position-wise, every chain that
    ends before `k` must hold the same id *set* in both, and the chain straddling position `k`
    may hold different ids as long as they score inside that chain's band.
    """
    if len(a) < k or len(b) < k:
        return CompareResult(
            False, [f"ranking shorter than k={k}: len(a)={len(a)}, len(b)={len(b)}"]
        )

    reasons = [
        f"score drift at position {i}: {a[i][1]:.9g} vs {b[i][1]:.9g}"
        for i in range(k)
        if abs(a[i][1] - b[i][1]) > eps
    ][:_MAX_REASONS]

    chain_reasons: list[str] = []
    for start, end in _eps_chains(a, k, eps):
        if end < k:
            chain_reasons.extend(_set_mismatch(a, b, start, end))
        else:
            chain_reasons.extend(_boundary_mismatch(a, b, start, end, k, eps))
    reasons.extend(chain_reasons[:_MAX_REASONS])
    return CompareResult(not reasons, reasons)


def _eps_chains(a: Sequence[tuple[UUID, float]], k: int, eps: float) -> Iterator[tuple[int, int]]:
    """Inclusive index spans of `a`'s eps-chains, stopping after the one covering position k-1."""
    start = 0
    for i in range(1, len(a)):
        if a[i - 1][1] - a[i][1] > eps:
            yield start, i - 1
            if i - 1 >= k - 1:
                return
            start = i
    yield start, len(a) - 1


def _set_mismatch(
    a: Sequence[tuple[UUID, float]], b: Sequence[tuple[UUID, float]], start: int, end: int
) -> list[str]:
    ids_a = {doc_id for doc_id, _ in a[start : end + 1]}
    ids_b = {doc_id for doc_id, _ in b[start : end + 1]}
    if ids_a == ids_b:
        return []
    return [
        f"id set differs in positions {start}-{end}: "
        f"only in a {sorted(ids_a - ids_b)}, only in b {sorted(ids_b - ids_a)}"
    ]


def _boundary_mismatch(
    a: Sequence[tuple[UUID, float]],
    b: Sequence[tuple[UUID, float]],
    start: int,
    end: int,
    k: int,
    eps: float,
) -> list[str]:
    """Ids `b` promotes into the chain crossing `k` must score inside the chain's band — in both."""
    chain = a[start : end + 1]
    chain_ids = {doc_id for doc_id, _ in chain}
    low = min(score for _, score in chain) - eps
    high = max(score for _, score in chain) + eps
    a_scores = dict(a)
    out = []
    for pos in range(start, k):
        doc_id, score = b[pos]
        if doc_id in chain_ids:
            continue
        seen = [score] if doc_id not in a_scores else [score, a_scores[doc_id]]
        if any(s < low or s > high for s in seen):
            out.append(
                f"position {pos}: {doc_id} is outside the tie chain {start}-{end} "
                f"[{low:.9g}, {high:.9g}] with scores {[round(s, 9) for s in seen]}"
            )
    return out
