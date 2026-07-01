"""Pure-Python TREC gain-based metrics.

No I/O, no external deps — this is the auditable reference implementation.
`ranx` is used for production compute and sweep comparisons (run.py, sweep.py);
this module is what the unit tests pin both against.

All functions take:
  ranking : list[UUID]  — retrieved docs, best-first
  labels  : dict[UUID, int]  — relevance grades 0-3 for the query
"""

from math import log2
from uuid import UUID


def dcg(ranking: list[UUID], labels: dict[UUID, int], k: int) -> float:
    """DCG@k with exponential gain: sum_{i=1..k} (2^grade_i - 1) / log2(i + 1)."""
    score = 0.0
    for i, doc_id in enumerate(ranking[:k], start=1):
        grade = labels.get(doc_id, 0)
        score += (2**grade - 1) / log2(i + 1)
    return score


def ndcg(ranking: list[UUID], labels: dict[UUID, int], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k.

    IDCG is the DCG of the perfect ordering (grades sorted descending).
    Returns 0.0 when IDCG is 0 — no relevant docs in labels, or all grades are 0.
    """
    ideal_grades = sorted(labels.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / log2(i + 1) for i, g in enumerate(ideal_grades, start=1) if g > 0)
    if idcg == 0.0:
        return 0.0
    return dcg(ranking, labels, k) / idcg


def mrr(ranking: list[UUID], labels: dict[UUID, int], rel_threshold: int = 1) -> float:
    """MRR: 1/position of the first doc with grade ≥ rel_threshold, else 0.0."""
    for i, doc_id in enumerate(ranking, start=1):
        if labels.get(doc_id, 0) >= rel_threshold:
            return 1.0 / i
    return 0.0


def precision_at_k(
    ranking: list[UUID], labels: dict[UUID, int], k: int, rel_threshold: int = 2
) -> float:
    """Fraction of the top-k results with grade ≥ rel_threshold."""
    top_k = ranking[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if labels.get(doc_id, 0) >= rel_threshold)
    return hits / k


def recall_at_k(
    ranking: list[UUID], labels: dict[UUID, int], k: int, rel_threshold: int = 2
) -> float:
    """Fraction of all relevant docs (grade ≥ rel_threshold) that appear in top-k.

    Denominator is the total relevant count in `labels` — the recall ceiling —
    not just what was retrieved. Returns 0.0 when there are no relevant docs.
    """
    total_relevant = sum(1 for g in labels.values() if g >= rel_threshold)
    if total_relevant == 0:
        return 0.0
    hits = sum(1 for doc_id in ranking[:k] if labels.get(doc_id, 0) >= rel_threshold)
    return hits / total_relevant


def bpref(
    ranking: list[UUID], labels: dict[UUID, int], rel_threshold: int = 2
) -> float:
    """BPref: robust to incomplete judgments (Buckley & Voorhees 2004).

    Only judged documents count — unlabeled docs are ignored entirely.
    This keeps scores stable when pool-based labeling leaves gaps in qrels,
    which is always the case for pool-sourced annotation.

    Formula: (1/R) * sum_r(1 - |judged non-rel before r| / min(R, N))
    where R = |relevant in labels|, N = |non-relevant in labels|.
    """
    relevant = {doc_id for doc_id, g in labels.items() if g >= rel_threshold}
    non_relevant = {doc_id for doc_id, g in labels.items() if g < rel_threshold}
    R = len(relevant)
    if R == 0:
        return 0.0
    norm = min(R, len(non_relevant))
    score = 0.0
    non_rel_seen = 0
    for doc_id in ranking:
        if doc_id in non_relevant:
            non_rel_seen += 1
        elif doc_id in relevant:
            score += 1.0 - (non_rel_seen / norm if norm > 0 else 0.0)
    return score / R
