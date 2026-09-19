"""Metric assembly, the paired document bootstrap and the negative controls. Pure.

The metric functions themselves live in `eval/metrics.py` — the auditable reference
implementation shared with the regression eval; this module only composes them.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from uuid import UUID

import numpy as np

from eval.metrics import average_precision, bpref, ndcg, precision_at_k, recall_at_k

REL_THRESHOLD = 2


def judged_at_k(ranking: Sequence[UUID], labels: dict[UUID, int], k: int) -> float:
    """Share of the top-k that carries a label. Denominator is `k`, as in `precision_at_k`."""
    if not ranking:
        return 0.0
    return sum(1 for doc_id in ranking[:k] if doc_id in labels) / k


# One entry per metric: adding a metric is adding a line here (design §3.2).
METRICS: dict[str, Callable[[Sequence[UUID], dict[UUID, int]], float]] = {
    "ndcg@10": partial(ndcg, k=10),
    "ndcg@50": partial(ndcg, k=50),
    "ndcg@100": partial(ndcg, k=100),
    "p@10": partial(precision_at_k, k=10),
    "p@50": partial(precision_at_k, k=50),
    "p@100": partial(precision_at_k, k=100),
    "recall@50": partial(recall_at_k, k=50),
    "recall@100": partial(recall_at_k, k=100),
    "ap": average_precision,
    "bpref": bpref,
    "judged@10": partial(judged_at_k, k=10),
    "judged@100": partial(judged_at_k, k=100),
}

_COMPLETE_KEYS = (
    "ndcg@100",
    "p@100",
    "ap",
    "ndcg@10",
    "recall@50",
    "recall@100",
    "bpref",
    "p@10",
    "judged@10",
    "judged@100",
)
# Partial views deliberately stop at @50: see `compute_metrics`.
_PARTIAL_CONDENSED_KEYS = ("ndcg@10", "ndcg@50", "p@10", "p@50", "recall@50", "ap")
_PARTIAL_FULL_KEYS = ("bpref", "judged@10", "judged@100")


def condense(ranking: Sequence[UUID], labels: dict[UUID, int]) -> list[UUID]:
    """The ranking restricted to judged documents, order preserved."""
    return [doc_id for doc_id in ranking if doc_id in labels]


def _relevant_count(labels: dict[UUID, int], rel_threshold: int = REL_THRESHOLD) -> int:
    return sum(1 for grade in labels.values() if grade >= rel_threshold)


def recall_ceiling(
    labels: dict[UUID, int], k: int = 100, rel_threshold: int = REL_THRESHOLD
) -> float:
    """`min(1, k/R)` — the best Recall@k any ranking can reach. R is 430 to 660 here (F25)."""
    relevant = _relevant_count(labels, rel_threshold)
    if relevant == 0:
        return 1.0
    return min(1.0, k / relevant)


def compute_metrics(
    ranking: Sequence[UUID], labels: dict[UUID, int], partial: bool
) -> dict[str, float]:
    """Metrics for one ranking under one label view.

    With `partial` (the `human` view, where only some documents are judged) the ranking is
    condensed to the judged documents first, and no @100 metric is produced at all. Raw
    metrics that score an unjudged document as grade 0 measure *coverage of the labeled set*,
    not quality: deliberately degraded embedders outscored the incumbent on exactly those
    numbers (F26/R-M1), so they are not computed rather than computed and captioned.
    """
    ranked = list(ranking)
    if not partial:
        scores = {key: METRICS[key](ranked, labels) for key in _COMPLETE_KEYS}
        scores["r"] = float(_relevant_count(labels))
        scores["recall_ceiling@100"] = recall_ceiling(labels, 100)
        return scores

    condensed = condense(ranked, labels)
    scores = {key: METRICS[key](condensed, labels) for key in _PARTIAL_CONDENSED_KEYS}
    scores.update({key: METRICS[key](ranked, labels) for key in _PARTIAL_FULL_KEYS})
    scores["n_judged"] = float(len(condensed))
    scores["r"] = float(_relevant_count(labels))
    return scores


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    sd: float
    low: float
    high: float


def paired_bootstrap(
    rankings: dict[str, Sequence[UUID]],
    labels: dict[UUID, int],
    baseline: str,
    metric: str,
    n: int,
    seed: int,
) -> dict[str, BootstrapResult]:
    """Seeded paired document bootstrap of every ranking's `metric` difference vs `baseline`.

    Documents are resampled with replacement from the baseline ranking's universe, the same
    resample for every candidate (paired), and a document drawn m times appears m times in
    each ranking's own order. Unlabeled documents count as grade 0, so a partial view must be
    condensed by the caller (`condense`) before it gets here.
    """
    if baseline not in rankings:
        raise ValueError(f"baseline {baseline!r} is not one of {sorted(rankings)}")
    if metric not in _BOOT_METRICS:
        raise ValueError(f"unknown bootstrap metric {metric!r}; known: {sorted(_BOOT_METRICS)}")

    docs = list(rankings[baseline])
    if not docs:
        raise ValueError("the baseline ranking is empty")
    index = {doc_id: i for i, doc_id in enumerate(docs)}
    if len(index) != len(docs):
        raise ValueError("the baseline ranking contains duplicate documents")

    orders = {}
    for name, ranking in rankings.items():
        ranked = list(ranking)
        if len(ranked) != len(docs) or set(ranked) != set(docs):
            raise ValueError(f"ranking {name!r} does not cover the baseline's documents")
        orders[name] = np.fromiter((index[d] for d in ranked), dtype=np.int64, count=len(ranked))

    grades = np.fromiter((labels.get(d, 0) for d in docs), dtype=np.int64, count=len(docs))
    counts = _resample_counts(len(docs), n, seed)
    replicate = _BOOT_METRICS[metric]
    base = replicate(_expand(grades, orders[baseline], counts))

    out = {}
    for name, order in orders.items():
        if name == baseline:
            continue
        diff = replicate(_expand(grades, order, counts)) - base
        out[name] = BootstrapResult(
            mean_diff=float(diff.mean()),
            sd=float(diff.std(ddof=1)) if n > 1 else 0.0,
            low=float(np.percentile(diff, 2.5)),
            high=float(np.percentile(diff, 97.5)),
        )
    return out


def _resample_counts(d: int, n: int, seed: int) -> np.ndarray:
    """(n, d) multiplicities: `n` draws of `d` documents with replacement."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, d, size=(n, d), dtype=np.int64)
    offsets = (np.arange(n, dtype=np.int64) * d)[:, None]
    return np.bincount((draws + offsets).ravel(), minlength=n * d).reshape(n, d)


def _expand(grades: np.ndarray, order: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """(n, d) grades of each replicate's multiset ranking, in `order`'s own document order."""
    n = counts.shape[0]
    reps = counts[:, order]
    return np.repeat(np.tile(grades[order], n), reps.ravel()).reshape(n, counts.shape[1])


# The bootstrap needs the same metrics computed over (n, d) grade matrices — a per-replicate
# call into eval.metrics is ~10^4 times too slow. `test_embedding_scoring` pins them together.
def _boot_ndcg(g: np.ndarray, k: int) -> np.ndarray:
    top = min(k, g.shape[1])
    discount = 1.0 / np.log2(np.arange(2, top + 2))
    gains = 2.0**g - 1.0
    dcg = gains[:, :top] @ discount
    idcg = np.sort(gains, axis=1)[:, ::-1][:, :top] @ discount
    return np.divide(dcg, idcg, out=np.zeros(g.shape[0]), where=idcg > 0.0)


def _boot_precision(g: np.ndarray, k: int) -> np.ndarray:
    return (g[:, :k] >= REL_THRESHOLD).sum(axis=1) / k


def _boot_recall(g: np.ndarray, k: int) -> np.ndarray:
    relevant = g >= REL_THRESHOLD
    total = relevant.sum(axis=1)
    hits = relevant[:, :k].sum(axis=1)
    return np.divide(hits, total, out=np.zeros(g.shape[0]), where=total > 0)


def _boot_ap(g: np.ndarray) -> np.ndarray:
    relevant = g >= REL_THRESHOLD
    ranks = np.arange(1, g.shape[1] + 1)
    score = (relevant * (np.cumsum(relevant, axis=1) / ranks)).sum(axis=1)
    total = relevant.sum(axis=1)
    return np.divide(score, total, out=np.zeros(g.shape[0]), where=total > 0)


_BOOT_METRICS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "ndcg@10": partial(_boot_ndcg, k=10),
    "ndcg@50": partial(_boot_ndcg, k=50),
    "ndcg@100": partial(_boot_ndcg, k=100),
    "p@10": partial(_boot_precision, k=10),
    "p@50": partial(_boot_precision, k=50),
    "p@100": partial(_boot_precision, k=100),
    "recall@50": partial(_boot_recall, k=50),
    "recall@100": partial(_boot_recall, k=100),
    "ap": _boot_ap,
}


def random_ranking(ids: Sequence[UUID], seed: int) -> list[UUID]:
    """Negative control: a ranking any real embedder must beat."""
    rng = np.random.default_rng(seed)
    return [ids[i] for i in rng.permutation(len(ids))]


def shuffled_tail(ranking: Sequence[UUID], keep: int, seed: int) -> list[UUID]:
    """Negative control: the head kept, everything below `keep` shuffled."""
    rng = np.random.default_rng(seed)
    tail = list(ranking[keep:])
    return list(ranking[:keep]) + [tail[i] for i in rng.permutation(len(tail))]


@dataclass(frozen=True)
class OrderStability:
    orders: dict[str, list[str]]
    agree: bool
    disagreements: dict[str, int]


def order_stability(scores: dict[str, dict[str, float]]) -> OrderStability:
    """Does the embedder order survive a change of query recipe (F27)?

    `disagreements` counts inverted embedder pairs against the first recipe's order.
    """
    recipes = list(scores)
    orders = {
        recipe: sorted(values, key=lambda name: (-values[name], name))
        for recipe, values in scores.items()
    }
    reference = orders[recipes[0]] if recipes else []
    if any(set(order) != set(reference) for order in orders.values()):
        raise ValueError("every recipe must score the same embedders")
    return OrderStability(
        orders=orders,
        agree=all(order == reference for order in orders.values()),
        disagreements={recipe: _inversions(reference, orders[recipe]) for recipe in recipes},
    )


def _inversions(reference: Sequence[str], order: Sequence[str]) -> int:
    rank = {name: i for i, name in enumerate(order)}
    return sum(
        1
        for i in range(len(reference))
        for j in range(i + 1, len(reference))
        if rank[reference[i]] > rank[reference[j]]
    )
