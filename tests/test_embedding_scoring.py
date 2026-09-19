from math import log2
from uuid import UUID, uuid4

import numpy as np
import pytest

from eval.embedding.scoring import (
    _BOOT_METRICS,
    _expand,
    _resample_counts,
    compute_metrics,
    condense,
    judged_at_k,
    order_stability,
    paired_bootstrap,
    random_ranking,
    recall_ceiling,
    shuffled_tail,
)
from eval.metrics import average_precision, ndcg, precision_at_k, recall_at_k


def uid(n: int) -> UUID:
    return UUID(int=n)


# Four ranked documents, one relevant document (5) the ranking never reaches.
COMPLETE_RANKING = [uid(1), uid(2), uid(3), uid(4)]
COMPLETE_LABELS = {uid(1): 3, uid(2): 0, uid(3): 2, uid(4): 1, uid(5): 3}


def test_complete_view_keys():
    scores = compute_metrics(COMPLETE_RANKING, COMPLETE_LABELS, partial=False)

    assert set(scores) == {
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
        "r",
        "recall_ceiling@100",
    }


def test_complete_view_values():
    scores = compute_metrics(COMPLETE_RANKING, COMPLETE_LABELS, partial=False)

    # Ranked gains 7, 0, 3, 1; the ideal ordering of all five labels is 3, 3, 2, 1, 0.
    dcg = 7 / log2(2) + 0 / log2(3) + 3 / log2(4) + 1 / log2(5)
    idcg = 7 / log2(2) + 7 / log2(3) + 3 / log2(4) + 1 / log2(5) + 0 / log2(6)
    assert scores["ndcg@100"] == pytest.approx(dcg / idcg)
    assert scores["ndcg@10"] == pytest.approx(dcg / idcg)
    # Two of the three relevant documents (1 and 3) are retrieved; document 5 is not.
    assert scores["p@10"] == pytest.approx(0.2)
    assert scores["p@100"] == pytest.approx(0.02)
    assert scores["recall@50"] == pytest.approx(2 / 3)
    assert scores["recall@100"] == pytest.approx(2 / 3)
    assert scores["ap"] == pytest.approx((1 / 1 + 2 / 3) / 3)
    # bpref: R=3, N=2, norm=2; doc 1 scores 1, doc 3 scores 1 - 1/2 after one non-relevant.
    assert scores["bpref"] == pytest.approx((1.0 + 0.5) / 3)
    assert scores["judged@10"] == pytest.approx(0.4)
    assert scores["judged@100"] == pytest.approx(0.04)
    assert scores["r"] == 3.0
    assert scores["recall_ceiling@100"] == 1.0


# Twelve ranked documents, three of them judged, one of those below rank 10.
PARTIAL_RANKING = [uid(n) for n in range(1, 13)]
PARTIAL_LABELS = {uid(2): 3, uid(4): 0, uid(12): 2}


def test_partial_view_keys_exclude_the_at_100_metrics():
    scores = compute_metrics(PARTIAL_RANKING, PARTIAL_LABELS, partial=True)

    assert "ndcg@100" not in scores
    assert "recall@100" not in scores
    assert set(scores) == {
        "ndcg@10",
        "ndcg@50",
        "p@10",
        "p@50",
        "recall@50",
        "ap",
        "bpref",
        "judged@10",
        "judged@100",
        "n_judged",
        "r",
    }


def test_partial_view_equals_the_condensed_computation():
    condensed = condense(PARTIAL_RANKING, PARTIAL_LABELS)
    assert condensed == [uid(2), uid(4), uid(12)]

    scores = compute_metrics(PARTIAL_RANKING, PARTIAL_LABELS, partial=True)

    assert scores["ndcg@10"] == pytest.approx(ndcg(condensed, PARTIAL_LABELS, 10))
    assert scores["ndcg@50"] == pytest.approx(ndcg(condensed, PARTIAL_LABELS, 50))
    assert scores["p@10"] == pytest.approx(precision_at_k(condensed, PARTIAL_LABELS, 10))
    assert scores["p@50"] == pytest.approx(precision_at_k(condensed, PARTIAL_LABELS, 50))
    assert scores["recall@50"] == pytest.approx(recall_at_k(condensed, PARTIAL_LABELS, 50))
    assert scores["ap"] == pytest.approx(average_precision(condensed, PARTIAL_LABELS))


def test_partial_view_values():
    scores = compute_metrics(PARTIAL_RANKING, PARTIAL_LABELS, partial=True)

    # Condensed gains 7, 0, 3; the ideal ordering of the three labels is 3, 2, 0.
    dcg = 7 / log2(2) + 0 / log2(3) + 3 / log2(4)
    idcg = 7 / log2(2) + 3 / log2(3) + 0 / log2(4)
    assert scores["ndcg@10"] == pytest.approx(dcg / idcg)
    assert scores["p@10"] == pytest.approx(0.2)
    assert scores["recall@50"] == 1.0
    assert scores["ap"] == pytest.approx((1 / 1 + 2 / 3) / 2)
    # bpref and judged@k read the FULL ranking: only 2 of the top 10 carry a label.
    assert scores["judged@10"] == pytest.approx(0.2)
    assert scores["judged@100"] == pytest.approx(0.03)
    assert scores["bpref"] == pytest.approx((1.0 + 0.0) / 2)
    assert scores["n_judged"] == 3.0
    assert scores["r"] == 2.0


def test_judged_at_k():
    ranking = [uid(n) for n in range(1, 21)]
    labels = {uid(n): 1 for n in (1, 3, 15)}

    assert judged_at_k(ranking, labels, 10) == pytest.approx(0.2)
    assert judged_at_k(ranking, labels, 20) == pytest.approx(0.15)
    assert judged_at_k([], labels, 10) == 0.0


def test_recall_ceiling():
    assert recall_ceiling({}, 100) == 1.0
    assert recall_ceiling({uid(1): 1, uid(2): 0}, 100) == 1.0  # R = 0: nothing to miss
    assert recall_ceiling({uid(n): 2 for n in range(300)}, 100) == pytest.approx(1 / 3)
    assert recall_ceiling({uid(n): 3 for n in range(4)}, 100) == 1.0
    assert recall_ceiling({uid(n): 2 for n in range(10)}, 5) == pytest.approx(0.5)


def bootstrap_case(n_docs: int = 40):
    docs = [uid(n) for n in range(n_docs)]
    labels = {doc: (3 if i < n_docs // 2 else 0) for i, doc in enumerate(docs)}
    good = docs[: n_docs // 2] + docs[n_docs // 2 :]
    bad = docs[n_docs // 2 :] + docs[: n_docs // 2]
    return docs, labels, good, bad


def test_paired_bootstrap_is_deterministic():
    _, labels, good, bad = bootstrap_case()
    rankings = {"base": bad, "good": good}

    first = paired_bootstrap(rankings, labels, "base", "ap", n=200, seed=3)
    second = paired_bootstrap(rankings, labels, "base", "ap", n=200, seed=3)

    assert first == second
    assert set(first) == {"good"}


def test_paired_bootstrap_is_paired():
    _, labels, good, _ = bootstrap_case()

    result = paired_bootstrap({"base": good, "copy": list(good)}, labels, "base", "ndcg@100", 50, 1)

    assert result["copy"].mean_diff == 0.0
    assert result["copy"].sd == 0.0
    assert result["copy"].low == 0.0
    assert result["copy"].high == 0.0


def test_paired_bootstrap_separates_a_clearly_better_ranking():
    _, labels, good, bad = bootstrap_case()

    result = paired_bootstrap({"base": bad, "good": good}, labels, "base", "ndcg@10", 300, 5)

    assert result["good"].mean_diff > 0.5
    assert result["good"].low > 0.0


def test_paired_bootstrap_rejects_bad_input():
    _, labels, good, bad = bootstrap_case()

    with pytest.raises(ValueError, match="unknown bootstrap metric"):
        paired_bootstrap({"base": good}, labels, "base", "mrr", 10, 0)
    with pytest.raises(ValueError, match="is not one of"):
        paired_bootstrap({"base": good}, labels, "nope", "ap", 10, 0)
    with pytest.raises(ValueError, match="does not cover"):
        paired_bootstrap({"base": good, "short": bad[:-1]}, labels, "base", "ap", 10, 0)


def test_resample_counts_are_a_seeded_resample_with_replacement():
    counts = _resample_counts(6, 4, seed=2)

    assert counts.shape == (4, 6)
    assert counts.sum(axis=1).tolist() == [6, 6, 6, 6]
    assert counts.max() > 1  # with replacement
    assert (counts == _resample_counts(6, 4, seed=2)).all()
    assert (counts != _resample_counts(6, 4, seed=3)).any()


@pytest.mark.parametrize("metric", ["ndcg@10", "ndcg@100", "p@10", "p@100", "recall@100", "ap"])
def test_vectorized_replicate_metrics_match_eval_metrics(metric):
    # A hand-built resample: document i appears counts[i] times, in each ranking's own order.
    grades = np.array([3, 0, 2, 1, 0, 3, 2, 0], dtype=np.int64)
    order = np.array([5, 0, 3, 6, 2, 7, 1, 4], dtype=np.int64)
    counts = np.array([[2, 0, 1, 3, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int64)

    expanded = _expand(grades, order, counts)
    got = _BOOT_METRICS[metric](expanded)

    # Reference: give every copy its own id so eval.metrics can score the multiset directly.
    for replicate, row in enumerate(expanded):
        ranking = [uuid4() for _ in row]
        labels = dict(zip(ranking, (int(g) for g in row), strict=True))
        reference = {
            "ndcg@10": lambda r, lab: ndcg(r, lab, 10),
            "ndcg@100": lambda r, lab: ndcg(r, lab, 100),
            "p@10": lambda r, lab: precision_at_k(r, lab, 10),
            "p@100": lambda r, lab: precision_at_k(r, lab, 100),
            "recall@100": lambda r, lab: recall_at_k(r, lab, 100),
            "ap": average_precision,
        }[metric]
        assert got[replicate] == pytest.approx(reference(ranking, labels))


def test_expand_repeats_documents_in_each_rankings_own_order():
    grades = np.array([1, 2, 3], dtype=np.int64)
    counts = np.array([[2, 1, 0]], dtype=np.int64)

    assert _expand(grades, np.array([0, 1, 2]), counts).tolist() == [[1, 1, 2]]
    assert _expand(grades, np.array([2, 1, 0]), counts).tolist() == [[2, 1, 1]]


def test_random_ranking_is_a_seeded_permutation():
    ids = [uid(n) for n in range(30)]

    shuffled = random_ranking(ids, seed=4)

    assert sorted(shuffled) == sorted(ids)
    assert shuffled != ids
    assert shuffled == random_ranking(ids, seed=4)
    assert shuffled != random_ranking(ids, seed=5)


def test_shuffled_tail_keeps_the_head():
    ids = [uid(n) for n in range(30)]

    control = shuffled_tail(ids, keep=10, seed=4)

    assert control[:10] == ids[:10]
    assert sorted(control[10:]) == sorted(ids[10:])
    assert control[10:] != ids[10:]
    assert control == shuffled_tail(ids, keep=10, seed=4)


def test_order_stability_agreement():
    scores = {
        "hyde_mean": {"bge-m3": 0.5, "qwen3": 0.4, "nomic": 0.3},
        "hyde_text:0": {"bge-m3": 0.6, "qwen3": 0.5, "nomic": 0.4},
    }

    stability = order_stability(scores)

    assert stability.orders["hyde_mean"] == ["bge-m3", "qwen3", "nomic"]
    assert stability.agree
    assert stability.disagreements == {"hyde_mean": 0, "hyde_text:0": 0}


def test_order_stability_counts_inverted_pairs():
    scores = {
        "hyde_mean": {"bge-m3": 0.5, "qwen3": 0.4, "nomic": 0.3},
        "hyde_text:0": {"bge-m3": 0.2, "qwen3": 0.5, "nomic": 0.4},
    }

    stability = order_stability(scores)

    assert stability.orders["hyde_text:0"] == ["qwen3", "nomic", "bge-m3"]
    assert not stability.agree
    assert stability.disagreements == {"hyde_mean": 0, "hyde_text:0": 2}


def test_order_stability_breaks_ties_by_name_and_checks_the_embedder_set():
    stability = order_stability({"hyde_mean": {"qwen3": 0.5, "bge-m3": 0.5}})

    assert stability.orders["hyde_mean"] == ["bge-m3", "qwen3"]

    with pytest.raises(ValueError, match="same embedders"):
        order_stability({"a": {"x": 0.1}, "b": {"y": 0.1}})
