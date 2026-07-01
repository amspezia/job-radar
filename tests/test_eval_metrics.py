"""Unit tests for eval/metrics.py — the auditable pure-Python metric reference.

All expected values are hand-computed from the DCG formula:
  DCG@k = sum_{i=1..k} (2^grade_i - 1) / log2(i + 1)

No database, no I/O.
"""

from uuid import UUID

import pytest

from eval.metrics import bpref, dcg, mrr, ndcg, precision_at_k, recall_at_k

# Stable readable ids.
A, B, C, D = (UUID(int=i) for i in range(1, 5))


class TestDCG:
    def test_perfect_ordering(self) -> None:
        # [A:3, B:2, C:1] in that order.
        # DCG = 7/1 + 3/log2(3) + 1/log2(4) ≈ 9.3928
        labels = {A: 3, B: 2, C: 1}
        assert dcg([A, B, C], labels, k=3) == pytest.approx(9.3928, rel=1e-3)

    def test_reversed_ordering(self) -> None:
        # Same docs, reversed — worst possible for these grades.
        # DCG = 1/1 + 3/log2(3) + 7/log2(4) ≈ 6.3928
        labels = {A: 3, B: 2, C: 1}
        assert dcg([C, B, A], labels, k=3) == pytest.approx(6.3928, rel=1e-3)

    def test_unlabeled_docs_contribute_zero(self) -> None:
        labels = {A: 3}
        # B is unlabeled → grade defaults to 0 → gain is 0.
        assert dcg([A, B], labels, k=2) == pytest.approx(dcg([A], labels, k=1))

    def test_k_truncates_ranking(self) -> None:
        labels = {A: 3, B: 3}
        assert dcg([A, B], labels, k=1) < dcg([A, B], labels, k=2)

    def test_grade_zero_contributes_nothing(self) -> None:
        labels = {A: 0, B: 0}
        assert dcg([A, B], labels, k=2) == 0.0

    def test_empty_ranking_is_zero(self) -> None:
        assert dcg([], {A: 3}, k=10) == 0.0


class TestNDCG:
    def test_perfect_ranking_scores_one(self) -> None:
        labels = {A: 3, B: 2, C: 1}
        assert ndcg([A, B, C], labels, k=3) == pytest.approx(1.0)

    def test_reversed_ranking_known_value(self) -> None:
        # Hand-computed: 6.3928 / 9.3928 ≈ 0.6806
        labels = {A: 3, B: 2, C: 1}
        assert ndcg([C, B, A], labels, k=3) == pytest.approx(0.6806, rel=1e-3)

    def test_no_relevant_docs_returns_zero_not_crash(self) -> None:
        # All grades 0 → IDCG = 0 → guard must return 0.0, not ZeroDivisionError.
        labels = {A: 0, B: 0, C: 0}
        assert ndcg([A, B, C], labels, k=3) == 0.0

    def test_empty_labels_returns_zero(self) -> None:
        assert ndcg([A, B], {}, k=5) == 0.0

    def test_single_relevant_at_top_scores_one(self) -> None:
        labels = {A: 2}
        assert ndcg([A], labels, k=1) == pytest.approx(1.0)

    def test_k_larger_than_ranking_is_fine(self) -> None:
        labels = {A: 3}
        # Asking for k=10 when ranking has 1 item should not crash.
        val = ndcg([A], labels, k=10)
        assert 0.0 <= val <= 1.0

    def test_ndcg_bounded_zero_to_one(self) -> None:
        labels = {A: 3, B: 1, C: 2}
        for ranking in [[A, B, C], [C, B, A], [B, A, C]]:
            val = ndcg(ranking, labels, k=3)
            assert 0.0 <= val <= 1.0 + 1e-9


class TestMRR:
    def test_first_result_relevant(self) -> None:
        labels = {A: 2}
        assert mrr([A, B, C], labels) == pytest.approx(1.0)

    def test_relevant_at_position_three(self) -> None:
        # [grade0, grade0, grade2] → first relevant at rank 3 → MRR = 1/3.
        labels = {A: 0, B: 0, C: 2}
        assert mrr([A, B, C], labels) == pytest.approx(1.0 / 3)

    def test_no_relevant_returns_zero(self) -> None:
        labels = {A: 0, B: 0}
        assert mrr([A, B], labels) == 0.0

    def test_rel_threshold_respected(self) -> None:
        # Grade-1 doc should NOT count when threshold=2.
        labels = {A: 1, B: 2}
        assert mrr([A, B], labels, rel_threshold=2) == pytest.approx(0.5)  # B at pos 2

    def test_empty_ranking_returns_zero(self) -> None:
        assert mrr([], {A: 3}) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        labels = {A: 2, B: 3, C: 2}
        assert precision_at_k([A, B, C], labels, k=3) == pytest.approx(1.0)

    def test_none_relevant(self) -> None:
        labels = {A: 0, B: 1, C: 0}
        assert precision_at_k([A, B, C], labels, k=3, rel_threshold=2) == 0.0

    def test_mixed(self) -> None:
        # [A:3, B:0, C:2] at threshold=2 → 2 hits out of 3 → P@3 = 2/3
        labels = {A: 3, B: 0, C: 2}
        assert precision_at_k([A, B, C], labels, k=3) == pytest.approx(2.0 / 3)

    def test_k_truncates_to_top(self) -> None:
        # Only [A] is relevant; it's at position 1. P@1=1.0, P@2=0.5.
        labels = {A: 3, B: 0}
        assert precision_at_k([A, B], labels, k=1) == pytest.approx(1.0)
        assert precision_at_k([A, B], labels, k=2) == pytest.approx(0.5)

    def test_empty_ranking_returns_zero(self) -> None:
        assert precision_at_k([], {A: 3}, k=5) == 0.0


class TestRecallAtK:
    def test_all_retrieved(self) -> None:
        labels = {A: 2, B: 3}
        assert recall_at_k([A, B, C], labels, k=2) == pytest.approx(1.0)

    def test_partial_retrieved(self) -> None:
        # Labels: A:3, B:2 (both relevant at threshold=2), C:0.
        # Retrieved top-1: [A] → recall = 1/2 = 0.5.
        labels = {A: 3, B: 2, C: 0}
        assert recall_at_k([A], labels, k=1) == pytest.approx(0.5)

    def test_no_relevant_docs_returns_zero(self) -> None:
        labels = {A: 0, B: 1}
        assert recall_at_k([A, B], labels, k=2, rel_threshold=2) == 0.0

    def test_denominator_is_total_relevant_not_retrieved(self) -> None:
        # D is relevant but never retrieved — denominator is 3 (A, B, D).
        labels = {A: 2, B: 2, C: 0, D: 3}
        retrieved = [A, B]  # only 2 of 3 relevant docs retrieved
        assert recall_at_k(retrieved, labels, k=2) == pytest.approx(2.0 / 3)

    def test_k_limits_count(self) -> None:
        labels = {A: 2, B: 2, C: 2}
        # All three in ranking but k=1 — only A counts.
        assert recall_at_k([A, B, C], labels, k=1) == pytest.approx(1.0 / 3)


class TestBPref:
    """BPref: robust to incomplete judgments (Buckley & Voorhees 2004).

    Uses threshold=2 matching P@k / Recall@k conventions.
    relevant = docs with grade >= 2; non_relevant = docs with grade < 2.
    """

    def test_perfect_ordering_scores_one(self) -> None:
        # relevant={A:3, B:2}, non_relevant={C:1, D:0}, R=2, N=2, norm=2
        # A at pos 1: non_rel_seen=0 → 1 - 0/2 = 1.0
        # B at pos 2: non_rel_seen=0 → 1 - 0/2 = 1.0
        # bpref = (1.0 + 1.0) / 2 = 1.0
        labels = {A: 3, B: 2, C: 1, D: 0}
        assert bpref([A, B, C, D], labels) == pytest.approx(1.0)

    def test_worst_ordering_scores_zero(self) -> None:
        # Non-relevant docs all before relevant ones.
        # A at pos 3: non_rel_seen=2, norm=2 → 1 - 2/2 = 0.0
        # B at pos 4: non_rel_seen=2 → 0.0  → bpref = 0.0
        labels = {A: 3, B: 2, C: 1, D: 0}
        assert bpref([D, C, A, B], labels) == pytest.approx(0.0)

    def test_interleaved_ordering_hand_computed(self) -> None:
        # [C, A, D, B]: relevant={A,B}, non_relevant={C,D}
        # A at pos 2: non_rel_seen=1 (C) → 1 - 1/2 = 0.5
        # B at pos 4: non_rel_seen=2 (C,D) → 1 - 2/2 = 0.0
        # bpref = (0.5 + 0.0) / 2 = 0.25
        labels = {A: 3, B: 2, C: 1, D: 0}
        assert bpref([C, A, D, B], labels) == pytest.approx(0.25)

    def test_no_relevant_docs_returns_zero(self) -> None:
        labels = {A: 0, B: 1}
        assert bpref([A, B], labels) == 0.0

    def test_relevant_not_retrieved_scores_zero(self) -> None:
        # Relevant A and B are labeled but never appear in ranking.
        labels = {A: 3, B: 2, C: 0}
        assert bpref([C], labels) == pytest.approx(0.0)

    def test_no_non_relevant_in_labels_scores_one(self) -> None:
        # norm = min(R, N) = min(2, 0) = 0 → no penalty possible → 1.0
        labels = {A: 3, B: 2}
        assert bpref([A, B], labels) == pytest.approx(1.0)

    def test_empty_ranking_scores_zero(self) -> None:
        labels = {A: 3, B: 0}
        assert bpref([], labels) == pytest.approx(0.0)
