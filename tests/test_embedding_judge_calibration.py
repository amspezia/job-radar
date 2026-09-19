"""Calibration statistics, checked against hand-computed values.

Quadratic weights are w(i, j) = (i - j)^2 / (k - 1)^2, k = 4, so w ranges 0 … 1.
kappa = 1 - sum(w * observed) / sum(w * expected), expected[i][j] = row[i] * col[j] / n.
"""

import pytest

from eval.embedding.judge.calibration import (
    agreement,
    binary_prf,
    confusion_matrix,
    quadratic_weighted_kappa,
)


class TestConfusionMatrix:
    def test_rows_are_the_true_grade(self) -> None:
        assert confusion_matrix([0, 1, 2, 3, 3], [0, 2, 2, 3, 0]) == [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
        ]

    def test_rejects_a_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            confusion_matrix([0, 1], [0])

    def test_rejects_an_out_of_range_grade(self) -> None:
        with pytest.raises(ValueError, match=r"outside 0\.\.3"):
            confusion_matrix([0, 4], [0, 3])

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="no labeled pairs"):
            confusion_matrix([], [])


class TestQuadraticWeightedKappa:
    def test_perfect_agreement(self) -> None:
        assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0

    def test_maximal_disagreement(self) -> None:
        # observed: w(0,3) + w(3,0) = 2; expected: (w(0,0)+w(0,3)+w(3,0)+w(3,3))/2 = 1.
        assert quadratic_weighted_kappa([0, 3], [3, 0]) == -1.0

    def test_worked_example(self) -> None:
        # y_true = 0,1,2,3   y_pred = 0,2,2,3 — one off-by-one error.
        # row totals = [1,1,1,1]; col totals = [1,0,2,1]; n = 4.
        # numerator   = w(1,2) = 1/9.
        # denominator = (1/4) * (17/9 + 7/9 + 5/9 + 11/9) = 10/9.
        # kappa = 1 - (1/9)/(10/9) = 0.9.
        assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 2, 2, 3]) == pytest.approx(0.9)

    def test_constant_agreeing_raters_have_no_expected_disagreement(self) -> None:
        # Undefined mathematically; 0.0 by convention (documented in the module).
        assert quadratic_weighted_kappa([2, 2, 2], [2, 2, 2]) == 0.0

    def test_a_constant_predictor_scores_zero(self) -> None:
        assert quadratic_weighted_kappa([0, 1, 2, 3], [2, 2, 2, 2]) == pytest.approx(0.0)


class TestAgreement:
    def test_exact_and_within_one(self) -> None:
        # exact: 0==0, 3==3 → 2/5; within 1 adds 1↔2 and 2↔3, but not 0↔2.
        assert agreement([0, 1, 2, 3, 0], [0, 2, 3, 3, 2]) == {"exact": 0.4, "within1": 0.8}


class TestBinaryPrf:
    def test_hand_computed(self) -> None:
        # threshold 2 → true relevant {0,1,4}, predicted relevant {0,2,4}:
        # tp = 2, fp = 1, fn = 1 → precision = recall = f1 = 2/3, both prevalences 3/5.
        scores = binary_prf([3, 2, 1, 0, 2], [2, 1, 2, 0, 3])
        assert scores == pytest.approx(
            {
                "precision": 2 / 3,
                "recall": 2 / 3,
                "f1": 2 / 3,
                "prevalence_true": 0.6,
                "prevalence_pred": 0.6,
            }
        )

    def test_no_predicted_positives_is_zero_not_a_division_error(self) -> None:
        scores = binary_prf([3, 2, 0], [1, 0, 0])
        assert (scores["precision"], scores["recall"], scores["f1"]) == (0.0, 0.0, 0.0)
        assert scores["prevalence_pred"] == 0.0

    def test_threshold_is_configurable(self) -> None:
        assert binary_prf([1, 0], [1, 0], threshold=1)["precision"] == 1.0
