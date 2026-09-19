"""Judge-vs-human agreement statistics (pure; no numpy needed at k=4).

Reported by `judge-calibrate` and stored on the `judge_calibration` check.
Quadratic-weighted kappa is the gate (design §2.4: kappa_w >= 0.6); the
confusion matrix, the within-1 rate and the binary (grade >= 2) scores are what
make a failing kappa actionable.
"""

from collections.abc import Sequence


def _checked(y_true: Sequence[int], y_pred: Sequence[int], k: int) -> int:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} predicted")
    if not y_true:
        raise ValueError("no labeled pairs to compare")
    for value in (*y_true, *y_pred):
        if not 0 <= value < k:
            raise ValueError(f"grade {value!r} outside 0..{k - 1}")
    return len(y_true)


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], k: int = 4) -> list[list[int]]:
    """Counts with rows = true grade, columns = predicted grade."""
    _checked(y_true, y_pred, k)
    matrix = [[0] * k for _ in range(k)]
    for true, pred in zip(y_true, y_pred, strict=True):
        matrix[true][pred] += 1
    return matrix


def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int], k: int = 4) -> float:
    """Cohen's kappa with quadratic weights: 1.0 perfect, 0.0 chance, -1.0 opposite.

    Kappa is undefined when the expected disagreement is zero (both raters used a
    single, identical grade); by convention that returns 0.0 — no agreement beyond
    chance has been demonstrated.
    """
    n = _checked(y_true, y_pred, k)
    observed = confusion_matrix(y_true, y_pred, k)
    true_totals = [sum(row) for row in observed]
    pred_totals = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    numerator = 0.0
    denominator = 0.0
    for i in range(k):
        for j in range(k):
            weight = (i - j) ** 2 / (k - 1) ** 2
            numerator += weight * observed[i][j]
            denominator += weight * true_totals[i] * pred_totals[j] / n
    if denominator == 0:
        return 0.0
    return 1.0 - numerator / denominator


def agreement(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    n = _checked(y_true, y_pred, k=4)
    pairs = list(zip(y_true, y_pred, strict=True))
    return {
        "exact": sum(true == pred for true, pred in pairs) / n,
        "within1": sum(abs(true - pred) <= 1 for true, pred in pairs) / n,
    }


def binary_prf(
    y_true: Sequence[int], y_pred: Sequence[int], threshold: int = 2
) -> dict[str, float]:
    """Precision/recall/F1 of the coarse "relevant" call, plus both prevalences.

    The prevalences are the over-rating check: a judge can score well here while
    calling far more postings relevant than the human did (F24).
    """
    n = _checked(y_true, y_pred, k=4)
    pairs = list(zip(y_true, y_pred, strict=True))
    tp = sum(true >= threshold and pred >= threshold for true, pred in pairs)
    fp = sum(true < threshold and pred >= threshold for true, pred in pairs)
    fn = sum(true >= threshold and pred < threshold for true, pred in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prevalence_true": sum(true >= threshold for true, _ in pairs) / n,
        "prevalence_pred": sum(pred >= threshold for _, pred in pairs) / n,
    }
