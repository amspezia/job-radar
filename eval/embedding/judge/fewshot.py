"""Deterministic few-shot selection and the dev/held-out split.

Both are seeded and operate on sorted input, so a judge run can be reproduced
from `(labels, per_grade, seed)` alone. Exemplars are excluded from calibration;
the prompt is iterated on the dev half and reported on the held-out half.
"""

import random
from collections.abc import Hashable, Sequence

# Highest grade first: the order examples are drawn in, before shuffling.
GRADES = (3, 2, 1, 0)


def select_fewshot(labels: dict[Hashable, int], per_grade: int = 2, seed: int = 0) -> list:
    """Pick `per_grade` exemplars of each grade 0-3, shuffled, reproducibly."""
    rnd = random.Random(seed)
    picked: list[Hashable] = []
    for grade in GRADES:
        candidates = sorted((key for key, g in labels.items() if g == grade), key=str)
        if len(candidates) < per_grade:
            raise ValueError(
                f"grade {grade} has {len(candidates)} labeled items, need {per_grade} exemplars"
            )
        picked.extend(rnd.sample(candidates, per_grade))
    rnd.shuffle(picked)
    return picked


def split_dev_test(ids: Sequence[Hashable], seed: int = 0) -> tuple[list, list]:
    """Halve the ids into (dev, held-out); an odd one goes to the held-out half."""
    shuffled = sorted(ids, key=str)
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]
