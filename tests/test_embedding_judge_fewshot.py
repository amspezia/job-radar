"""Few-shot selection and the dev/held-out split are reproducible from the seed."""

from uuid import UUID

import pytest

from eval.embedding.judge.fewshot import select_fewshot, split_dev_test

# Twelve labeled ids, three per grade.
LABELS: dict = {UUID(int=i): i % 4 for i in range(1, 13)}


class TestSelectFewshot:
    def test_is_deterministic_for_a_seed(self) -> None:
        assert select_fewshot(LABELS, per_grade=2, seed=7) == select_fewshot(
            LABELS, per_grade=2, seed=7
        )

    def test_covers_every_grade_the_requested_number_of_times(self) -> None:
        picked = select_fewshot(LABELS, per_grade=2, seed=7)
        assert len(picked) == 8
        assert len(set(picked)) == 8
        assert sorted(LABELS[key] for key in picked) == [0, 0, 1, 1, 2, 2, 3, 3]

    def test_insertion_order_does_not_change_the_selection(self) -> None:
        reversed_labels = dict(reversed(list(LABELS.items())))
        assert select_fewshot(LABELS, seed=3) == select_fewshot(reversed_labels, seed=3)

    def test_another_seed_selects_differently(self) -> None:
        seeds = {tuple(select_fewshot(LABELS, per_grade=2, seed=s)) for s in range(6)}
        assert len(seeds) > 1

    def test_raises_when_a_grade_has_too_few_examples(self) -> None:
        sparse = {UUID(int=1): 3, UUID(int=2): 2, UUID(int=3): 1, UUID(int=4): 0}
        with pytest.raises(ValueError, match="grade 3 has 1 labeled items"):
            select_fewshot(sparse, per_grade=2, seed=0)


class TestSplitDevTest:
    def test_is_deterministic_disjoint_and_complete(self) -> None:
        ids = list(LABELS)
        dev, held_out = split_dev_test(ids, seed=5)
        assert (dev, held_out) == split_dev_test(ids, seed=5)
        assert not set(dev) & set(held_out)
        assert sorted(dev + held_out, key=str) == sorted(ids, key=str)
        assert len(dev) == len(held_out) == 6

    def test_input_order_does_not_matter(self) -> None:
        ids = list(LABELS)
        assert split_dev_test(ids, seed=5) == split_dev_test(list(reversed(ids)), seed=5)

    def test_odd_count_puts_the_extra_id_in_the_held_out_half(self) -> None:
        dev, held_out = split_dev_test([UUID(int=i) for i in range(5)], seed=1)
        assert (len(dev), len(held_out)) == (2, 3)

    def test_empty_input(self) -> None:
        assert split_dev_test([], seed=1) == ([], [])
