import random
from itertools import count
from uuid import UUID, uuid4

import pytest

from eval.embedding.labels import (
    HUMAN_SOURCES,
    SOURCES,
    VIEWS,
    LabelRow,
    is_partial,
    resolve_effective,
)

RUN_A = uuid4()
RUN_B = uuid4()
_ids = count(1)


def row(job: UUID, grade: int, source: str, run: UUID | None = None, id: int | None = None):
    if source == "llm_judge" and run is None:
        run = RUN_A
    return LabelRow(job, grade, source, run, next(_ids) if id is None else id)


def test_contract_constants():
    assert SOURCES == ("human_blind", "constructed", "llm_judge")
    assert HUMAN_SOURCES == ("human_blind", "constructed")
    assert VIEWS == ("silver", "human", "effective")


def test_precedence_across_all_sources():
    job = uuid4()
    rows = [
        row(job, 0, "llm_judge", id=4),
        row(job, 2, "constructed", id=2),
        row(job, 3, "human_blind", id=1),
    ]
    assert resolve_effective(rows, "effective", RUN_A) == {job: 3}
    assert resolve_effective(rows[:2], "effective", RUN_A) == {job: 2}
    assert resolve_effective(rows[:1], "effective", RUN_A) == {job: 0}


def test_precedence_beats_recency():
    job = uuid4()
    rows = [row(job, 3, "human_blind", id=1), row(job, 0, "llm_judge", id=99)]
    assert resolve_effective(rows, "effective", RUN_A) == {job: 3}


def test_latest_wins_within_a_source():
    job = uuid4()
    rows = [row(job, 1, "constructed", id=5), row(job, 3, "constructed", id=9)]
    assert resolve_effective(rows, "human") == {job: 3}


def test_latest_is_by_id_not_by_position():
    job = uuid4()
    rows = [row(job, 3, "human_blind", id=9), row(job, 1, "human_blind", id=5)]
    assert resolve_effective(rows, "human") == {job: 3}


def test_retest_lowering_a_grade_supersedes_the_older_row():
    job = uuid4()
    rows = [row(job, 3, "human_blind", id=1), row(job, 1, "human_blind", id=2)]
    assert resolve_effective(rows, "human") == {job: 1}


def test_blind_row_supersedes_older_constructed_row_even_when_lowering():
    job = uuid4()
    rows = [row(job, 3, "constructed", id=1), row(job, 0, "human_blind", id=2)]
    assert resolve_effective(rows, "human") == {job: 0}
    assert resolve_effective(rows, "effective", RUN_A) == {job: 0}


def test_silver_view_counts_only_judge_rows():
    judged, human_only = uuid4(), uuid4()
    rows = [
        row(judged, 2, "llm_judge"),
        row(judged, 3, "human_blind"),
        row(human_only, 3, "constructed"),
    ]
    assert resolve_effective(rows, "silver", RUN_A) == {judged: 2}


def test_human_view_excludes_judge_rows():
    judged, labeled = uuid4(), uuid4()
    rows = [
        row(judged, 2, "llm_judge"),
        row(labeled, 1, "constructed"),
        row(labeled, 0, "llm_judge"),
    ]
    assert resolve_effective(rows, "human") == {labeled: 1}


def test_effective_view_covers_every_source():
    a, b, c = (uuid4() for _ in range(3))
    rows = [row(a, 3, "human_blind"), row(b, 2, "constructed"), row(c, 0, "llm_judge")]
    assert resolve_effective(rows, "effective", RUN_A) == {a: 3, b: 2, c: 0}


def test_job_present_only_in_a_lower_source_still_resolves():
    high, low = uuid4(), uuid4()
    rows = [row(high, 3, "human_blind"), row(low, 1, "llm_judge")]
    assert resolve_effective(rows, "effective", RUN_A) == {high: 3, low: 1}


def test_judge_rows_of_another_run_are_ignored():
    job = uuid4()
    rows = [
        row(job, 3, "llm_judge", RUN_B, id=10),
        row(job, 1, "llm_judge", RUN_A, id=5),
    ]
    assert resolve_effective(rows, "silver", RUN_A) == {job: 1}
    assert resolve_effective(rows, "silver", RUN_B) == {job: 3}


def test_job_labeled_only_by_another_judge_run_is_absent():
    job = uuid4()
    rows = [row(job, 2, "llm_judge", RUN_B)]
    assert resolve_effective(rows, "silver", RUN_A) == {}
    assert resolve_effective(rows, "effective", RUN_A) == {}


def test_pin_does_not_affect_non_judge_sources():
    job = uuid4()
    rows = [row(job, 2, "constructed")]
    assert resolve_effective(rows, "effective", RUN_A) == {job: 2}
    assert resolve_effective(rows, "effective", RUN_B) == {job: 2}


def test_human_view_ignores_the_pin():
    job = uuid4()
    rows = [row(job, 2, "human_blind"), row(job, 3, "llm_judge", RUN_B)]
    assert resolve_effective(rows, "human") == {job: 2}
    assert resolve_effective(rows, "human", RUN_A) == {job: 2}


@pytest.mark.parametrize("view", ["silver", "effective"])
def test_judge_views_require_a_pin(view):
    with pytest.raises(ValueError, match="judge_run_id"):
        resolve_effective([row(uuid4(), 1, "llm_judge")], view)


def test_human_view_needs_no_pin():
    assert resolve_effective([], "human") == {}


def test_unknown_view_rejected():
    with pytest.raises(ValueError, match="unknown view"):
        resolve_effective([], "gold", RUN_A)


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="unknown label source"):
        resolve_effective([row(uuid4(), 1, "crowd")], "human")


def test_unknown_source_rejected_even_when_the_view_would_exclude_it():
    with pytest.raises(ValueError, match="unknown label source"):
        resolve_effective([row(uuid4(), 1, "crowd")], "silver", RUN_A)


@pytest.mark.parametrize("grade", [-1, 4, 10])
def test_grade_outside_range_rejected(grade):
    with pytest.raises(ValueError, match=r"outside 0\.\.3"):
        resolve_effective([row(uuid4(), grade, "human_blind")], "human")


def test_empty_rows_resolve_to_empty():
    assert resolve_effective([], "effective", RUN_A) == {}


def test_result_is_deterministic_when_row_order_is_shuffled():
    jobs = [uuid4() for _ in range(20)]
    rng = random.Random(0)
    rows = [
        row(job, rng.randint(0, 3), source, RUN_A if source == "llm_judge" else None)
        for job in jobs
        for source in rng.choices(SOURCES, k=6)
    ]
    expected = resolve_effective(rows, "effective", RUN_A)
    for seed in range(10):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        assert resolve_effective(shuffled, "effective", RUN_A) == expected


def test_only_the_human_view_is_partial():
    assert {view for view in VIEWS if is_partial(view)} == {"human"}


def test_is_partial_rejects_unknown_view():
    with pytest.raises(ValueError, match="unknown view"):
        is_partial("gold")
