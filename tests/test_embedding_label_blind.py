"""label_blind: candidate ordering, pooled/random strata, resume, skips and crash safety.

A scripted `input_fn` stands in for the person; nothing here touches Ollama or a real database.
"""

import re
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import insert, select

from eval.embedding.judge.base import JobView
from eval.embedding.judge.prompt import render_label_view
from eval.embedding.label_blind import _candidates, label_blind
from eval.embedding.models import (
    EmbeddingJob,
    EmbeddingLabel,
    EmbeddingRun,
    EmbeddingRunRanking,
    EmbeddingTopic,
    EmbeddingTopicJob,
)

TOPIC = "blind-topic"
N = 60
T0 = datetime(2026, 9, 1, tzinfo=UTC)

# alpha's top 30 is docs 0-29, beta's is docs 15-44: the runs disagree on 0-14 and 30-44
# (pooled), agree on 15-29, and neither reaches 45-59.
POOLED = set(range(0, 15)) | set(range(30, 45))
RANDOM = set(range(N)) - POOLED

SPANISH = frozenset({3, 20, 33, 50, 58})  # pooled and random ones
SPANISH_TEXT = "Buscamos un desarrollador con experiencia en el desarrollo para el equipo"


class Script:
    """Answers each prompt in turn; end of input (EOFError) when the script runs out."""

    def __init__(self, *answers: str) -> None:
        self._answers = iter(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        try:
            return next(self._answers)
        except StopIteration:
            raise EOFError from None


class Recorder:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str) -> None:
        self.lines.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def text_lines(self) -> list[str]:
        return self.text.splitlines()

    @property
    def shown(self) -> list[int]:
        return [int(m) for m in re.findall(r"Title   : posting-(\d+)", self.text)]


def grades(*pairs: tuple[str, str]) -> Script:
    """`grades(("3", ""), ("1", "note"))` answers grade 3 with no note, then grade 1 with one."""
    return Script(*(answer for pair in pairs for answer in pair))


async def _add_run(session, topic_id, docs, name, order, *, hours, status="completed"):
    run_id = uuid4()
    session.add(
        EmbeddingRun(
            id=run_id,
            topic_id=topic_id,
            name=name,
            method_fingerprint=f"mf-{name}",
            method_config={},
            status=status,
            started_at=T0 + timedelta(hours=hours),
        )
    )
    await session.flush()
    await session.execute(
        insert(EmbeddingRunRanking),
        [
            {"run_id": run_id, "job_id": docs[i], "rank": rank, "score": 1.0 / rank}
            for rank, i in enumerate(order, start=1)
        ],
    )


async def make_world(session, *, runs: bool = True, spanish: frozenset[int] = frozenset()):
    topic = EmbeddingTopic(
        id=uuid4(),
        tier="A",
        name=TOPIC,
        status="frozen",
        profile_snapshot={},
        query_inputs={},
        builder="test",
    )
    session.add(topic)
    docs = [
        EmbeddingJob(
            id=uuid4(),
            origin="test",
            origin_id=f"o-{i}",
            source="himalayas",
            title=f"posting-{i:03d}",
            company="Acme",
            url=f"https://example.test/{i}",
            description=SPANISH_TEXT if i in spanish else "builds backend services",
            requirements="python",
            embed_text="text",
            embed_text_sha=f"sha-{i}",
        )
        for i in range(N)
    ]
    session.add_all(docs)
    await session.flush()
    session.add_all(EmbeddingTopicJob(topic_id=topic.id, job_id=doc.id) for doc in docs)
    ids = [doc.id for doc in docs]
    if runs:
        rest = list(range(N))
        alpha = rest
        beta = list(range(15, N)) + list(range(15))
        stale = list(reversed(rest))  # superseded by the newer alpha: must not count
        variant = list(range(50, N)) + list(range(50))  # not primary: must not count
        failed = list(range(45, N)) + list(range(45))  # not completed: must not count
        await _add_run(session, topic.id, ids, "alpha", stale, hours=0)
        await _add_run(session, topic.id, ids, "alpha", alpha, hours=1)
        await _add_run(session, topic.id, ids, "beta", beta, hours=1)
        await _add_run(session, topic.id, ids, "beta:hyde_text:0", variant, hours=2)
        await _add_run(session, topic.id, ids, "gamma", failed, hours=3, status="failed")
    await session.commit()
    return topic.id, ids


@pytest.fixture
async def world(evals_session):
    return await make_world(evals_session)


def _index(ids, job_id) -> int:
    return ids.index(job_id)


async def _order(session, topic_id, ids, bucket="mixed", seed=0):
    unlabeled, strata = await _candidates(session, topic_id, bucket, seed)
    return [_index(ids, j) for j in unlabeled], {_index(ids, j): s for j, s in strata.items()}


async def _human_rows(session):
    rows = await session.scalars(
        select(EmbeddingLabel)
        .where(EmbeddingLabel.source == "human_blind")
        .order_by(EmbeddingLabel.id)
    )
    return list(rows)


# --- ordering and strata ---------------------------------------------------------------------


async def test_the_order_is_deterministic_per_seed(evals_session, world):
    topic_id, ids = world

    first, _ = await _order(evals_session, topic_id, ids, seed=4)
    again, _ = await _order(evals_session, topic_id, ids, seed=4)
    other, _ = await _order(evals_session, topic_id, ids, seed=5)

    assert first == again
    assert first != other
    assert sorted(first) == list(range(N))


async def test_pooled_is_in_the_top_30_of_some_but_not_all_latest_primary_runs(
    evals_session, world
):
    topic_id, ids = world

    order, strata = await _order(evals_session, topic_id, ids, bucket="pooled")

    assert set(order) == POOLED and len(order) == len(POOLED)
    assert {i for i, stratum in strata.items() if stratum == "pooled"} == POOLED
    assert {i for i, stratum in strata.items() if stratum == "random"} == RANDOM


async def test_random_is_every_document_outside_pooled(evals_session, world):
    topic_id, ids = world

    order, _ = await _order(evals_session, topic_id, ids, bucket="random")

    assert set(order) == RANDOM and len(order) == len(RANDOM)


async def test_random_needs_no_runs(evals_session):
    topic_id, ids = await make_world(evals_session, runs=False)

    order, strata = await _order(evals_session, topic_id, ids, bucket="random")

    assert sorted(order) == list(range(N))
    assert set(strata.values()) == {"random"}


async def test_mixed_interleaves_about_sixty_percent_pooled_with_forty_percent_random(
    evals_session, world
):
    topic_id, ids = world

    order, strata = await _order(evals_session, topic_id, ids, bucket="mixed")

    assert sorted(order) == list(range(N))
    head = order[:50]
    assert [strata[i] for i in head].count("pooled") == 30
    for start in range(0, 50, 5):  # blocks of 3 pooled + 2 random, shuffled within
        block = [strata[i] for i in order[start : start + 5]]
        assert (block.count("pooled"), block.count("random")) == (3, 2)


async def test_pooled_ignores_documents_that_are_already_labeled_but_they_drop_out(
    evals_session, world
):
    topic_id, ids = world
    full, _ = await _order(evals_session, topic_id, ids, bucket="pooled")
    evals_session.add_all(
        [
            EmbeddingLabel(topic_id=topic_id, job_id=ids[0], grade=2, source="human_blind"),
            EmbeddingLabel(topic_id=topic_id, job_id=ids[1], grade=1, source="constructed"),
            EmbeddingLabel(topic_id=topic_id, job_id=ids[2], grade=3, source="llm_judge"),
        ]
    )
    await evals_session.commit()

    order, _ = await _order(evals_session, topic_id, ids, bucket="pooled")

    assert 0 not in order and 1 not in order  # human-labeled: excluded
    assert 2 in order  # an LLM label is not a human label
    assert order == [i for i in full if i not in (0, 1)]  # the rest keep their seeded order


async def test_pooled_errors_without_two_completed_primary_runs(evals_session):
    await make_world(evals_session, runs=False)

    with pytest.raises(ValueError, match="completed primary runs"):
        await label_blind(evals_session, TOPIC, bucket="pooled", input_fn=Script(), output_fn=print)
    with pytest.raises(ValueError, match="completed primary runs"):
        await label_blind(evals_session, TOPIC, bucket="mixed", input_fn=Script(), output_fn=print)
    assert not await _human_rows(evals_session)


async def test_only_non_primary_or_unfinished_runs_do_not_count_as_runs(evals_session):
    topic_id, ids = await make_world(evals_session, runs=False)
    await _add_run(evals_session, topic_id, ids, "alpha", list(range(N)), hours=1)
    await _add_run(evals_session, topic_id, ids, "beta", list(range(N)), hours=1, status="running")
    await _add_run(evals_session, topic_id, ids, "beta:hyde_text:0", list(range(N)), hours=1)
    await evals_session.commit()

    with pytest.raises(ValueError, match="at least two completed primary runs"):
        await label_blind(evals_session, TOPIC, bucket="pooled", input_fn=Script())


# --- the session -----------------------------------------------------------------------------


async def test_a_label_is_one_human_blind_row_with_its_bucket_and_note(evals_session, world):
    topic_id, ids = world
    order, strata = await _order(evals_session, topic_id, ids, seed=1)
    out = Recorder()

    result = await label_blind(
        evals_session,
        TOPIC,
        n=3,
        seed=1,
        input_fn=grades(("3", "great remote role"), ("0", ""), (" 2 ", "  padded  ")),
        output_fn=out,
    )

    rows = await _human_rows(evals_session)
    assert [(ids.index(r.job_id), r.grade, r.source) for r in rows] == [
        (order[0], 3, "human_blind"),
        (order[1], 0, "human_blind"),
        (order[2], 2, "human_blind"),
    ]
    assert [r.rationale for r in rows] == [
        f"bucket={strata[order[0]]}; great remote role",
        f"bucket={strata[order[1]]}",
        f"bucket={strata[order[2]]}; padded",
    ]
    assert all(r.topic_id == topic_id and r.judge_run_id is None for r in rows)
    assert out.shown == order[:3]
    assert result["labeled"] == 3 and result["skipped"] == 0
    assert result["remaining"] == N - 3
    assert sum(result["buckets"].values()) == 3


async def test_the_bucket_of_a_single_stratum_session_is_recorded(evals_session, world):
    grades_in = grades(("1", ""), ("1", ""))

    result = await label_blind(evals_session, TOPIC, n=2, bucket="pooled", input_fn=grades_in)

    rows = await _human_rows(evals_session)
    assert {r.rationale for r in rows} == {"bucket=pooled"}
    assert result["buckets"] == {"pooled": 2}


async def test_n_caps_the_session_and_reports_the_rest(evals_session, world):
    result = await label_blind(
        evals_session, TOPIC, n=4, input_fn=grades(*[("2", "")] * 10), output_fn=Recorder()
    )

    assert result["labeled"] == 4 and result["remaining"] == N - 4
    assert len(await _human_rows(evals_session)) == 4


async def test_rerunning_with_the_same_seed_continues_where_the_user_stopped(evals_session, world):
    topic_id, ids = world
    order, _ = await _order(evals_session, topic_id, ids, seed=2)
    first, second = Recorder(), Recorder()

    await label_blind(
        evals_session, TOPIC, n=5, seed=2, input_fn=grades(*[("1", "")] * 5), output_fn=first
    )
    result = await label_blind(
        evals_session, TOPIC, n=5, seed=2, input_fn=grades(*[("1", "")] * 5), output_fn=second
    )

    assert first.shown == order[:5]
    assert second.shown == order[5:10]
    assert result["remaining"] == N - 10
    assert len(await _human_rows(evals_session)) == 10


async def test_skip_writes_nothing_and_does_not_use_up_a_slot(evals_session, world):
    topic_id, ids = world
    order, _ = await _order(evals_session, topic_id, ids)
    out = Recorder()

    result = await label_blind(
        evals_session,
        TOPIC,
        n=2,
        input_fn=Script("s", "3", "", "2", ""),
        output_fn=out,
    )

    rows = await _human_rows(evals_session)
    assert [ids.index(r.job_id) for r in rows] == order[1:3]
    assert order[0] not in [ids.index(r.job_id) for r in rows]
    assert (result["labeled"], result["skipped"]) == (2, 1)
    assert result["remaining"] == N - 2  # the skipped document is still unlabeled
    assert out.shown == order[:3]


async def test_a_skipped_document_comes_back_at_the_end_of_the_session(evals_session):
    topic_id, ids = await make_world(evals_session)
    evals_session.add_all(
        EmbeddingLabel(topic_id=topic_id, job_id=ids[i], grade=1, source="human_blind")
        for i in range(N - 2)
    )
    await evals_session.commit()
    order, _ = await _order(evals_session, topic_id, ids)
    assert sorted(order) == [N - 2, N - 1]
    out = Recorder()

    result = await label_blind(
        evals_session, TOPIC, n=2, input_fn=Script("s", "2", "", "3", ""), output_fn=out
    )

    assert out.shown == [order[0], order[1], order[0]]  # first, second, then the skipped one again
    assert result["labeled"] == 2 and result["skipped"] == 0 and result["remaining"] == 0
    rows = await _human_rows(evals_session)
    assert {ids.index(r.job_id): r.grade for r in rows if ids.index(r.job_id) >= N - 2} == {
        order[1]: 2,
        order[0]: 3,
    }


async def test_a_document_skipped_twice_is_not_offered_a_third_time(evals_session):
    topic_id, ids = await make_world(evals_session)
    evals_session.add_all(
        EmbeddingLabel(topic_id=topic_id, job_id=ids[i], grade=1, source="constructed")
        for i in range(N - 1)
    )
    await evals_session.commit()
    script = Script("s", "s", "3", "")

    result = await label_blind(evals_session, TOPIC, n=1, input_fn=script, output_fn=Recorder())

    assert len(script.prompts) == 2  # offered twice, never a third time
    assert (result["labeled"], result["skipped"], result["remaining"]) == (0, 1, 1)
    assert not await _human_rows(evals_session)


@pytest.mark.parametrize("stop", ["q", "Q", " q "])
async def test_quit_ends_the_session_after_the_labels_already_given(evals_session, world, stop):
    result = await label_blind(
        evals_session,
        TOPIC,
        n=10,
        input_fn=Script("2", "", "1", "", stop, "3", ""),
        output_fn=Recorder(),
    )

    assert result["labeled"] == 2
    assert len(await _human_rows(evals_session)) == 2


async def test_end_of_input_ends_the_session_cleanly(evals_session, world):
    result = await label_blind(
        evals_session, TOPIC, n=10, input_fn=Script("2", "", "1"), output_fn=Recorder()
    )

    # The grade "1" had no note prompt answered: the label is still kept, then the session ends.
    assert result["labeled"] == 2
    assert [r.grade for r in await _human_rows(evals_session)] == [2, 1]


async def test_ctrl_c_ends_the_session_cleanly(evals_session, world):
    answers = iter(["3", ""])

    def input_fn(prompt: str) -> str:
        try:
            return next(answers)
        except StopIteration:
            raise KeyboardInterrupt from None

    result = await label_blind(evals_session, TOPIC, n=10, input_fn=input_fn, output_fn=Recorder())

    assert result["labeled"] == 1
    assert [r.grade for r in await _human_rows(evals_session)] == [3]


async def test_every_label_is_committed_when_it_is_given(evals_session, world):
    """A rollback after an interrupted session must not lose a label that was already given."""
    calls = 0

    def input_fn(prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 5:  # the third grade prompt: the person stops here
            raise KeyboardInterrupt
        return "2" if calls % 2 == 1 else ""

    await label_blind(evals_session, TOPIC, n=10, input_fn=input_fn, output_fn=Recorder())
    await evals_session.rollback()

    assert len(await _human_rows(evals_session)) == 2


async def test_bad_input_asks_again_and_writes_nothing(evals_session, world):
    script = Script("x", "7", "-1", "", "1.5", "  3 ", "")
    out = Recorder()

    result = await label_blind(evals_session, TOPIC, n=1, input_fn=script, output_fn=out)

    assert result["labeled"] == 1
    assert [r.grade for r in await _human_rows(evals_session)] == [3]
    assert script.prompts[:6] == ["Grade [0-3 | s=skip | q=quit]: "] * 6
    assert script.prompts[6] == "Note (Enter = none): "
    assert out.lines.count("enter 0, 1, 2, 3, s or q") == 5


async def test_nothing_shown_reveals_a_grade_score_rank_run_or_bucket(evals_session, world):
    topic_id, ids = world
    evals_session.add_all(
        EmbeddingLabel(
            topic_id=topic_id,
            job_id=job_id,
            grade=3,
            source="llm_judge",
            rationale="ZZ-JUDGE-REASON",
        )
        for job_id in ids
    )
    await evals_session.commit()
    order, _ = await _order(evals_session, topic_id, ids, seed=3)
    out = Recorder()
    script = Script(*["2", ""] * 6)

    await label_blind(evals_session, TOPIC, n=6, seed=3, input_fn=script, output_fn=out)

    shown = out.text.lower()
    for forbidden in (
        "score",
        "rank",
        "bucket",
        "pooled",
        "random",
        "llm",
        "zz-",
        "alpha",
        "beta",
        "gamma",
    ):
        assert forbidden not in shown, forbidden
    assert all("bucket" not in p.lower() for p in script.prompts)
    # Every line is the rubric, a [k/n] counter, a blank, or the posting exactly as labelers see it.
    posting = render_label_view(
        JobView("posting-000", "Acme", "himalayas", None, "python", None, "x")
    ).splitlines()
    allowed = re.compile(
        r"^(|\[\d+/6\]|Grade each.*|  [0-3] .*|Location is not shown.*|Title   : posting-\d+|"
        + re.escape(posting[1])
        + r"|Source  : himalayas  \|  Seniority: unknown|Requires: python)$"
    )
    assert [line for line in out.text_lines if not allowed.match(line)] == []
    assert out.shown == order[:6]


async def test_the_header_states_the_rubric_and_the_hidden_location_once(evals_session, world):
    out = Recorder()

    await label_blind(evals_session, TOPIC, n=3, input_fn=grades(*[("1", "")] * 3), output_fn=out)

    text = out.text
    assert text.count("Location is not shown; judge role/stack fit from what you see.") == 1
    for line in ("3 Strong", "2 Relevant", "1 Marginal", "0 Not relevant"):
        assert text.count(line) == 1
    assert [line for line in out.text_lines if re.fullmatch(r"\[\d+/3\]", line)] == [
        "[1/3]",
        "[2/3]",
        "[3/3]",
    ]


async def test_nothing_left_to_label_returns_zero_counts(evals_session, world):
    topic_id, ids = world
    evals_session.add_all(
        EmbeddingLabel(topic_id=topic_id, job_id=job_id, grade=1, source="human_blind")
        for job_id in ids
    )
    await evals_session.commit()
    out = Recorder()

    result = await label_blind(evals_session, TOPIC, input_fn=Script(), output_fn=out)

    assert result == {"labeled": 0, "skipped": 0, "remaining": 0, "buckets": {}}
    assert out.lines == ["nothing left to label"]


async def test_invalid_arguments_and_unknown_topics_are_rejected(evals_session, world):
    with pytest.raises(ValueError, match="unknown bucket"):
        await label_blind(evals_session, TOPIC, bucket="hard", input_fn=Script())
    with pytest.raises(ValueError, match="at least 1"):
        await label_blind(evals_session, TOPIC, n=0, input_fn=Script())
    with pytest.raises(LookupError, match="no topic"):
        await label_blind(evals_session, "nope", input_fn=Script())


# --- language --------------------------------------------------------------------------------


async def _spanish_world(session):
    return await make_world(session, spanish=SPANISH)


@pytest.mark.parametrize("bucket", ["mixed", "pooled", "random"])
async def test_candidates_leave_out_other_languages_and_keep_the_seeded_order(
    evals_session, bucket
):
    topic_id, ids = await _spanish_world(evals_session)

    every, strata = await _candidates(evals_session, topic_id, bucket, 0)
    english, english_strata = await _candidates(evals_session, topic_id, bucket, 0, "en")
    spanish, _ = await _candidates(evals_session, topic_id, bucket, 0, "es")

    assert english == [j for j in every if ids.index(j) not in SPANISH]
    assert {ids.index(j) for j in spanish} == {ids.index(j) for j in every} & SPANISH
    assert english_strata == strata
    assert not {ids.index(j) for j in english} & SPANISH


async def test_the_default_never_offers_a_non_english_posting(evals_session):
    await _spanish_world(evals_session)
    shown = Recorder()

    counts = await label_blind(
        evals_session, TOPIC, n=N, input_fn=Script(*["1", ""] * N), output_fn=shown
    )

    assert sorted(shown.shown) == sorted(set(range(N)) - SPANISH)
    assert counts["labeled"] == N - len(SPANISH)
    assert counts["remaining"] == 0


async def test_skipped_postings_do_not_bring_a_non_english_one_back(evals_session):
    await _spanish_world(evals_session)
    shown = Recorder()

    await label_blind(evals_session, TOPIC, n=N, input_fn=Script(*["s", "s"] * N), output_fn=shown)

    assert set(shown.shown) == set(range(N)) - SPANISH


async def test_any_language_offers_every_posting(evals_session):
    await _spanish_world(evals_session)
    shown = Recorder()

    counts = await label_blind(
        evals_session, TOPIC, n=N, input_fn=Script(*["1", ""] * N), output_fn=shown, language=None
    )

    assert sorted(shown.shown) == list(range(N))
    assert counts["labeled"] == N


async def test_another_language_can_be_chosen(evals_session):
    await _spanish_world(evals_session)
    shown = Recorder()

    await label_blind(
        evals_session, TOPIC, n=N, input_fn=Script(*["1", ""] * N), output_fn=shown, language="es"
    )

    assert set(shown.shown) == SPANISH
