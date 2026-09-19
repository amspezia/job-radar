from uuid import uuid4

from eval.embedding.checks import (
    REQUIRED_CHECKS,
    calibration_for_run,
    latest_checks,
    record_check,
)
from eval.embedding.models import EmbeddingJudgeRun, EmbeddingTopic


async def _topic(session, name: str = "topic"):
    topic = EmbeddingTopic(
        id=uuid4(),
        tier="A",
        name=name,
        status="frozen",
        profile_snapshot={},
        query_inputs={},
        builder="test",
    )
    session.add(topic)
    await session.commit()
    return topic.id


async def _judge_run(session, topic_id):
    run = EmbeddingJudgeRun(
        id=uuid4(),
        topic_id=topic_id,
        model="judge",
        prompt_version="v1",
        prompt_sha="sha",
        params={},
        fewshot=[],
        status="completed",
    )
    session.add(run)
    await session.commit()
    return run.id


def test_required_checks_are_the_two_parity_checks():
    assert REQUIRED_CHECKS == ("parity_ranking", "parity_reconstruction")


async def test_latest_checks_is_empty_before_anything_is_recorded(evals_session):
    topic_id = await _topic(evals_session)

    assert await latest_checks(evals_session, topic_id) == {}


async def test_record_check_round_trips_every_field(evals_session):
    topic_id = await _topic(evals_session)

    await record_check(evals_session, topic_id, "parity_ranking", True, {"max_abs_delta": 2e-7})

    row = (await latest_checks(evals_session, topic_id))["parity_ranking"]
    assert (row.name, row.passed, row.detail) == ("parity_ranking", True, {"max_abs_delta": 2e-7})
    assert row.judge_run_id is None
    assert row.created_at is not None


async def test_latest_checks_keeps_the_newest_row_per_name(evals_session):
    topic_id = await _topic(evals_session)
    await record_check(evals_session, topic_id, "parity_ranking", False, {"attempt": 1})
    await record_check(evals_session, topic_id, "parity_reconstruction", True, {"attempt": 1})
    await record_check(evals_session, topic_id, "parity_ranking", True, {"attempt": 2})

    latest = await latest_checks(evals_session, topic_id)

    assert set(latest) == {"parity_ranking", "parity_reconstruction"}
    assert (latest["parity_ranking"].passed, latest["parity_ranking"].detail) == (
        True,
        {"attempt": 2},
    )
    assert latest["parity_reconstruction"].detail == {"attempt": 1}


async def test_latest_checks_is_scoped_to_the_topic(evals_session):
    mine = await _topic(evals_session, "mine")
    other = await _topic(evals_session, "other")
    await record_check(evals_session, other, "parity_ranking", True, {})

    assert await latest_checks(evals_session, mine) == {}


async def test_calibration_for_run_returns_the_newest_for_that_run_only(evals_session):
    topic_id = await _topic(evals_session)
    run_a = await _judge_run(evals_session, topic_id)
    run_b = await _judge_run(evals_session, topic_id)
    await record_check(evals_session, topic_id, "judge_calibration", False, {"kappa": 0.4}, run_a)
    await record_check(evals_session, topic_id, "judge_calibration", True, {"kappa": 0.7}, run_a)
    await record_check(evals_session, topic_id, "judge_calibration", False, {"kappa": 0.1}, run_b)
    await record_check(evals_session, topic_id, "parity_ranking", True, {}, run_a)

    row = await calibration_for_run(evals_session, topic_id, run_a)

    assert row is not None
    assert (row.passed, row.detail, row.judge_run_id) == (True, {"kappa": 0.7}, run_a)


async def test_calibration_for_run_is_none_when_never_calibrated(evals_session):
    topic_id = await _topic(evals_session)
    run = await _judge_run(evals_session, topic_id)
    await record_check(evals_session, topic_id, "parity_ranking", True, {}, run)

    assert await calibration_for_run(evals_session, topic_id, run) is None
