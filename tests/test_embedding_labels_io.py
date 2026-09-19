"""labels_io: human labels survive a JSON round trip; judge labels and posting text never do."""

import json

import pytest
from sqlalchemy import select

from eval.embedding.labels_io import export_labels, import_labels
from eval.embedding.models import EmbeddingJob, EmbeddingJudgeRun, EmbeddingLabel, EmbeddingTopic
from eval.embedding.tiers.base import JobRecord, LabelRecord, TopicPayload, persist_topic


def make_job(origin_id: str, sha: str = "sha-a") -> JobRecord:
    return JobRecord(
        origin="prod",
        origin_id=origin_id,
        source="greenhouse",
        title=f"SECRET TITLE {origin_id}",
        company="Acme",
        url=f"https://example.com/{origin_id}",
        location=None,
        remote=True,
        seniority=None,
        description="SECRET DESCRIPTION",
        requirements=None,
        responsibilities=None,
        content_hash=None,
        embed_text=f"SECRET EMBED TEXT {origin_id}",
        embed_text_sha=sha,
        prod_embedding=None,
    )


async def make_topic(session, name: str, jobs: list[JobRecord], labels: list[LabelRecord]):
    payload = TopicPayload("A", name, {}, {}, jobs, labels, "test")
    return (await persist_topic(session, payload)).topic_id


async def labels_of(session, topic_id) -> list[tuple[str, int, str]]:
    rows = await session.execute(
        select(EmbeddingJob.origin_id, EmbeddingLabel.grade, EmbeddingLabel.source)
        .join(EmbeddingJob, EmbeddingJob.id == EmbeddingLabel.job_id)
        .where(EmbeddingLabel.topic_id == topic_id)
        .order_by(EmbeddingLabel.id)
    )
    return [tuple(row) for row in rows]


async def seed_labelled_topic(session):
    jobs = [make_job("1"), make_job("2"), make_job("3")]
    labels = [
        LabelRecord("1", 3, "human_blind"),
        LabelRecord("2", 1, "constructed"),
        LabelRecord("1", 2, "human_blind"),
    ]
    topic_id = await make_topic(session, "src", jobs, labels)
    run = EmbeddingJudgeRun(
        topic_id=topic_id,
        model="m",
        prompt_version="v1",
        prompt_sha="p",
        params={},
        fewshot=[],
        status="completed",
    )
    session.add(run)
    await session.flush()
    session.add(
        EmbeddingLabel(
            topic_id=topic_id,
            job_id=(await first_job_id(session)),
            grade=0,
            source="llm_judge",
            judge_run_id=run.id,
        )
    )
    await session.commit()
    return topic_id


async def first_job_id(session):
    return await session.scalar(select(EmbeddingJob.id).limit(1))


async def test_export_writes_only_human_labels_and_no_posting_text(evals_session, tmp_path):
    await seed_labelled_topic(evals_session)
    path = tmp_path / "labels.json"

    assert await export_labels(evals_session, "src", path) == 3

    text = path.read_text()
    assert "SECRET" not in text and "example.com" not in text
    records = json.loads(text)
    assert [(r["origin_id"], r["grade"], r["source"]) for r in records] == [
        ("1", 3, "human_blind"),
        ("2", 1, "constructed"),
        ("1", 2, "human_blind"),
    ]
    assert all(
        set(r) == {"origin_id", "embed_text_sha", "grade", "source", "rationale", "created_at"}
        for r in records
    )
    assert all(r["embed_text_sha"] == "sha-a" for r in records)


async def test_round_trip_into_a_rebuilt_topic_restores_labels_in_order(evals_session, tmp_path):
    await seed_labelled_topic(evals_session)
    path = tmp_path / "labels.json"
    await export_labels(evals_session, "src", path)
    rebuilt = await make_topic(evals_session, "rebuilt", [make_job("1"), make_job("2")], [])

    assert await import_labels(evals_session, "rebuilt", path) == 3

    assert await labels_of(evals_session, rebuilt) == [
        ("1", 3, "human_blind"),
        ("2", 1, "constructed"),
        ("1", 2, "human_blind"),
    ]
    again = tmp_path / "again.json"
    assert await export_labels(evals_session, "rebuilt", again) == 3
    first, second = json.loads(path.read_text()), json.loads(again.read_text())
    assert first == second  # created_at included: the timeline is restored, not restamped


async def test_import_skips_records_that_match_no_member_of_the_topic(evals_session, tmp_path):
    await seed_labelled_topic(evals_session)
    path = tmp_path / "labels.json"
    await export_labels(evals_session, "src", path)
    # "2" is absent, and "1" now has different embed text: only the matching sha attaches.
    topic_id = await make_topic(
        evals_session, "changed", [make_job("1", sha="sha-b"), make_job("3")], []
    )

    assert await import_labels(evals_session, "changed", path) == 0
    assert await labels_of(evals_session, topic_id) == []


async def test_import_is_idempotent(evals_session, tmp_path):
    await seed_labelled_topic(evals_session)
    path = tmp_path / "labels.json"
    await export_labels(evals_session, "src", path)
    target = await make_topic(evals_session, "target", [make_job("1"), make_job("2")], [])

    assert await import_labels(evals_session, "target", path) == 3
    assert await import_labels(evals_session, "target", path) == 0
    assert len(await labels_of(evals_session, target)) == 3
    assert await import_labels(evals_session, "src", path) == 0  # its own labels are not doubled


async def test_export_never_includes_the_judge_and_import_refuses_it(evals_session, tmp_path):
    await seed_labelled_topic(evals_session)
    path = tmp_path / "labels.json"
    await export_labels(evals_session, "src", path)
    assert "llm_judge" not in path.read_text()

    forged = tmp_path / "forged.json"
    forged.write_text(
        json.dumps(
            [
                {
                    "origin_id": "1",
                    "embed_text_sha": "sha-a",
                    "grade": 1,
                    "source": "llm_judge",
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="human source"):
        await import_labels(evals_session, "src", forged)


async def test_an_unknown_topic_is_an_error(evals_session, tmp_path):
    with pytest.raises(LookupError, match="nope"):
        await export_labels(evals_session, "nope", tmp_path / "x.json")
    with pytest.raises(LookupError, match="nope"):
        await import_labels(evals_session, "nope", tmp_path / "x.json")
    assert (await evals_session.scalars(select(EmbeddingTopic))).all() == []


async def test_round_trip_keeps_the_rationale(evals_session, tmp_path):
    # The rationale carries the labeling bucket and the user's note: without it a backup
    # could no longer tell a pooled label from a random one.
    labels = [LabelRecord("1", 2, "human_blind", "bucket=pooled; odd one")]
    await make_topic(evals_session, "src", [make_job("1")], labels)
    path = tmp_path / "labels.json"
    assert await export_labels(evals_session, "src", path) == 1
    assert json.loads(path.read_text())[0]["rationale"] == "bucket=pooled; odd one"

    rebuilt = await make_topic(evals_session, "rebuilt", [make_job("1")], [])
    assert await import_labels(evals_session, "rebuilt", path) == 1
    restored = await evals_session.scalar(
        select(EmbeddingLabel.rationale).where(EmbeddingLabel.topic_id == rebuilt)
    )
    assert restored == "bucket=pooled; odd one"
