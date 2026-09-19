"""The tier-agnostic topic framework: job identity and persist_topic, on a throwaway database."""

import random
from uuid import UUID

import pytest
from sqlalchemy import func, select

from eval.embedding.models import EmbeddingJob, EmbeddingLabel, EmbeddingTopic, EmbeddingTopicJob
from eval.embedding.tiers.base import (
    JobRecord,
    LabelRecord,
    TopicExists,
    TopicPayload,
    job_uuid,
    persist_topic,
)


def make_job(origin_id: str, sha: str = "sha-a", embedding: list[float] | None = None) -> JobRecord:
    return JobRecord(
        origin="prod",
        origin_id=origin_id,
        source="greenhouse",
        title=f"Engineer {origin_id}",
        company="Acme",
        url=f"https://example.com/{origin_id}",
        location="Remote",
        remote=True,
        seniority=None,
        description="desc",
        requirements=None,
        responsibilities=None,
        content_hash=None,
        embed_text=f"text {origin_id} {sha}",
        embed_text_sha=sha,
        prod_embedding=embedding,
    )


def make_payload(name: str, jobs: list[JobRecord], labels: list[LabelRecord] | None = None):
    return TopicPayload(
        tier="A",
        name=name,
        profile_snapshot={"seniority": "senior"},
        query_inputs={"hyde_texts": ["h"]},
        jobs=jobs,
        labels=labels or [],
        builder="test",
    )


async def count(session, model) -> int:
    return await session.scalar(select(func.count()).select_from(model))


def test_job_uuid_is_deterministic_and_keyed_on_all_three_parts():
    base = job_uuid("prod", "1", "abc")
    assert base == job_uuid("prod", "1", "abc")
    assert isinstance(base, UUID)
    assert len({base, job_uuid("other", "1", "abc"), job_uuid("prod", "2", "abc")}) == 3
    assert job_uuid("prod", "1", "abd") != base


def test_job_uuid_parts_cannot_be_shifted_across_the_boundary():
    assert job_uuid("prod", "1x", "y") != job_uuid("prod", "1", "xy")


async def test_persist_topic_stores_a_frozen_topic(evals_session):
    payload = make_payload(
        "t1",
        [make_job("1", embedding=[0.1] * 768), make_job("2")],
        [LabelRecord("1", 3, "constructed", "great")],
    )
    summary = await persist_topic(evals_session, payload)

    assert (summary.n_jobs_new, summary.n_jobs_reused, summary.n_labels) == (2, 0, 1)
    topic = await evals_session.get(EmbeddingTopic, summary.topic_id)
    assert (topic.status, topic.tier, topic.name) == ("frozen", "A", "t1")
    assert topic.query_inputs == {"hyde_texts": ["h"]}
    assert await count(evals_session, EmbeddingTopicJob) == 2
    label = (await evals_session.scalars(select(EmbeddingLabel))).one()
    assert (label.grade, label.source, label.rationale) == (3, "constructed", "great")
    stored = await evals_session.get(EmbeddingJob, job_uuid("prod", "1", "sha-a"))
    assert len(stored.prod_embedding) == 768


async def test_second_persist_of_the_same_name_fails_and_changes_nothing(evals_session):
    await persist_topic(evals_session, make_payload("t1", [make_job("1")]))
    with pytest.raises(TopicExists):
        await persist_topic(evals_session, make_payload("t1", [make_job("1"), make_job("2")]))
    assert await count(evals_session, EmbeddingJob) == 1
    assert await count(evals_session, EmbeddingTopic) == 1


async def test_a_shared_posting_is_one_document_across_two_topics(evals_session):
    first = await persist_topic(evals_session, make_payload("t1", [make_job("1"), make_job("2")]))
    second = await persist_topic(evals_session, make_payload("t2", [make_job("2"), make_job("3")]))

    assert (first.n_jobs_new, first.n_jobs_reused) == (2, 0)
    assert (second.n_jobs_new, second.n_jobs_reused) == (1, 1)
    assert await count(evals_session, EmbeddingJob) == 3
    assert await count(evals_session, EmbeddingTopicJob) == 4


async def test_a_changed_embed_text_becomes_a_new_document(evals_session):
    await persist_topic(evals_session, make_payload("t1", [make_job("1", sha="old")]))
    second = await persist_topic(evals_session, make_payload("t2", [make_job("1", sha="new")]))

    assert (second.n_jobs_new, second.n_jobs_reused) == (1, 0)
    ids = set(await evals_session.scalars(select(EmbeddingJob.id)))
    assert ids == {job_uuid("prod", "1", "old"), job_uuid("prod", "1", "new")}


async def test_a_1400_job_payload_with_768_dim_vectors_persists(evals_session):
    rng = random.Random(0)
    jobs = [make_job(str(i), embedding=[rng.random() for _ in range(768)]) for i in range(1400)]
    labels = [LabelRecord(str(i), i % 4, "constructed") for i in range(0, 1400, 7)]

    summary = await persist_topic(evals_session, make_payload("big", jobs, labels))

    assert (summary.n_jobs_new, summary.n_labels) == (1400, len(labels))
    assert await count(evals_session, EmbeddingJob) == 1400
    assert await count(evals_session, EmbeddingTopicJob) == 1400
    assert await count(evals_session, EmbeddingLabel) == len(labels)


async def test_labels_are_attached_to_the_job_with_their_origin_id(evals_session):
    jobs = [make_job("a"), make_job("b")]
    await persist_topic(
        evals_session, make_payload("t1", jobs, [LabelRecord("b", 2, "constructed")])
    )

    row = (await evals_session.scalars(select(EmbeddingLabel))).one()
    assert row.job_id == job_uuid("prod", "b", "sha-a")


async def test_a_label_for_an_unknown_origin_id_is_an_error_and_writes_nothing(evals_session):
    payload = make_payload("t1", [make_job("1")], [LabelRecord("ghost", 1, "constructed")])
    with pytest.raises(ValueError, match="ghost"):
        await persist_topic(evals_session, payload)

    assert await count(evals_session, EmbeddingTopic) == 0
    assert await count(evals_session, EmbeddingJob) == 0


@pytest.mark.parametrize(
    "label",
    [LabelRecord("1", 4, "constructed"), LabelRecord("1", 1, "llm_judge")],
)
async def test_invalid_labels_are_rejected_before_anything_is_written(evals_session, label):
    with pytest.raises(ValueError):
        await persist_topic(evals_session, make_payload("t1", [make_job("1")], [label]))
    assert await count(evals_session, EmbeddingTopic) == 0


async def test_an_origin_id_with_two_embed_texts_in_one_payload_is_ambiguous(evals_session):
    jobs = [make_job("1", sha="x"), make_job("1", sha="y")]
    with pytest.raises(ValueError, match="appears twice"):
        await persist_topic(evals_session, make_payload("t1", jobs))
