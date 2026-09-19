"""The `embedding_*` schema as the rest of the harness relies on it.

Everything runs against a throwaway EVALS database (`evals_session`), which is truncated
before each test.
"""

from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.models import (
    EmbeddingCheck,
    EmbeddingEmbedder,
    EmbeddingJob,
    EmbeddingJudgeRun,
    EmbeddingLabel,
    EmbeddingRun,
    EmbeddingRunRanking,
    EmbeddingTopic,
    EmbeddingTopicJob,
    EmbeddingVector,
)

FINGERPRINT = "ollama:abc123:8192"


def _job(**overrides: object) -> EmbeddingJob:
    fields: dict[str, object] = {
        "id": uuid4(),
        "origin": "prod",
        "origin_id": "42",
        "source": "himalayas",
        "title": "Senior Python Engineer",
        "company": "Acme",
        "url": "https://example.test/42",
        "location": None,
        "remote": True,
        "seniority": "mid",
        "description": "Build things.",
        "requirements": "Python",
        "responsibilities": "Ship",
        "content_hash": None,
        "embed_text": "Senior Python Engineer\nPython\nShip",
        "embed_text_sha": "sha-42",
        "prod_embedding": None,
    }
    return EmbeddingJob(**(fields | overrides))


def _topic(name: str = "A-test") -> EmbeddingTopic:
    return EmbeddingTopic(
        tier="A",
        name=name,
        status="frozen",
        profile_snapshot={"seniority": "mid"},
        query_inputs={"hyde_texts": ["a"]},
        builder="tier_a",
    )


async def test_every_table_round_trips(evals_session: AsyncSession) -> None:
    topic, job = _topic(), _job()
    embedder = EmbeddingEmbedder(
        fingerprint=FINGERPRINT,
        name="nomic-v1.5",
        model="nomic-embed-text",
        digest="abc123",
        quantization="F16",
        runtime_version="0.30.11",
        config={"num_ctx": 8192},
    )
    evals_session.add_all([topic, job, embedder])
    await evals_session.flush()

    judge_run = EmbeddingJudgeRun(
        topic_id=topic.id,
        model="Gemma3:12b",
        prompt_version="v1",
        prompt_sha="sha-prompt",
        params={"clip": 400},
        fewshot=[str(job.id)],
        status="completed",
    )
    run = EmbeddingRun(
        topic_id=topic.id,
        name="nomic-v1.5",
        embedder_fingerprint=FINGERPRINT,
        method_fingerprint="dense:abc",
        method_config={"recipe": "hyde_mean"},
        status="completed",
    )
    evals_session.add_all(
        [
            EmbeddingTopicJob(topic_id=topic.id, job_id=job.id),
            judge_run,
            run,
            EmbeddingVector(
                fingerprint=FINGERPRINT, text_sha="sha-text", vector=np.zeros(3, dtype=np.float32)
            ),
        ]
    )
    await evals_session.flush()

    evals_session.add_all(
        [
            EmbeddingLabel(topic_id=topic.id, job_id=job.id, grade=3, source="constructed"),
            EmbeddingLabel(
                topic_id=topic.id,
                job_id=job.id,
                grade=2,
                source="llm_judge",
                judge_run_id=judge_run.id,
                rationale="close enough",
            ),
            EmbeddingRunRanking(run_id=run.id, job_id=job.id, rank=1, score=0.87),
            EmbeddingCheck(
                topic_id=topic.id, name="parity_ranking", passed=True, detail={"max_delta": 2.4e-7}
            ),
        ]
    )
    await evals_session.commit()

    # populate_existing: server defaults (`created_at`) are not loaded by the INSERT, and
    # an async session cannot lazily refresh them on attribute access.
    labels = (
        await evals_session.scalars(
            select(EmbeddingLabel)
            .order_by(EmbeddingLabel.id)
            .execution_options(populate_existing=True)
        )
    ).all()
    assert [label.source for label in labels] == ["constructed", "llm_judge"]
    # BIGINT IDENTITY, so "latest" is the largest id even inside one transaction, where
    # now() is constant.
    assert labels[0].id < labels[1].id
    assert labels[1].judge_run_id == judge_run.id
    assert all(label.created_at is not None for label in labels)

    assert (await evals_session.scalar(select(EmbeddingCheck))).detail == {"max_delta": 2.4e-7}
    assert (await evals_session.scalar(select(EmbeddingRunRanking))).score == pytest.approx(0.87)
    assert (await evals_session.scalar(select(EmbeddingTopicJob))).job_id == job.id
    assert (await evals_session.scalar(select(EmbeddingJob))).title == "Senior Python Engineer"
    assert (await evals_session.scalar(select(EmbeddingRun))).embedder_fingerprint == FINGERPRINT
    assert (await evals_session.scalar(select(EmbeddingJudgeRun))).params == {"clip": 400}
    assert (await evals_session.scalar(select(EmbeddingEmbedder))).quantization == "F16"


async def test_a_repeated_document_identity_is_rejected(evals_session: AsyncSession) -> None:
    # Same posting, same embed text — one document, whatever uuid is handed to it.
    evals_session.add_all([_job(), _job(id=uuid4())])

    with pytest.raises(IntegrityError, match="uq_embedding_job_identity"):
        await evals_session.flush()


async def test_a_grade_outside_the_rubric_is_rejected(evals_session: AsyncSession) -> None:
    topic, job = _topic(), _job()
    evals_session.add_all([topic, job])
    await evals_session.flush()
    evals_session.add(EmbeddingLabel(topic_id=topic.id, job_id=job.id, grade=4, source="llm_judge"))

    with pytest.raises(IntegrityError, match="ck_embedding_label_grade"):
        await evals_session.flush()


async def test_vectors_of_different_dimensions_share_one_column(
    evals_session: AsyncSession,
) -> None:
    # The whole point of an unconstrained `vector`: nomic (768) and bge-m3 (1024) cache
    # into the same table, and a 3-d test fixture works too.
    evals_session.add(
        EmbeddingEmbedder(
            fingerprint=FINGERPRINT,
            name="nomic-v1.5",
            model="nomic-embed-text",
            digest="abc123",
            config={},
        )
    )
    small = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    large = np.arange(1024, dtype=np.float32) / 1024
    evals_session.add_all(
        [
            EmbeddingVector(fingerprint=FINGERPRINT, text_sha="small", vector=small, n_tokens=7),
            EmbeddingVector(
                fingerprint=FINGERPRINT, text_sha="large", vector=large, truncated=True
            ),
        ]
    )
    await evals_session.commit()

    rows = {
        row.text_sha: row
        for row in (
            await evals_session.scalars(
                select(EmbeddingVector).execution_options(populate_existing=True)
            )
        ).all()
    }
    assert isinstance(rows["small"].vector, np.ndarray)
    assert rows["small"].vector.dtype == np.float32
    assert rows["small"].vector.shape == (3,)
    assert rows["large"].vector.shape == (1024,)
    np.testing.assert_allclose(rows["large"].vector, large)
    assert rows["small"].n_tokens == 7
    # `truncated` defaults to false in the database, not just in Python.
    assert rows["small"].truncated is False
    assert rows["large"].truncated is True
