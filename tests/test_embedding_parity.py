import hashlib
import json
from datetime import UTC, datetime
from functools import partial
from uuid import UUID, uuid4

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from eval.embedding import parity
from eval.embedding.checks import latest_checks
from eval.embedding.embedders.base import EmbedderSpec
from eval.embedding.models import (
    EmbeddingEmbedder,
    EmbeddingJob,
    EmbeddingLabel,
    EmbeddingRun,
    EmbeddingTopic,
    EmbeddingTopicJob,
    EmbeddingVector,
)
from eval.embedding.parity import ParityDoc, compare_prod_ranking, reconstruction_report
from eval.embedding.ranking import cosine_rank
from eval.evals_db import admin
from eval.evals_db.base import evals_session as open_evals_session
from eval.evals_db.base import make_engine
from eval.evals_db.prod_reader import prod_session as open_prod_session
from job_radar.db.models import Base as ProdBase
from job_radar.db.models import Job, Profile

DIM = 768  # production's jobs.embedding column is Vector(768)


def _unit(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _docs(vectors: np.ndarray, sources=None, extracted=None) -> list[ParityDoc]:
    n = len(vectors)
    return [
        ParityDoc(
            id=uuid4(),
            origin_id=uuid4(),
            source=sources[i] if sources else "src",
            requirements="req" if extracted is None or extracted[i] else None,
            responsibilities=None,
            prod_vector=vectors[i],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# parity_reconstruction (pure core)
# ---------------------------------------------------------------------------


def test_reconstruction_passes_when_every_vector_matches():
    rng = np.random.default_rng(0)
    stored = _unit(rng, 100, 16)
    cached = stored + rng.normal(scale=1e-6, size=stored.shape).astype(np.float32)
    docs = _docs(stored)
    labels = {doc.id: i % 4 for i, doc in enumerate(docs)}

    result = reconstruction_report(docs, cached, _unit(rng, 1, 16)[0], labels)

    detail = result["detail"]
    assert result["passed"] is True
    assert (detail["n"], detail["fraction"], detail["n_misses"]) == (100, 1.0, 0)
    assert detail["min_cosine"] > 0.999999
    assert detail["misses_by_source"] == {} and detail["misses_by_branch"] == {}


def test_reconstruction_fails_when_two_percent_of_vectors_are_degraded():
    rng = np.random.default_rng(1)
    stored = _unit(rng, 100, 16)
    cached = stored.copy()
    cached[:2] = _unit(rng, 2, 16)  # 2 of 100 no longer resemble production's vector
    docs = _docs(stored)

    result = reconstruction_report(docs, cached, _unit(rng, 1, 16)[0], {})

    assert result["passed"] is False
    assert result["detail"]["fraction"] == pytest.approx(0.98)
    assert result["detail"]["n_misses"] == 2
    assert result["detail"]["min_cosine"] < 0.999


def test_reconstruction_tolerates_one_miss_in_a_hundred():
    rng = np.random.default_rng(2)
    stored = _unit(rng, 100, 16)
    cached = stored.copy()
    cached[0] = _unit(rng, 1, 16)[0]

    result = reconstruction_report(_docs(stored), cached, _unit(rng, 1, 16)[0], {})

    assert result["passed"] is True
    assert result["detail"]["n_misses"] == 1


def test_reconstruction_groups_misses_by_source_and_by_branch():
    rng = np.random.default_rng(3)
    stored = _unit(rng, 6, 16)
    cached = stored.copy()
    cached[:4] = _unit(rng, 4, 16)
    sources = ["greenhouse", "greenhouse", "lever", "lever", "lever", "lever"]
    extracted = [True, False, False, False, True, True]  # docs 0-3 miss: 1 extracted, 3 fallback
    docs = _docs(stored, sources=sources, extracted=extracted)

    detail = reconstruction_report(docs, cached, _unit(rng, 1, 16)[0], {})["detail"]

    assert detail["misses_by_source"] == {"greenhouse": 2, "lever": 2}
    assert detail["misses_by_branch"] == {"extracted": 1, "fallback": 3}


def test_a_document_with_blank_requirements_and_responsibilities_is_the_fallback_branch():
    doc = ParityDoc(uuid4(), uuid4(), "s", "  ", "", np.zeros(2))
    extracted = ParityDoc(uuid4(), uuid4(), "s", None, "do things", np.zeros(2))

    assert parity._branch(doc) == "fallback"
    assert parity._branch(extracted) == "extracted"


def test_reconstruction_reports_a_noise_floor_of_about_zero_when_vectors_match():
    rng = np.random.default_rng(4)
    stored = _unit(rng, 60, 16)
    cached = stored + rng.normal(scale=1e-6, size=stored.shape).astype(np.float32)
    docs = _docs(stored)
    labels = {doc.id: i % 4 for i, doc in enumerate(docs)}

    noise = reconstruction_report(docs, cached, _unit(rng, 1, 16)[0], labels)["detail"][
        "noise_floor"
    ]

    assert noise["n_labeled"] == 60
    assert set(noise["stored"]) == set(noise["reembedded"]) == set(noise["delta"])
    assert "n_judged" in noise["stored"]
    assert noise["max_abs_delta"] == pytest.approx(0.0, abs=1e-9)


def test_reconstruction_noise_floor_shows_the_delta_when_the_vectors_differ():
    rng = np.random.default_rng(5)
    stored = _unit(rng, 60, 16)
    cached = _unit(rng, 60, 16)  # unrelated: the ranking, hence the metrics, move
    docs = _docs(stored)
    labels = {doc.id: i % 4 for i, doc in enumerate(docs)}

    detail = reconstruction_report(docs, cached, _unit(rng, 1, 16)[0], labels)["detail"]

    assert detail["noise_floor"]["max_abs_delta"] > 0.0


def test_reconstruction_without_human_labels_says_so_instead_of_failing():
    rng = np.random.default_rng(6)
    stored = _unit(rng, 10, 16)

    detail = reconstruction_report(_docs(stored), stored, _unit(rng, 1, 16)[0], {})["detail"]

    assert detail["noise_floor"]["n_labeled"] == 0
    json.dumps(detail)


def test_reconstruction_detail_is_json_serializable():
    rng = np.random.default_rng(7)
    stored = _unit(rng, 20, 16)
    docs = _docs(stored)
    labels = {doc.id: i % 4 for i, doc in enumerate(docs)}

    result = reconstruction_report(docs, stored, _unit(rng, 1, 16)[0], labels)

    json.dumps(result)


# ---------------------------------------------------------------------------
# parity_ranking (pure core, with a fake production result)
# ---------------------------------------------------------------------------


def _ranking_fixture(n: int = 8):
    """A harness ranking over embedding ids, and production ids that map onto them."""
    rng = np.random.default_rng(10)
    ids = [uuid4() for _ in range(n)]
    harness = cosine_rank(_unit(rng, n, 16), ids, _unit(rng, 1, 16)[0])
    origin_to_id = {uuid4(): doc_id for doc_id, _ in harness}
    id_to_origin = {doc_id: origin for origin, doc_id in origin_to_id.items()}
    prod = [(id_to_origin[doc_id], score) for doc_id, score in harness]
    return harness, prod, origin_to_id


def test_identical_rankings_are_ok_after_mapping_production_ids_to_embedding_ids():
    harness, prod, origin_to_id = _ranking_fixture()
    assert all(origin not in {doc_id for doc_id, _ in harness} for origin, _ in prod)

    result = compare_prod_ranking(harness, prod, origin_to_id, k=8)

    assert result["passed"] is True
    assert result["detail"]["ok"] is True and result["detail"]["reasons"] == []
    assert result["detail"]["max_abs_delta"] == 0.0
    assert result["detail"]["top_k_overlap"] == 8
    assert (result["detail"]["n_harness"], result["detail"]["n_prod"]) == (8, 8)


def test_sub_epsilon_score_noise_is_ok_and_reported():
    harness, prod, origin_to_id = _ranking_fixture()
    noisy = [(origin, score + 2.4e-7) for origin, score in prod]

    result = compare_prod_ranking(harness, noisy, origin_to_id, k=8)

    assert result["passed"] is True
    assert result["detail"]["max_abs_delta"] == pytest.approx(2.4e-7, abs=1e-9)


def test_byte_identical_ties_in_a_different_order_are_ok():
    a, b, c, d = (uuid4() for _ in range(4))
    harness = [(a, 0.9), (b, 0.8), (c, 0.8), (d, 0.7)]
    origin_to_id = {uuid4(): doc_id for doc_id in (a, b, c, d)}
    to_origin = {doc_id: origin for origin, doc_id in origin_to_id.items()}
    prod = [(to_origin[a], 0.9), (to_origin[c], 0.8), (to_origin[b], 0.8), (to_origin[d], 0.7)]

    assert compare_prod_ranking(harness, prod, origin_to_id, k=4)["passed"] is True


def test_swapped_non_tied_documents_are_not_ok():
    harness, prod, origin_to_id = _ranking_fixture()
    swapped = [prod[1], prod[0], *prod[2:]]

    result = compare_prod_ranking(harness, swapped, origin_to_id, k=8)

    assert result["passed"] is False
    assert result["detail"]["ok"] is False and result["detail"]["reasons"]


def test_a_document_production_does_not_return_is_not_ok():
    harness, prod, origin_to_id = _ranking_fixture()

    result = compare_prod_ranking(harness, prod[:-1], origin_to_id, k=8)

    assert result["passed"] is False
    assert result["detail"]["n_prod"] == 7


def test_ids_production_returns_outside_the_topic_are_not_ok():
    harness, prod, origin_to_id = _ranking_fixture()
    stranger = [(uuid4(), prod[0][1]), *prod[1:]]

    result = compare_prod_ranking(harness, stranger, origin_to_id, k=8)

    assert result["passed"] is False
    assert any("outside the topic" in reason for reason in result["detail"]["reasons"])


def test_compare_prod_ranking_detail_is_json_serializable():
    harness, prod, origin_to_id = _ranking_fixture()

    json.dumps(compare_prod_ranking(harness, prod, origin_to_id, k=8))


# ---------------------------------------------------------------------------
# verify_topic, end to end, against a second throwaway database that has the prod schema
# ---------------------------------------------------------------------------

SPEC = EmbedderSpec(
    name="incumbent", model="m", doc_prefix="doc: ", hyde_prefix="query: ", incumbent=True
)
FINGERPRINT = "fp-incumbent"
N_DOCS = 12


def _sha(text_: str) -> str:
    return hashlib.sha256(text_.encode()).hexdigest()


@pytest_asyncio.fixture
async def prod_url(evals_db_url):
    """An empty database with production's schema. Never the real one: its name is `_test_`."""
    url = admin.with_database(evals_db_url, f"job_radar_evals_test_{uuid4().hex[:8]}")
    admin.ensure_database(url)
    engine = make_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(ProdBase.metadata.create_all)
        yield url
    finally:
        await engine.dispose()
        admin.drop_database(url)


async def _seed_prod(url: str, origin_ids: list[UUID], vectors: np.ndarray) -> None:
    engine = make_engine(url)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            session.add(
                Profile(
                    source="real",
                    full_name="Test Person",
                    email="test@example.com",
                    links={},
                    work_history={},
                    cv_text="cv",
                    target_titles=[],
                    seniority="senior",
                    domains_keywords=[],
                    location_rules={},
                    remote_required=False,
                )
            )
            session.add_all(
                Job(
                    id=origin_id,
                    source="src",
                    source_type="board",
                    ingested_via="test",
                    url=f"https://example.com/{i}",
                    title=f"Job {i}",
                    company="Co",
                    description="d",
                    remote=True,
                    collected_at=datetime.now(UTC),
                    content_hash=f"hash-{i}",
                    embedding=vectors[i],
                )
                for i, origin_id in enumerate(origin_ids)
            )
            await session.commit()
    finally:
        await engine.dispose()


async def _seed_evals(session, origin_ids: list[UUID], vectors: np.ndarray, hyde: np.ndarray):
    """A topic whose incumbent run is complete and whose cache holds `vectors` and `hyde`."""
    topic = EmbeddingTopic(
        id=uuid4(),
        tier="A",
        name="topic",
        status="frozen",
        profile_snapshot={},
        query_inputs={"hyde_texts": ["hyde 0", "hyde 1"]},
        builder="test",
    )
    session.add_all(
        [
            topic,
            EmbeddingEmbedder(
                fingerprint=FINGERPRINT, name=SPEC.name, model=SPEC.model, digest="d", config={}
            ),
        ]
    )
    await session.flush()
    jobs = [
        EmbeddingJob(
            id=uuid4(),
            origin="prod",
            origin_id=str(origin_id),
            source="a" if i % 2 else "b",
            title=f"Job {i}",
            company="Co",
            url=f"https://example.com/{i}",
            description="d",
            requirements="req" if i % 3 else None,
            embed_text=f"embed text {i}",
            embed_text_sha=_sha(f"embed text {i}"),
            prod_embedding=vectors[i],
        )
        for i, origin_id in enumerate(origin_ids)
    ]
    session.add_all(jobs)
    await session.flush()
    session.add_all(
        [EmbeddingTopicJob(topic_id=topic.id, job_id=job.id) for job in jobs]
        + [
            EmbeddingLabel(topic_id=topic.id, job_id=job.id, grade=i % 4, source="human_blind")
            for i, job in enumerate(jobs)
        ]
        + [
            EmbeddingVector(
                fingerprint=FINGERPRINT, text_sha=_sha(SPEC.doc_prefix + job.embed_text), vector=v
            )
            for job, v in zip(jobs, vectors, strict=True)
        ]
        + [
            EmbeddingVector(
                fingerprint=FINGERPRINT,
                text_sha=_sha(SPEC.effective_hyde_prefix + f"hyde {i}"),
                vector=v,
            )
            for i, v in enumerate(hyde)
        ]
    )
    return topic


async def _add_completed_run(session, topic_id: UUID) -> None:
    session.add(
        EmbeddingRun(
            topic_id=topic_id,
            name=SPEC.name,
            embedder_fingerprint=FINGERPRINT,
            method_fingerprint="m",
            method_config={},
            status="completed",
        )
    )
    await session.commit()


@pytest.fixture
def point_at(monkeypatch, evals_db_url):
    def _point(prod_database_url: str | None = None) -> None:
        monkeypatch.setattr(parity, "load_specs", lambda: [SPEC])
        monkeypatch.setattr(parity, "evals_session", partial(open_evals_session, evals_db_url))
        if prod_database_url:
            monkeypatch.setattr(
                parity, "prod_session", partial(open_prod_session, prod_database_url)
            )

    return _point


async def test_verify_topic_passes_then_records_a_failure_when_the_cache_drifts(
    evals_session, prod_url, point_at
):
    rng = np.random.default_rng(20)
    vectors = _unit(rng, N_DOCS, DIM)
    origin_ids = [uuid4() for _ in range(N_DOCS)]
    await _seed_prod(prod_url, origin_ids, vectors)
    topic = await _seed_evals(evals_session, origin_ids, vectors, _unit(rng, 2, DIM))
    await _add_completed_run(evals_session, topic.id)
    point_at(prod_url)

    results = await parity.verify_topic("topic")

    assert set(results) == {"parity_ranking", "parity_reconstruction"}
    assert results["parity_ranking"]["passed"] is True, results["parity_ranking"]["detail"]
    assert results["parity_ranking"]["detail"]["max_abs_delta"] < 1e-5
    assert results["parity_reconstruction"]["passed"] is True
    assert results["parity_reconstruction"]["detail"]["noise_floor"]["n_labeled"] == N_DOCS
    recorded = await latest_checks(evals_session, topic.id)
    assert {name: row.passed for name, row in recorded.items()} == {
        "parity_ranking": True,
        "parity_reconstruction": True,
    }

    # Re-embedding "drifts": 4 of 12 cached document vectors stop matching production's.
    for i in range(4):
        await evals_session.execute(
            update(EmbeddingVector)
            .where(EmbeddingVector.text_sha == _sha(SPEC.doc_prefix + f"embed text {i}"))
            .values(vector=_unit(rng, 1, DIM)[0])
        )
    await evals_session.commit()

    again = await parity.verify_topic("topic")

    assert again["parity_reconstruction"]["passed"] is False
    assert again["parity_reconstruction"]["detail"]["n_misses"] == 4
    recorded = await latest_checks(evals_session, topic.id)
    assert recorded["parity_reconstruction"].passed is False


async def test_verify_topic_fails_ranking_when_production_disagrees(
    evals_session, prod_url, point_at
):
    rng = np.random.default_rng(21)
    vectors = _unit(rng, N_DOCS, DIM)
    origin_ids = [uuid4() for _ in range(N_DOCS)]
    production = vectors.copy()
    production[[0, 1]] = _unit(rng, 2, DIM)  # production's stored vectors differ from the import
    await _seed_prod(prod_url, origin_ids, production)
    topic = await _seed_evals(evals_session, origin_ids, vectors, _unit(rng, 2, DIM))
    await _add_completed_run(evals_session, topic.id)
    point_at(prod_url)

    results = await parity.verify_topic("topic")

    assert results["parity_ranking"]["passed"] is False
    assert results["parity_ranking"]["detail"]["reasons"]


async def test_verify_topic_asks_for_an_incumbent_run_first(evals_session, point_at):
    rng = np.random.default_rng(22)
    origin_ids = [uuid4() for _ in range(N_DOCS)]
    await _seed_evals(evals_session, origin_ids, _unit(rng, N_DOCS, DIM), _unit(rng, 2, DIM))
    await evals_session.commit()
    point_at()

    with pytest.raises(LookupError, match="run embedding-run for the incumbent first"):
        await parity.verify_topic("topic")


async def test_verify_topic_names_an_unknown_topic(evals_session, point_at):
    point_at()

    with pytest.raises(LookupError, match="no topic named 'nope'"):
        await parity.verify_topic("nope")
