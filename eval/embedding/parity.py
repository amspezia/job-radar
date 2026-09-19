"""Verify that the harness reproduces production's dense arm before any comparison is trusted.

Prod-touching, and one of only three modules allowed to import `job_radar.*` (and only
`db.models`, `retrieval.vector` and `retrieval.filters`). It never calls `build_hyde_embedding`:
the query vector is rebuilt from the cached HyDE embeddings, so nothing here needs Ollama.

Two checks, both recorded by `verify_topic`:

- `parity_ranking`: the incumbent's query vector ranked two ways — by production's own
  `search_vector` (profile filter, restricted to the topic's jobs) and by `cosine_rank` over the
  imported `prod_embedding` vectors — must agree under `compare_rankings` (tie-aware, eps 1e-5).
- `parity_reconstruction`: the incumbent's re-embedded documents must match `prod_embedding`
  (cosine >= 0.999 for >= 99% of documents), and the metric difference between the two vector
  sets is stored as the noise floor.
"""

import hashlib
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.cache import VectorCache
from eval.embedding.checks import record_check
from eval.embedding.embedders.base import EmbedderSpec, apply_dim
from eval.embedding.embedders.registry import load_specs
from eval.embedding.labels import LabelRow, resolve_effective
from eval.embedding.models import (
    EmbeddingJob,
    EmbeddingLabel,
    EmbeddingRun,
    EmbeddingTopic,
    EmbeddingTopicJob,
)
from eval.embedding.ranking import compare_rankings, cosine_rank
from eval.embedding.scoring import compute_metrics
from eval.evals_db.base import evals_session
from eval.evals_db.prod_reader import prod_session
from job_radar.db.models import Job, Profile
from job_radar.retrieval.filters import build_profile_filter
from job_radar.retrieval.vector import search_vector

RANKING_CHECK = "parity_ranking"
RECONSTRUCTION_CHECK = "parity_reconstruction"

TOP_K = 100
SCORE_EPS = 1e-5
COSINE_THRESHOLD = 0.999
MIN_MATCH_FRACTION = 0.99

_RERUN_HINT = "run embedding-run for the incumbent first"


@dataclass(frozen=True)
class ParityDoc:
    """One topic document that has a production vector to compare against."""

    id: UUID  # embedding_job.id
    origin_id: UUID  # production's jobs.id
    source: str
    requirements: str | None
    responsibilities: str | None
    prod_vector: np.ndarray


@dataclass(frozen=True)
class _TopicData:
    topic_id: UUID
    docs: list[ParityDoc]
    cached: np.ndarray  # incumbent's re-embedded documents, one row per doc, native dimension
    query: np.ndarray
    labels: dict[UUID, int]  # human view


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _branch(doc: ParityDoc) -> str:
    """Which embed-text branch built the document: the raw-description fallback or the extracted
    requirements/responsibilities."""
    empty = not (doc.requirements or "").strip() and not (doc.responsibilities or "").strip()
    return "fallback" if empty else "extracted"


def _row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        np.einsum("ij,ij->i", a, b), denom, out=np.zeros(a.shape[0]), where=denom > 0.0
    )


def compare_prod_ranking(
    harness: list[tuple[UUID, float]],
    prod: list[tuple[UUID, float]],
    origin_to_id: dict[UUID, UUID],
    *,
    k: int = TOP_K,
    eps: float = SCORE_EPS,
) -> dict:
    """Compare the harness ranking with production's, translating production's ids first.

    `prod` is `search_vector`'s `[(jobs.id, score)]`; `origin_to_id` maps `jobs.id` to
    `embedding_job.id`. Returns `{"passed": bool, "detail": dict}` (see `verify_topic`).
    """
    mapped = [(origin_to_id[job_id], score) for job_id, score in prod if job_id in origin_to_id]
    reasons = []
    if len(mapped) != len(prod):
        reasons.append(f"production returned {len(prod) - len(mapped)} ids outside the topic")
    result = compare_rankings(harness, mapped, k, eps)
    reasons.extend(result.reasons)

    shared = min(k, len(harness), len(mapped))
    deltas = [abs(harness[i][1] - mapped[i][1]) for i in range(shared)]
    top_harness = {doc_id for doc_id, _ in harness[:k]}
    top_prod = {doc_id for doc_id, _ in mapped[:k]}
    return {
        "passed": not reasons,
        "detail": {
            "ok": not reasons,
            "reasons": reasons,
            "k": k,
            "eps": eps,
            "max_abs_delta": max(deltas, default=0.0),
            "n_harness": len(harness),
            "n_prod": len(prod),
            "top_k_overlap": len(top_harness & top_prod),
        },
    }


def reconstruction_report(
    docs: list[ParityDoc],
    cached: np.ndarray,
    query: np.ndarray,
    labels: dict[UUID, int],
    *,
    threshold: float = COSINE_THRESHOLD,
    min_fraction: float = MIN_MATCH_FRACTION,
) -> dict:
    """Cosine of each cached (re-embedded) vector against its production vector.

    `cached[i]` belongs to `docs[i]`. Returns `{"passed": bool, "detail": dict}` (see
    `verify_topic`); `detail["noise_floor"]` is how much the human-view metrics move when the
    stored vectors are swapped for the re-embedded ones.
    """
    stored = np.vstack([doc.prod_vector for doc in docs])
    cosines = _row_cosines(cached, stored)
    missed = [doc for doc, cosine in zip(docs, cosines, strict=True) if cosine < threshold]
    fraction = float((cosines >= threshold).mean()) if len(docs) else 0.0

    by_source: dict[str, int] = {}
    by_branch: dict[str, int] = {}
    for doc in missed:
        by_source[doc.source] = by_source.get(doc.source, 0) + 1
        by_branch[_branch(doc)] = by_branch.get(_branch(doc), 0) + 1

    passed = bool(len(docs) and fraction >= min_fraction)
    return {
        "passed": passed,
        "detail": {
            "n": len(docs),
            "fraction": fraction,
            "threshold": threshold,
            "min_fraction": min_fraction,
            "min_cosine": float(cosines.min()) if len(docs) else None,
            "n_misses": len(missed),
            "misses_by_source": by_source,
            "misses_by_branch": by_branch,
            "noise_floor": _noise_floor(docs, stored, cached, query, labels),
        },
    }


def _noise_floor(
    docs: list[ParityDoc],
    stored: np.ndarray,
    cached: np.ndarray,
    query: np.ndarray,
    labels: dict[UUID, int],
) -> dict:
    """Human-view metrics under the stored vs the re-embedded vectors; their delta is the noise
    any candidate's gain has to clear."""
    if not labels:
        return {"n_labeled": 0, "note": "no human-view labels: noise floor not measured"}
    ids = [doc.id for doc in docs]
    metrics = {
        name: compute_metrics(
            [doc_id for doc_id, _ in cosine_rank(matrix, ids, query)], labels, partial=True
        )
        for name, matrix in (("stored", stored), ("reembedded", cached))
    }
    delta = {key: metrics["reembedded"][key] - metrics["stored"][key] for key in metrics["stored"]}
    return {
        "n_labeled": len(labels),
        **{name: {k: float(v) for k, v in scores.items()} for name, scores in metrics.items()},
        "delta": {k: float(v) for k, v in delta.items()},
        "max_abs_delta": max((abs(v) for v in delta.values()), default=0.0),
    }


async def _load_topic_data(session: AsyncSession, name: str, spec: EmbedderSpec) -> _TopicData:
    topic = (
        await session.execute(select(EmbeddingTopic).where(EmbeddingTopic.name == name))
    ).scalar_one_or_none()
    if topic is None:
        raise LookupError(f"no topic named {name!r}")

    fingerprint = await session.scalar(
        select(EmbeddingRun.embedder_fingerprint)
        .where(
            EmbeddingRun.topic_id == topic.id,
            EmbeddingRun.name == spec.name,
            EmbeddingRun.status == "completed",
            EmbeddingRun.embedder_fingerprint.is_not(None),
        )
        .order_by(EmbeddingRun.started_at.desc(), EmbeddingRun.id)
        .limit(1)
    )
    if fingerprint is None:
        raise LookupError(
            f"topic {name!r} has no completed run of the incumbent {spec.name!r}: {_RERUN_HINT}"
        )

    jobs = (
        await session.execute(
            select(EmbeddingJob)
            .join(EmbeddingTopicJob, EmbeddingTopicJob.job_id == EmbeddingJob.id)
            .where(EmbeddingTopicJob.topic_id == topic.id, EmbeddingJob.prod_embedding.is_not(None))
            .order_by(EmbeddingJob.id)
        )
    ).scalars()
    jobs = list(jobs)
    if not jobs:
        raise LookupError(f"topic {name!r} has no documents with a production embedding")
    hyde_texts = topic.query_inputs.get("hyde_texts") or []
    if not hyde_texts:
        raise ValueError(f"topic {name!r} has no query_inputs['hyde_texts']")

    doc_shas = [_sha(spec.doc_prefix + job.embed_text) for job in jobs]
    hyde_shas = [_sha(spec.effective_hyde_prefix + text) for text in hyde_texts]
    vectors = await VectorCache(session).get_many(fingerprint, doc_shas + hyde_shas)
    missing = {sha for sha in doc_shas + hyde_shas if sha not in vectors}
    if missing:
        raise LookupError(
            f"{len(missing)} vector(s) of the incumbent {spec.name!r} are not cached: {_RERUN_HINT}"
        )

    label_rows = await session.scalars(
        select(EmbeddingLabel).where(EmbeddingLabel.topic_id == topic.id)
    )
    labels = resolve_effective(
        [
            LabelRow(row.job_id, row.grade, row.source, row.judge_run_id, row.id)
            for row in label_rows
        ],
        "human",
    )
    return _TopicData(
        topic_id=topic.id,
        docs=[
            ParityDoc(
                id=job.id,
                origin_id=UUID(job.origin_id),
                source=job.source,
                requirements=job.requirements,
                responsibilities=job.responsibilities,
                prod_vector=job.prod_embedding,
            )
            for job in jobs
        ],
        cached=np.vstack([vectors[sha][0] for sha in doc_shas]),
        query=np.mean([apply_dim(vectors[sha][0], spec.dim) for sha in hyde_shas], axis=0),
        labels=labels,
    )


async def _prod_ranking(
    prod: AsyncSession, query: np.ndarray, origin_ids: list[UUID]
) -> list[tuple[UUID, float]]:
    """Production's own dense search, restricted to the topic's jobs."""
    profile = (
        (await prod.execute(select(Profile).where(Profile.source == "real"))).scalars().first()
    )
    if profile is None:
        raise LookupError("production has no real profile to build the retrieval filter from")
    # `build_profile_filter` returns None for a profile with no constraints, and and_(None, ...)
    # would compile to NULL, so None must be dropped rather than passed through.
    clauses = [c for c in (build_profile_filter(profile), Job.id.in_(origin_ids)) if c is not None]
    return await search_vector(prod, query.tolist(), TOP_K, and_(*clauses))


async def verify_topic(name: str) -> dict:
    """Run both parity checks for a topic against the incumbent, record them, and return them.

    Returns `{check_name: {"passed": bool, "detail": dict}}` with exactly the keys
    `parity_ranking` and `parity_reconstruction` (`checks.REQUIRED_CHECKS`); the CLI prints
    and gates on this shape. `detail` is JSON-serializable and is what was stored.

    Raises LookupError when the topic, the incumbent's completed run, or any cached vector is
    missing ("run embedding-run for the incumbent first").
    """
    spec = next(spec for spec in load_specs() if spec.incumbent)
    async with evals_session() as session:
        data = await _load_topic_data(session, name, spec)
        origin_to_id = {doc.origin_id: doc.id for doc in data.docs}
        if len(origin_to_id) != len(data.docs):
            raise ValueError(f"topic {name!r} holds the same production job more than once")
        async with prod_session() as prod:
            prod_ranking = await _prod_ranking(prod, data.query, list(origin_to_id))

        harness_ranking = cosine_rank(
            np.vstack([doc.prod_vector for doc in data.docs]),
            [doc.id for doc in data.docs],
            data.query,
        )
        results = {
            RANKING_CHECK: compare_prod_ranking(
                harness_ranking,
                prod_ranking,
                origin_to_id,
                k=min(TOP_K, len(data.docs)),
            ),
            RECONSTRUCTION_CHECK: reconstruction_report(
                data.docs, data.cached, data.query, data.labels
            ),
        }
        for check, outcome in results.items():
            await record_check(session, data.topic_id, check, outcome["passed"], outcome["detail"])
    return results
