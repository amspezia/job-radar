"""The generality claims of the embedding-eval harness, tested rather than asserted.

Tier-agnostic: an authored, closed-corpus topic and a Tier-A-shaped topic live in one database
and go through the same unchanged core (run, judge, calibrate, evaluate). Method-agnostic: a
ranker that is not an embedder goes through `run_method` and `evaluate_topic` with no embedder
anywhere.
"""

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import func, select

import eval.embedding as embedding_pkg
from eval.embedding.cache import VectorCache
from eval.embedding.checks import record_check
from eval.embedding.embedders.base import EmbedderSpec, EmbedResult, fingerprint
from eval.embedding.embedders.registry import load_specs
from eval.embedding.evaluate import evaluate_topic, format_report, write_report
from eval.embedding.judge.base import JobView, Judgment
from eval.embedding.judge.runner import calibrate, label_topic
from eval.embedding.labels import VIEWS
from eval.embedding.methods.base import TopicView
from eval.embedding.methods.dense import DenseMethod
from eval.embedding.models import (
    EmbeddingEmbedder,
    EmbeddingJob,
    EmbeddingLabel,
    EmbeddingRun,
    EmbeddingRunRanking,
    EmbeddingTopic,
    EmbeddingTopicJob,
    EmbeddingVector,
)
from eval.embedding.runner import run_method
from eval.embedding.tiers.base import JobRecord, LabelRecord, TopicPayload, persist_topic

N_DOCS = 40
X_NAME = "authored-closed"
A_NAME = "tier-a-shaped"
HYDE = ["python postgres remote async engineer", "remote python backend role", "async postgres"]
TOKENS = ("python", "postgres", "remote", "async")
GRADE_CYCLE = (0, 0, 1, 0, 2, 1, 3, 0, 2, 3)
DIM = 16
SNAPSHOT = {
    "seniority": "senior",
    "years_experience": 8,
    "target_titles": ["Platform Engineer"],
    "tech_stack": ["python", "postgres"],
    "domains": ["infra"],
    "remote_required": True,
    "cv_text": "scrubbed cv text",
}
INCUMBENT_SPEC = next(spec for spec in load_specs() if spec.incumbent)
OTHER_SPEC = EmbedderSpec(name="fake-other", model="fake-other", doc_prefix="doc: ", mrl=True)

# Reading any of these in a branch condition would make the core depend on where a topic came from.
TOPIC_ORIGIN_NAMES = {"tier", "origin", "builder"}
CORE_MODULES = (
    "runner.py",
    "cache.py",
    "checks.py",
    "labels.py",
    "scoring.py",
    "ranking.py",
    "evaluate.py",
    "methods/base.py",
    "methods/dense.py",
    "judge/runner.py",
    "tiers/base.py",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _job(origin: str, name: str, i: int, grade: int, prod_embedding=None) -> JobRecord:
    text = f"doc {name} {i} grade={grade} " + " ".join(TOKENS[: grade + 1])
    return JobRecord(
        origin=origin,
        origin_id=f"{name}-{i}",
        source="synthetic",
        title=f"Role {name} {i:03d}",
        company="Acme",
        url=f"https://example.test/{name}/{i}",
        location=None,
        remote=True,
        seniority="senior",
        description=f"Description of {name} {i}",
        requirements=None,
        responsibilities=None,
        content_hash=None,
        embed_text=text,
        embed_text_sha=_sha(text),
        prod_embedding=prod_embedding,
    )


def _grade(i: int, shift: int) -> int:
    return GRADE_CYCLE[(i + shift) % len(GRADE_CYCLE)]


class FakeTierBuilder:
    """An authored tier: a closed corpus, every document labeled by construction."""

    tier = "X"

    async def build(self, name: str, **opts) -> TopicPayload:
        grades = [_grade(i, shift=0) for i in range(N_DOCS)]
        jobs = [_job("authored", name, i, g) for i, g in enumerate(grades)]
        labels = [
            LabelRecord(job.origin_id, g, "constructed", "grade planted by the author")
            for job, g in zip(jobs, grades, strict=True)
        ]
        return TopicPayload(
            tier=self.tier,
            name=name,
            profile_snapshot=SNAPSHOT,
            query_inputs={"hyde_texts": HYDE, "closed_corpus": True},
            jobs=jobs,
            labels=labels,
            builder="fake-tier-x",
            notes="authored closed corpus",
        )


class FakeTierABuilder:
    """Shaped like Tier A: production postings, blind labels on part of the pool."""

    tier = "A"

    async def build(self, name: str, **opts) -> TopicPayload:
        grades = [_grade(i, shift=3) for i in range(N_DOCS)]
        jobs = [
            _job("prod", name, i, g, prod_embedding=[float(i % 5), 1.0, 0.0, 0.5])
            for i, g in enumerate(grades)
        ]
        labels = [
            LabelRecord(job.origin_id, g, "human_blind")
            for i, (job, g) in enumerate(zip(jobs, grades, strict=True))
            if i % 4 != 3
        ]
        return TopicPayload(
            tier=self.tier,
            name=name,
            profile_snapshot=SNAPSHOT,
            query_inputs={"hyde_texts": HYDE, "search_query": "remote python role"},
            jobs=jobs,
            labels=labels,
            builder="fake-tier-a",
        )


def _noise(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    vector = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


class SignalEmbedder:
    """Deterministic vectors: a document leans toward one axis in proportion to its grade."""

    def __init__(self, spec: EmbedderSpec, strength: float, digest: str) -> None:
        self.spec = spec
        self._strength = strength
        self._digest = digest
        self.calls: list[str] = []

    async def ready(self) -> None:
        return None

    async def embed(self, text: str) -> EmbedResult:
        self.calls.append(text)
        graded = re.search(r"grade=(\d)", text)
        lean = int(graded[1]) / 3 if graded else 1.0
        axis = np.zeros(DIM, dtype=np.float32)
        axis[0] = self._strength * lean
        vector = axis + _noise(text) * (1.0 if graded else 0.2)
        return EmbedResult(
            vector / np.linalg.norm(vector), n_tokens=len(text.split()), truncated=False
        )

    def describe(self) -> dict:
        return {
            "model": self.spec.model,
            "digest": self._digest,
            "quantization": "F16",
            "runtime_version": "0.0.0",
            "num_ctx": self.spec.num_ctx,
            "fingerprint": fingerprint(self.spec.backend, self._digest, self.spec.num_ctx),
        }


class FakeJudge:
    """Grades from a title -> grade table, off by one on every seventh title."""

    name = "fake-judge"

    def __init__(self, truth: dict[str, int]) -> None:
        self._truth = truth

    def describe(self) -> dict:
        return {"model": "fake-judge", "prompt_version": "c1", "prompt_sha": "contract-sha"}

    async def grade(self, job: JobView) -> Judgment:
        index = int(job.title.rsplit(" ", 1)[1])
        grade = self._truth[job.title]
        if index % 7 == 0:
            grade = min(3, grade + 1) if grade < 3 else 2
        return Judgment(grade=grade, rationale=f"because {job.title}")


class FakeMethod:
    """A ranker that is not an embedder: query-word overlap, ties broken by document id."""

    name = "lexical-fake"

    def __init__(self) -> None:
        self._query: set[str] = set()

    def fingerprint(self) -> str:
        return _sha("lexical-fake|v1")

    def describe(self) -> dict:
        return {"kind": "lexical-fake", "name": self.name}

    async def prepare(self, topic: TopicView) -> None:
        self._query = set(" ".join(topic.query_inputs["hyde_texts"]).split())

    def rank(self, topic: TopicView) -> list[tuple[UUID, float]]:
        scored = [
            (doc.id, float(len(self._query & set(doc.embed_text.split())))) for doc in topic.docs
        ]
        return sorted(scored, key=lambda item: (-item[1], str(item[0])))

    def stats(self) -> dict:
        return {}


@dataclass
class World:
    payloads: dict[str, TopicPayload]
    topic_ids: dict[str, UUID]

    def judge(self, name: str) -> FakeJudge:
        return FakeJudge({job.title: g for job, g in self._graded(name)})

    def _graded(self, name: str):
        shift = 0 if name == X_NAME else 3
        return [(job, _grade(i, shift)) for i, job in enumerate(self.payloads[name].jobs)]


@pytest_asyncio.fixture
async def world(evals_session):
    """Both topics, built and persisted by the identical code path, in one database."""
    payloads, topic_ids = {}, {}
    for builder, name in ((FakeTierBuilder(), X_NAME), (FakeTierABuilder(), A_NAME)):
        payloads[name] = await builder.build(name)
        topic_ids[name] = (await persist_topic(evals_session, payloads[name])).topic_id
    return World(payloads, topic_ids)


async def _record_parity(session, topic_id: UUID) -> None:
    await record_check(session, topic_id, "parity_ranking", True, {"max_abs_delta": 1e-7})
    await record_check(session, topic_id, "parity_reconstruction", True, {"n_docs": N_DOCS})


async def run_core(session, world: World, name: str) -> dict:
    """The whole unchanged core for one topic. It is handed a name and nothing about the tier."""
    topic_id = world.topic_ids[name]
    embedders = {
        INCUMBENT_SPEC.name: SignalEmbedder(INCUMBENT_SPEC, strength=1.2, digest="sha256:inc"),
        OTHER_SPEC.name: SignalEmbedder(OTHER_SPEC, strength=0.0, digest="sha256:other"),
    }
    runs = {}
    for embedder in embedders.values():
        method = DenseMethod(embedder, VectorCache(session))
        runs[method.name] = await run_method(session, name, method)

    judge_run = await label_topic(session, topic_id, world.judge(name), [])
    calibration = await calibrate(session, name, judge_run)
    await _record_parity(session, topic_id)
    reports = {
        view: await evaluate_topic(session, name, view, bootstrap=50, seed=3) for view in VIEWS
    }
    return {
        "runs": runs,
        "judge_run": judge_run,
        "calibration": calibration,
        "reports": reports,
        "n_embed_calls": {name: len(embedder.calls) for name, embedder in embedders.items()},
    }


@pytest_asyncio.fixture
async def core(evals_session, world):
    return {name: await run_core(evals_session, world, name) for name in (X_NAME, A_NAME)}


# ------------------------------------------------------------------ tier-agnostic


def _branch_reads(source: str) -> set[str]:
    """Names read inside any branch condition or comparison of `source`."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        tests = []
        if isinstance(node, ast.If | ast.IfExp | ast.While):
            tests.append(node.test)
        elif isinstance(node, ast.Compare):
            tests.append(node)
        for test in tests:
            for leaf in ast.walk(test):
                if isinstance(leaf, ast.Attribute) and leaf.attr in TOPIC_ORIGIN_NAMES:
                    found.add(leaf.attr)
                elif isinstance(leaf, ast.Name) and leaf.id in TOPIC_ORIGIN_NAMES:
                    found.add(leaf.id)
                elif isinstance(leaf, ast.Constant) and leaf.value in TOPIC_ORIGIN_NAMES:
                    found.add(leaf.value)
    return found


def test_the_branch_detector_sees_a_tier_branch():
    assert _branch_reads("x = 1 if topic.tier == 'A' else 2") == {"tier"}
    assert _branch_reads("if job.origin != 'prod':\n    pass") == {"origin"}
    assert _branch_reads("if row['builder']:\n    pass") == {"builder"}
    assert _branch_reads("rows = [j for j in jobs if j.ok]\nname = topic.tier") == set()


@pytest.mark.parametrize("module", CORE_MODULES)
def test_no_core_module_branches_on_where_a_topic_came_from(module):
    source = (Path(embedding_pkg.__file__).parent / module).read_text()

    assert _branch_reads(source) == set(), f"{module} reads tier/origin/builder in a condition"


async def test_two_tiers_persist_side_by_side_with_separate_corpora(evals_session, world):
    topics = {t.name: t for t in (await evals_session.scalars(select(EmbeddingTopic))).all()}
    assert {name: (t.tier, t.status) for name, t in topics.items()} == {
        X_NAME: ("X", "frozen"),
        A_NAME: ("A", "frozen"),
    }

    corpus = {
        name: set(
            await evals_session.scalars(
                select(EmbeddingTopicJob.job_id).where(EmbeddingTopicJob.topic_id == topic_id)
            )
        )
        for name, topic_id in world.topic_ids.items()
    }
    assert {len(ids) for ids in corpus.values()} == {N_DOCS}
    assert corpus[X_NAME].isdisjoint(corpus[A_NAME])
    origins = dict(
        (await evals_session.execute(select(EmbeddingJob.id, EmbeddingJob.origin))).all()
    )
    assert {origins[j] for j in corpus[X_NAME]} == {"authored"}
    assert {origins[j] for j in corpus[A_NAME]} == {"prod"}

    sources = {
        name: set(
            await evals_session.scalars(
                select(EmbeddingLabel.source).where(EmbeddingLabel.topic_id == topic_id)
            )
        )
        for name, topic_id in world.topic_ids.items()
    }
    assert sources == {X_NAME: {"constructed"}, A_NAME: {"human_blind"}}


async def test_the_same_core_produces_a_report_for_every_view_of_both_topics(core):
    for name, tier, n_labeled in ((X_NAME, "X", N_DOCS), (A_NAME, "A", 30)):
        reports = core[name]["reports"]
        assert set(reports) == set(VIEWS)
        for view, report in reports.items():
            assert (report["topic"], report["tier"], report["view"]) == (name, tier, view)
            assert report["stamps"] == []
            assert report["n_docs"] == N_DOCS
            assert report["incumbent"] == INCUMBENT_SPEC.name
            assert report["variants"] == [INCUMBENT_SPEC.name, OTHER_SPEC.name]
            assert set(report["metrics"]) == set(report["variants"])
            assert "ap" in report["metrics"][INCUMBENT_SPEC.name]
            assert format_report(report).startswith(f"Embedding eval: topic {name} (tier {tier})")
        # Human labels cover the whole closed corpus of X, but only what was blind-labeled for A.
        assert reports["human"]["labels"]["n_labeled"] == n_labeled
        assert reports["human"]["partial"] and not reports["silver"]["partial"]
        assert reports["silver"]["labels"]["n_labeled"] == N_DOCS


async def test_the_judge_and_its_calibration_ran_on_each_topic_alone(evals_session, world, core):
    for name, topic_id in world.topic_ids.items():
        judge_run = core[name]["judge_run"]
        judged = set(
            await evals_session.scalars(
                select(EmbeddingLabel.job_id).where(
                    EmbeddingLabel.topic_id == topic_id, EmbeddingLabel.judge_run_id == judge_run
                )
            )
        )
        corpus = set(
            await evals_session.scalars(
                select(EmbeddingTopicJob.job_id).where(EmbeddingTopicJob.topic_id == topic_id)
            )
        )
        assert judged == corpus
        assert core[name]["calibration"].passed
        assert core[name]["reports"]["silver"]["judge_run"]["id"] == str(judge_run)


async def test_every_ranking_covers_exactly_its_own_topic(evals_session, world, core):
    for name, topic_id in world.topic_ids.items():
        corpus = set(
            await evals_session.scalars(
                select(EmbeddingTopicJob.job_id).where(EmbeddingTopicJob.topic_id == topic_id)
            )
        )
        for run_id in core[name]["runs"].values():
            run = await evals_session.get(EmbeddingRun, run_id)
            ranked = list(
                await evals_session.scalars(
                    select(EmbeddingRunRanking.job_id).where(EmbeddingRunRanking.run_id == run_id)
                )
            )
            assert run.topic_id == topic_id and run.status == "completed"
            assert len(ranked) == N_DOCS and set(ranked) == corpus


async def test_one_topic_cannot_leak_into_the_other(evals_session, world, core):
    baseline = {
        view: await evaluate_topic(evals_session, X_NAME, view, bootstrap=50, seed=3)
        for view in VIEWS
    }
    assert baseline == core[X_NAME]["reports"]

    # Wreck topic A: every label and every ranking gone, judge labels included.
    a_id = world.topic_ids[A_NAME]
    await evals_session.execute(
        EmbeddingLabel.__table__.delete().where(EmbeddingLabel.topic_id == a_id)
    )
    await evals_session.execute(
        EmbeddingRunRanking.__table__.delete().where(
            EmbeddingRunRanking.run_id.in_(core[A_NAME]["runs"].values())
        )
    )
    await evals_session.commit()

    for view in VIEWS:
        assert (
            await evaluate_topic(evals_session, X_NAME, view, bootstrap=50, seed=3)
            == baseline[view]
        )
    with pytest.raises(ValueError):
        await evaluate_topic(evals_session, A_NAME, "human", bootstrap=0)


async def test_the_vector_cache_is_shared_across_topics_by_text_content(evals_session, core):
    # Topic A is embedded second: its documents are new texts, the HyDE texts are cache hits.
    for name in (INCUMBENT_SPEC.name, OTHER_SPEC.name):
        assert core[X_NAME]["n_embed_calls"][name] == N_DOCS + len(HYDE)
        assert core[A_NAME]["n_embed_calls"][name] == N_DOCS
    assert await evals_session.scalar(select(func.count()).select_from(EmbeddingEmbedder)) == 2
    n_vectors = await evals_session.scalar(select(func.count()).select_from(EmbeddingVector))
    # Per embedder: 2 x 40 distinct documents + the 3 shared HyDE texts.
    assert n_vectors == 2 * (2 * N_DOCS + len(HYDE))


# ------------------------------------------------------------------ method-agnostic


async def test_a_non_embedding_method_runs_and_evaluates_with_no_embedder_anywhere(
    evals_session, world, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "eval.embedding.evaluate.load_specs",
        lambda: [EmbedderSpec(name=FakeMethod.name, model="none", incumbent=True)],
    )
    method = FakeMethod()

    run_id = await run_method(evals_session, X_NAME, method, git_sha="abc1234")

    run = await evals_session.get(EmbeddingRun, run_id)
    assert run.status == "completed" and run.name == "lexical-fake"
    assert run.embedder_fingerprint is None
    assert run.docs_per_s is None and run.n_truncated is None
    assert run.method_config == method.describe() and run.method_fingerprint == method.fingerprint()
    ranks = list(
        (
            await evals_session.execute(
                select(EmbeddingRunRanking.rank, EmbeddingRunRanking.score)
                .where(EmbeddingRunRanking.run_id == run_id)
                .order_by(EmbeddingRunRanking.rank)
            )
        ).all()
    )
    assert [rank for rank, _ in ranks] == list(range(1, N_DOCS + 1))
    scores = [score for _, score in ranks]
    assert scores == sorted(scores, reverse=True)
    assert await evals_session.scalar(select(func.count()).select_from(EmbeddingEmbedder)) == 0
    assert await evals_session.scalar(select(func.count()).select_from(EmbeddingVector)) == 0

    topic_id = world.topic_ids[X_NAME]
    await _record_parity(evals_session, topic_id)
    judge_run = await label_topic(evals_session, topic_id, world.judge(X_NAME), [])
    await calibrate(evals_session, X_NAME, judge_run)
    for view in VIEWS:
        report = await evaluate_topic(evals_session, X_NAME, view, bootstrap=30)

        assert report["incumbent"] == report["variants"][0] == "lexical-fake"
        assert report["runs"]["lexical-fake"]["embedder"] is None
        assert report["runs"]["lexical-fake"]["git_sha"] == "abc1234"
        assert report["stamps"] == []
        assert "lexical-fake" in format_report(report)
        written = json.loads(write_report(report, tmp_path).read_text())
        assert written["runs"]["lexical-fake"]["embedder"] is None
    # Ranked by word overlap, the fake beats the random control: the numbers are real.
    silver = await evaluate_topic(evals_session, X_NAME, "silver", bootstrap=0)
    assert silver["controls"]["random"]["ap"] < silver["metrics"]["lexical-fake"]["ap"]


async def test_a_non_embedding_method_sits_beside_embedders_in_one_report(evals_session, world):
    topic_id = world.topic_ids[A_NAME]
    incumbent = SignalEmbedder(INCUMBENT_SPEC, strength=1.2, digest="sha256:inc")
    await run_method(evals_session, A_NAME, DenseMethod(incumbent, VectorCache(evals_session)))
    await run_method(evals_session, A_NAME, FakeMethod())
    await _record_parity(evals_session, topic_id)

    report = await evaluate_topic(evals_session, A_NAME, "human", bootstrap=30)

    assert report["incumbent"] == INCUMBENT_SPEC.name
    assert report["variants"] == [INCUMBENT_SPEC.name, "lexical-fake"]
    assert report["runs"]["lexical-fake"]["embedder"] is None
    dense_meta = report["runs"][INCUMBENT_SPEC.name]["embedder"]
    assert dense_meta["digest"] == "sha256:inc"
    assert set(report["bootstrap"]["metrics"]["ap"]) >= {"lexical-fake"}
    table = format_report(report)
    assert "lexical-fake" in table and "sha256:inc"[:12] in table
