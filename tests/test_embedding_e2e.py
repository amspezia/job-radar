"""One end-to-end scenario for the embedding-eval harness, fakes only.

A synthetic topic goes through persist, three fake embedder variants (with per-text HyDE recipes),
recorded parity checks, a good and a deliberately bad judge with calibration, and `evaluate_topic`
in every view, so the modules are proven to work together and the report gates are proven to bite.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from uuid import UUID

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select

from eval.embedding.cache import VectorCache
from eval.embedding.checks import calibration_for_run, record_check
from eval.embedding.embedders.base import EmbedderSpec, EmbedResult, fingerprint
from eval.embedding.embedders.registry import load_specs
from eval.embedding.evaluate import MissingChecks, evaluate_topic, format_report, write_report
from eval.embedding.judge.base import JobView, Judgment
from eval.embedding.judge.runner import CalibrationResult, calibrate, label_topic
from eval.embedding.labels_io import HUMAN_SOURCES, export_labels, import_labels
from eval.embedding.methods.dense import DenseMethod
from eval.embedding.models import EmbeddingLabel, EmbeddingVector
from eval.embedding.runner import run_method
from eval.embedding.tiers.base import (
    JobRecord,
    LabelRecord,
    TopicPayload,
    job_uuid,
    persist_topic,
)

TOPIC = "e2e-topic"
N_DOCS = 120
DIM = 16
GRADE_CYCLE = (0, 0, 1, 0, 2, 1, 3, 0, 2, 3)
HYDE = ["remote python role", "postgres backend engineer", "async remote"]
RECIPES = ("hyde_mean", "hyde_text:0", "hyde_text:1", "hyde_text:2")
INCUMBENT_SPEC = next(spec for spec in load_specs() if spec.incumbent)
INCUMBENT = INCUMBENT_SPEC.name
BETTER = "fake-better"
RANDOM = "fake-random"
VARIANTS = {
    INCUMBENT: (INCUMBENT_SPEC, 0.45, "sha256:incumbent"),
    BETTER: (
        EmbedderSpec(name=BETTER, model=BETTER, doc_prefix="d: ", hyde_prefix="q: "),
        4.0,
        "sha256:better",
    ),
    RANDOM: (EmbedderSpec(name=RANDOM, model=RANDOM), 0.0, "sha256:random"),
}
FEWSHOT_INDEXES = (0, 1, 3, 4)
SNAPSHOT = {"seniority": "senior", "target_titles": ["Platform Engineer"], "cv_text": "scrubbed"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _grade(i: int) -> int:
    return GRADE_CYCLE[i % len(GRADE_CYCLE)]


def _is_human_labeled(i: int) -> bool:
    return i % 3 != 2


def _payload() -> TopicPayload:
    jobs, labels = [], []
    for i in range(N_DOCS):
        text = f"doc {i} grade={_grade(i)}"
        jobs.append(
            JobRecord(
                origin="synthetic",
                origin_id=f"{TOPIC}-{i}",
                source="synthetic",
                title=f"Role {i:03d}",
                company="Acme",
                url=f"https://example.test/{i}",
                location=None,
                remote=True,
                seniority="senior",
                description=f"Description {i}",
                requirements=None,
                responsibilities=None,
                content_hash=None,
                embed_text=text,
                embed_text_sha=_sha(text),
                prod_embedding=None,
            )
        )
        if _is_human_labeled(i):
            labels.append(LabelRecord(f"{TOPIC}-{i}", _grade(i), "constructed"))
    return TopicPayload(
        tier="T",
        name=TOPIC,
        profile_snapshot=SNAPSHOT,
        query_inputs={"hyde_texts": HYDE},
        jobs=jobs,
        labels=labels,
        builder="e2e-fake",
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
    """`agreeing` follows the human grade, off by one on every seventh title; else inverts it."""

    name = "fake-judge"

    def __init__(self, *, agreeing: bool) -> None:
        self._agreeing = agreeing

    def describe(self) -> dict:
        version = "good" if self._agreeing else "bad"
        return {"model": "fake-judge", "prompt_version": version, "prompt_sha": f"sha-{version}"}

    async def grade(self, job: JobView) -> Judgment:
        index = int(job.title.removeprefix("Role "))
        truth = _grade(index)
        if not self._agreeing:
            grade = 3 - truth
        elif index % 7 == 0:
            grade = truth + 1 if truth < 3 else 2
        else:
            grade = truth
        return Judgment(grade=grade, rationale=f"{job.title}: {grade}")


@dataclass
class World:
    topic_id: UUID
    embedders: dict[str, SignalEmbedder]
    runs: dict[str, UUID]
    n_calls: dict[str, int]
    good_run: UUID
    bad_run: UUID
    good: CalibrationResult
    bad: CalibrationResult


@pytest_asyncio.fixture
async def world(evals_session) -> World:
    """Persisted, embedded and ranked, judged twice and calibrated. No parity checks yet."""
    session = evals_session
    topic_id = (await persist_topic(session, _payload())).topic_id

    embedders, runs = {}, {}
    for name, (spec, strength, digest) in VARIANTS.items():
        embedders[name] = SignalEmbedder(spec, strength, digest)
        for recipe in RECIPES:
            method = DenseMethod(embedders[name], VectorCache(session), recipe)
            runs[method.name] = await run_method(session, TOPIC, method, git_sha="e2e0001")
    n_calls = {name: len(embedder.calls) for name, embedder in embedders.items()}

    fewshot = [
        job_uuid("synthetic", f"{TOPIC}-{i}", _sha(f"doc {i} grade={_grade(i)}"))
        for i in FEWSHOT_INDEXES
    ]
    good_run = await label_topic(session, topic_id, FakeJudge(agreeing=True), fewshot)
    good = await calibrate(session, TOPIC, good_run)
    # The bad judge runs last, so it is the newest: a default pin has to skip it.
    bad_run = await label_topic(session, topic_id, FakeJudge(agreeing=False), fewshot)
    bad = await calibrate(session, TOPIC, bad_run)
    return World(topic_id, embedders, runs, n_calls, good_run, bad_run, good, bad)


async def _record_parity(session, topic_id: UUID) -> None:
    await record_check(session, topic_id, "parity_ranking", True, {"max_abs_delta": 1e-7})
    await record_check(session, topic_id, "parity_reconstruction", True, {"n_docs": N_DOCS})


@pytest_asyncio.fixture
async def verified(evals_session, world) -> World:
    await _record_parity(evals_session, world.topic_id)
    return world


async def _evaluate(session, view="silver", **kwargs) -> dict:
    kwargs.setdefault("bootstrap", 300)
    return await evaluate_topic(session, TOPIC, view, seed=5, **kwargs)


# ------------------------------------------------------------------ what the stages left behind


async def test_every_variant_and_recipe_has_a_completed_run_and_the_cache_holds_it_all(
    evals_session, world
):
    assert set(world.runs) == {
        name if recipe == "hyde_mean" else f"{name}:{recipe}"
        for name in VARIANTS
        for recipe in RECIPES
    }
    # Documents plus HyDE texts, once each: the three recipe runs cost no embed call.
    assert set(world.n_calls.values()) == {N_DOCS + len(HYDE)}
    n_vectors = await evals_session.scalar(select(func.count()).select_from(EmbeddingVector))
    assert n_vectors == len(VARIANTS) * (N_DOCS + len(HYDE))


async def test_calibration_passes_for_the_agreeing_judge_and_fails_for_the_bad_one(
    evals_session, world
):
    assert world.good.passed and world.good.kappa >= 0.6
    assert not world.bad.passed and world.bad.kappa < 0.6
    assert world.good.n_excluded_exemplars == len(FEWSHOT_INDEXES)  # the judge saw those grades

    good_check = await calibration_for_run(evals_session, world.topic_id, world.good_run)
    bad_check = await calibration_for_run(evals_session, world.topic_id, world.bad_run)
    assert good_check.passed and not bad_check.passed
    assert good_check.detail["kappa"] == pytest.approx(world.good.kappa)


# ------------------------------------------------------------------ the report gates


async def test_a_missing_or_failed_parity_check_blocks_the_report_until_it_passes(
    evals_session, world
):
    with pytest.raises(MissingChecks) as missing:
        await _evaluate(evals_session, bootstrap=0)
    assert missing.value.checks == ["parity_ranking (missing)", "parity_reconstruction (missing)"]

    stamped = await _evaluate(evals_session, bootstrap=0, allow_unverified=True)
    assert stamped["stamps"] == ["UNVERIFIED"]
    assert "UNVERIFIED" in format_report(stamped)

    await record_check(evals_session, world.topic_id, "parity_ranking", False, {"max_abs_delta": 1})
    await record_check(evals_session, world.topic_id, "parity_reconstruction", True, {})
    with pytest.raises(MissingChecks) as failed:
        await _evaluate(evals_session, bootstrap=0)
    assert failed.value.checks == ["parity_ranking (failed)"]

    # Checks append: the newest row per name decides, so a re-run that passes clears the gate.
    await record_check(evals_session, world.topic_id, "parity_ranking", True, {})
    assert (await _evaluate(evals_session, bootstrap=0))["stamps"] == []


async def test_silver_defaults_to_the_calibrated_judge_run_not_the_newest(evals_session, verified):
    report = await _evaluate(evals_session)

    assert report["stamps"] == []
    assert report["judge_run"]["id"] == str(verified.good_run)
    assert report["checks"]["judge_calibration"]["status"] == "passed"
    assert report["labels"]["n_labeled"] == N_DOCS and not report["partial"]
    assert report["variants"] == [INCUMBENT, BETTER, RANDOM]


async def test_a_run_pinned_to_the_failed_calibration_is_refused_or_stamped_uncalibrated(
    evals_session, verified
):
    with pytest.raises(MissingChecks) as refused:
        await _evaluate(evals_session, judge_run_id=verified.bad_run)
    assert len(refused.value.checks) == 1
    assert refused.value.checks[0].startswith(f"judge_calibration for judge run {verified.bad_run}")
    assert "(failed)" in refused.value.checks[0]

    stamped = await _evaluate(evals_session, judge_run_id=verified.bad_run, allow_unverified=True)
    assert stamped["stamps"] == ["UNCALIBRATED"]
    assert stamped["judge_run"]["id"] == str(verified.bad_run)
    assert stamped["checks"]["judge_calibration"]["status"] == "failed"
    banner = format_report(stamped)
    assert "UNCALIBRATED" in banner and "Do not quote these numbers" in banner

    # The same holds for the effective view, which also includes the judge.
    with pytest.raises(MissingChecks):
        await _evaluate(evals_session, "effective", judge_run_id=verified.bad_run)


async def test_the_effective_view_needs_the_same_gates_and_labels_every_document(
    evals_session, verified
):
    report = await _evaluate(evals_session, "effective")

    assert report["stamps"] == [] and not report["partial"]
    assert report["judge_run"]["id"] == str(verified.good_run)
    assert report["labels"]["n_labeled"] == N_DOCS
    assert report["labels"]["recall_ceiling@100"] is not None


# ------------------------------------------------------------------ what the numbers say


async def test_the_better_variant_beats_the_incumbent_and_the_random_one_does_not(
    evals_session, verified
):
    report = await _evaluate(evals_session)

    ap = {name: report["metrics"][name]["ap"] for name in report["variants"]}
    assert ap[BETTER] > ap[INCUMBENT] > ap[RANDOM]
    assert report["bootstrap"]["n"] == 300 and report["bootstrap"]["condensed"] is False
    for metric in ("ap", "ndcg@100"):
        better = report["bootstrap"]["metrics"][metric][BETTER]
        random = report["bootstrap"]["metrics"][metric][RANDOM]
        assert better["excludes_zero"] and better["low"] > 0 and better["mean_diff"] > 0
        assert random["excludes_zero"] and random["high"] < 0 < better["low"]


async def test_negative_controls_score_below_the_incumbent_so_the_comparison_can_separate(
    evals_session, verified
):
    for view in ("silver", "effective"):
        report = await _evaluate(evals_session, view, bootstrap=0)

        incumbent = report["metrics"][INCUMBENT]
        for control in ("random", "shuffled_tail"):
            for metric in ("ndcg@100", "p@100", "ap"):
                assert report["controls"][control][metric] < incumbent[metric], (control, metric)
        assert report["control_warning"] is None


async def test_the_human_view_is_partial_and_produces_no_at_100_metric(evals_session, verified):
    report = await _evaluate(evals_session, "human")

    assert report["partial"] and report["judge_run"] is None
    assert "judge_calibration" not in report["checks"]
    n_human = sum(_is_human_labeled(i) for i in range(N_DOCS))
    assert report["labels"]["n_labeled"] == n_human
    assert report["labels"]["recall_ceiling@100"] is None
    assert report["bootstrap"]["condensed"] is True
    scored = [*report["metrics"].values(), *report["controls"].values()]
    for metrics in scored:
        assert "ap" in metrics
        assert not {"ndcg@100", "p@100", "recall@100"} & set(metrics), sorted(metrics)
    silver = await _evaluate(evals_session, bootstrap=0)
    assert "ndcg@100" in silver["metrics"][INCUMBENT] and "recall@100" in silver["metrics"][BETTER]


async def test_per_text_recipes_are_reported_as_query_sensitivity(evals_session, verified):
    report = await _evaluate(evals_session, bootstrap=0)

    sensitivity = report["query_sensitivity"]
    assert set(sensitivity["variants"]) == set(VARIANTS)
    assert all(list(recipes) == list(RECIPES) for recipes in sensitivity["variants"].values())
    assert sensitivity["stability"]["recipes"] == list(RECIPES)
    assert sensitivity["stability"]["agree"]  # better > incumbent > random under every recipe
    assert report["variants"] == [INCUMBENT, BETTER, RANDOM]  # recipe runs are not variants


# ------------------------------------------------------------------ reporting


async def test_the_report_renders_and_its_json_round_trips_in_every_view(
    evals_session, verified, tmp_path
):
    for view in ("human", "silver", "effective"):
        report = await _evaluate(evals_session, view, bootstrap=100)

        text = format_report(report)
        assert f"topic {TOPIC}" in text and f"view: {view}" in text
        for name in VARIANTS:
            assert name in text
        path = write_report(report, tmp_path)
        loaded = json.loads(path.read_text())
        assert path.name.startswith(f"embedding-{TOPIC}-{view}-") and path.suffix == ".json"
        for key in ("topic", "view", "incumbent", "variants", "stamps", "metrics", "controls"):
            assert loaded[key] == report[key], key
        assert loaded["bootstrap"] == json.loads(json.dumps(report["bootstrap"]))
        assert loaded["runs"][BETTER]["embedder"]["digest"] == "sha256:better"


# ------------------------------------------------------------------ the cache


async def test_a_second_run_of_any_variant_is_a_pure_cache_hit(evals_session, verified):
    n_vectors = await evals_session.scalar(select(func.count()).select_from(EmbeddingVector))

    for name, embedder in verified.embedders.items():
        for recipe in RECIPES:
            method = DenseMethod(embedder, VectorCache(evals_session), recipe)
            first = verified.runs[method.name]
            second = await run_method(evals_session, TOPIC, method)

            assert second != first
            assert method.stats()["docs_per_s"] is None  # nothing was embedded this call
        assert len(embedder.calls) == verified.n_calls[name]
    assert (
        await evals_session.scalar(select(func.count()).select_from(EmbeddingVector)) == n_vectors
    )

    # A rerun changes no number: the newest completed run of a name replaces the older one.
    before = await _evaluate(evals_session, bootstrap=0)
    assert before["metrics"][BETTER]["ap"] > before["metrics"][INCUMBENT]["ap"]


# ------------------------------------------------------------------ human label backup


async def test_exported_human_labels_are_restored_after_a_wipe(evals_session, verified, tmp_path):
    def _human_rows():
        return select(EmbeddingLabel.job_id, EmbeddingLabel.grade, EmbeddingLabel.source).where(
            EmbeddingLabel.topic_id == verified.topic_id, EmbeddingLabel.source.in_(HUMAN_SOURCES)
        )

    before_rows = set((await evals_session.execute(_human_rows())).all())
    before_report = await _evaluate(evals_session, "human", bootstrap=0)
    path = tmp_path / "labels.json"
    n_human = sum(_is_human_labeled(i) for i in range(N_DOCS))

    assert await export_labels(evals_session, TOPIC, path) == n_human == len(before_rows)
    assert "grade" in path.read_text() and "Description" not in path.read_text()  # no posting text

    await evals_session.execute(
        delete(EmbeddingLabel).where(
            EmbeddingLabel.topic_id == verified.topic_id, EmbeddingLabel.source.in_(HUMAN_SOURCES)
        )
    )
    await evals_session.commit()
    assert not (await evals_session.execute(_human_rows())).all()
    with pytest.raises(ValueError, match="no human labels yet"):
        await _evaluate(evals_session, "human", bootstrap=0)

    assert await import_labels(evals_session, TOPIC, path) == n_human
    assert set((await evals_session.execute(_human_rows())).all()) == before_rows
    assert (await _evaluate(evals_session, "human", bootstrap=0))["metrics"] == before_report[
        "metrics"
    ]
    assert await import_labels(evals_session, TOPIC, path) == 0  # idempotent
