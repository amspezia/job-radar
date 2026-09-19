import json
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest
import pytest_asyncio
from sqlalchemy import delete, insert, select, update

from eval.embedding.checks import record_check
from eval.embedding.embedders.base import EmbedderSpec
from eval.embedding.evaluate import (
    MissingChecks,
    evaluate_topic,
    format_report,
    list_topics,
    write_report,
)
from eval.embedding.models import (
    EmbeddingEmbedder,
    EmbeddingJob,
    EmbeddingJudgeRun,
    EmbeddingLabel,
    EmbeddingRun,
    EmbeddingRunRanking,
    EmbeddingTopic,
    EmbeddingTopicJob,
)

TOPIC = "topic"
INCUMBENT = "nomic-v1.5"
GOOD = "qwen3-0.6b-instruct"
BAD = "bge-m3"
N = 240  # documents: 0-39 grade 3, 40-109 grade 2 (R=110), 110-149 grade 1, the rest 0
R = 110
T0 = datetime(2026, 1, 1, tzinfo=UTC)
SPANISH = frozenset(range(30, 60))  # 10 grade-3 and 20 grade-2 documents, all relevant
R_EN = R - len(SPANISH)
SPANISH_TEXT = (
    "Buscamos un desarrollador con experiencia en el desarrollo de servicios para el equipo"
)
SPANISH_FIELDS = "all skills, technologies and qualifications the team will need"  # prompt echo


def _grade(i: int) -> int:
    return 3 if i < 40 else 2 if i < R else 1 if i < 150 else 0


def _source(i: int) -> str:
    return "alpha" if i < 120 else "beta" if i < 225 else "gamma"  # gamma: 15 docs


def _perfect() -> list[int]:
    return list(range(N))  # grade order by construction


def _rrn() -> list[int]:
    """Relevant, relevant, not — repeated: hand-checkable precision at every cutoff."""
    relevant, other, out = list(range(R)), list(range(R, N)), []
    while relevant:
        out += [relevant.pop(0), relevant.pop(0), other.pop(0)]
    return out + other


def _random(seed: int) -> list[int]:
    return random.Random(seed).sample(range(N), N)


def _rrn_ap() -> float:
    total = 0.0
    for i in range(1, R + 1):
        block, within = divmod(i - 1, 2)
        total += i / (3 * block + within + 1)
    return total / R


@pytest.fixture(autouse=True)
def _incumbent(monkeypatch):
    specs = [EmbedderSpec(name=INCUMBENT, model="m", incumbent=True)]
    monkeypatch.setattr("eval.embedding.evaluate.load_specs", lambda: specs)


@dataclass
class World:
    topic_id: UUID
    docs: list[UUID]
    judge_old: UUID
    judge_new: UUID
    runs: dict[str, UUID]
    n_completed: int


async def _add_run(session, topic_id, docs, name, order, *, hours, status="completed", **extra):
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
            **extra,
        )
    )
    await session.flush()
    if order:
        await session.execute(
            insert(EmbeddingRunRanking),
            [
                {"run_id": run_id, "job_id": docs[i], "rank": rank, "score": 1.0 / rank}
                for rank, i in enumerate(order, start=1)
            ],
        )
    return run_id


def _labels(topic_id, docs, source, grades, judge_run_id=None):
    return [
        EmbeddingLabel(
            topic_id=topic_id,
            job_id=docs[i],
            grade=grade,
            source=source,
            judge_run_id=judge_run_id,
        )
        for i, grade in grades
    ]


async def build_world(
    session,
    *,
    parity: bool = True,
    calibration: str = "passed",
    incumbent: bool = True,
    incumbent_order: list[int] | None = None,
    name: str = TOPIC,
    spanish: frozenset[int] = frozenset(),
) -> World:
    """One topic: 240 labeled docs, three variants, recipe runs, traps and every label source.

    Traps a correct evaluator ignores: an older completed run of BAD (perfect ranking), a
    newer failed run of GOOD (partial ranking) and a newer running run of the incumbent.
    """
    topic_id = uuid4()
    session.add(
        EmbeddingTopic(
            id=topic_id,
            tier="A",
            name=name,
            status="frozen",
            profile_snapshot={},
            query_inputs={},
            builder="test",
            created_at=T0,
        )
    )
    await session.flush()
    docs = [uuid5(NAMESPACE_URL, f"{name}/{i}") for i in range(N)]
    for i, doc_id in enumerate(docs):
        session.add(
            EmbeddingJob(
                id=doc_id,
                origin="test",
                origin_id=f"{name}-{i}",
                source=_source(i),
                title=f"job {i}",
                company="acme",
                url=f"https://example.test/{name}/{i}",
                description=SPANISH_TEXT
                if i in spanish
                else "x" * 4001
                if 100 <= i < 120
                else "short",
                requirements=SPANISH_FIELDS if i in spanish else None if i % 2 == 0 else "req",
                responsibilities="  " if i % 2 == 0 else ("resp" if i >= 30 else None),
                embed_text="text",
                embed_text_sha=f"sha-{name}-{i}",
            )
        )
    await session.flush()
    session.add_all(EmbeddingTopicJob(topic_id=topic_id, job_id=doc_id) for doc_id in docs)

    judge_old, judge_new, judge_running = uuid4(), uuid4(), uuid4()
    for judge_id, days, status in (
        (judge_old, 1, "completed"),
        (judge_new, 2, "completed"),
        (judge_running, 3, "running"),
    ):
        session.add(
            EmbeddingJudgeRun(
                id=judge_id,
                topic_id=topic_id,
                model="judge-model",
                prompt_version="v1",
                prompt_sha="sha",
                params={},
                fewshot=[],
                status=status,
                created_at=T0 + timedelta(days=days),
            )
        )
    session.add(
        EmbeddingEmbedder(
            fingerprint="fp-good",
            name=GOOD,
            model="qwen3-embedding:0.6b",
            digest="sha256:0123456789abcdef",
            quantization="Q8_0",
            runtime_version="0.12.3",
            config={},
        )
    )
    await session.flush()

    session.add_all(
        _labels(topic_id, docs, "llm_judge", [(i, _grade(i)) for i in range(N)], judge_old)
    )
    session.add_all(
        _labels(
            topic_id, docs, "llm_judge", [(i, 2 if i % 2 == 0 else 0) for i in range(N)], judge_new
        )
    )
    human = [(i, _grade(i)) for i in range(0, N, 6)]
    session.add_all(_labels(topic_id, docs, "constructed", human))
    session.add_all(_labels(topic_id, docs, "human_blind", [(6, 1)]))  # grade 3 -> 1
    await session.flush()

    runs: dict[str, UUID] = {}
    if incumbent:
        order = _rrn() if incumbent_order is None else incumbent_order
        runs[INCUMBENT] = await _add_run(session, topic_id, docs, INCUMBENT, order, hours=1)
    runs[GOOD] = await _add_run(
        session,
        topic_id,
        docs,
        GOOD,
        _perfect(),
        hours=2,
        embedder_fingerprint="fp-good",
        docs_per_s=12.5,
        n_truncated=3,
        seconds=42.0,
        git_sha="abc1234def",
    )
    await _add_run(session, topic_id, docs, BAD, _perfect(), hours=0)  # older, ignored
    runs[BAD] = await _add_run(session, topic_id, docs, BAD, _random(7), hours=3)
    runs[f"{INCUMBENT}:hyde_text:0"] = await _add_run(
        session, topic_id, docs, f"{INCUMBENT}:hyde_text:0", _perfect(), hours=4
    )
    runs[f"{GOOD}:hyde_text:0"] = await _add_run(
        session, topic_id, docs, f"{GOOD}:hyde_text:0", _rrn(), hours=4
    )
    runs[f"{GOOD}:hyde_text:1"] = await _add_run(
        session, topic_id, docs, f"{GOOD}:hyde_text:1", _random(3), hours=4
    )
    await _add_run(session, topic_id, docs, GOOD, [5, 4, 3], hours=5, status="failed")
    await _add_run(session, topic_id, docs, INCUMBENT, [], hours=6, status="running")
    await session.commit()

    if parity:
        await record_check(session, topic_id, "parity_ranking", True, {"max_abs_delta": 1e-7})
        await record_check(session, topic_id, "parity_reconstruction", True, {})
    if calibration != "none":
        await record_check(
            session,
            topic_id,
            "judge_calibration",
            calibration == "passed",
            {"kappa": 0.71},
            judge_run_id=judge_old,
        )
    return World(topic_id, docs, judge_old, judge_new, runs, n_completed=7 - (not incumbent))


@pytest_asyncio.fixture
async def world(evals_session):
    return await build_world(evals_session)


async def _report(session, view="silver", **kwargs):
    kwargs.setdefault("bootstrap", 0)
    return await evaluate_topic(session, TOPIC, view, **kwargs)


# ---------------------------------------------------------------- metric values


async def test_silver_metrics_match_hand_computed_values(evals_session, world):
    report = await _report(evals_session)

    perfect = report["metrics"][GOOD]
    assert (perfect["p@100"], perfect["ap"]) == (1.0, 1.0)
    assert perfect["ndcg@100"] == pytest.approx(1.0)
    assert perfect["ndcg@10"] == pytest.approx(1.0)
    assert perfect["p@10"] == 1.0
    assert perfect["recall@100"] == pytest.approx(100 / R)
    assert perfect["judged@10"] == 1.0
    assert perfect["r"] == R

    # R R N R R N ...: 7 relevant in the top 10, 67 in the top 100, 34 in the top 50.
    rrn = report["metrics"][INCUMBENT]
    assert rrn["p@10"] == pytest.approx(0.7)
    assert rrn["p@100"] == pytest.approx(0.67)
    assert rrn["recall@50"] == pytest.approx(34 / R)
    assert rrn["ap"] == pytest.approx(_rrn_ap())
    assert 0.0 < rrn["ndcg@100"] < 1.0

    assert report["metrics"][BAD]["ap"] < rrn["ap"]


async def test_variants_are_the_primary_runs_incumbent_first(evals_session, world):
    report = await _report(evals_session)

    assert report["incumbent"] == INCUMBENT
    assert report["variants"] == [INCUMBENT, BAD, GOOD]
    assert set(report["metrics"]) == {INCUMBENT, BAD, GOOD}


async def test_latest_completed_run_wins_and_failed_or_running_runs_are_ignored(
    evals_session, world
):
    report = await _report(evals_session)

    # BAD's older run was perfect; its latest is random. GOOD's newer run failed (partial
    # ranking) and the incumbent has a newer running run with no rows at all.
    assert report["runs"][BAD]["run_id"] == str(world.runs[BAD])
    assert report["metrics"][BAD]["ap"] < 0.9
    assert report["runs"][GOOD]["run_id"] == str(world.runs[GOOD])
    assert report["metrics"][GOOD]["ap"] == 1.0
    assert report["runs"][INCUMBENT]["run_id"] == str(world.runs[INCUMBENT])


async def test_a_run_that_does_not_rank_the_whole_topic_is_an_error(evals_session, world):
    await evals_session.execute(
        delete(EmbeddingRunRanking).where(
            EmbeddingRunRanking.run_id == world.runs[BAD],
            EmbeddingRunRanking.rank > 200,
        )
    )
    await evals_session.commit()

    with pytest.raises(ValueError, match=BAD):
        await _report(evals_session)


async def test_run_metadata_comes_from_the_run_and_the_embedder_row(evals_session, world):
    report = await _report(evals_session)

    meta = report["runs"][GOOD]
    assert meta["docs_per_s"] == 12.5
    assert meta["n_truncated"] == 3
    assert meta["seconds"] == 42.0
    assert meta["git_sha"] == "abc1234def"
    assert meta["method_fingerprint"] == f"mf-{GOOD}"
    assert meta["embedder"] == {
        "fingerprint": "fp-good",
        "model": "qwen3-embedding:0.6b",
        "digest": "sha256:0123456789abcdef",
        "quantization": "Q8_0",
        "runtime_version": "0.12.3",
    }
    assert report["runs"][INCUMBENT]["embedder"] is None


# ---------------------------------------------------------------- views and labels


async def test_silver_carries_ceiling_r_and_recall_at_100(evals_session, world):
    report = await _report(evals_session)

    assert report["labels"]["n_labeled"] == N
    assert report["labels"]["r"] == R
    assert report["labels"]["recall_ceiling@100"] == pytest.approx(100 / R)
    assert report["labels"]["coverage"] == 1.0
    assert report["metrics"][GOOD]["recall@100"] == pytest.approx(
        report["labels"]["recall_ceiling@100"]
    )


async def test_effective_view_lets_human_labels_supersede_the_judge(evals_session, world):
    report = await _report(evals_session, "effective")

    # Document 6 is judged 3 but a human graded it 1.
    assert report["labels"]["n_labeled"] == N
    assert report["labels"]["r"] == R - 1
    assert report["labels"]["recall_ceiling@100"] == pytest.approx(100 / (R - 1))
    assert "ndcg@100" in report["metrics"][GOOD]


async def test_human_view_has_no_raw_at_100_metrics_anywhere(evals_session, world):
    report = await _report(evals_session, "human", bootstrap=50)

    assert report["partial"] is True
    labels = report["labels"]
    assert (labels["n_labeled"], labels["r"]) == (N // 6, 18)
    assert labels["recall_ceiling@100"] is None
    # Condensed to the 40 judged documents in index order, the human-demoted document 6
    # sits between the first two relevant ones: hits at positions 1, 3, 4, ..., 19.
    assert report["metrics"][GOOD]["ap"] == pytest.approx(
        (1 + sum(i / (i + 1) for i in range(2, 19))) / 18
    )
    assert report["metrics"][GOOD]["n_judged"] == N // 6
    assert set(report["bootstrap"]["metrics"]) == {"ap", "p@50", "ndcg@50"}

    text = json.dumps(report)
    assert "ndcg@100" not in text
    assert "recall@100" not in text
    assert "p@100" not in text
    assert report["judge_run"] is None
    assert "judge_calibration" not in report["checks"]


async def test_human_view_ignores_the_judge_run_pin(evals_session, world):
    report = await _report(evals_session, "human", judge_run_id=uuid4())

    assert report["judge_run"] is None


async def test_human_view_without_human_labels_says_to_run_label_blind(evals_session, world):
    await evals_session.execute(
        delete(EmbeddingLabel).where(EmbeddingLabel.source.in_(("human_blind", "constructed")))
    )
    await evals_session.commit()

    with pytest.raises(ValueError, match=r"no human labels yet.*label-blind"):
        await _report(evals_session, "human")
    # The judge-backed views still score: silver never needed a human label.
    assert (await _report(evals_session, "silver"))["view"] == "silver"


async def test_incomplete_labels_on_a_complete_view_are_an_error(evals_session, world):
    await evals_session.execute(
        delete(EmbeddingLabel).where(
            EmbeddingLabel.source == "llm_judge",
            EmbeddingLabel.job_id.in_(world.docs[200:]),
        )
    )
    await evals_session.commit()

    for view in ("silver", "effective"):
        with pytest.raises(ValueError, match="needs a label for every document"):
            await _report(evals_session, view)
    assert (await _report(evals_session, "human"))["view"] == "human"


async def test_unknown_topic_and_view_and_missing_incumbent(evals_session):
    with pytest.raises(LookupError, match="no topic"):
        await evaluate_topic(evals_session, "absent", "silver")

    world = await build_world(evals_session, incumbent=False)
    assert world.topic_id
    with pytest.raises(ValueError, match="unknown view"):
        await _report(evals_session, "gold")
    with pytest.raises(LookupError, match=INCUMBENT):
        await _report(evals_session)


# ---------------------------------------------------------------- judge run pinning


async def test_default_judge_run_prefers_a_passed_calibration_over_a_newer_run(
    evals_session, world
):
    report = await _report(evals_session)

    # judge_new is newer and judge_running newest, but only judge_old passed calibration.
    assert report["judge_run"]["id"] == str(world.judge_old)
    assert report["labels"]["r"] == R
    assert report["stamps"] == []
    assert report["checks"]["judge_calibration"]["status"] == "passed"


async def test_default_judge_run_falls_back_to_the_newest_completed_one(evals_session):
    world = await build_world(evals_session, calibration="failed")

    with pytest.raises(MissingChecks, match=str(world.judge_new)):
        await _report(evals_session)
    report = await _report(evals_session, allow_unverified=True)

    # judge_old's calibration failed, so nothing is calibrated; judge_running is not completed.
    assert report["judge_run"]["id"] == str(world.judge_new)
    assert report["stamps"] == ["UNCALIBRATED"]


async def test_an_explicit_judge_run_is_pinned_and_labels_follow_it(evals_session, world):
    report = await _report(evals_session, judge_run_id=world.judge_new, allow_unverified=True)

    assert report["judge_run"]["id"] == str(world.judge_new)
    assert report["labels"]["r"] == N // 2
    assert report["stamps"] == ["UNCALIBRATED"]

    with pytest.raises(LookupError, match="does not belong"):
        await _report(evals_session, judge_run_id=uuid4())


async def test_a_topic_without_a_completed_judge_run_cannot_be_scored_on_silver(
    evals_session, world
):
    await evals_session.execute(update(EmbeddingJudgeRun).values(status="failed"))
    await evals_session.commit()

    with pytest.raises(LookupError, match="no completed judge run"):
        await _report(evals_session)
    assert (await _report(evals_session, "human"))["view"] == "human"


# ---------------------------------------------------------------- required checks


async def test_missing_parity_refuses_and_names_both_checks(evals_session):
    await build_world(evals_session, parity=False)

    with pytest.raises(MissingChecks) as raised:
        await _report(evals_session)

    assert raised.value.checks == ["parity_ranking (missing)", "parity_reconstruction (missing)"]
    assert "parity_ranking (missing)" in str(raised.value)


async def test_a_failed_parity_check_refuses(evals_session, world):
    await record_check(evals_session, world.topic_id, "parity_reconstruction", False, {})

    with pytest.raises(MissingChecks, match=r"parity_reconstruction \(failed\)"):
        await _report(evals_session, "human")


async def test_missing_or_failed_calibration_refuses_silver_and_effective(evals_session):
    await build_world(evals_session, calibration="none")

    for view in ("silver", "effective"):
        with pytest.raises(MissingChecks, match=r"judge_calibration .*\(missing\)"):
            await _report(evals_session, view)
    # `human` does not need a calibrated judge.
    assert (await _report(evals_session, "human"))["stamps"] == []


async def test_a_failed_calibration_refuses(evals_session):
    world = await build_world(evals_session, calibration="failed")

    with pytest.raises(MissingChecks, match=r"judge_calibration .*\(failed\)"):
        await _report(evals_session, judge_run_id=world.judge_old)


async def test_allow_unverified_stamps_the_report(evals_session):
    await build_world(evals_session, parity=False, calibration="none")

    both = await _report(evals_session, allow_unverified=True)
    human = await _report(evals_session, "human", allow_unverified=True)

    assert both["stamps"] == ["UNVERIFIED", "UNCALIBRATED"]
    assert both["checks"]["parity_ranking"]["status"] == "missing"
    assert human["stamps"] == ["UNVERIFIED"]


# ---------------------------------------------------------------- bootstrap, controls, slices


async def test_bootstrap_is_seeded_and_the_good_variant_excludes_zero(evals_session, world):
    first = await _report(evals_session, bootstrap=200, seed=5)
    again = await _report(evals_session, bootstrap=200, seed=5)
    other = await _report(evals_session, bootstrap=200, seed=6)

    assert first["bootstrap"] == again["bootstrap"]
    assert first["bootstrap"] != other["bootstrap"]
    assert first["bootstrap"]["n"] == 200
    assert set(first["bootstrap"]["metrics"]) == {"ndcg@100", "p@100", "ap"}

    ap = first["bootstrap"]["metrics"]["ap"]
    assert INCUMBENT not in ap
    good = ap[GOOD]
    assert good["low"] > 0.0
    assert good["excludes_zero"] is True
    assert good["mean_diff"] == pytest.approx(1.0 - _rrn_ap(), abs=0.08)
    assert good["low"] <= good["mean_diff"] <= good["high"]
    assert ap[BAD]["high"] < 0.0
    assert first["bootstrap"]["metrics"]["p@100"][GOOD]["excludes_zero"] is True


async def test_bootstrap_can_be_switched_off(evals_session, world):
    report = await _report(evals_session, bootstrap=0)

    assert report["bootstrap"] is None
    assert "Bootstrap" not in format_report(report)


async def test_controls_are_scored_and_do_not_beat_a_good_incumbent(evals_session, world):
    report = await _report(evals_session, seed=0)

    assert set(report["controls"]) == {"random", "shuffled_tail"}
    for scores in report["controls"].values():
        assert set(scores) == set(report["metrics"][INCUMBENT])
        assert scores["ap"] < report["metrics"][INCUMBENT]["ap"]
    assert report["control_warning"] is None
    assert "WARNING" not in format_report(report)


async def test_a_control_that_beats_the_incumbent_is_flagged(evals_session):
    await build_world(evals_session, incumbent_order=list(reversed(range(N))))

    report = await _report(evals_session)

    assert report["control_warning"] is not None
    assert "random" in report["control_warning"]
    assert "WARNING: " + report["control_warning"] in format_report(report)


async def test_slices_exist_only_for_at_least_twenty_documents(evals_session, world):
    report = await _report(evals_session)

    slices = report["slices"]
    assert set(slices) == {
        "extraction:fallback",
        "extraction:structured",
        "description>4000",
        "source:alpha",
        "source:beta",
    }  # one_field_only has 15 documents and source gamma 15
    assert slices["description>4000"]["n_docs"] == 20  # the boundary is inclusive
    assert slices["extraction:fallback"]["n_docs"] == N // 2
    assert slices["source:alpha"]["n_docs"] == 120
    assert slices["source:beta"]["n_docs"] == 105

    fallback = slices["extraction:fallback"]
    assert set(fallback["metrics"]) == {INCUMBENT, BAD, GOOD}
    assert set(fallback["metrics"][GOOD]) == {"ndcg@100", "p@100", "ap"}
    # Even documents only: 55 relevant come first, the other 65 follow.
    assert fallback["metrics"][GOOD]["ap"] == 1.0
    assert fallback["metrics"][GOOD]["p@100"] == pytest.approx(0.55)
    assert fallback["r"] == 55


async def test_human_slices_use_the_condensed_headline_metrics(evals_session, world):
    report = await _report(evals_session, "human")

    info = report["slices"]["source:alpha"]
    assert set(info["metrics"][GOOD]) == {"ap", "p@50", "ndcg@50"}
    assert info["n_labeled"] == 20  # documents 0, 6, ..., 114


# ---------------------------------------------------------------- query sensitivity


async def test_query_sensitivity_compares_recipes_and_embedder_order(evals_session, world):
    report = await _report(evals_session)

    sensitivity = report["query_sensitivity"]
    assert sensitivity["metric"] == "ap"
    # BAD ran a single recipe, so it is not part of the comparison.
    assert list(sensitivity["variants"]) == [INCUMBENT, GOOD]
    good = sensitivity["variants"][GOOD]
    assert list(good) == ["hyde_mean", "hyde_text:0", "hyde_text:1"]
    assert good["hyde_mean"] == 1.0
    assert good["hyde_text:0"] == pytest.approx(_rrn_ap())
    assert sensitivity["variants"][INCUMBENT]["hyde_text:0"] == 1.0

    stability = sensitivity["stability"]
    assert stability["recipes"] == ["hyde_mean", "hyde_text:0"]
    assert stability["orders"]["hyde_mean"] == [GOOD, INCUMBENT]
    assert stability["orders"]["hyde_text:0"] == [INCUMBENT, GOOD]
    assert stability["agree"] is False
    assert stability["disagreements"] == {"hyde_mean": 0, "hyde_text:0": 1}


async def test_no_query_sensitivity_without_multi_recipe_variants(evals_session):
    await build_world(evals_session)
    recipe_runs = select(EmbeddingRun.id).where(EmbeddingRun.name.contains(":"))
    await evals_session.execute(
        delete(EmbeddingRunRanking).where(EmbeddingRunRanking.run_id.in_(recipe_runs))
    )
    await evals_session.execute(delete(EmbeddingRun).where(EmbeddingRun.name.contains(":")))
    await evals_session.commit()

    report = await _report(evals_session)

    assert report["query_sensitivity"] is None
    assert "Query sensitivity" not in format_report(report)


# ---------------------------------------------------------------- format_report


async def test_format_report_shows_the_ceiling_beside_every_recall_at_100(evals_session, world):
    report = await _report(evals_session, bootstrap=100)
    text = format_report(report)

    assert "Recall@100 ceiling" in text
    assert f"{100 / R:.3f}" in text
    for line in text.splitlines():
        if "recall@100" in line.lower():
            assert "ceiling" in line.lower(), line
    assert f"{INCUMBENT} *" in text
    assert "Δ vs incumbent" in text
    assert "Negative controls" in text
    assert "Query sensitivity" in text
    assert "Slices" in text
    assert str(world.judge_old) in text
    assert "UNVERIFIED" not in text

    # A small ceiling is still printed with every Recall@100 figure.
    report["labels"]["recall_ceiling@100"] = 0.5
    text = format_report(report)
    assert "ceiling 0.500" in text
    for line in text.splitlines():
        if "recall@100" in line.lower():
            assert "ceiling" in line.lower(), line


async def test_format_report_flags_intervals_that_exclude_zero(evals_session, world):
    text = format_report(await _report(evals_session, bootstrap=200))

    ap_block = text.split("\n  ap\n", 1)[1].split("\n\n", 1)[0]
    good_line = next(line for line in ap_block.splitlines() if line.strip().startswith(GOOD))
    assert good_line.endswith("!")
    assert "[+" in good_line


async def test_format_report_is_loud_when_unverified_or_uncalibrated(evals_session):
    await build_world(evals_session, parity=False, calibration="none")

    text = format_report(await _report(evals_session, allow_unverified=True))

    assert "!! UNVERIFIED: parity_ranking (missing); parity_reconstruction (missing)" in text
    assert "!! UNCALIBRATED: judge_calibration for judge run" in text
    assert text.index("UNVERIFIED") < text.index("Labels")
    assert "PARITY_RANKING: MISSING" in text.upper()


async def test_format_report_for_the_human_view_never_mentions_the_at_100_metrics(
    evals_session, world
):
    text = format_report(await _report(evals_session, "human", bootstrap=50))

    assert "partial view" in text
    assert "recall@100" not in text.lower()
    assert "ndcg@100" not in text.lower()
    assert "p@100" not in text.lower()
    assert "n_judged" in text
    assert "condensed to judged documents" in text


async def test_format_report_lists_run_metadata(evals_session, world):
    text = format_report(await _report(evals_session))

    runs = text.split("\nRuns\n", 1)[1]
    good_line = next(line for line in runs.splitlines() if line.strip().startswith(GOOD))
    assert "sha256:01234" in good_line
    assert "Q8_0" in good_line
    assert "0.12.3" in good_line
    assert "12.5" in good_line
    assert "abc1234d" in good_line


# ---------------------------------------------------------------- write_report / list_topics


async def test_write_report_round_trips_the_json(evals_session, world, tmp_path):
    report = await _report(evals_session, bootstrap=20)

    path = write_report(report, tmp_path / "nested" / "results")

    assert path.parent == tmp_path / "nested" / "results"
    assert re.fullmatch(r"embedding-topic-silver-\d{8}T\d{6}Z\.json", path.name)
    assert json.loads(path.read_text()) == report


def test_write_report_serializes_uuids_and_datetimes_with_sorted_keys(tmp_path):
    ident = uuid4()
    when = datetime(2026, 5, 1, 12, 30, tzinfo=UTC)
    report = {"topic": "a/b c", "view": "human", "zeta": ident, "alpha": when}

    path = write_report(report, tmp_path)

    text = path.read_text()
    assert path.name.startswith("embedding-a_b_c-human-")
    assert json.loads(text)["zeta"] == str(ident)
    assert json.loads(text)["alpha"] == when.isoformat()
    assert text.index('"alpha"') < text.index('"zeta"')


async def test_list_topics_summarises_every_topic(evals_session, world):
    evals_session.add(
        EmbeddingTopic(
            id=uuid4(),
            tier="B",
            name="aaa-empty",
            status="draft",
            profile_snapshot={},
            query_inputs={},
            builder="test",
        )
    )
    await evals_session.commit()

    rows = await list_topics(evals_session)

    assert [row["name"] for row in rows] == ["aaa-empty", TOPIC]
    empty, full = rows
    assert (empty["tier"], empty["status"], empty["n_docs"], empty["labels"], empty["runs"]) == (
        "B",
        "draft",
        0,
        {},
        0,
    )
    assert full["tier"] == "A"
    assert full["status"] == "frozen"
    assert full["n_docs"] == N
    assert full["labels"] == {"constructed": N // 6, "human_blind": 1, "llm_judge": 2 * N}
    assert full["runs"] == world.n_completed
    assert full["created_at"] == T0.isoformat()
    json.dumps(rows)


# ---------------------------------------------------------------- language filter


def _leaked(order: list[int]) -> int:
    return sum(1 for i in order[:100] if i in SPANISH)


async def test_language_filter_restricts_corpus_labels_r_and_the_ceiling(evals_session):
    await build_world(evals_session, spanish=SPANISH)
    report = await _report(evals_session, language="en")

    assert report["n_docs"] == N - len(SPANISH)
    assert report["labels"]["n_labeled"] == N - len(SPANISH)
    assert report["labels"]["coverage"] == 1.0
    assert report["labels"]["r"] == R_EN
    assert report["labels"]["recall_ceiling@100"] == 1.0  # min(1, 100/80); it was 100/110

    # Perfect order, English documents only: the 80 relevant ones fill the top 80.
    perfect = report["metrics"][GOOD]
    assert (perfect["r"], perfect["ap"], perfect["recall@100"]) == (R_EN, 1.0, 1.0)
    assert perfect["p@100"] == pytest.approx(R_EN / 100)

    english_rrn = [i for i in _rrn() if i not in SPANISH]
    rrn = report["metrics"][INCUMBENT]
    assert rrn["p@10"] == sum(1 for i in english_rrn[:10] if _grade(i) >= 2) / 10
    assert rrn["p@100"] == sum(1 for i in english_rrn[:100] if _grade(i) >= 2) / 100
    assert rrn["r"] == R_EN

    unrestricted = await _report(evals_session)
    assert unrestricted["labels"]["r"] == R
    assert unrestricted["labels"]["recall_ceiling@100"] == pytest.approx(100 / R)
    assert unrestricted["metrics"][GOOD]["ap"] == 1.0
    assert unrestricted["metrics"][GOOD]["p@100"] == 1.0
    assert perfect["p@100"] != unrestricted["metrics"][GOOD]["p@100"]


async def test_language_filter_counts_what_it_left_out_and_the_top_100_leakage(evals_session):
    await build_world(evals_session, spanish=SPANISH)
    report = await _report(evals_session, language="en")

    block = report["language"]
    assert (block["filter"], block["n_included"], block["n_excluded"]) == ("en", 210, 30)
    assert block["excluded"] == {"es": 30}
    assert block["slices"] == {}

    # Counted in each variant's own unrestricted top 100, not in the restricted one.
    expected = {INCUMBENT: _leaked(_rrn()), GOOD: _leaked(_perfect()), BAD: _leaked(_random(7))}
    assert block["off_language_top100"] == expected
    assert all(count > 0 for count in expected.values())
    assert set(block["off_language_top100"]) == set(report["variants"])


async def test_language_filter_also_restricts_controls_slices_and_bootstrap(evals_session):
    await build_world(evals_session, spanish=SPANISH)
    report = await _report(evals_session, language="en", bootstrap=20)

    assert report["bootstrap"]["n"] == 20
    assert report["slices"]["source:alpha"]["n_docs"] == 120 - len(SPANISH)
    assert report["slices"]["source:beta"]["n_docs"] == 105
    assert report["slices"]["source:alpha"]["n_labeled"] == 120 - len(SPANISH)
    for scores in report["controls"].values():
        assert scores["r"] == R_EN


async def test_language_filter_in_the_partial_view_drops_the_excluded_labels(evals_session):
    await build_world(evals_session, spanish=SPANISH)
    every = await _report(evals_session, "human")
    english = await _report(evals_session, "human", language="en")

    spanish_labeled = sum(1 for i in range(0, N, 6) if i in SPANISH)
    assert spanish_labeled == 5
    assert english["labels"]["n_labeled"] == every["labels"]["n_labeled"] - spanish_labeled
    assert english["labels"]["recall_ceiling@100"] is None
    assert set(every["language"]["slices"]) == {"en", "es"}


async def test_without_a_filter_the_report_is_unchanged_and_slices_by_language(evals_session):
    await build_world(evals_session, spanish=SPANISH)
    report = await _report(evals_session)

    assert report["n_docs"] == N
    assert report["labels"]["r"] == R
    block = report["language"]
    assert (block["filter"], block["n_included"], block["n_excluded"]) == (None, N, 0)
    assert block["excluded"] == {}
    assert block["off_language_top100"] is None
    assert not any("language" in name or name in ("en", "es") for name in report["slices"])

    slices = block["slices"]
    assert set(slices) == {"en", "es"}
    assert (slices["es"]["n_docs"], slices["es"]["r"]) == (len(SPANISH), len(SPANISH))
    assert slices["en"]["r"] == R_EN
    assert slices["es"]["metrics"][GOOD]["ap"] == 1.0


async def test_an_all_english_topic_scores_the_same_with_and_without_the_filter(
    evals_session, world
):
    base = await _report(evals_session)
    english = await _report(evals_session, language="en")

    assert english["language"]["n_excluded"] == 0
    assert english["language"]["off_language_top100"] == dict.fromkeys(base["variants"], 0)
    for key in ("n_docs", "labels", "metrics", "controls", "slices"):
        assert english[key] == base[key]
    assert set(base["language"]["slices"]) == {"en"}


async def test_the_runs_are_checked_against_the_whole_topic_before_the_filter(evals_session):
    # Missing document 30 is Spanish: the English subset would look complete, the run is not.
    await build_world(
        evals_session, spanish=SPANISH, incumbent_order=[i for i in range(N) if i != 30]
    )
    with pytest.raises(ValueError, match=INCUMBENT):
        await _report(evals_session, language="en")


async def test_language_filter_rejects_an_unknown_or_empty_language(evals_session, world):
    with pytest.raises(ValueError, match="unknown language"):
        await _report(evals_session, language="fr")
    with pytest.raises(ValueError, match="no 'es' documents"):
        await _report(evals_session, language="es")


async def test_format_report_states_the_filter_and_the_leakage(evals_session):
    await build_world(evals_session, spanish=SPANISH)
    text = format_report(await _report(evals_session, language="en"))

    assert "language filter: en (210 of 240 documents)" in text
    assert "Off-language leakage (excluded: 30 es)" in text
    assert "off-language@100" in text
    assert "Language slices" not in text
    for line in text.splitlines():
        if "recall@100" in line.lower():
            assert "ceiling" in line.lower(), line

    unfiltered = format_report(await _report(evals_session))
    assert "language filter" not in unfiltered
    assert "Off-language leakage" not in unfiltered
    assert "Language slices" in unfiltered


async def test_format_report_of_an_english_topic_has_no_language_section(evals_session, world):
    text = format_report(await _report(evals_session))
    assert "language" not in text.lower()
