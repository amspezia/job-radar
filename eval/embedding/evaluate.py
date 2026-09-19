"""Evaluate a topic's stored rankings and report them so the table cannot be misread.

Metrics are functions of `(ranking, labels)` recomputed on demand (`scoring.py`); this module
only loads what is stored, refuses a comparison whose checks are missing, and lays the numbers
out. No `job_radar` imports.
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.checks import REQUIRED_CHECKS, CheckRow, calibration_for_run, latest_checks
from eval.embedding.embedders.registry import load_specs
from eval.embedding.labels import VIEWS, LabelRow, is_partial, resolve_effective
from eval.embedding.language import LANGUAGES, doc_language
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
from eval.embedding.scoring import (
    REL_THRESHOLD,
    compute_metrics,
    condense,
    order_stability,
    paired_bootstrap,
    random_ranking,
    recall_ceiling,
    shuffled_tail,
)

STAMP_UNVERIFIED = "UNVERIFIED"
STAMP_UNCALIBRATED = "UNCALIBRATED"

DEFAULT_RECIPE = "hyde_mean"
MIN_SLICE_DOCS = 20
LONG_DESCRIPTION = 4000
TAIL_KEEP = 20
LEAKAGE_DEPTH = 100

# Headline metrics per kind of view. The partial view stops at @50 because scoring never
# produces an @100 metric for it (unjudged documents would count as grade 0).
_HEADLINE = ("ndcg@100", "p@100", "ap")
_HEADLINE_PARTIAL = ("ap", "p@50", "ndcg@50")
_TABLE = ("ndcg@100", "p@100", "ap", "ndcg@10", "recall@50", "bpref", "p@10")
_TABLE_PARTIAL = ("ap", "p@50", "ndcg@50", "ndcg@10", "p@10", "recall@50", "bpref", "n_judged")
_JUDGED = ("judged@10", "judged@100")
_COUNTS = ("n_judged", "r")


class MissingChecks(Exception):
    """A required check is absent or failed; `checks` names each one and why."""

    def __init__(self, checks: list[str]) -> None:
        self.checks = list(checks)
        super().__init__("required checks not satisfied: " + "; ".join(self.checks))


@dataclass(frozen=True)
class _Doc:
    id: UUID
    source: str
    title: str
    description: str
    requirements: str | None
    responsibilities: str | None

    @property
    def language(self) -> str:
        return doc_language(self.title, self.description, self.requirements, self.responsibilities)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _headline(partial: bool) -> tuple[str, ...]:
    return _HEADLINE_PARTIAL if partial else _HEADLINE


def _split_name(name: str) -> tuple[str, str]:
    variant, _, recipe = name.partition(":")
    return variant, recipe or DEFAULT_RECIPE


def _recipe_key(recipe: str) -> tuple[int, int, str]:
    if recipe == DEFAULT_RECIPE:
        return (0, 0, recipe)
    tail = recipe.rpartition(":")[2]
    return (1, int(tail) if tail.isdigit() else 0, recipe)


async def _latest_runs(session: AsyncSession, topic_id: UUID) -> dict[str, EmbeddingRun]:
    """The newest completed run per name; older, running and failed runs never count."""
    rows = await session.scalars(
        select(EmbeddingRun)
        .where(EmbeddingRun.topic_id == topic_id, EmbeddingRun.status == "completed")
        .order_by(EmbeddingRun.started_at, EmbeddingRun.id)
    )
    return {run.name: run for run in rows}


async def _rankings(session: AsyncSession, runs: dict[str, EmbeddingRun]) -> dict[str, list[UUID]]:
    by_id = {run.id: name for name, run in runs.items()}
    out: dict[str, list[UUID]] = {name: [] for name in runs}
    rows = await session.execute(
        select(EmbeddingRunRanking.run_id, EmbeddingRunRanking.job_id)
        .where(EmbeddingRunRanking.run_id.in_(list(by_id)))
        .order_by(EmbeddingRunRanking.run_id, EmbeddingRunRanking.rank)
    )
    for run_id, job_id in rows:
        out[by_id[run_id]].append(job_id)
    return out


async def _pin_judge_run(
    session: AsyncSession, topic_id: UUID, requested: UUID | str | None
) -> EmbeddingJudgeRun:
    """The judge run a silver/effective view is pinned to: the requested one, else the newest
    completed run with a passed calibration, else the newest completed run."""
    if requested is not None:
        run = await session.get(EmbeddingJudgeRun, UUID(str(requested)))
        if run is None or run.topic_id != topic_id:
            raise LookupError(f"judge run {requested} does not belong to this topic")
        return run
    runs = (
        await session.scalars(
            select(EmbeddingJudgeRun)
            .where(EmbeddingJudgeRun.topic_id == topic_id, EmbeddingJudgeRun.status == "completed")
            .order_by(EmbeddingJudgeRun.created_at.desc(), EmbeddingJudgeRun.id.desc())
        )
    ).all()
    if not runs:
        raise LookupError("the topic has no completed judge run; run the judge first")
    for run in runs:
        calibration = await calibration_for_run(session, topic_id, run.id)
        if calibration is not None and calibration.passed:
            return run
    return runs[0]


def _check_entry(row: CheckRow | None) -> dict:
    if row is None:
        return {"status": "missing", "created_at": None, "detail": {}}
    return {
        "status": "passed" if row.passed else "failed",
        "created_at": _iso(row.created_at),
        "detail": row.detail,
    }


def _bootstrap(
    rankings: dict[str, list[UUID]],
    labels: dict[UUID, int],
    incumbent: str,
    partial: bool,
    n: int,
    seed: int,
) -> dict:
    """Paired document bootstrap vs the incumbent; a partial view is condensed first."""
    scored = {name: condense(r, labels) for name, r in rankings.items()} if partial else rankings
    out: dict[str, dict] = {}
    for metric in _headline(partial):
        results = paired_bootstrap(scored, labels, incumbent, metric, n, seed)
        out[metric] = {
            name: {
                "mean_diff": r.mean_diff,
                "sd": r.sd,
                "low": r.low,
                "high": r.high,
                "excludes_zero": r.low > 0.0 or r.high < 0.0,
            }
            for name, r in results.items()
        }
    return {"n": n, "seed": seed, "condensed": partial, "metrics": out}


def _slices(
    docs: list[_Doc],
    rankings: dict[str, list[UUID]],
    labels: dict[UUID, int],
    partial: bool,
) -> dict[str, dict]:
    def empty(text: str | None) -> bool:
        return text is None or not text.strip()

    groups: dict[str, set[UUID]] = defaultdict(set)
    for doc in docs:
        no_req, no_resp = empty(doc.requirements), empty(doc.responsibilities)
        groups["extraction:fallback" if no_req and no_resp else "extraction:structured"].add(doc.id)
        if len(doc.description) > LONG_DESCRIPTION:
            groups[f"description>{LONG_DESCRIPTION}"].add(doc.id)
        if no_req != no_resp:
            groups["one_field_only"].add(doc.id)
        groups[f"source:{doc.source}"].add(doc.id)
    return _score_groups(groups, rankings, labels, partial)


def _language_slices(
    langs: dict[UUID, str],
    rankings: dict[str, list[UUID]],
    labels: dict[UUID, int],
    partial: bool,
) -> dict[str, dict]:
    groups: dict[str, set[UUID]] = defaultdict(set)
    for doc_id, language in langs.items():
        groups[language].add(doc_id)
    return _score_groups(groups, rankings, labels, partial)


def _score_groups(
    groups: dict[str, set[UUID]],
    rankings: dict[str, list[UUID]],
    labels: dict[UUID, int],
    partial: bool,
) -> dict[str, dict]:
    """Headline metrics per variant inside each group of at least `MIN_SLICE_DOCS` documents."""
    headline = _headline(partial)
    out: dict[str, dict] = {}
    for name in sorted(groups):
        members = groups[name]
        if len(members) < MIN_SLICE_DOCS:
            continue
        sliced = {doc_id: grade for doc_id, grade in labels.items() if doc_id in members}
        metrics = {}
        for variant, ranking in rankings.items():
            scores = compute_metrics([d for d in ranking if d in members], sliced, partial)
            metrics[variant] = {key: scores[key] for key in headline}
        out[name] = {
            "n_docs": len(members),
            "n_labeled": len(sliced),
            "r": sum(1 for grade in sliced.values() if grade >= REL_THRESHOLD),
            "metrics": metrics,
        }
    return out


def _query_sensitivity(all_metrics: dict[str, dict[str, float]]) -> dict | None:
    """`ap` per query recipe for every variant that ran several, and whether the order holds."""
    by_variant: dict[str, dict[str, float]] = defaultdict(dict)
    for name, scores in all_metrics.items():
        variant, recipe = _split_name(name)
        by_variant[variant][recipe] = scores["ap"]
    multi = {variant: recipes for variant, recipes in by_variant.items() if len(recipes) > 1}
    if not multi:
        return None

    common = set.intersection(*(set(recipes) for recipes in multi.values()))
    stability = None
    if len(common) > 1:
        ordered = sorted(common, key=_recipe_key)
        result = order_stability(
            {recipe: {variant: multi[variant][recipe] for variant in multi} for recipe in ordered}
        )
        stability = {
            "recipes": ordered,
            "orders": result.orders,
            "agree": result.agree,
            "disagreements": result.disagreements,
        }
    return {
        "metric": "ap",
        "variants": {
            variant: {r: multi[variant][r] for r in sorted(multi[variant], key=_recipe_key)}
            for variant in sorted(multi)
        },
        "stability": stability,
    }


async def _embedders(session: AsyncSession, runs: dict[str, EmbeddingRun]) -> dict[str, dict]:
    fingerprints = {r.embedder_fingerprint for r in runs.values() if r.embedder_fingerprint}
    rows = await session.scalars(
        select(EmbeddingEmbedder).where(EmbeddingEmbedder.fingerprint.in_(list(fingerprints)))
    )
    return {e.fingerprint: e for e in rows}


def _run_metadata(run: EmbeddingRun, embedders: dict) -> dict:
    embedder = embedders.get(run.embedder_fingerprint)
    return {
        "run_id": str(run.id),
        "embedder": None
        if embedder is None
        else {
            "fingerprint": embedder.fingerprint,
            "model": embedder.model,
            "digest": embedder.digest,
            "quantization": embedder.quantization,
            "runtime_version": embedder.runtime_version,
        },
        "docs_per_s": run.docs_per_s,
        "n_truncated": run.n_truncated,
        "git_sha": run.git_sha,
        "seconds": run.seconds,
        "method_fingerprint": run.method_fingerprint,
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
    }


async def evaluate_topic(
    session: AsyncSession,
    topic_name: str,
    view: str,
    judge_run_id: UUID | None = None,
    bootstrap: int = 1000,
    seed: int = 0,
    allow_unverified: bool = False,
    language: str | None = None,
) -> dict:
    """Score every stored ranking of `topic_name` under `view`.

    Raises `MissingChecks` unless every required check passed, or `allow_unverified` is set, in
    which case the report is stamped UNVERIFIED and/or UNCALIBRATED instead.

    With `language` (e.g. `"en"`) only documents detected in that language are scored: the
    rankings, labels, R and the Recall@100 ceiling are all restricted to them, after the runs
    were checked against the whole topic. The `language` block of the report counts what was
    left out and how many of it still sits in each variant's unrestricted top 100.
    """
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    if language is not None and language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}; expected one of {LANGUAGES}")
    partial = is_partial(view)

    topic = await session.scalar(select(EmbeddingTopic).where(EmbeddingTopic.name == topic_name))
    if topic is None:
        raise LookupError(f"no topic named {topic_name!r}")
    corpus = sorted(
        await session.scalars(
            select(EmbeddingTopicJob.job_id).where(EmbeddingTopicJob.topic_id == topic.id)
        )
    )
    if not corpus:
        raise ValueError(f"topic {topic_name!r} has no documents")
    corpus_set = set(corpus)

    runs = await _latest_runs(session, topic.id)
    incumbent = next(spec.name for spec in load_specs() if spec.incumbent)
    if incumbent not in runs:
        raise LookupError(
            f"incumbent {incumbent!r} has no completed run for topic {topic_name!r}; "
            "run it first, every comparison is stated against it"
        )
    rankings = await _rankings(session, runs)
    for name, ranking in rankings.items():
        if len(ranking) != len(corpus) or set(ranking) != corpus_set:
            raise ValueError(
                f"run {name!r} ranks {len(ranking)} documents, but the topic has {len(corpus)}"
            )
    primary = [incumbent, *sorted(n for n in runs if ":" not in n and n != incumbent)]

    doc_rows = await session.execute(
        select(
            EmbeddingJob.id,
            EmbeddingJob.source,
            EmbeddingJob.title,
            EmbeddingJob.description,
            EmbeddingJob.requirements,
            EmbeddingJob.responsibilities,
        ).where(EmbeddingJob.id.in_(corpus))
    )
    docs = [_Doc(*row) for row in doc_rows]
    langs = {doc.id: doc.language for doc in docs}
    n_corpus = len(corpus)
    excluded: dict[str, int] = {}
    off_language_top100 = None
    if language is not None:
        keep = {doc_id for doc_id, lang in langs.items() if lang == language}
        if not keep:
            raise ValueError(f"topic {topic_name!r} has no {language!r} documents")
        excluded = dict(
            sorted(Counter(lang for lang in langs.values() if lang != language).items())
        )
        off_language_top100 = {
            name: sum(1 for d in rankings[name][:LEAKAGE_DEPTH] if d not in keep)
            for name in primary
        }
        corpus = [doc_id for doc_id in corpus if doc_id in keep]
        corpus_set = keep
        rankings = {name: [d for d in ranking if d in keep] for name, ranking in rankings.items()}
        docs = [doc for doc in docs if doc.id in keep]
        langs = {doc_id: lang for doc_id, lang in langs.items() if doc_id in keep}

    judge_run = None if partial else await _pin_judge_run(session, topic.id, judge_run_id)
    rows = await session.scalars(select(EmbeddingLabel).where(EmbeddingLabel.topic_id == topic.id))
    resolved = resolve_effective(
        [LabelRow(r.job_id, r.grade, r.source, r.judge_run_id, r.id) for r in rows],
        view,
        None if judge_run is None else judge_run.id,
    )
    labels = {doc_id: grade for doc_id, grade in resolved.items() if doc_id in corpus_set}
    if not labels:
        if view == "human":
            raise ValueError(
                f"topic {topic_name!r} has no human labels yet — run `label-blind` "
                "(the `human` view is only your blind grades and constructed labels)"
            )
        raise ValueError(f"topic {topic_name!r} has no labels in view {view!r}")
    if not partial and len(labels) != len(corpus):
        raise ValueError(
            f"view {view!r} needs a label for every document, but {len(corpus) - len(labels)} "
            f"of {len(corpus)} are unlabeled (judge run {judge_run.id})"
        )

    latest = await latest_checks(session, topic.id)
    checks = {name: _check_entry(latest.get(name)) for name in REQUIRED_CHECKS}
    reasons: dict[str, list[str]] = {}
    unverified = [
        f"{n} ({checks[n]['status']})" for n in REQUIRED_CHECKS if checks[n]["status"] != "passed"
    ]
    if unverified:
        reasons[STAMP_UNVERIFIED] = unverified
    if judge_run is not None:
        calibration = _check_entry(await calibration_for_run(session, topic.id, judge_run.id))
        checks["judge_calibration"] = calibration
        if calibration["status"] != "passed":
            reasons[STAMP_UNCALIBRATED] = [
                f"judge_calibration for judge run {judge_run.id} ({calibration['status']})"
            ]
    if reasons and not allow_unverified:
        raise MissingChecks([problem for problems in reasons.values() for problem in problems])

    all_metrics = {name: compute_metrics(rankings[name], labels, partial) for name in runs}
    headline = _headline(partial)
    controls = {
        "random": random_ranking(corpus, seed),
        "shuffled_tail": shuffled_tail(rankings[incumbent], TAIL_KEEP, seed),
    }
    control_metrics = {name: compute_metrics(r, labels, partial) for name, r in controls.items()}
    beaten = [
        f"{control} {metric} {scores[metric]:.3f} >= incumbent {all_metrics[incumbent][metric]:.3f}"
        for control, scores in control_metrics.items()
        for metric in headline
        if scores[metric] >= all_metrics[incumbent][metric]
    ]
    control_warning = None
    if beaten:
        control_warning = (
            "a negative control reaches the incumbent (" + "; ".join(beaten) + "): this "
            "comparison cannot separate embedders from noise"
        )

    primary_rankings = {name: rankings[name] for name in primary}
    embedders = await _embedders(session, runs)

    return {
        "topic": topic.name,
        "topic_id": str(topic.id),
        "tier": topic.tier,
        "view": view,
        "partial": partial,
        "judge_run": None
        if judge_run is None
        else {
            "id": str(judge_run.id),
            "model": judge_run.model,
            "prompt_version": judge_run.prompt_version,
            "status": judge_run.status,
            "created_at": _iso(judge_run.created_at),
        },
        "stamps": list(reasons),
        "stamp_reasons": reasons,
        "incumbent": incumbent,
        "variants": primary,
        "n_docs": len(corpus),
        "language": {
            "filter": language,
            "n_included": len(corpus),
            "n_excluded": n_corpus - len(corpus),
            "excluded": excluded,
            "off_language_top100": off_language_top100,
            "slices": {}
            if language is not None
            else _language_slices(langs, primary_rankings, labels, partial),
        },
        "labels": {
            "n_labeled": len(labels),
            "r": sum(1 for grade in labels.values() if grade >= REL_THRESHOLD),
            "recall_ceiling@100": None if partial else recall_ceiling(labels, 100),
            "coverage": len(labels) / len(corpus),
        },
        "checks": checks,
        "metrics": {name: all_metrics[name] for name in primary},
        "bootstrap": _bootstrap(primary_rankings, labels, incumbent, partial, bootstrap, seed)
        if bootstrap > 0
        else None,
        "controls": control_metrics,
        "control_warning": control_warning,
        "slices": _slices(docs, primary_rankings, labels, partial),
        "query_sensitivity": _query_sensitivity(all_metrics),
        "runs": {name: _run_metadata(run, embedders) for name, run in runs.items()},
    }


async def list_topics(session: AsyncSession) -> list[dict]:
    topics = (await session.scalars(select(EmbeddingTopic).order_by(EmbeddingTopic.name))).all()
    n_docs = dict(
        (
            await session.execute(
                select(EmbeddingTopicJob.topic_id, func.count()).group_by(
                    EmbeddingTopicJob.topic_id
                )
            )
        ).all()
    )
    label_counts: dict[UUID, dict[str, int]] = defaultdict(dict)
    label_rows = await session.execute(
        select(EmbeddingLabel.topic_id, EmbeddingLabel.source, func.count()).group_by(
            EmbeddingLabel.topic_id, EmbeddingLabel.source
        )
    )
    for topic_id, source, count in label_rows:
        label_counts[topic_id][source] = count
    n_runs = dict(
        (
            await session.execute(
                select(EmbeddingRun.topic_id, func.count())
                .where(EmbeddingRun.status == "completed")
                .group_by(EmbeddingRun.topic_id)
            )
        ).all()
    )
    return [
        {
            "name": t.name,
            "tier": t.tier,
            "status": t.status,
            "n_docs": n_docs.get(t.id, 0),
            "labels": dict(sorted(label_counts[t.id].items())),
            "runs": n_runs.get(t.id, 0),
            "created_at": _iso(t.created_at),
        }
        for t in topics
    ]


# ---------------------------------------------------------------------------- rendering


def _num(key: str, value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}" if key in _COUNTS else f"{value:.3f}"


def _table(header: list[str], rows: list[list[str]], indent: str = "  ") -> list[str]:
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        parts = [
            cell.ljust(width) if i == 0 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ]
        return (indent + "  ".join(parts)).rstrip()

    return [line(header), indent + "  ".join("-" * w for w in widths), *(line(r) for r in rows)]


def _label(name: str, incumbent: str) -> str:
    return f"{name} *" if name == incumbent else name


def format_report(report: dict) -> str:
    partial = report["partial"]
    incumbent = report["incumbent"]
    variants = report["variants"]
    headline = _headline(partial)
    labels = report["labels"]
    judge = report["judge_run"]
    out: list[str] = []

    out.append(f"Embedding eval: topic {report['topic']} (tier {report['tier']})")
    judge_text = f"{judge['id']} ({judge['model']}, {judge['prompt_version']})" if judge else "n/a"
    out.append(f"view: {report['view']}   judge run: {judge_text}")
    lang = report["language"]
    if lang["filter"] is not None:
        out.append(
            f"language filter: {lang['filter']} "
            f"({lang['n_included']} of {lang['n_included'] + lang['n_excluded']} documents)"
        )
    if report["stamps"]:
        out.append("!" * 72)
        for stamp in report["stamps"]:
            out.append(f"!! {stamp}: " + "; ".join(report["stamp_reasons"][stamp]))
        out.append("!! Do not quote these numbers as a verified comparison.")
        out.append("!" * 72)

    out.append("")
    out.append("Labels")
    out.append(
        f"  {labels['n_labeled']} of {report['n_docs']} documents labeled "
        f"(coverage {labels['coverage']:.1%});  R (grade >= {REL_THRESHOLD}) = {labels['r']}"
    )
    if partial:
        out.append("  partial view: metrics are computed on the human-labeled documents only")
    else:
        out.append(
            f"  Recall@100 ceiling = min(1, 100/R) = {labels['recall_ceiling@100']:.3f}: "
            "no ranking can exceed it"
        )

    if lang["filter"] is not None:
        excluded = ", ".join(f"{n} {name}" for name, n in lang["excluded"].items()) or "none"
        out.append("")
        out.append(
            f"Off-language leakage (excluded: {excluded}): documents outside "
            f"{lang['filter']} in each variant's unrestricted top {LEAKAGE_DEPTH}"
        )
        rows = [[_label(n, incumbent), str(c)] for n, c in lang["off_language_top100"].items()]
        out += _table(["variant", f"off-language@{LEAKAGE_DEPTH}"], rows)

    out.append("")
    out.append(f"Metrics (relevant = grade >= {REL_THRESHOLD}; * = incumbent)")
    columns = _TABLE_PARTIAL if partial else (*_TABLE, *_JUDGED)
    rows = [
        [_label(n, incumbent), *(_num(c, report["metrics"][n].get(c)) for c in columns)]
        for n in variants
    ]
    out += _table(["variant", *columns], rows)

    if not partial:
        ceiling = labels["recall_ceiling@100"]
        out.append("")
        out.append(f"Recall@100 (ceiling {ceiling:.3f}; * = incumbent)")
        rows = [
            [
                _label(n, incumbent),
                _num("recall@100", report["metrics"][n]["recall@100"]),
                f"{report['metrics'][n]['recall@100'] / ceiling:.1%}",
            ]
            for n in variants
        ]
        out += _table(["variant", "Recall@100", "of ceiling"], rows)

    boot = report["bootstrap"]
    if boot is not None:
        out.append("")
        scope = "condensed to judged documents, " if boot["condensed"] else ""
        out.append(
            f"Bootstrap: Δ vs incumbent {incumbent} [low, high]  "
            f"({scope}n={boot['n']}, seed={boot['seed']}, 95% interval; ! = excludes 0)"
        )
        for metric in headline:
            rows = []
            for name in variants:
                cell = boot["metrics"][metric].get(name)
                if cell is not None:
                    rows.append(
                        [
                            name,
                            f"{cell['mean_diff']:+.3f}",
                            f"[{cell['low']:+.3f}, {cell['high']:+.3f}]",
                            "!" if cell["excludes_zero"] else "",
                        ]
                    )
            if rows:
                out.append(f"  {metric}")
                out += _table(["variant", "Δ mean", "[low, high]", ""], rows, indent="    ")

    out.append("")
    out.append("Negative controls (must score below the incumbent)")
    rows = [
        [
            _label(incumbent, incumbent),
            *(_num(m, report["metrics"][incumbent][m]) for m in headline),
        ]
    ]
    rows += [
        [name, *(_num(m, scores[m]) for m in headline)]
        for name, scores in report["controls"].items()
    ]
    out += _table(["ranking", *headline], rows)
    if report["control_warning"]:
        out.append(f"  WARNING: {report['control_warning']}")

    for title, slices in (
        ("Slices (headline metrics)", report["slices"]),
        ("Language slices (headline metrics)", lang["slices"] if len(lang["slices"]) > 1 else {}),
    ):
        if slices:
            out.append("")
            out.append(title)
        for name, info in slices.items():
            out.append(
                f"  {name}: {info['n_docs']} docs, {info['n_labeled']} labeled, R={info['r']}"
            )
            rows = [
                [_label(v, incumbent), *(_num(m, info["metrics"][v][m]) for m in headline)]
                for v in variants
            ]
            out += _table(["variant", *headline], rows, indent="    ")

    sensitivity = report["query_sensitivity"]
    if sensitivity is not None:
        out.append("")
        out.append("Query sensitivity (ap per query recipe)")
        recipes = sorted(
            {r for by_recipe in sensitivity["variants"].values() for r in by_recipe},
            key=_recipe_key,
        )
        rows = [
            [name, *(_num("ap", by_recipe.get(r)) for r in recipes)]
            for name, by_recipe in sensitivity["variants"].items()
        ]
        out += _table(["variant", *recipes], rows)
        stability = sensitivity["stability"]
        if stability is not None:
            verdict = "holds" if stability["agree"] else "CHANGES"
            out.append(f"  embedder order {verdict} across recipes")
            for recipe in stability["recipes"]:
                out.append(
                    f"    {recipe}: {' > '.join(stability['orders'][recipe])}"
                    f"  (inversions vs first: {stability['disagreements'][recipe]})"
                )

    out.append("")
    out.append("Checks")
    for name, entry in report["checks"].items():
        out.append(f"  {name}: {entry['status'].upper()}")

    out.append("")
    out.append("Runs")
    rows = []
    for name in variants:
        meta = report["runs"][name]
        embedder = meta["embedder"] or {}
        rows.append(
            [
                name,
                (embedder.get("digest") or "-")[:12],
                embedder.get("quantization") or "-",
                embedder.get("runtime_version") or "-",
                "-" if meta["docs_per_s"] is None else f"{meta['docs_per_s']:.1f}",
                "-" if meta["n_truncated"] is None else str(meta["n_truncated"]),
                "-" if meta["seconds"] is None else f"{meta['seconds']:.0f}",
                (meta["git_sha"] or "-")[:8],
            ]
        )
    out += _table(["variant", "digest", "quant", "runtime", "docs/s", "trunc", "secs", "git"], rows)
    return "\n".join(out) + "\n"


def _json_default(value: object) -> str:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    topic = re.sub(r"[^\w.-]", "_", report["topic"])
    path = out_dir / f"embedding-{topic}-{report['view']}-{stamp}.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path
