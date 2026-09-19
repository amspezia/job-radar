"""`job-radar-eval-embedding` — import, embed, rank, judge and evaluate embedding-eval topics.

Every command imports its own dependencies inside its function: `--help` stays instant, and the
two prod-touching modules (`tiers.tier_a`, `parity`) are loaded only by the command that needs
them, so no other command can reach `job_radar` through this entry point.
"""

import argparse
import asyncio
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from eval.embedding.labels import VIEWS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eval.embedding.embedders.base import EmbedderSpec
    from eval.embedding.methods.base import TopicView


def _git_sha() -> str | None:
    """Short HEAD sha, recorded on every run; None when git cannot answer."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def _select_specs(specs: "Sequence[EmbedderSpec]", name: str) -> "list[EmbedderSpec]":
    """The configured embedder(s) `name` asks for; `all` means every one."""
    if name == "all":
        return list(specs)
    chosen = [spec for spec in specs if spec.name == name]
    if not chosen:
        known = ", ".join(spec.name for spec in specs)
        raise ValueError(f"unknown embedder {name!r}; configured: {known}, or 'all'")
    return chosen


def _recipes(topic: "TopicView", per_text: bool) -> list[str]:
    """`hyde_mean`, plus one recipe per frozen HyDE text for the query-sensitivity runs."""
    recipes = ["hyde_mean"]
    if per_text:
        recipes += [f"hyde_text:{i}" for i in range(len(topic.query_inputs["hyde_texts"]))]
    return recipes


def _result_line(name: str, recipe: str, stats: dict, run_id: UUID | None) -> str:
    rate = stats.get("docs_per_s")
    rate_text = f"{rate:.1f}" if rate else "-"
    line = (
        f"{name:<26} {recipe:<14} {rate_text:>6} docs/s  truncated {stats.get('n_truncated') or 0}"
    )
    return line if run_id is None else f"{line}  run {run_id}"


async def _import_topic(args: argparse.Namespace) -> int:
    from eval.embedding.tiers.tier_a import import_topic

    summary = await import_topic(args.name, args.tier, profile_id=args.profile_id)
    print(
        f"topic {summary.topic_id}  jobs {summary.n_jobs_new} new, {summary.n_jobs_reused} reused"
    )
    return 0


async def _embed_or_run(args: argparse.Namespace, *, execute: bool, per_text: bool) -> int:
    from eval.embedding.cache import VectorCache
    from eval.embedding.embedders.registry import build_embedder, load_specs
    from eval.embedding.methods.base import load_topic_view
    from eval.embedding.methods.dense import DenseMethod
    from eval.embedding.runner import run_method
    from eval.evals_db.base import evals_session
    from eval.evals_db.settings import get_settings
    from eval.llm.ollama import OllamaClient

    try:
        specs = _select_specs(load_specs(), args.embedder)
    except ValueError as exc:
        print(exc)
        return 1

    git_sha = _git_sha() if execute else None
    client = OllamaClient(get_settings().ollama_base_url)
    try:
        async with evals_session() as session:
            topic = await load_topic_view(session, args.topic)
            cache = VectorCache(session)
            for spec in specs:
                embedder = build_embedder(spec, client)
                await embedder.ready()
                for recipe in _recipes(topic, per_text):
                    method = DenseMethod(embedder, cache, query_recipe=recipe)
                    run_id = None
                    if execute:
                        run_id = await run_method(session, args.topic, method, git_sha=git_sha)
                    else:
                        await method.prepare(topic)
                    print(_result_line(spec.name, recipe, method.stats(), run_id))
    finally:
        await client.aclose()
    return 0


async def _embed(args: argparse.Namespace) -> int:
    return await _embed_or_run(args, execute=False, per_text=False)


async def _run(args: argparse.Namespace) -> int:
    return await _embed_or_run(args, execute=True, per_text=args.per_text)


async def _verify(args: argparse.Namespace) -> int:
    from eval.embedding.checks import REQUIRED_CHECKS
    from eval.embedding.parity import verify_topic

    results = await verify_topic(args.topic)
    failed = []
    for name, result in sorted(results.items()):
        passed = bool(result["passed"])
        print(f"{name:<26} {'PASSED' if passed else 'FAILED'}")
        if result.get("detail"):
            print(f"  {result['detail']}")
        if not passed and name in REQUIRED_CHECKS:
            failed.append(name)
    if failed:
        print(f"required check(s) failed: {', '.join(failed)}")
        return 1
    return 0


async def _judge(args: argparse.Namespace) -> int:
    from eval.embedding.judge.runner import build_judge, label_topic
    from eval.embedding.methods.base import load_topic_view
    from eval.embedding.models import EmbeddingJudgeRun
    from eval.evals_db.base import evals_session
    from eval.evals_db.settings import get_settings
    from eval.llm.ollama import OllamaClient

    client = OllamaClient(get_settings().ollama_base_url)
    try:
        await client.warm_chat(args.model)
        async with evals_session() as session:
            topic = await load_topic_view(session, args.topic)
            judge, fewshot_ids = await build_judge(
                session,
                args.topic,
                client,
                args.model,
                per_grade=args.fewshot_per_grade,
                fewshot_seed=args.fewshot_seed,
            )
            run_id = await label_topic(
                session,
                topic.topic_id,
                judge,
                fewshot_ids,
                concurrency=args.concurrency,
                limit=args.limit,
                resume_run_id=args.resume,
                labeled_only=args.labeled_only,
            )
            run = await session.get(EmbeddingJudgeRun, run_id)
    finally:
        await client.aclose()
    print(f"judge run {run_id}  {run.status}  labeled {run.n_labeled}  failed {run.n_failed}")
    return 0


async def _judge_calibrate(args: argparse.Namespace) -> int:
    from eval.embedding.judge.runner import calibrate, caveats
    from eval.evals_db.base import evals_session

    async with evals_session() as session:
        result = await calibrate(
            session, args.topic, args.judge_run, args.min_kappa, use_split=args.held_out_half
        )
    scope = "the held-out half of the" if args.held_out_half else "all"
    print(
        f"kappa_w {result.kappa:.3f} over {result.n} ({scope} non-exemplar blind labels; "
        f"gate {args.min_kappa:.2f})"
    )
    print(f"exact {result.exact:.3f}  within-1 {result.within1:.3f}")
    print(
        f"grade >= 2  precision {result.binary['precision']:.3f}  "
        f"recall {result.binary['recall']:.3f}"
    )
    print("confusion (row = human grade, column = judge grade)")
    for grade, row in enumerate(result.confusion):
        print(f"  {grade}  " + "  ".join(f"{cell:>4}" for cell in row))
    print("PASSED" if result.passed else "FAILED")
    for caveat in caveats(result.n):
        print(f"caveat: {caveat}")
    return 0


async def _label_blind(args: argparse.Namespace) -> int:
    import signal

    from eval.embedding.label_blind import label_blind
    from eval.evals_db.base import evals_session

    # asyncio's own handler would swallow the first Ctrl-C while `input()` blocks.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    async with evals_session() as session:
        counts = await label_blind(
            session,
            args.topic,
            n=args.n,
            bucket=args.bucket,
            seed=args.seed,
            language=None if args.language == "any" else args.language,
        )
    print(
        f"labeled {counts['labeled']}, skipped {counts['skipped']}, "
        f"{counts['remaining']} unlabeled remain"
    )
    return 0


async def _evaluate(args: argparse.Namespace) -> int:
    from eval.embedding.evaluate import MissingChecks, evaluate_topic, format_report, write_report
    from eval.evals_db.base import evals_session

    views = list(VIEWS) if args.view == "all" else [args.view]
    refused = []
    async with evals_session() as session:
        for view in views:
            try:
                report = await evaluate_topic(
                    session,
                    args.topic,
                    view,
                    judge_run_id=args.judge_run,
                    bootstrap=args.bootstrap,
                    seed=0,
                    allow_unverified=args.allow_unverified,
                    language=args.language,
                )
            except MissingChecks as exc:
                print(f"{view}: refused — {exc}")
                refused.append(view)
                continue
            print(format_report(report))
            print(f"wrote {write_report(report, Path(args.out_dir))}")
    if refused:
        print(f"refused: {', '.join(refused)} — pass --allow-unverified to report them anyway")
        return 1
    return 0


async def _topics(args: argparse.Namespace) -> int:
    from eval.embedding.evaluate import list_topics
    from eval.evals_db.base import evals_session

    async with evals_session() as session:
        rows = await list_topics(session)
    if not rows:
        print("no topics — run `import-topic` first")
        return 0
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(str(row.get(c, ""))) for row in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    return 0


async def _labels_export(args: argparse.Namespace) -> int:
    from eval.embedding.labels_io import export_labels
    from eval.evals_db.base import evals_session

    path = Path(args.out) if args.out else Path("data") / f"labels-{args.topic}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    async with evals_session() as session:
        written = await export_labels(session, args.topic, path)
    print(f"exported {written} human label(s) to {path}")
    return 0


async def _labels_import(args: argparse.Namespace) -> int:
    from eval.embedding.labels_io import import_labels
    from eval.evals_db.base import evals_session

    async with evals_session() as session:
        read = await import_labels(session, args.topic, Path(args.file))
    print(f"imported {read} label(s) into {args.topic}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-radar-eval-embedding",
        description="Embedding-eval harness: import a topic, embed it, judge it, evaluate it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("import-topic", help="copy a tier's pool and queries in")
    p.add_argument("--tier", required=True, choices=("A",), help="tier builder to run")
    p.add_argument("--name", required=True, help="topic name, e.g. A-real-2026-09-18")
    p.add_argument("--profile-id", type=UUID, help="profile to import (default: the real profile)")
    p.set_defaults(func=_import_topic)

    p = commands.add_parser("embed", help="fill the vector cache for a topic, creating no runs")
    p.add_argument("--topic", required=True, help="topic name")
    p.add_argument("--embedder", required=True, help="configured embedder name, or 'all'")
    p.set_defaults(func=_embed)

    p = commands.add_parser("run", help="embed, rank and store a full ranking per embedder")
    p.add_argument("--topic", required=True, help="topic name")
    p.add_argument("--embedder", required=True, help="configured embedder name, or 'all'")
    p.add_argument(
        "--per-text",
        action="store_true",
        help="also run one query recipe per frozen HyDE text (query sensitivity)",
    )
    p.set_defaults(func=_run)

    p = commands.add_parser("verify", help="run the parity checks against production")
    p.add_argument("--topic", required=True, help="topic name")
    p.set_defaults(func=_verify)

    p = commands.add_parser(
        "label-blind", help="grade postings yourself, blind to every model score (the gold)"
    )
    p.add_argument("--topic", required=True, help="topic name")
    p.add_argument("--n", type=int, default=50, help="postings to label this session (default: 50)")
    p.add_argument(
        "--bucket",
        choices=("mixed", "pooled", "random"),
        default="mixed",
        help="which postings: ~60%% pooled + ~40%% random (default), or only one kind",
    )
    p.add_argument("--seed", type=int, default=0, help="ordering seed; reuse it to resume")
    p.add_argument(
        "--language",
        choices=("en", "es", "pt", "any"),
        default="en",
        help="offer only postings detected in this language; 'any' offers all (default: en)",
    )
    p.set_defaults(func=_label_blind)

    p = commands.add_parser("judge", help="grade a topic's pool with the LLM judge")
    p.add_argument("--topic", required=True, help="topic name")
    p.add_argument("--model", required=True, help="Ollama chat model, e.g. gemma3:12b")
    p.add_argument("--limit", type=int, help="grade at most this many jobs (timing spikes)")
    p.add_argument(
        "--labeled-only",
        action="store_true",
        help="grade only human-labeled jobs (fast prompt iteration; --resume finishes the pool)",
    )
    p.add_argument("--resume", type=UUID, help="judge run to continue instead of a new one")
    p.add_argument("--concurrency", type=int, default=4, help="parallel judgments (default: 4)")
    p.add_argument("--fewshot-seed", type=int, default=13, help="exemplar draw seed (default: 13)")
    p.add_argument(
        "--fewshot-per-grade", type=int, default=1, help="exemplars per grade 0-3 (default: 1)"
    )
    p.set_defaults(func=_judge)

    p = commands.add_parser("judge-calibrate", help="score a judge run against your blind labels")
    p.add_argument("--topic", required=True, help="topic name")
    p.add_argument("--judge-run", required=True, type=UUID, help="judge run to calibrate")
    p.add_argument("--min-kappa", type=float, default=0.6, help="pass threshold (default: 0.6)")
    p.add_argument(
        "--held-out-half",
        action="store_true",
        help="calibrate on the held-out half of the non-exemplar labels (default: all of them)",
    )
    p.set_defaults(func=_judge_calibrate)

    p = commands.add_parser("evaluate", help="metrics, intervals and controls for a topic")
    p.add_argument("--topic", required=True, help="topic name")
    p.add_argument("--view", required=True, choices=(*VIEWS, "all"), help="label view to score")
    p.add_argument("--judge-run", type=UUID, help="judge run pinning the silver labels")
    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap samples (default: 1000)")
    p.add_argument(
        "--allow-unverified",
        action="store_true",
        help="report even without the required checks, stamped UNVERIFIED",
    )
    p.add_argument(
        "--language",
        choices=("en", "es", "pt"),
        help="score only documents detected in this language and report the off-language "
        "leakage; use 'en' for the English-only corpus (default: score every document)",
    )
    p.add_argument("--out-dir", default="eval/results", help="report directory")
    p.set_defaults(func=_evaluate)

    p = commands.add_parser("topics", help="list the imported topics")
    p.set_defaults(func=_topics)

    p = commands.add_parser("labels-export", help="back up the human label rows as JSON")
    p.add_argument("--topic", required=True, help="topic name")
    p.add_argument("--out", help="output path (default: data/labels-<topic>.json)")
    p.set_defaults(func=_labels_export)

    p = commands.add_parser("labels-import", help="restore human label rows from a JSON backup")
    p.add_argument("--topic", required=True, help="topic name")
    p.add_argument("--file", required=True, help="JSON file written by labels-export")
    p.set_defaults(func=_labels_import)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(args.func(args))
