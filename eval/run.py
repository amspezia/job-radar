"""Eval runner — one config set → all metrics → JSON result artifact.

Measures the three §12 configurations (hybrid / vector-only / keyword-only)
plus any named config supplied via --config.  Metrics are computed with our
pure-Python reference implementation (eval/metrics.py) and cross-validated
against ranx.  Results are written to eval/results/ as timestamped JSON.

Usage
-----
    uv run job-radar-eval-run [options]

Options
-------
  --profile-id UUID   Profile to evaluate (default: the single stored profile)
  --out-dir PATH      Where to write the JSON result (default: eval/results/)
  --min-labels N      Abort if fewer than N labeled pairs exist (default: 10)
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eval import metrics as m
from eval.qrels import (
    HYBRID,
    HYBRID_PROD,
    HYDE_ONLY,
    KEYWORD_ONLY,
    SearchConfig,
    build_run,
    load_qrels,
)
from job_radar.db.base import async_session_factory
from job_radar.db.models import Profile
from job_radar.fit.pipeline import build_hyde_embedding, build_lexical_query

logger = logging.getLogger(__name__)

_DEFAULT_OUT = Path("eval/results")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _ranking(run: dict[UUID, float]) -> list[UUID]:
    """Return job IDs sorted best-first by RRF score."""
    return sorted(run, key=lambda jid: -run[jid])


def _compute_metrics(run: dict[UUID, float], qrels: dict[UUID, int]) -> dict:
    """Compute all eval metrics using the pure-Python reference implementation."""
    ranking = _ranking(run)
    return {
        "ndcg_at_10": round(m.ndcg(ranking, qrels, 10), 4),
        "bpref": round(m.bpref(ranking, qrels, rel_threshold=2), 4),
        "mrr": round(m.mrr(ranking, qrels, rel_threshold=1), 4),
        "p_at_5": round(m.precision_at_k(ranking, qrels, 5, rel_threshold=2), 4),
        "p_at_10": round(m.precision_at_k(ranking, qrels, 10, rel_threshold=2), 4),
        "recall_at_20": round(m.recall_at_k(ranking, qrels, 20, rel_threshold=2), 4),
        "recall_at_50": round(m.recall_at_k(ranking, qrels, 50, rel_threshold=2), 4),
        "recall_at_100": round(m.recall_at_k(ranking, qrels, 100, rel_threshold=2), 4),
    }


def _validate_ranx(run: dict[UUID, float], qrels: dict[UUID, int]) -> None:
    """Cross-validate nDCG@10 against ranx — logs a warning if they diverge."""
    try:
        from ranx import Qrels, Run, evaluate  # type: ignore[import]

        q_key = "q"
        ranx_qrels = Qrels({q_key: {str(jid): g for jid, g in qrels.items()}})
        ranx_run = Run({q_key: {str(jid): s for jid, s in run.items()}})
        ranx_val = evaluate(ranx_qrels, ranx_run, "ndcg@10")
        our_val = m.ndcg(_ranking(run), qrels, 10)
        if abs(ranx_val - our_val) > 1e-3:
            logger.warning(
                "ranx nDCG@10=%.4f diverges from reference %.4f — check metric impl",
                ranx_val,
                our_val,
            )
        else:
            logger.debug("ranx cross-validation passed: nDCG@10=%.4f", ranx_val)
    except ImportError:
        logger.debug("ranx not installed — skipping cross-validation")


def _ranx_compare(
    qrels: dict[UUID, int],
    runs: list[tuple[str, dict[UUID, float]]],
) -> None:
    """Print ranx significance table if ranx is installed.

    Student's t-test requires multiple query topics to be meaningful.
    With a single profile → single topic (n=1), the test is undefined and
    suppressed. It activates once multi-query synthetic eval is in place.
    """
    try:
        from ranx import Qrels, Run, compare  # type: ignore[import]
    except ImportError:
        logger.debug("ranx not installed — skipping significance table")
        return

    q_key = "q"
    ranx_qrels = Qrels({q_key: {str(jid): g for jid, g in qrels.items()}})
    if len(ranx_qrels.qrels) < 2:
        logger.debug(
            "skipping ranx significance test — need ≥2 query topics, have %d",
            len(ranx_qrels.qrels),
        )
        return
    try:
        ranx_runs = [
            Run({q_key: {str(jid): s for jid, s in run.items()}}, name=name)
            for name, run in runs
        ]
        report = compare(
            ranx_qrels,
            ranx_runs,
            metrics=["ndcg@10", "recall@50"],
            stat_test="student",
            rounding_digits=4,
        )
        print("\nranx significance table:")
        print(report)
    except Exception as exc:
        logger.debug("ranx compare failed: %s", exc)


def _print_table(results: dict[str, dict]) -> None:
    configs = list(results.keys())
    metric_keys = [
        ("Recall@100", "recall_at_100"),
        ("Recall@50 ", "recall_at_50"),
        ("Recall@20 ", "recall_at_20"),
        ("nDCG@10  ", "ndcg_at_10"),
        ("BPref    ", "bpref"),
        ("MRR      ", "mrr"),
        ("P@5      ", "p_at_5"),
        ("P@10     ", "p_at_10"),
    ]
    col_w = 14
    header = " " * 10 + "  ".join(c.rjust(col_w) for c in configs)
    print()
    print(header)
    print("-" * len(header))
    for label, key in metric_keys:
        row = label + "  " + "  ".join(
            f"{results[c].get(key, 0):.4f}".rjust(col_w) for c in configs
        )
        print(row)
    print()


async def _evaluate_one(
    session: AsyncSession,
    profile: Profile,
    config: SearchConfig,
    qrels: dict[UUID, int],
    *,
    query: str,
    hyde_embedding: list[float] | None,
    config_name: str,
) -> dict:
    run = await build_run(
        session, profile, config, query=query, hyde_embedding=hyde_embedding
    )
    _validate_ranx(run, qrels)
    result_metrics = _compute_metrics(run, qrels)
    return {
        "config_name": config_name,
        "config": {
            "arms": config.arms,
            "k": config.k,
            "pool": config.pool,
            "limit": config.limit,
            "weights": config.weights,
            "field_boosts": config.field_boosts,
        },
        "pool_size": len(run),
        "num_labeled": len(qrels),
        **result_metrics,
    }


async def _run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    async with async_session_factory() as session:
        if args.profile_id:
            profile = (
                await session.execute(
                    select(Profile).where(Profile.id == UUID(args.profile_id))
                )
            ).scalars().first()
            if profile is None:
                print(f"Profile {args.profile_id} not found.")
                sys.exit(1)
        else:
            profile = (await session.execute(select(Profile))).scalars().first()
            if profile is None:
                print("No profile found — run job-radar-profile first.")
                sys.exit(1)

        qrels = await load_qrels(session, profile.id)
        if len(qrels) < args.min_labels:
            print(
                f"Only {len(qrels)} labeled pairs (need ≥{args.min_labels}). "
                "Run job-radar-eval-label first."
            )
            sys.exit(1)

        grades_by_count: dict[int, int] = {}
        for g in qrels.values():
            grades_by_count[g] = grades_by_count.get(g, 0) + 1
        grade_dist = "  ".join(f"grade {g}: {n}" for g, n in sorted(grades_by_count.items()))
        print(f"\nQrels: {len(qrels)} labeled pairs  [{grade_dist}]")

        # Pre-compute embeddings once — shared across all config evaluations.
        print("Building query representations …")
        lexical_q = build_lexical_query(profile)
        hyde_embedding = await build_hyde_embedding(profile, session)
        print(
            f"  Lexical query: {len(lexical_q.split())} tokens  |  "
            f"HyDE: {'ok' if hyde_embedding else 'skipped'}"
        )

        configs: list[tuple[str, SearchConfig]] = [
            ("hybrid", HYBRID),
            ("hybrid_prod", HYBRID_PROD),
            ("hyde_only", HYDE_ONLY),
            ("keyword_only", KEYWORD_ONLY),
        ]

        print("Evaluating all configs concurrently …")
        results = await asyncio.gather(*[
            _evaluate_one(
                session,
                profile,
                config,
                qrels,
                query=lexical_q,
                hyde_embedding=hyde_embedding,
                config_name=name,
            )
            for name, config in configs
        ])
        all_results: dict[str, dict] = {}
        for (name, _), result in zip(configs, results):
            all_results[name] = result
            print(f"  {name}: nDCG@10={result['ndcg_at_10']:.4f}  Recall@50={result['recall_at_50']:.4f}")

    _print_table(all_results)

    sha = _git_sha()
    run_at = datetime.now(UTC).isoformat()
    artifact = {
        "profile_id": str(profile.id),
        "run_at": run_at,
        "git_sha": sha,
        "configurations": list(all_results.values()),
    }
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"eval-{ts}.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"Result written to {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid search against labeled qrels; write JSON artifact."
    )
    parser.add_argument("--profile-id", default=None, help="Profile UUID to evaluate")
    parser.add_argument(
        "--out-dir", default=str(_DEFAULT_OUT), help="Output directory for JSON artifact"
    )
    parser.add_argument(
        "--min-labels",
        type=int,
        default=10,
        help="Minimum labeled pairs required before aborting (default: 10)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
