"""One-at-a-time (OAT) parameter sweep for hybrid search tuning.

Follows the impact-priority order from M6_EVAL_IMPLEMENTATION_PLAN.md §4.5:
  1. RRF k         — highest-sensitivity parameter
  2. Arm weights   — fixes the 2:1 vector:lexical bias in the default config
  3. Pool size     — elbow-of-recall vs latency
  4. BM25 field boosts — title/requirements/responsibilities weights

Each dimension is swept while holding all others at their default.  Results
are printed as a table and written to eval/results/sweep-<timestamp>.json.

If ranx is installed, each dimension's runs are compared with a paired
significance test (Student's t) via ranx.compare().

Usage
-----
    uv run job-radar-eval-sweep [--dimension k|weights|pool|boosts|all] [options]

Options
-------
  --dimension    Which dimension to sweep (default: all)
  --min-labels N Minimum labeled pairs required (default: 10)
  --profile-id   Profile UUID to evaluate (default: stored profile)
  --out-dir PATH Output directory (default: eval/results/)
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
from eval.qrels import SearchConfig, build_run, load_qrels
from eval.run import _ranx_compare
from job_radar.adapters.embeddings import embed
from job_radar.db.base import async_session_factory
from job_radar.db.models import Profile
from job_radar.fit.pipeline import build_dense_query, build_lexical_query

logger = logging.getLogger(__name__)

_DEFAULT_OUT = Path("eval/results")

# OAT sweep grids — derived from M6 plan §4.5.
_K_VALUES = [20, 30, 40, 60, 90, 120]

_WEIGHT_CONFIGS: list[tuple[str, list[float]]] = [
    ("1/1/1 equal", [1.0, 1.0, 1.0]),
    ("2/1/1 lexical↑", [2.0, 1.0, 1.0]),
    ("1/2/1 HyDE↑", [1.0, 2.0, 1.0]),
    ("1/1/2 CV↑", [1.0, 1.0, 2.0]),
    ("2/2/1 lexical+HyDE↑", [2.0, 2.0, 1.0]),
    ("1/2/2 vector↑", [1.0, 2.0, 2.0]),
]

_POOL_VALUES = [20, 50, 100]

_BOOST_CONFIGS: list[tuple[str, dict[str, int]]] = [
    ("title^3 req^1 resp^1", {"title": 3, "requirements": 1, "responsibilities": 1}),
    ("title^5 req^2 resp^1", {"title": 5, "requirements": 2, "responsibilities": 1}),  # default
    ("title^7 req^2 resp^1", {"title": 7, "requirements": 2, "responsibilities": 1}),
    ("title^5 req^3 resp^1", {"title": 5, "requirements": 3, "responsibilities": 1}),
    ("title^5 req^2 resp^2", {"title": 5, "requirements": 2, "responsibilities": 2}),
    ("title^7 req^3 resp^2", {"title": 7, "requirements": 3, "responsibilities": 2}),
]


def _ranking(run: dict[UUID, float]) -> list[UUID]:
    return sorted(run, key=lambda jid: -run[jid])


def _score(run: dict[UUID, float], qrels: dict[UUID, int]) -> dict[str, float]:
    r = _ranking(run)
    return {
        "recall_at_50": round(m.recall_at_k(r, qrels, 50, rel_threshold=2), 4),
        "ndcg_at_10": round(m.ndcg(r, qrels, 10), 4),
        "mrr": round(m.mrr(r, qrels, rel_threshold=1), 4),
        "p_at_5": round(m.precision_at_k(r, qrels, 5, rel_threshold=2), 4),
    }


def _print_sweep_table(dimension: str, rows: list[dict]) -> None:
    col_w = 16
    label_w = max(len(r["label"]) for r in rows)
    header = (
        f"\n{'':>{label_w}}  "
        + "  ".join(k.rjust(col_w) for k in ["Recall@50", "nDCG@10", "MRR", "P@5"])
    )
    print(f"\n=== Sweep: {dimension} ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            row["label"].ljust(label_w)
            + "  "
            + "  ".join(
                f"{row.get(k, 0):.4f}".rjust(col_w)
                for k in ["recall_at_50", "ndcg_at_10", "mrr", "p_at_5"]
            )
        )
        print(line)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


async def _sweep_k(
    session: AsyncSession,
    profile: Profile,
    qrels: dict[UUID, int],
    query: str,
    hyde_embedding: list[float] | None,
) -> list[dict]:
    rows: list[dict] = []
    raw_runs: list[tuple[str, dict[UUID, float]]] = []
    for k_val in _K_VALUES:
        config = SearchConfig(arms=["lexical", "hyde", "cv"], k=k_val, pool=100, limit=100)
        run = await build_run(session, profile, config, query=query, hyde_embedding=hyde_embedding)
        scores = _score(run, qrels)
        label = f"k={k_val}"
        rows.append({"label": label, **scores})
        raw_runs.append((label, run))
    _print_sweep_table("RRF k", rows)
    _ranx_compare(qrels, raw_runs)
    return rows


async def _sweep_weights(
    session: AsyncSession,
    profile: Profile,
    qrels: dict[UUID, int],
    query: str,
    hyde_embedding: list[float] | None,
) -> list[dict]:
    rows: list[dict] = []
    raw_runs: list[tuple[str, dict[UUID, float]]] = []

    for label, weights in _WEIGHT_CONFIGS:
        if len(weights) != 3:
            continue  # skip malformed entries
        # Pass the full 3-element weight vector keyed to ["lexical","hyde","cv"].
        # build_run slices it by active_indices so inactive arms are excluded
        # correctly even when they're not the last arm in the list.
        config = SearchConfig(
            arms=["lexical", "hyde", "cv"], k=60, pool=100, limit=100, weights=weights
        )
        run = await build_run(session, profile, config, query=query, hyde_embedding=hyde_embedding)
        scores = _score(run, qrels)
        rows.append({"label": label, **scores})
        raw_runs.append((label, run))

    _print_sweep_table("Arm weights (lexical / HyDE / CV)", rows)
    _ranx_compare(qrels, raw_runs)
    return rows


async def _sweep_pool(
    session: AsyncSession,
    profile: Profile,
    qrels: dict[UUID, int],
    query: str,
    hyde_embedding: list[float] | None,
) -> list[dict]:
    rows: list[dict] = []
    raw_runs: list[tuple[str, dict[UUID, float]]] = []
    for pool_val in _POOL_VALUES:
        config = SearchConfig(
            arms=["lexical", "hyde", "cv"], k=60, pool=pool_val, limit=pool_val  # pool varies
        )
        run = await build_run(session, profile, config, query=query, hyde_embedding=hyde_embedding)
        scores = _score(run, qrels)
        label = f"pool={pool_val}"
        rows.append({"label": label, **scores})
        raw_runs.append((label, run))
    _print_sweep_table("Pool size", rows)
    _ranx_compare(qrels, raw_runs)
    return rows


async def _sweep_boosts(
    session: AsyncSession,
    profile: Profile,
    qrels: dict[UUID, int],
    query: str,
    hyde_embedding: list[float] | None,
) -> list[dict]:
    rows: list[dict] = []
    raw_runs: list[tuple[str, dict[UUID, float]]] = []
    for label, boosts in _BOOST_CONFIGS:
        config = SearchConfig(
            arms=["lexical", "hyde", "cv"], k=60, pool=100, limit=100, field_boosts=boosts
        )
        run = await build_run(session, profile, config, query=query, hyde_embedding=hyde_embedding)
        scores = _score(run, qrels)
        rows.append({"label": label, **scores})
        raw_runs.append((label, run))
    _print_sweep_table("BM25 field boosts", rows)
    _ranx_compare(qrels, raw_runs)
    return rows


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

        print(f"Sweep: profile={profile.full_name}  labeled={len(qrels)} pairs")

        lexical_q = build_lexical_query(profile)
        dense_text = await build_dense_query(profile, session)
        hyde_embedding = await embed(dense_text, task="document") if dense_text else None

        sweep_results: dict[str, list[dict]] = {}
        dims = args.dimension if args.dimension != "all" else ["k", "weights", "pool", "boosts"]
        dims_set = set(dims)

        if "k" in dims_set:
            sweep_results["k"] = await _sweep_k(
                session, profile, qrels, lexical_q, hyde_embedding
            )
        if "weights" in dims_set:
            sweep_results["weights"] = await _sweep_weights(
                session, profile, qrels, lexical_q, hyde_embedding
            )
        if "pool" in dims_set:
            sweep_results["pool"] = await _sweep_pool(
                session, profile, qrels, lexical_q, hyde_embedding
            )
        if "boosts" in dims_set:
            sweep_results["boosts"] = await _sweep_boosts(
                session, profile, qrels, lexical_q, hyde_embedding
            )

    artifact = {
        "profile_id": str(profile.id),
        "run_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "dimensions_swept": list(sweep_results.keys()),
        "results": sweep_results,
    }
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"sweep-{ts}.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nSweep results written to {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="OAT parameter sweep for hybrid search tuning."
    )
    parser.add_argument(
        "--dimension",
        nargs="+",
        default=["all"],
        choices=["k", "weights", "pool", "boosts", "all"],
        help="Dimension(s) to sweep (default: all)",
    )
    parser.add_argument("--profile-id", default=None, help="Profile UUID to evaluate")
    parser.add_argument(
        "--out-dir", default=str(_DEFAULT_OUT), help="Output directory for JSON artifact"
    )
    parser.add_argument(
        "--min-labels",
        type=int,
        default=10,
        help="Minimum labeled pairs required (default: 10)",
    )
    args = parser.parse_args()
    # Normalize "all" in the list.
    if "all" in args.dimension:
        args.dimension = "all"
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
