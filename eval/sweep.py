"""One-at-a-time (OAT) parameter sweep for hybrid search tuning.

Production baseline: 2-arm (lexical + HyDE), k=60, pool=100,
field_boosts={title:5, requirements:3, responsibilities:1}, no explicit weights.
Each dimension varies one parameter while holding the rest at baseline defaults.

Dimensions:
  1. RRF k         — smoothing constant; higher → more uniform contribution
  2. Arm weights   — lexical vs HyDE relative importance
  3. Pool size     — per-arm candidate count before fusion
  4. BM25 field boosts — title/requirements/responsibilities weights
  5. Query construction — which profile fields compose the BM25 query token bag

Results are printed as a table and written to eval/results/sweep-<timestamp>.json.

Usage
-----
    uv run job-radar-eval-sweep [--dimension k|weights|pool|boosts|query|all] [options]

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
from job_radar.db.base import async_session_factory
from job_radar.db.models import Profile
from job_radar.fit.pipeline import build_hyde_embedding, build_lexical_query

logger = logging.getLogger(__name__)

_DEFAULT_OUT = Path("eval/results")

# OAT sweep grids — each dimension varies one parameter from the production
# baseline: arms=["lexical","hyde"], k=60, pool=100, limit=100, boosts=default.
# Production uses no explicit weights (2 arms, equal by default in RRF).
_K_VALUES = [20, 30, 40, 60, 90, 120]

_WEIGHT_CONFIGS: list[tuple[str, list[float]]] = [
    ("1/1 equal", [1.0, 1.0]),
    ("2/1 lexical↑", [2.0, 1.0]),
    ("1/2 HyDE↑", [1.0, 2.0]),
    ("3/2 lexical↑", [3.0, 2.0]),
    ("2/3 HyDE↑", [2.0, 3.0]),
]

_POOL_VALUES = [20, 50, 100, 150, 200]

def _build_query_variant(profile: Profile, variant: str) -> str:
    """Build a BM25 token bag for the given query construction variant.

    All variants deduplicate tokens (same logic as production build_lexical_query)
    so repeated words don't accumulate artificial IDF weight.
    """
    keywords = profile.domains_keywords or {}
    tech_stack = keywords.get("tech_stack", [])
    domains = keywords.get("domains", [])
    titles = profile.target_titles or []

    parts: list[str]
    if variant == "baseline":
        parts = [*titles, *tech_stack]
    elif variant == "+domains":
        parts = [*titles, *tech_stack, *domains]
    elif variant == "titles_only":
        parts = list(titles)
    elif variant == "stack_only":
        parts = list(tech_stack)
    elif variant == "+domains_no_stack":
        parts = [*titles, *domains]
    else:
        raise ValueError(f"unknown query variant: {variant!r}")

    seen: set[str] = set()
    tokens: list[str] = []
    for part in parts:
        for tok in part.split():
            if tok.lower() not in seen:
                seen.add(tok.lower())
                tokens.append(tok)
    return " ".join(tokens)


_QUERY_VARIANTS: list[str] = [
    "baseline",
    "+domains",
    "titles_only",
    "stack_only",
    "+domains_no_stack",
]

_BOOST_CONFIGS: list[tuple[str, dict[str, int]]] = [
    ("title^3 req^1 resp^1", {"title": 3, "requirements": 1, "responsibilities": 1}),
    ("title^5 req^2 resp^1", {"title": 5, "requirements": 2, "responsibilities": 1}),
    ("title^5 req^3 resp^1", {"title": 5, "requirements": 3, "responsibilities": 1}),
    ("title^7 req^2 resp^1", {"title": 7, "requirements": 2, "responsibilities": 1}),
    ("title^7 req^3 resp^1", {"title": 7, "requirements": 3, "responsibilities": 1}),
    ("title^7 req^3 resp^2", {"title": 7, "requirements": 3, "responsibilities": 2}),
]


def _ranking(run: dict[UUID, float]) -> list[UUID]:
    return sorted(run, key=lambda jid: -run[jid])


def _score(run: dict[UUID, float], qrels: dict[UUID, int]) -> dict[str, float]:
    r = _ranking(run)
    return {
        "recall_at_100": round(m.recall_at_k(r, qrels, 100, rel_threshold=2), 4),
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
        + "  ".join(k.rjust(col_w) for k in ["Recall@100", "Recall@50", "nDCG@10", "MRR", "P@5"])
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
                for k in ["recall_at_100", "recall_at_50", "ndcg_at_10", "mrr", "p_at_5"]
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
        config = SearchConfig(arms=["lexical", "hyde"], k=k_val, pool=100, limit=100)
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
        config = SearchConfig(
            arms=["lexical", "hyde"], k=60, pool=100, limit=100, weights=weights
        )
        run = await build_run(session, profile, config, query=query, hyde_embedding=hyde_embedding)
        scores = _score(run, qrels)
        rows.append({"label": label, **scores})
        raw_runs.append((label, run))

    _print_sweep_table("Arm weights (lexical / HyDE)", rows)
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
        config = SearchConfig(arms=["lexical", "hyde"], k=60, pool=pool_val, limit=100)
        run = await build_run(session, profile, config, query=query, hyde_embedding=hyde_embedding)
        scores = _score(run, qrels)
        label = f"pool={pool_val}"
        rows.append({"label": label, **scores})
        raw_runs.append((label, run))
    _print_sweep_table("Pool size", rows)
    _ranx_compare(qrels, raw_runs)
    return rows


async def _sweep_query(
    session: AsyncSession,
    profile: Profile,
    qrels: dict[UUID, int],
    hyde_embedding: list[float] | None,
) -> list[dict]:
    rows: list[dict] = []
    raw_runs: list[tuple[str, dict[UUID, float]]] = []
    for variant in _QUERY_VARIANTS:
        query = _build_query_variant(profile, variant)
        config = SearchConfig(arms=["lexical", "hyde"], k=60, pool=100, limit=100)
        run = await build_run(session, profile, config, query=query, hyde_embedding=hyde_embedding)
        scores = _score(run, qrels)
        rows.append({"label": variant, **scores})
        raw_runs.append((variant, run))
    _print_sweep_table("Query construction", rows)
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
            arms=["lexical", "hyde"], k=60, pool=100, limit=100, field_boosts=boosts
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
        hyde_embedding = await build_hyde_embedding(profile, session)

        sweep_results: dict[str, list[dict]] = {}
        all_dims = ["k", "weights", "pool", "boosts", "query"]
        dims = args.dimension if args.dimension != "all" else all_dims
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
        if "query" in dims_set:
            sweep_results["query"] = await _sweep_query(
                session, profile, qrels, hyde_embedding
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
        choices=["k", "weights", "pool", "boosts", "query", "all"],
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
    if "all" in args.dimension:
        args.dimension = "all"
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
