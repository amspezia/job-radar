"""Per-persona retrieval eval over the 5 synthetic personas (B1).

    uv run job-radar-eval-run-synthetic [--out-dir PATH]

Requires eval/inject_synthetic.py to have been run first. Deliberately reuses
the exact machinery real-profile eval uses — eval/qrels.py's
load_qrels/build_run, eval/metrics.py — unmodified: a synthetic persona
differs from a real profile only in how its EvalLabel rows got there
(construction, not human judgment), so nothing about how it's scored should
differ either.

This is the actual fix for the n=1-query-topic problem flagged in
docs/ASSESSMENT.md: every prior retrieval-tuning conclusion in this codebase
was drawn from one profile. Reports nDCG@10 + BPref per persona — stratified,
not just averaged, since domain heterogeneity across personas can hide a
regression that only hits one persona's slice — for the HYBRID_PROD config
(the one that actually mirrors fit/pipeline.py's production weights/limit,
see eval/qrels.py).

Reports two nDCG@10 variants per persona: the "mixed" one (ranked against the
whole corpus, real unlabeled postings included — what a real user's top-10
actually looks like) and an "isolated" one (ranked among only this persona's
own 20 labeled jobs) — the isolated variant is the one comparable across
personas regardless of how common a persona's stack vocabulary happens to be
in the real corpus (see eval/results/RETRIEVAL_ANALYSIS.md finding #3).
"""

import asyncio
import json
import logging
import subprocess
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eval import metrics as m
from eval import personas_lib
from eval.qrels import HYBRID_PROD, build_run, load_qrels
from job_radar.db.base import async_session_factory
from job_radar.db.models import Profile
from job_radar.fit.pipeline import build_hyde_embedding, build_lexical_query

logger = logging.getLogger(__name__)
_DEFAULT_OUT = Path("eval/results")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _ranking(run: dict) -> list:
    return sorted(run, key=lambda jid: -run[jid])


async def _evaluate_persona(session: AsyncSession, persona_id: str) -> dict | None:
    profile_id = personas_lib.profile_uuid(persona_id)
    profile = (
        (await session.execute(select(Profile).where(Profile.id == profile_id))).scalars().first()
    )
    if profile is None:
        logger.warning("Persona %s not injected — run eval/inject_synthetic.py first", persona_id)
        return None

    qrels = await load_qrels(session, profile_id)
    if not qrels:
        logger.warning("No EvalLabel rows for %s — run eval/inject_synthetic.py first", persona_id)
        return None

    lexical_q = build_lexical_query(profile)
    hyde_embedding = await build_hyde_embedding(profile, session)
    run = await build_run(
        session, profile, HYBRID_PROD, query=lexical_q, hyde_embedding=hyde_embedding
    )
    ranking = _ranking(run)
    # Ranked among only this persona's own labeled jobs — every id in `qrels` is one
    # of that persona's 20 synthetic postings (load_qrels filters by profile_id), so
    # this isolates ranking quality from how much real, unlabeled corpus competes for
    # the same top-10 window (see RETRIEVAL_ANALYSIS.md finding #3).
    isolated_ranking = [job_id for job_id in ranking if job_id in qrels]

    return {
        "persona_id": persona_id,
        "num_labeled": len(qrels),
        "pool_size": len(run),
        "ndcg_at_10": round(m.ndcg(ranking, qrels, 10), 4),
        "ndcg_at_10_isolated": round(m.ndcg(isolated_ranking, qrels, 10), 4),
        "bpref": round(m.bpref(ranking, qrels, rel_threshold=2), 4),
        "mrr": round(m.mrr(ranking, qrels, rel_threshold=1), 4),
        "p_at_5": round(m.precision_at_k(ranking, qrels, 5, rel_threshold=2), 4),
        "recall_at_20": round(m.recall_at_k(ranking, qrels, 20, rel_threshold=2), 4),
    }


def _print_table(results: list[dict]) -> None:
    cols = [
        ("persona", "persona_id", "s"),
        ("nDCG@10", "ndcg_at_10", "f"),
        ("nDCG@10 (iso)", "ndcg_at_10_isolated", "f"),
        ("BPref", "bpref", "f"),
        ("MRR", "mrr", "f"),
        ("P@5", "p_at_5", "f"),
        ("Recall@20", "recall_at_20", "f"),
        ("labeled", "num_labeled", "d"),
    ]
    widths = [max(len(label), 12) for label, _, _ in cols]
    header = "  ".join(label.rjust(w) for (label, _, _), w in zip(cols, widths, strict=True))
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        cells = []
        for (_, key, kind), w in zip(cols, widths, strict=True):
            val = r[key]
            cells.append((f"{val:.4f}" if kind == "f" else str(val)).rjust(w))
        print("  ".join(cells))
    avg_ndcg = round(sum(r["ndcg_at_10"] for r in results) / len(results), 4)
    avg_ndcg_isolated = round(sum(r["ndcg_at_10_isolated"] for r in results) / len(results), 4)
    avg_bpref = round(sum(r["bpref"] for r in results) / len(results), 4)
    print("-" * len(header))
    print(
        f"Average across {len(results)} personas: nDCG@10={avg_ndcg}  "
        f"nDCG@10 (iso)={avg_ndcg_isolated}  BPref={avg_bpref}"
    )


async def _run(out_dir: Path) -> None:
    async with async_session_factory() as session:
        results = []
        for persona_id in personas_lib.persona_ids():
            r = await _evaluate_persona(session, persona_id)
            if r is not None:
                results.append(r)

    if not results:
        print("No synthetic personas evaluated — run eval/inject_synthetic.py first.")
        return

    _print_table(results)

    avg_ndcg = round(sum(r["ndcg_at_10"] for r in results) / len(results), 4)
    avg_ndcg_isolated = round(sum(r["ndcg_at_10_isolated"] for r in results) / len(results), 4)
    avg_bpref = round(sum(r["bpref"] for r in results) / len(results), 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"synthetic-eval-{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "run_at": datetime.now(UTC).isoformat(),
                "git_sha": _git_sha(),
                "config_name": "hybrid_prod",
                "per_persona": results,
                "avg_ndcg_at_10": avg_ndcg,
                "avg_ndcg_at_10_isolated": avg_ndcg_isolated,
                "avg_bpref": avg_bpref,
            },
            indent=2,
        )
    )
    print(f"\nResult written to {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(description="Per-persona retrieval eval over the synthetic personas.")
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUT), help="Output directory")
    args = parser.parse_args()
    asyncio.run(_run(Path(args.out_dir)))


if __name__ == "__main__":
    main()
