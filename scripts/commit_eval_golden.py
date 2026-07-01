"""Populate eval/golden/ from the latest eval run result and the live DB qrels.

Run after a stable labeling + eval cycle to activate the CI regression gate:

    uv run python scripts/commit_eval_golden.py [--run PATH]

The script:
  1. Reads qrels from the DB for the stored profile.
  2. Reads the best eval run JSON (latest by timestamp unless --run is given).
  3. Picks the "hybrid" config entry from that run (or the first if absent).
  4. Writes eval/golden/qrels.json and eval/golden/result.json.

Once these files are committed, test_eval_gate.py::test_ndcg_meets_golden_threshold
activates in CI and enforces nDCG@10 >= golden - 0.02 on every future change.
"""

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from eval.qrels import load_qrels
from job_radar.db.base import async_session_factory
from job_radar.db.models import Profile

_REPO_ROOT = Path(__file__).parent.parent
_GOLDEN_QRELS = _REPO_ROOT / "eval" / "golden" / "qrels.json"
_GOLDEN_RESULT = _REPO_ROOT / "eval" / "golden" / "result.json"
_RESULTS_DIR = _REPO_ROOT / "eval" / "results"


def _latest_eval_run() -> Path:
    runs = sorted(_RESULTS_DIR.glob("eval-*.json"))
    if not runs:
        raise FileNotFoundError(
            "No eval run files found in eval/results/. Run `just eval-run` first."
        )
    return runs[-1]


async def _main(run_path: Path) -> None:
    run_data = json.loads(run_path.read_text())

    configs: list[dict] = run_data.get("configurations", [])
    if not configs:
        raise ValueError(f"No configurations found in {run_path}")

    # Prefer the "hybrid" config; fall back to the first entry.
    best = next((c for c in configs if c["config_name"] == "hybrid"), configs[0])

    if best.get("ndcg_at_10") is None:
        raise ValueError(f"Selected config has no ndcg_at_10 in {run_path}")

    profile_id_str: str = run_data["profile_id"]

    async with async_session_factory() as session:
        from uuid import UUID

        profile = (
            await session.execute(
                select(Profile).where(Profile.id == UUID(profile_id_str))
            )
        ).scalars().first()
        if profile is None:
            raise RuntimeError(
                f"Profile {profile_id_str} not found in DB. "
                "Make sure the DB is seeded before committing goldens."
            )

        qrels = await load_qrels(session, profile.id)

    if not qrels:
        raise RuntimeError(
            "No labeled pairs found for this profile. "
            "Run `just eval-label` before committing goldens."
        )

    grade_dist = {}
    for g in qrels.values():
        grade_dist[g] = grade_dist.get(g, 0) + 1
    relevant = sum(n for g, n in grade_dist.items() if g >= 2)
    print(f"Qrels: {len(qrels)} pairs  grade distribution: {dict(sorted(grade_dist.items()))}")
    print(f"Relevant (grade>=2): {relevant}")

    if relevant < 15:
        raise RuntimeError(
            f"Only {relevant} grade>=2 pairs — target is >=15 for stable metrics. "
            "Label more jobs before committing goldens."
        )

    # Write golden qrels — job IDs + grades only, no job text or PII.
    qrels_out = {
        "_comment": (
            "Golden qrels — committed after labeling (M6 build step 7)."
            " Format: profile_id + list of {job_id, grade} pairs (grades 0-3)."
            " No job text or PII."
        ),
        "profile_id": profile_id_str,
        "labels": [
            {"job_id": str(jid), "grade": grade}
            for jid, grade in sorted(qrels.items(), key=lambda x: str(x[0]))
        ],
    }
    _GOLDEN_QRELS.write_text(json.dumps(qrels_out, indent=2) + "\n")
    print(f"Wrote {_GOLDEN_QRELS}  ({len(qrels)} labels)")

    # Write golden result — config + nDCG@10 only.
    result_out = {
        "_comment": (
            "Golden result — committed after sweep identifies the best config (M6 build step 7)."
            " The CI gate asserts nDCG@10 >= ndcg_at_10 - 0.02."
        ),
        "profile_id": profile_id_str,
        "run_at": run_data.get("run_at"),
        "git_sha": run_data.get("git_sha"),
        "config_name": best["config_name"],
        "config": best["config"],
        "ndcg_at_10": best["ndcg_at_10"],
        "bpref": best.get("bpref"),
        "mrr": best.get("mrr"),
        "p_at_5": best.get("p_at_5"),
        "p_at_10": best.get("p_at_10"),
        "recall_at_20": best.get("recall_at_20"),
        "recall_at_50": best.get("recall_at_50"),
    }
    _GOLDEN_RESULT.write_text(json.dumps(result_out, indent=2) + "\n")
    print(f"Wrote {_GOLDEN_RESULT}  (nDCG@10={best['ndcg_at_10']:.4f})")

    print(
        "\nGolden files written. CI regression gate is now active.\n"
        "Commit both files to enable test_ndcg_meets_golden_threshold in CI:\n"
        "  git add eval/golden/qrels.json eval/golden/result.json\n"
        "  git commit -m 'eval: commit golden baseline'"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Commit eval golden baseline files.")
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Eval run JSON to use (default: latest eval-*.json in eval/results/)",
    )
    args = parser.parse_args()
    run_path = args.run or _latest_eval_run()
    print(f"Using eval run: {run_path}")
    asyncio.run(_main(run_path))


if __name__ == "__main__":
    main()
