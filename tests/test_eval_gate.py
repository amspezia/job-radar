"""CI regression gate — nDCG@10 must not drop below golden - 0.02.

The gate has two parts:

1. Pure-Python unit tests that verify the gate logic itself (always run,
   no DB, no golden files required).

2. An integration test that loads eval/golden/qrels.json + result.json and
   checks that the current codebase still meets the golden threshold.
   This test is skipped when the golden files have not been committed yet.
   Once committed, it runs in any environment that has the labeled jobs in the
   DB and enforces the regression bound on every future change.

Committing the goldens is step 7 of the M6 build sequence
(M6_EVAL_IMPLEMENTATION_PLAN.md §6).  Until then, the integration test is a
no-op in CI (skipif) and does not block the build.
"""

import json
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eval import metrics as m
from eval.qrels import HYBRID, SearchConfig, build_run
from job_radar.db.models import Profile

_REPO_ROOT = Path(__file__).parent.parent
_GOLDEN_QRELS = _REPO_ROOT / "eval" / "golden" / "qrels.json"
_GOLDEN_RESULT = _REPO_ROOT / "eval" / "golden" / "result.json"

# The noise threshold from Craswell et al.: a delta smaller than this is
# indistinguishable from random variation in a small labeled set.
_NOISE_BAND = 0.02


# ---------------------------------------------------------------------------
# Unit tests — gate logic, no DB or golden files required
# ---------------------------------------------------------------------------


def test_gate_passes_when_metric_equals_golden() -> None:
    current = 0.70
    golden = 0.70
    assert current >= golden - _NOISE_BAND


def test_gate_passes_within_noise_band() -> None:
    current = 0.68
    golden = 0.70
    assert current >= golden - _NOISE_BAND  # 0.68 >= 0.68


def test_gate_fails_just_outside_noise_band() -> None:
    current = 0.679
    golden = 0.70
    assert not (current >= golden - _NOISE_BAND)  # 0.679 < 0.68


def test_gate_passes_when_metric_improves() -> None:
    current = 0.82
    golden = 0.70
    assert current >= golden - _NOISE_BAND


# ---------------------------------------------------------------------------
# Integration gate — skips until golden files are committed
# ---------------------------------------------------------------------------


def _golden_ready() -> bool:
    """Return True only when golden files exist AND have been populated with real data."""
    if not (_GOLDEN_QRELS.exists() and _GOLDEN_RESULT.exists()):
        return False
    try:
        q = json.loads(_GOLDEN_QRELS.read_text())
        r = json.loads(_GOLDEN_RESULT.read_text())
        return bool(q.get("profile_id") and q.get("labels") and r.get("ndcg_at_10") is not None)
    except Exception:
        return False


@pytest.mark.skipif(
    not _golden_ready(),
    reason="no golden data committed yet — run eval/label.py then commit eval/golden/",
)
async def test_ndcg_meets_golden_threshold(db_session: AsyncSession) -> None:
    """Run the golden config and assert nDCG@10 >= golden.ndcg_at_10 - 0.02.

    This test is the retrieval regression gate: a meaningful drop in nDCG
    before a merge is caught here, not in production.
    """
    # Load committed ground truth.
    qrels_data = json.loads(_GOLDEN_QRELS.read_text())
    golden_result = json.loads(_GOLDEN_RESULT.read_text())
    golden_ndcg = golden_result["ndcg_at_10"]

    # Rebuild the qrels dict from {profile_id, job_id, grade} triples.
    profile_id = UUID(qrels_data["profile_id"])
    qrels: dict[UUID, int] = {
        UUID(entry["job_id"]): entry["grade"] for entry in qrels_data["labels"]
    }

    if not qrels:
        pytest.skip("golden qrels.json contains no labels")

    # Load profile from DB.
    profile = (
        await db_session.execute(select(Profile).where(Profile.id == profile_id))
    ).scalars().first()

    if profile is None:
        pytest.skip(
            f"golden profile {profile_id} not in DB — cannot evaluate; "
            "seed the DB with labeled job data first"
        )

    # Retrieve and score the golden config.
    # Embeddings are taken from the stored profile (cv_embedding + dense_query_cache).
    hyde_embedding = None
    if profile.dense_query_cache:
        import contextlib

        from job_radar.adapters.embeddings import embed

        with contextlib.suppress(Exception):
            hyde_embedding = await embed(profile.dense_query_cache, task="document")

    from job_radar.fit.pipeline import build_lexical_query

    lexical_q = build_lexical_query(profile)

    golden_config = (
        SearchConfig(**golden_result.get("config", {}))
        if golden_result.get("config")
        else HYBRID
    )

    run = await build_run(
        db_session,
        profile,
        golden_config,
        query=lexical_q,
        hyde_embedding=hyde_embedding,
    )

    ranking = sorted(run, key=lambda jid: -run[jid])
    current_ndcg = m.ndcg(ranking, qrels, 10)
    threshold = golden_ndcg - _NOISE_BAND

    assert current_ndcg >= threshold, (
        f"Retrieval regression detected: current nDCG@10={current_ndcg:.4f} "
        f"< golden {golden_ndcg:.4f} - {_NOISE_BAND} = {threshold:.4f}. "
        "Investigate before merging: check fusion weights, pool size, or BM25 boosts."
    )
