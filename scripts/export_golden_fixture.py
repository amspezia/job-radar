"""Export the CI golden-gate fixture (profile + labeled jobs) from the live DB.

    uv run python scripts/export_golden_fixture.py

`tests/test_eval_gate.py::test_ndcg_meets_golden_threshold` needs the exact Profile
and Job rows `eval/golden/qrels.json` labels to exist in whatever DB it runs
against — which CI's ephemeral, freshly-migrated Postgres never has. This script
snapshots them into `eval/golden/fixture.json`, which a pytest fixture loads into
the test DB before the gate runs (see `_load_golden_fixture` in test_eval_gate.py).

Re-run this whenever `eval/golden/qrels.json` changes (different labeled job set)
or whenever the live profile's search-criteria fields change in a way that should
be reflected in the golden baseline. It does NOT regenerate qrels.json or
result.json — those are `scripts/commit_eval_golden.py`'s job, and should be
regenerated together with this script when the intent is to establish a new
baseline, so the three files describe one coherent, reproducible run.

Privacy (CLAUDE.md: "Profile data is runtime data in Postgres, never in git"):
job postings are already public data and exported verbatim, but the Profile row
is redacted before writing. `full_name`, `email`, `cv_text`, `work_history`, and
`cv_embedding` never leave this script. `target_titles` / `domains_keywords` /
`seniority` / `location_rules` are kept, because they're the search-criteria
fields the golden labels' relevance judgments are actually about — the same
category of data `docs/plans/SYNTHETIC_EVAL_DESIGN.md` already treats as safe,
non-PII fixture content for its synthetic personas. `dense_query_cache` is kept
too: it holds LLM-synthesized, employer-voice posting text derived from those
criteria fields, not copied from the CV.
"""

import asyncio
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.base import async_session_factory
from job_radar.db.models import Job, Profile
from job_radar.fit.pipeline import build_hyde_embedding, build_lexical_query

_GOLDEN_DIR = Path(__file__).parent.parent / "eval" / "golden"
_QRELS_PATH = _GOLDEN_DIR / "qrels.json"
_FIXTURE_PATH = _GOLDEN_DIR / "fixture.json"


def _job_to_dict(job: Job) -> dict:
    return {
        "id": str(job.id),
        "source": job.source,
        "source_type": job.source_type,
        "ingested_via": job.ingested_via,
        "source_id": job.source_id,
        "url": job.url,
        "title": job.title,
        "company": job.company,
        "description": job.description,
        "requirements": job.requirements,
        "responsibilities": job.responsibilities,
        "seniority": job.seniority,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "currency": job.currency,
        "location": job.location,
        "remote": job.remote,
        "job_type": job.job_type,
        "published_at": job.published_at.isoformat() if job.published_at else None,
        "collected_at": job.collected_at.isoformat(),
        "embedding": [float(x) for x in job.embedding] if job.embedding is not None else None,
        "content_hash": job.content_hash,
    }


def _profile_to_dict(profile: Profile) -> dict:
    """Redacted per the module docstring — PII fields never leave this function."""
    return {
        "id": str(profile.id),
        "full_name": "Golden Fixture Profile",
        "email": "golden-fixture@example.invalid",
        "links": {},
        "work_history": [],
        "cv_text": "(redacted for the golden fixture — unused by retrieval-only scoring)",
        "cv_embedding": None,
        "target_titles": profile.target_titles,
        "seniority": profile.seniority,
        "years_experience": profile.years_experience,
        "domains_keywords": profile.domains_keywords,
        "salary_floor": profile.salary_floor,
        "currency": profile.currency,
        "location_rules": profile.location_rules,
        "seniority_rules": profile.seniority_rules,
        "remote_required": profile.remote_required,
        "dense_query_cache": profile.dense_query_cache,
    }


async def _export(session: AsyncSession) -> None:
    qrels_data = json.loads(_QRELS_PATH.read_text())
    profile_id: str = qrels_data["profile_id"]
    job_ids: list[str] = [entry["job_id"] for entry in qrels_data["labels"]]

    profile = (
        (await session.execute(select(Profile).where(Profile.id == profile_id))).scalars().first()
    )
    if profile is None:
        raise SystemExit(f"profile {profile_id} not found — cannot export golden fixture")

    jobs = (await session.execute(select(Job).where(Job.id.in_(job_ids)))).scalars().all()
    found_ids = {str(j.id) for j in jobs}
    missing = [jid for jid in job_ids if jid not in found_ids]
    if missing:
        raise SystemExit(
            f"{len(missing)} labeled job(s) missing from the DB, cannot export a complete "
            f"fixture: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    # Populate/reuse the HyDE cache so fixture.json's dense_query_cache matches
    # whatever result.json's nDCG was measured against, if regenerated alongside it.
    build_lexical_query(profile)
    await build_hyde_embedding(profile, session)

    fixture = {
        "_comment": (
            "Redacted profile + labeled job snapshot for the CI golden gate. "
            "Regenerate with scripts/export_golden_fixture.py — see that file's "
            "docstring for what is and isn't safe to export."
        ),
        "profile": _profile_to_dict(profile),
        "jobs": [_job_to_dict(j) for j in jobs],
    }
    _FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Wrote {_FIXTURE_PATH} ({len(jobs)} jobs, profile {profile_id})")


async def main() -> None:
    async with async_session_factory() as session:
        await _export(session)


if __name__ == "__main__":
    asyncio.run(main())
