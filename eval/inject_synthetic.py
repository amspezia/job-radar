"""Inject the 5 synthetic personas + their 100 job postings into the live DB.

    uv run job-radar-eval-inject-synthetic [--teardown]

Upserts (ON CONFLICT DO UPDATE, keyed on deterministic UUIDs — see
personas_lib.py) a Profile row + up to 20 Job rows per persona, every Job row
tagged source="synthetic" so it's identifiable and excludable from production
ranking. Also writes an EvalLabel row per (persona, job) pair from each job's
construction-defined `grade`, labeled_by="synthetic-construction" — so the
existing eval/qrels.py::load_qrels and eval/run.py machinery work against
synthetic personas completely unmodified; see run_synthetic.py.

Idempotent: safe to re-run after editing a persona's fixture files — deterministic
IDs mean re-running updates the same rows rather than duplicating them.

--teardown removes everything this script ever inserted: EvalLabel and
FitJudgmentCache rows first, then Job, then Profile, respecting FK order.
"""

import argparse
import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from eval import personas_lib
from eval.label import upsert_label
from job_radar.adapters.embeddings import embed
from job_radar.db.base import async_session_factory
from job_radar.db.models import EvalLabel, FitJudgmentCache, Job, Profile

logger = logging.getLogger(__name__)

_SOURCE = "synthetic"
_LABELED_BY = "synthetic-construction"


async def _build_profile_row(persona: personas_lib.Persona) -> dict:
    gt = persona.ground_truth
    embedding = await embed(persona.cv_text, task="document")

    work_history = []
    for entry in gt["work_history"]:
        start = personas_lib.parse_mon_year(entry["start"])
        end = personas_lib.parse_mon_year(entry["end"]) if entry["end"] else personas_lib.today()
        years = (
            entry["years"] if entry["years"] is not None else personas_lib.years_between(start, end)
        )
        work_history.append(
            {
                "role": entry["role"],
                "company": entry["company"],
                "years": years,
                "start": entry["start"],
                "end": entry["end"],
                "highlights": [],
            }
        )

    career_start = personas_lib.parse_mon_year(gt["career_start"])
    years_experience = personas_lib.years_between(career_start, personas_lib.today())

    prefs = gt["search_preferences"]
    return {
        "id": personas_lib.profile_uuid(persona.persona_id),
        "source": _SOURCE,
        "full_name": gt["full_name"],
        "email": gt["email"],
        "links": {},
        "work_history": work_history,
        "cv_text": persona.cv_text,
        "cv_embedding": embedding,
        "target_titles": gt["target_titles"]["expected_any_of"],
        "seniority": gt["seniority"],
        "years_experience": years_experience,
        "domains_keywords": {
            "tech_stack": gt["tech_stack"],
            "domains": gt["domains"]["expected_any_of"],
        },
        "salary_floor": prefs["salary_floor"],
        "currency": prefs["currency"],
        "location_rules": prefs["location_rules"],
        "seniority_rules": None,
        "remote_required": prefs["remote_required"],
        # Left unset deliberately: build_hyde_embedding (fit/pipeline.py) populates
        # and caches this itself on the first real run against this profile.
        "dense_query_cache": None,
    }


async def _build_job_row(persona_id: str, index: int, posting: dict) -> dict:
    # Matches production's embed_text construction exactly (ingest/pipeline.py):
    # title + requirements + responsibilities, since these postings skip the
    # extract_fields step entirely — requirements/responsibilities are already
    # hand-constructed, not derived from a raw description.
    embed_text = f"{posting['title']}\n{posting['requirements']}\n{posting['responsibilities']}"
    embedding = await embed(embed_text, task="document")
    return {
        "id": personas_lib.job_uuid(persona_id, index),
        "source": _SOURCE,
        "source_type": _SOURCE,
        "ingested_via": "synthetic",
        "source_id": f"{persona_id}:{index}",
        "url": f"https://synthetic.job-radar.local/{persona_id}/{index}",
        "title": posting["title"],
        "company": posting["company"],
        "description": f"{posting['requirements']}\n\n{posting['responsibilities']}",
        "requirements": posting["requirements"],
        "responsibilities": posting["responsibilities"],
        "seniority": posting["seniority"],
        "salary_min": None,
        "salary_max": None,
        "currency": None,
        "location": "Remote",
        "remote": True,
        "job_type": None,
        "published_at": None,
        "collected_at": datetime.now(UTC),
        "embedding": embedding,
        "content_hash": f"synthetic:{persona_id}:{index}",
    }


async def _upsert_row(session: AsyncSession, model: type, row: dict) -> None:
    update_set = {k: v for k, v in row.items() if k != "id"}
    await session.execute(
        insert(model).values(**row).on_conflict_do_update(index_elements=["id"], set_=update_set)
    )


async def _inject_persona(session: AsyncSession, persona_id: str) -> int:
    persona = personas_lib.load_persona(persona_id)
    if not persona.synthetic_jobs:
        logger.warning("No synthetic jobs found for %s — skipping", persona_id)
        return 0

    profile_row = await _build_profile_row(persona)
    await _upsert_row(session, Profile, profile_row)

    job_rows = await asyncio.gather(
        *(_build_job_row(persona_id, i, p) for i, p in enumerate(persona.synthetic_jobs))
    )
    for row in job_rows:
        await _upsert_row(session, Job, row)
    await session.commit()

    for i, posting in enumerate(persona.synthetic_jobs):
        await upsert_label(
            session,
            profile_row["id"],
            job_rows[i]["id"],
            posting["grade"],
            _LABELED_BY,
            posting.get("construction_rationale"),
        )

    return len(job_rows)


async def _teardown(session: AsyncSession) -> None:
    ids = [personas_lib.profile_uuid(pid) for pid in personas_lib.persona_ids()]
    await session.execute(delete(EvalLabel).where(EvalLabel.profile_id.in_(ids)))
    await session.execute(delete(FitJudgmentCache).where(FitJudgmentCache.profile_id.in_(ids)))
    await session.execute(delete(Job).where(Job.source == _SOURCE))
    await session.execute(delete(Profile).where(Profile.id.in_(ids)))
    await session.commit()
    print(f"Removed synthetic data for {len(ids)} personas.")


async def _run(teardown: bool) -> None:
    async with async_session_factory() as session:
        if teardown:
            await _teardown(session)
            return

        total_jobs = 0
        persona_ids = personas_lib.persona_ids()
        for persona_id in persona_ids:
            n = await _inject_persona(session, persona_id)
            total_jobs += n
            print(f"  {persona_id}: {n} jobs injected")
        print(f"\nInjected {len(persona_ids)} personas, {total_jobs} jobs total.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Inject synthetic personas + jobs into the DB (idempotent)."
    )
    parser.add_argument(
        "--teardown", action="store_true", help="Remove all synthetic data instead of injecting"
    )
    args = parser.parse_args()
    asyncio.run(_run(args.teardown))


if __name__ == "__main__":
    main()
