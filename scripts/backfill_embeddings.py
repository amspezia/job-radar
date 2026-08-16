"""Re-embed all stored job postings and the CV profile with nomic task prefixes.

Run once after upgrading to the prefixed embed() call sites. Every stored vector
was produced without a prefix; vectors must all live in the same space, so
a half-applied backfill (some prefixed, some not) degrades retrieval quality.

Usage:
    uv run python scripts/backfill_embeddings.py [--dry-run] [--batch-size N]

The script is idempotent: re-running it re-embeds everything, which is safe
(same prefix, same model → same vectors). Use --dry-run to count rows first.
"""

import argparse
import asyncio
import logging

from sqlalchemy import select

from job_radar.adapters.embeddings import embed
from job_radar.db.base import async_session_factory
from job_radar.db.models import Job, Profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def backfill_jobs(batch_size: int, dry_run: bool) -> int:
    async with async_session_factory() as session:
        rows = (
            (await session.execute(select(Job).where(Job.embedding.is_not(None)))).scalars().all()
        )
        logger.info("Found %d jobs with embeddings to re-embed", len(rows))
        if dry_run:
            return len(rows)

        for i, job in enumerate(rows, 1):
            text = f"{job.title}\n{job.description}"
            job.embedding = await embed(text, task="document")
            if i % batch_size == 0:
                await session.commit()
                logger.info("  committed %d / %d jobs", i, len(rows))

        await session.commit()
        logger.info("Jobs re-embedded: %d total", len(rows))
        return len(rows)


async def backfill_profile(dry_run: bool) -> int:
    async with async_session_factory() as session:
        # Unscoped would risk grabbing a synthetic eval persona instead of the
        # real profile now that eval/inject_synthetic.py puts several rows in
        # this same table (see db/models.py's Profile.source docstring).
        profile = (
            (await session.execute(select(Profile).where(Profile.source == "real")))
            .scalars()
            .first()
        )
        if profile is None:
            logger.info("No profile found — skipping")
            return 0
        if profile.cv_embedding is None:
            logger.info("Profile has no embedding — skipping")
            return 0
        logger.info("Re-embedding CV profile")
        if dry_run:
            return 1
        profile.cv_embedding = await embed(profile.cv_text, task="document")
        profile.dense_query_cache = None  # force re-synthesis with the new vectors
        await session.commit()
        logger.info("Profile re-embedded")
        return 1


async def main(batch_size: int, dry_run: bool) -> None:
    if dry_run:
        logger.info("DRY RUN — no writes")
    n_jobs = await backfill_jobs(batch_size, dry_run)
    n_profile = await backfill_profile(dry_run)
    logger.info(
        "%s complete: %d jobs, %d profiles",
        "Dry run" if dry_run else "Backfill",
        n_jobs,
        n_profile,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Count rows without writing")
    parser.add_argument("--batch-size", type=int, default=50, help="Commit every N jobs")
    args = parser.parse_args()
    asyncio.run(main(batch_size=args.batch_size, dry_run=args.dry_run))
