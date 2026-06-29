import argparse
import asyncio
import logging

from job_radar.db.base import async_session_factory
from job_radar.fit.pipeline import run_fit_pipeline

logger = logging.getLogger(__name__)


async def _run(query: str | None, limit: int) -> None:
    async with async_session_factory() as session:
        results = await run_fit_pipeline(session, query, limit=limit)

    if not results:
        print("No matching jobs found.")
        return

    for job, assessment in results:
        score = assessment.score if assessment.score is not None else "—"
        gate = " (gate failed)" if assessment.gate_failed else ""
        print(f"[{score:>3}] {assessment.verdict:<8}{gate}  {job.title} @ {job.company}")
        print(f"        {job.url}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Search jobs for the stored profile and rank them by fit."
    )
    parser.add_argument(
        "query", nargs="?", default=None, help="search query (default: profile target_titles)"
    )
    parser.add_argument("--limit", type=int, default=20, help="max jobs to retrieve and score")
    args = parser.parse_args()
    try:
        asyncio.run(_run(args.query, args.limit))
    except Exception:
        logger.exception("Fit pipeline failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
