import argparse
import asyncio
import logging
from pathlib import Path

from job_radar.db.base import async_session_factory
from job_radar.profile.loader import load_profile

logger = logging.getLogger(__name__)


async def _run(path: Path) -> None:
    async with async_session_factory() as session:
        profile = await load_profile(session, path)
    skills = (profile.domains_keywords or {}).get("tech_stack", [])
    print(f"Loaded profile: {profile.full_name or '(no name)'} — {profile.seniority}")
    print(f"  target titles: {', '.join(profile.target_titles) or '(none)'}")
    print(f"  skills ({len(skills)}): {', '.join(skills) or '(none)'}")
    print(f"  work history entries: {len(profile.work_history or [])}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Parse a CV file into the stored profile.")
    parser.add_argument("path", type=Path, help="path to a CV (.pdf, .txt, or .md)")
    args = parser.parse_args()
    try:
        asyncio.run(_run(args.path))
    except Exception:
        logger.exception("Failed to load profile from %s", args.path)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
