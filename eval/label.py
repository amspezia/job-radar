"""LLM-seeded, human-confirmed labeling CLI.

Produces EVAL_LABEL rows (profile_id, job_id, grade 0-3) that become the
qrels ground truth for the eval harness.  The human is the authority; the
LLM's analyze_fit verdict is an optional pre-label that makes grading ~3-4x
faster by surfacing a suggested grade.

Usage
-----
    uv run job-radar-eval-label [options]

Options
-------
  --limit N          Cap how many jobs to show (default: 50)
  --skip-labeled     Skip jobs already labeled for this profile (default: on)
  --no-skip-labeled  Re-show already-labeled jobs
  --auto-prelabel    Run analyze_fit on each job and show the suggested grade
  --pool N           Retrieval pool size (default: 100)

Label grades
------------
  3 — Strong:   squarely the role searched for
  2 — Relevant: would belong on the results page
  1 — Marginal: adjacent domain / off-by-one level / one key skill missing
  0 — Not relevant: wrong stack/domain, or gated by region/seniority

Target: 50-80 graded pairs per profile, >= 15 at grade >= 2.
"""

import argparse
import asyncio
import logging
import sys
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.qrels import SearchConfig, build_run
from job_radar.db.base import async_session_factory
from job_radar.db.models import EvalLabel, Job, Profile
from job_radar.fit.analyze import analyze_fit
from job_radar.fit.pipeline import build_hyde_embedding, build_lexical_query

logger = logging.getLogger(__name__)

# Match OLLAMA_NUM_PARALLEL in justfile so we saturate the eval Ollama.
_MAX_CONCURRENT_ANALYSIS = 12

# Verdict from analyze_fit → suggested grade (seeding, not overriding human).
_VERDICT_GRADE: dict[str, int] = {
    "strong": 3,
    "moderate": 2,
    "weak": 1,
    "none": 0,
}


def _truncate(text: str | None, max_chars: int = 400) -> str:
    if not text:
        return "(none)"
    text = text.strip()
    return text[:max_chars] + "…" if len(text) > max_chars else text


def _show_job(job: Job, seed_grade: int | None, current_grade: int | None = None) -> None:
    print()
    print(f"  Title   : {job.title}")
    print(f"  Company : {job.company}")
    print(f"  Source  : {job.source}  |  Seniority: {job.seniority or 'unknown'}")
    if job.requirements:
        print(f"  Requires: {_truncate(job.requirements)}")
    if job.responsibilities:
        print(f"  Resp    : {_truncate(job.responsibilities)}")
    if not job.requirements and not job.responsibilities:
        print(f"  Desc    : {_truncate(job.description)}")
    if current_grade is not None:
        print(f"  Current grade: {current_grade}")
    if seed_grade is not None:
        print(f"  Suggested grade: {seed_grade}  (from analyze_fit — override freely)")


async def _already_labeled_ids(session: AsyncSession, profile_id: UUID) -> set[UUID]:
    rows = (
        await session.execute(
            select(EvalLabel.job_id).where(EvalLabel.profile_id == profile_id)
        )
    ).scalars().all()
    return set(rows)


async def _prompt_grade(job: Job, seed_grade: int | None) -> tuple[int, str] | None:
    """Prompt the human for a grade. Returns (grade, notes) or None to quit."""
    default_str = str(seed_grade) if seed_grade is not None else ""
    prompt = f"  Grade [0/1/2/3 | s=skip | q=quit]{f' [{default_str}]' if default_str else ''}: "
    while True:
        try:
            raw = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if raw == "q":
            return None
        if raw == "s":
            return "skip", ""  # type: ignore[return-value]
        if raw == "" and seed_grade is not None:
            raw = str(seed_grade)

        if raw in {"0", "1", "2", "3"}:
            notes_raw = input("  Notes (optional, Enter to skip): ").strip()
            return int(raw), notes_raw

        print("  Enter 0, 1, 2, 3, s, or q.")


async def _upsert_label(
    session: AsyncSession,
    profile_id: UUID,
    job_id: UUID,
    grade: int,
    labeled_by: str,
    notes: str | None = None,
) -> None:
    existing = (
        await session.execute(
            select(EvalLabel).where(
                EvalLabel.profile_id == profile_id,
                EvalLabel.job_id == job_id,
            )
        )
    ).scalars().first()

    if existing is not None:
        existing.label = str(grade)
        existing.labeled_by = labeled_by
        existing.notes = notes
    else:
        session.add(
            EvalLabel(
                profile_id=profile_id,
                job_id=job_id,
                label=str(grade),
                labeled_by=labeled_by,
                notes=notes,
            )
        )
    await session.commit()


async def _analyze_one(
    profile: Profile,
    job: Job,
    semaphore: asyncio.Semaphore,
) -> tuple[Job, int | None]:
    async with semaphore:
        try:
            assessment = await analyze_fit(profile, job)
            return job, _VERDICT_GRADE.get(assessment.verdict)
        except Exception:
            logger.debug("analyze_fit failed for job %s", job.id)
            return job, None


async def _label_jobs(
    session: AsyncSession,
    profile: Profile,
    jobs: list[Job],
    *,
    skip_labeled: bool,
    auto_prelabel: bool,
    fully_auto: bool,
    labeled_by: str,
) -> int:
    already_labeled = await _already_labeled_ids(session, profile.id)
    pending = [j for j in jobs if not (skip_labeled and j.id in already_labeled)]
    count = 0

    if fully_auto:
        print(
            f"Running analyze_fit on {len(pending)} jobs"
            f" (concurrency={_MAX_CONCURRENT_ANALYSIS})..."
        )
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ANALYSIS)
        tasks = [asyncio.ensure_future(_analyze_one(profile, j, semaphore)) for j in pending]
        for coro in asyncio.as_completed(tasks):
            job, grade = await coro
            if grade is None:
                print(f"  [skip] {job.title} @ {job.company} — analyze_fit failed")
                continue
            await _upsert_label(session, profile.id, job.id, grade, labeled_by)
            count += 1
            print(f"  [{count}/{len(pending)}] {job.title} @ {job.company} → grade {grade}")
        return count

    for job in pending:
        seed_grade: int | None = None
        if auto_prelabel:
            try:
                assessment = await analyze_fit(profile, job)
                seed_grade = _VERDICT_GRADE.get(assessment.verdict)
            except Exception:
                logger.debug("analyze_fit failed for job %s; no seed grade", job.id)

        _show_job(job, seed_grade)

        result = await _prompt_grade(job, seed_grade)
        if result is None:
            print("\nLabeling stopped.")
            break
        grade, notes = result
        if grade == "skip":
            print("  Skipped.")
            continue

        await _upsert_label(session, profile.id, job.id, grade, labeled_by, notes or None)
        count += 1
        print(f"  Saved grade {grade}.")

    return count


async def _review_labels(
    session: AsyncSession,
    profile: Profile,
    labeled_by: str,
) -> int:
    labels = (
        await session.execute(
            select(EvalLabel).where(EvalLabel.profile_id == profile.id)
        )
    ).scalars().all()

    if not labels:
        print("No labeled jobs found for this profile.")
        return 0

    job_ids = [lbl.job_id for lbl in labels]
    jobs_by_id: dict[UUID, Job] = {
        j.id: j
        for j in (
            await session.execute(select(Job).where(Job.id.in_(job_ids)))
        ).scalars().all()
    }
    grade_by_job: dict[UUID, int] = {
        lbl.job_id: int(lbl.label) for lbl in labels if lbl.label.isdigit()
    }

    print(
        f"\nReviewing {len(labels)} labeled jobs."
        " Enter to keep current grade, or type a new one.\n"
    )
    changed = 0
    for lbl in labels:
        job = jobs_by_id.get(lbl.job_id)
        if job is None:
            continue
        current = grade_by_job.get(job.id)
        _show_job(job, seed_grade=None, current_grade=current)

        prompt = f"  Grade [0/1/2/3 | s=skip | q=quit] [{current}]: "
        while True:
            try:
                raw = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                print(f"\nReview stopped. {changed} label(s) changed.")
                return changed

            if raw == "q":
                print(f"\nReview stopped. {changed} label(s) changed.")
                return changed
            if raw == "s" or raw == "":
                print("  Kept.")
                break
            if raw in {"0", "1", "2", "3"}:
                new_grade = int(raw)
                if new_grade != current:
                    await _upsert_label(session, profile.id, job.id, new_grade, labeled_by)
                    changed += 1
                    print(f"  Updated: {current} → {new_grade}")
                else:
                    print("  Kept (same grade).")
                break
            print("  Enter 0, 1, 2, 3, s, or q.")

    print(f"\nReview complete. {changed} label(s) changed.")
    return changed


async def _load_profile(session: AsyncSession) -> Profile | None:
    return (await session.execute(select(Profile))).scalars().first()


async def _run(args: argparse.Namespace) -> None:
    async with async_session_factory() as session:
        profile = await _load_profile(session)
        if profile is None:
            print("No profile found — run job-radar-profile first.")
            sys.exit(1)

        print(f"Profile loaded: {profile.full_name} ({profile.seniority})")

        if args.review:
            await _review_labels(session, profile, labeled_by=args.labeled_by)
            return

        lexical_q = build_lexical_query(profile)
        hyde_embedding = await build_hyde_embedding(profile, session)

        # Union pool from all three system configs so labels cover the full
        # relevant document space, not just what one config surfaces (pool bias).
        print(
            f"Building union pool from 3 retrieval configs (pool={args.pool} each) …"
        )
        pool_configs = [
            SearchConfig(arms=["lexical", "hyde"], pool=args.pool, limit=args.pool),
            SearchConfig(arms=["hyde"], pool=args.pool, limit=args.pool),
            SearchConfig(arms=["lexical"], pool=args.pool, limit=args.pool),
        ]
        pool_runs = await asyncio.gather(*[
            build_run(session, profile, cfg, query=lexical_q, hyde_embedding=hyde_embedding)
            for cfg in pool_configs
        ])
        run_scores: dict[UUID, float] = {}
        for pr in pool_runs:
            for jid, score in pr.items():
                if score > run_scores.get(jid, -1):
                    run_scores[jid] = score
        print(f"Union pool: {len(run_scores)} unique jobs (hybrid + hyde-only + keyword-only).")

        if not run_scores:
            print("No jobs retrieved — check that ingest has run and BM25/vector indexes exist.")
            sys.exit(1)

        # Fetch Job objects for the retrieved ids, preserving fused order.
        ranked_ids = sorted(run_scores, key=lambda jid: -run_scores[jid])
        jobs_by_id: dict[UUID, Job] = {
            j.id: j
            for j in (
                await session.execute(select(Job).where(Job.id.in_(ranked_ids)))
            ).scalars().all()
        }
        jobs = [jobs_by_id[jid] for jid in ranked_ids if jid in jobs_by_id]

        if args.limit:
            jobs = jobs[: args.limit]

        print(f"Pool: {len(jobs)} jobs (limit={args.limit or 'none'}).")

        already = await _already_labeled_ids(session, profile.id)
        unlabeled = [j for j in jobs if j.id not in already]
        print(
            f"Already labeled for this profile: {len(already)}  |  "
            f"Unlabeled in pool: {len(unlabeled)}"
        )
        if args.skip_labeled:
            print("Skipping already-labeled jobs (pass --no-skip-labeled to re-grade).")

        saved = await _label_jobs(
            session,
            profile,
            jobs,
            skip_labeled=args.skip_labeled,
            auto_prelabel=args.auto_prelabel,
            fully_auto=args.fully_auto,
            labeled_by=args.labeled_by,
        )
        print(f"\nDone. {saved} new label(s) saved.")
        total = len(await _already_labeled_ids(session, profile.id))
        print(f"Total labeled pairs for this profile: {total}")
        if total < 50:
            print(f"Target: 50-80 pairs (>=15 at grade >=2). Need {50 - total} more.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(
        description="Label job postings for eval harness (interactive, human-confirmed)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Max jobs to present")
    parser.add_argument("--pool", type=int, default=100, help="Retrieval pool size (default: 100)")
    parser.add_argument(
        "--skip-labeled",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Skip jobs already labeled for this profile (default: on)",
    )
    parser.add_argument(
        "--auto-prelabel",
        action="store_true",
        default=False,
        help="Run analyze_fit to suggest a grade for each job",
    )
    parser.add_argument(
        "--fully-auto",
        action="store_true",
        default=False,
        help="Save analyze_fit grades directly with no human prompt (implies --auto-prelabel)",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        default=False,
        help="Review and correct existing labels one by one (no AI, no pool retrieval)",
    )
    parser.add_argument(
        "--labeled-by",
        default="human",
        help="Who is doing the labeling (stored in EVAL_LABEL.labeled_by)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
