from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import EvalLabel, Job, Profile


async def test_job_round_trip(db_session: AsyncSession) -> None:
    job = Job(
        source="example-board",
        source_type="board",
        ingested_via="scheduler",
        url="https://example.com/jobs/123",
        title="Senior Backend Engineer",
        company="Example Corp",
        description="Build and operate backend services.",
        remote=True,
        collected_at=datetime.now(UTC),
        embedding=[0.0] * 768,
        content_hash="abc123",
    )
    db_session.add(job)
    await db_session.flush()

    result = await db_session.execute(select(Job).where(Job.id == job.id))
    fetched = result.scalar_one()
    assert fetched.title == "Senior Backend Engineer"
    assert fetched.company == "Example Corp"
    assert fetched.remote is True


async def test_profile_round_trip(db_session: AsyncSession) -> None:
    profile = Profile(
        full_name="Jane Doe",
        email="jane@example.com",
        links={"github": "https://github.com/jane"},
        work_history=[{"company": "Example Corp", "title": "Engineer"}],
        cv_text="Experienced backend engineer.",
        cv_embedding=[0.0] * 768,
        target_titles=["Backend Engineer", "Platform Engineer"],
        seniority="senior",
        domains_keywords=["python", "distributed systems"],
        location_rules={"remote_only": True},
        remote_required=True,
    )
    db_session.add(profile)
    await db_session.flush()

    result = await db_session.execute(select(Profile).where(Profile.id == profile.id))
    fetched = result.scalar_one()
    assert fetched.full_name == "Jane Doe"
    assert fetched.seniority == "senior"


async def test_eval_label_round_trip(db_session: AsyncSession) -> None:
    job = Job(
        source="example-board",
        source_type="board",
        ingested_via="scheduler",
        url="https://example.com/jobs/456",
        title="Platform Engineer",
        company="Example Corp",
        description="Own the platform.",
        remote=True,
        collected_at=datetime.now(UTC),
        embedding=[0.0] * 768,
        content_hash="def456",
    )
    db_session.add(job)
    await db_session.flush()

    label = EvalLabel(
        job_id=job.id,
        label="good",
        labeled_by="afonso",
        notes="Strong match on stack and seniority.",
    )
    db_session.add(label)
    await db_session.flush()

    result = await db_session.execute(select(EvalLabel).where(EvalLabel.job_id == job.id))
    fetched = result.scalar_one()
    assert fetched.label == "good"
    assert fetched.labeled_by == "afonso"
