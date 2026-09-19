"""Tier A: pure assembly and CV scrubbing, plus the read-only fetch against a prod-shaped DB.

No real production database and no Ollama: the fetch runs against a second throwaway database
created with the prod schema (`job_radar_evals_test_*`, dropped afterwards).
"""

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from eval.embedding.models import EmbeddingLabel, EmbeddingTopic
from eval.embedding.tiers.tier_a import (
    ORIGIN,
    PoolJob,
    ProfileData,
    TierABuilder,
    assemble,
    import_topic,
    scrub_cv_text,
)
from eval.evals_db import admin
from eval.evals_db.prod_reader import alembic_revision, prod_session, table_counts
from job_radar.db.base import Base
from job_radar.db.models import EvalLabel, Job, Profile
from job_radar.ingest.embed_text import build_embed_text

NAME = "Zelda Quimby Farnsworth"
EMAIL = "zelda.farnsworth@example.com"
PHONE = "+55 (11) 91234-5678"
FAKE_CV = f"""{NAME}
Senior Backend Engineer
Email: {EMAIL} | Phone: {PHONE}
LinkedIn: https://www.linkedin.com/in/zelda-farnsworth | github.com/zquimby | zelda.dev

Experience
Acme Corp, 2019-2023: cut infrastructure cost by 40% and latency by 250%.
Globex, 05.2020 - 02.2024: built Node.js and ASP.NET services on Python 3.12.
{NAME.split()[0]} led a team of 8 (ISO date 2023-01-15).
"""
PII = [
    NAME,
    "Zelda",
    "Quimby",
    "Farnsworth",
    EMAIL,
    "91234",
    "linkedin.com",
    "zquimby",
    "zelda.dev",
]


def make_profile(**overrides) -> ProfileData:
    fields = {
        "id": uuid4(),
        "full_name": NAME,
        "cv_text": FAKE_CV,
        "target_titles": ["Backend Engineer"],
        "seniority": "senior",
        "years_experience": 6.5,
        "domains_keywords": {"tech_stack": ["python", "sql"], "domains": ["fintech"]},
        "work_history": [
            {"role": "Engineer", "company": "Acme", "years": 2.0, "start": "2019", "highlights": []}
        ],
        "location_rules": {"allowed_keywords": ["brazil"]},
        "remote_required": True,
        "salary_floor": 90000,
        "currency": "USD",
        "dense_query_cache": json.dumps(["hyde one", "hyde two", "hyde three"]),
    }
    return ProfileData(**{**fields, **overrides})


def make_pool_job(
    requirements: str | None = "reqs", responsibilities: str | None = None
) -> PoolJob:
    return PoolJob(
        id=uuid4(),
        source="greenhouse",
        title="Backend Engineer",
        company="Acme",
        url=f"https://example.com/{uuid4()}",
        location="Remote",
        remote=True,
        seniority="senior",
        description="Full description",
        requirements=requirements,
        responsibilities=responsibilities,
        content_hash="hash",
        embedding=[0.25] * 4,
    )


# --- assemble -------------------------------------------------------------------------------


def test_assemble_rebuilds_the_production_embed_text_and_its_sha():
    with_reqs = make_pool_job(requirements="reqs", responsibilities="resp")
    fallback = make_pool_job(requirements=None, responsibilities=None)

    payload, stats = assemble("t", make_profile(), [with_reqs, fallback])

    for record, job in zip(payload.jobs, [with_reqs, fallback], strict=True):
        expected = build_embed_text(
            job.title, job.description, job.requirements, job.responsibilities
        )
        assert record.embed_text == expected
        assert record.embed_text_sha == hashlib.sha256(expected.encode()).hexdigest()
        assert (record.origin, record.origin_id) == (ORIGIN, str(job.id))
        assert record.prod_embedding == job.embedding
    assert payload.jobs[1].embed_text == "Backend Engineer\nFull description"
    assert (payload.tier, payload.name, payload.builder) == ("A", "t", "tier_a")
    assert stats.pool_size == 2


def test_assemble_takes_the_frozen_hyde_texts_from_the_cache():
    payload, stats = assemble("t", make_profile(), [make_pool_job()])
    assert payload.query_inputs == {"hyde_texts": ["hyde one", "hyde two", "hyde three"]}
    assert stats.hyde_count == 3


@pytest.mark.parametrize("cache", [None, "", "not json", "{}", "[]", '["", 3]'])
def test_assemble_requires_hyde_texts_and_says_how_to_get_them(cache):
    with pytest.raises(ValueError, match="job-radar-fit"):
        assemble("t", make_profile(dense_query_cache=cache), [make_pool_job()])


def test_assemble_carries_no_labels_and_no_label_stats():
    payload, stats = assemble("t", make_profile(), [make_pool_job(), make_pool_job()])

    assert payload.labels == []
    assert set(vars(stats)) == {"pool_size", "hyde_count"}


def test_snapshot_has_exactly_the_judge_brief_keys_and_no_pii_anywhere():
    payload, _ = assemble("t", make_profile(), [make_pool_job()])
    snapshot = payload.profile_snapshot

    assert set(snapshot) == {
        "target_titles",
        "seniority",
        "years_experience",
        "tech_stack",
        "domains",
        "location_rules",
        "remote_required",
        "salary_floor",
        "currency",
        "work_history",
        "cv_text",
    }
    assert snapshot["tech_stack"] == ["python", "sql"]
    assert snapshot["domains"] == ["fintech"]
    assert snapshot["work_history"] == [{"role": "Engineer", "years": 2.0}]

    everything = json.dumps(payload.profile_snapshot) + json.dumps(payload.query_inputs)
    for secret in PII:
        assert secret.lower() not in everything.lower()
    for kept in ("2019-2023", "05.2020 - 02.2024", "40%", "250%", "2023-01-15", "Python 3.12"):
        assert kept in snapshot["cv_text"]
    for placeholder in ("[name]", "[email]", "[phone]", "[url]"):
        assert placeholder in snapshot["cv_text"]


# --- scrub_cv_text ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "scrubbed"),
    [
        ("mail a.b+c@sub.example.co.uk.", "mail [email]."),
        ("call +1 415 555 2671 now", "call [phone] now"),
        ("call (415) 555-2671", "call [phone]"),
        ("call 11912345678", "call [phone]"),
        ("see https://github.com/foo/bar.", "see [url]."),
        ("see www.example.org/x, ok", "see [url], ok"),
        ("see linkedin.com/in/foo", "see [url]"),
        ("see portfolio.dev", "see [url]"),
    ],
)
def test_scrub_removes_each_kind_of_pii(raw, scrubbed):
    assert scrub_cv_text(raw, None) == scrubbed


@pytest.mark.parametrize(
    "harmless",
    [
        "2019-2023",
        "2019 - 2023",
        "05.2020 - 02.2024",
        "2023-01-15",
        "cut cost by 40%",
        "up 250% since 2020",
        "Python 3.12.4",
        "ticket 1234567",
        "team of 12345",
        "Node.js, Next.js/React, ASP.NET, Socket.io",
        "2019\n2020\n2021\n2022",
    ],
)
def test_scrub_leaves_years_percentages_and_tech_names_alone(harmless):
    assert scrub_cv_text(harmless, "Some Body") == harmless


def test_scrub_removes_every_name_token_case_insensitively_as_one_placeholder():
    text = "ZELDA QUIMBY FARNSWORTH\nSenior at Zelda's shop; Quimby-Farnsworth Co, Z. Farnsworth"
    scrubbed = scrub_cv_text(text, "Zelda Quimby Farnsworth")
    assert scrubbed == "[name]\nSenior at [name]'s shop; [name]-[name] Co, Z. [name]"


def test_scrub_ignores_single_letter_name_tokens_and_a_missing_name():
    assert scrub_cv_text("A B C", "A B") == "A B C"
    assert scrub_cv_text("keep Zelda", None) == "keep Zelda"
    assert scrub_cv_text("keep Zelda", "") == "keep Zelda"


def test_scrub_removes_the_email_before_the_name_so_no_fragment_is_left_behind():
    scrubbed = scrub_cv_text(f"Contact {EMAIL} ({NAME})", NAME)
    assert scrubbed == "Contact [email] ([name])"


# --- fetch, against a prod-shaped throwaway database ---------------------------------------


@pytest_asyncio.fixture
async def prod_url(evals_db_url):
    """A second throwaway database with the production schema (same server as the EVALS one)."""
    url = admin.with_database(evals_db_url, f"job_radar_evals_test_{uuid4().hex[:8]}")
    admin.ensure_database(url)
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        yield url, engine
    finally:
        await engine.dispose()
        admin.drop_database(url)


def _job(embedding, *, remote=True, seniority=None) -> Job:
    tag = uuid4().hex
    return Job(
        id=uuid4(),
        source="greenhouse",
        source_type="ats",
        ingested_via="test",
        url=f"https://example.com/{tag}",
        title=f"Engineer {tag[:6]}",
        company="Acme",
        description="desc",
        requirements="reqs",
        seniority=seniority,
        remote=remote,
        collected_at=datetime.now(UTC),
        embedding=embedding,
        content_hash=tag,
    )


def _profile(source: str, *, remote_required: bool, cache: str | None) -> Profile:
    return Profile(
        id=uuid4(),
        source=source,
        full_name=NAME,
        email=EMAIL,
        links={"linkedin": "https://linkedin.com/in/x"},
        work_history=[{"role": "Engineer", "years": 2.0}],
        cv_text=FAKE_CV,
        target_titles=["Backend Engineer"],
        seniority="senior",
        years_experience=6.5,
        domains_keywords={"tech_stack": ["python"], "domains": []},
        location_rules={},
        remote_required=remote_required,
        dense_query_cache=cache,
    )


@pytest_asyncio.fixture
async def seeded(prod_url):
    url, engine = prod_url
    vec = [0.5] * 768
    in_a, in_b, in_c = _job(vec), _job(vec, seniority="senior"), _job(vec)
    onsite = _job(vec, remote=False)
    no_embedding = _job(None)
    real = _profile("real", remote_required=True, cache=json.dumps(["h1", "h2"]))
    synthetic = _profile("synthetic", remote_required=False, cache=json.dumps(["s1"]))
    labels = [
        EvalLabel(profile_id=real.id, job_id=in_a.id, label="strong", labeled_by="me"),
        EvalLabel(profile_id=real.id, job_id=in_b.id, label="weak", labeled_by="me"),
        EvalLabel(profile_id=real.id, job_id=in_c.id, label="bogus", labeled_by="me"),
        EvalLabel(profile_id=real.id, job_id=onsite.id, label="relevant", labeled_by="me"),
        EvalLabel(profile_id=real.id, job_id=no_embedding.id, label="none", labeled_by="me"),
        EvalLabel(profile_id=synthetic.id, job_id=in_a.id, label="moderate", labeled_by="x"),
    ]
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        session.add_all([in_a, in_b, in_c, onsite, no_embedding, real, synthetic])
        await session.flush()
        session.add_all(labels)
        await session.commit()
    pool = sorted([in_a.id, in_b.id, in_c.id])
    return {"url": url, "pool": pool, "in_a": in_a.id, "in_b": in_b.id, "synthetic": synthetic.id}


async def test_build_reads_the_filtered_pool_and_the_real_profile_only(seeded):
    payload = await TierABuilder(seeded["url"]).build("real-topic")

    assert [j.origin_id for j in payload.jobs] == [str(i) for i in seeded["pool"]]
    assert all(j.prod_embedding == [0.5] * 768 for j in payload.jobs)
    assert payload.query_inputs == {"hyde_texts": ["h1", "h2"]}
    assert payload.labels == []  # prod's eval_labels are LLM-made and never imported
    assert NAME.split()[0] not in json.dumps(payload.profile_snapshot)


async def test_build_for_an_explicit_profile_uses_that_profiles_filter(seeded):
    payload = await TierABuilder(seeded["url"]).build("synthetic", profile_id=seeded["synthetic"])

    assert len(payload.jobs) == 4  # remote_required is off, so the onsite job is in the pool
    assert payload.labels == []
    assert payload.query_inputs == {"hyde_texts": ["s1"]}


async def test_the_fetch_cannot_write_and_leaves_prod_untouched(seeded):
    async with prod_session(seeded["url"]) as session:
        before = (await table_counts(session), await alembic_revision(session))
    await TierABuilder(seeded["url"]).build("t")
    async with prod_session(seeded["url"]) as session:
        assert (await table_counts(session), await alembic_revision(session)) == before
    assert (before[0]["jobs"], before[0]["profile"], before[0]["eval_labels"]) == (5, 2, 6)


async def test_import_topic_persists_the_topic_and_reports(
    seeded, evals_session, evals_db_url, capsys
):
    summary = await import_topic("e2e", "A", prod_url=seeded["url"], evals_url=evals_db_url)

    topic = await evals_session.get(EmbeddingTopic, summary.topic_id)
    assert (topic.status, topic.tier) == ("frozen", "A")
    assert (summary.n_jobs_new, summary.n_labels) == (3, 0)
    assert await evals_session.scalar(select(func.count()).select_from(EmbeddingLabel)) == 0
    out = capsys.readouterr().out
    assert "pool size:        3" in out
    assert "HyDE texts:       2" in out
    assert "label" not in out.replace("'eval_labels'", "")
    assert "'jobs': 5" in out and "'eval_labels': 6" in out


async def test_import_topic_rejects_other_tiers():
    with pytest.raises(ValueError, match="tier"):
        await import_topic("x", "B")
