"""Backup and restore of the labels a human made (`human_blind`, `constructed`): JSON on disk,
no posting text. The rationale (the labeling bucket and any note) travels with each grade.

Human labels are the expensive, irreplaceable ones (`llm_judge` rows can be regenerated), so
they can be exported and later re-attached to a rebuilt topic. A label is keyed by the posting
it was made against, `origin_id` plus `embed_text_sha`, not by an id local to one database.
"""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.labels import HUMAN_SOURCES
from eval.embedding.models import EmbeddingJob, EmbeddingLabel, EmbeddingTopic, EmbeddingTopicJob


async def _topic_id(session: AsyncSession, topic_name: str):
    topic_id = await session.scalar(
        select(EmbeddingTopic.id).where(EmbeddingTopic.name == topic_name)
    )
    if topic_id is None:
        raise LookupError(f"no topic named {topic_name!r}")
    return topic_id


async def export_labels(session: AsyncSession, topic_name: str, path: str | Path) -> int:
    """Write the topic's human-source labels, oldest first, and return how many."""
    topic_id = await _topic_id(session, topic_name)
    rows = await session.execute(
        select(
            EmbeddingJob.origin_id,
            EmbeddingJob.embed_text_sha,
            EmbeddingLabel.grade,
            EmbeddingLabel.source,
            EmbeddingLabel.rationale,
            EmbeddingLabel.created_at,
        )
        .join(EmbeddingJob, EmbeddingJob.id == EmbeddingLabel.job_id)
        .where(EmbeddingLabel.topic_id == topic_id, EmbeddingLabel.source.in_(HUMAN_SOURCES))
        .order_by(EmbeddingLabel.id)
    )
    records = [
        {
            "origin_id": origin_id,
            "embed_text_sha": sha,
            "grade": grade,
            "source": source,
            "rationale": rationale,
            "created_at": created_at.isoformat(),
        }
        for origin_id, sha, grade, source, rationale, created_at in rows
    ]
    Path(path).write_text(json.dumps(records, indent=2) + "\n")
    return len(records)


async def import_labels(session: AsyncSession, topic_name: str, path: str | Path) -> int:
    """Re-insert exported labels into the topic and return how many were added.

    A record is matched to a topic member by (origin_id, embed_text_sha); one that matches
    none is skipped. Records already present (same job, grade, source and time) are skipped
    too, so importing the same file twice adds nothing.
    """
    topic_id = await _topic_id(session, topic_name)
    records = json.loads(Path(path).read_text())
    for record in records:
        if record["source"] not in HUMAN_SOURCES:
            raise ValueError(f"cannot import a {record['source']!r} label: not a human source")
        if not 0 <= record["grade"] <= 3:
            raise ValueError(f"grade {record['grade']} for {record['origin_id']!r} is outside 0..3")

    members = {
        (origin_id, sha): job_id
        for job_id, origin_id, sha in await session.execute(
            select(EmbeddingJob.id, EmbeddingJob.origin_id, EmbeddingJob.embed_text_sha)
            .join(EmbeddingTopicJob, EmbeddingTopicJob.job_id == EmbeddingJob.id)
            .where(EmbeddingTopicJob.topic_id == topic_id)
        )
    }
    seen = {
        tuple(row)
        for row in await session.execute(
            select(
                EmbeddingLabel.job_id,
                EmbeddingLabel.grade,
                EmbeddingLabel.source,
                EmbeddingLabel.created_at,
            ).where(EmbeddingLabel.topic_id == topic_id)
        )
    }

    added = 0
    for record in records:
        job_id = members.get((record["origin_id"], record["embed_text_sha"]))
        if job_id is None:
            continue
        created_at = datetime.fromisoformat(record["created_at"])
        key = (job_id, record["grade"], record["source"], created_at)
        if key in seen:
            continue
        seen.add(key)
        session.add(
            EmbeddingLabel(
                topic_id=topic_id,
                job_id=job_id,
                grade=record["grade"],
                source=record["source"],
                rationale=record.get("rationale"),
                created_at=created_at,
            )
        )
        added += 1
    await session.commit()
    return added
