import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.adapters.embeddings import embed
from job_radar.db.base import async_session_factory
from job_radar.db.models import Job
from job_radar.quality import metrics, relevance

_JSON_DIR = Path("data/quality")

# (row label, dict key, format kind) for the console table, top to bottom.
_DISPLAY: list[tuple[str, str, str]] = [
    ("count", "count", "int"),
    ("% of corpus", "pct_of_corpus", "pct"),
    ("published null %", "published_null_pct", "pct"),
    ("published future %", "published_future_pct", "pct"),
    ("salary %", "salary_pct", "pct"),
    ("currency %", "currency_pct", "pct"),
    ("location %", "location_pct", "pct"),
    ("job_type %", "job_type_pct", "pct"),
    ("desc median len", "desc_median_len", "num"),
    ("desc <200 chars %", "desc_short_pct", "pct"),
    ("desc residual-HTML %", "desc_html_pct", "pct"),
    ("salary min>max %", "salary_invalid_pct", "pct"),
    ("dev title % (keyword)", "dev_title_pct", "pct"),
    ("relevance mean", "relevance_mean", "sim"),
    ("dev embed % (vector)", "dev_embed_pct", "pct"),
]


async def _load_rows(session: AsyncSession) -> list[metrics.JobRow]:
    stmt = select(
        Job.source,
        Job.title,
        Job.description,
        Job.salary_min,
        Job.salary_max,
        Job.currency,
        Job.location,
        Job.job_type,
        Job.published_at,
    )
    return [metrics.JobRow(*row) for row in (await session.execute(stmt)).all()]


def _overall_relevance(
    quals: list[metrics.SourceQuality], rel: dict[str, relevance.SourceRelevance]
) -> relevance.SourceRelevance | None:
    pairs = [(q.count, rel[q.source]) for q in quals if q.source != "ALL" and q.source in rel]
    total = sum(count for count, _ in pairs)
    if not total:
        return None
    mean = sum(count * r.relevance_mean for count, r in pairs) / total
    pct = sum(count * r.dev_embed_pct for count, r in pairs) / total
    return relevance.SourceRelevance(round(mean, 4), round(pct, 1))


def _merge(
    quals: list[metrics.SourceQuality], rel: dict[str, relevance.SourceRelevance]
) -> list[dict]:
    overall = _overall_relevance(quals, rel)
    columns: list[dict] = []
    for q in quals:
        merged = metrics.as_dict(q)
        r = overall if q.source == "ALL" else rel.get(q.source)
        merged["relevance_mean"] = r.relevance_mean if r else None
        merged["dev_embed_pct"] = r.dev_embed_pct if r else None
        columns.append(merged)
    return columns


def _fmt(kind: str, value: object) -> str:
    if value is None:
        return "—"
    if kind == "int":
        return str(value)
    if kind == "sim":
        return f"{value:.3f}"
    if kind == "num":
        return f"{value:.0f}"
    return f"{value:.1f}"  # pct


def render_table(columns: list[dict]) -> str:
    label_w = max(len(label) for label, _, _ in _DISPLAY)
    headers = [c["source"] for c in columns]
    col_w = max(10, *(len(h) for h in headers))
    head = " " * label_w + "  " + "  ".join(h.rjust(col_w) for h in headers)
    lines = [head, "-" * len(head)]
    for label, key, kind in _DISPLAY:
        cells = [_fmt(kind, c.get(key)).rjust(col_w) for c in columns]
        lines.append(label.ljust(label_w) + "  " + "  ".join(cells))
    return "\n".join(lines)


def _write_json(columns: list[dict], *, threshold: float, generated_at: datetime) -> Path:
    report = {
        "generated_at": generated_at.isoformat(),
        "relevance_threshold": threshold,
        "relevance_anchors": relevance.ANCHORS,
        "total_jobs": next((c["count"] for c in columns if c["source"] == "ALL"), 0),
        "overall": next((c for c in columns if c["source"] == "ALL"), None),
        "sources": [c for c in columns if c["source"] != "ALL"],
    }
    _JSON_DIR.mkdir(parents=True, exist_ok=True)
    path = _JSON_DIR / f"quality-{generated_at:%Y%m%dT%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


async def _run(threshold: float) -> None:
    now = datetime.now(UTC)
    async with async_session_factory() as session:
        rows = await _load_rows(session)
        if not rows:
            print("No jobs in the database — run job-radar-ingest first.")
            return
        quals = metrics.compute(rows, now=now)
        centroid = await relevance.build_centroid(embed)
        rel = await relevance.relevance_by_source(session, centroid, threshold)

    columns = _merge(quals, rel)
    print(render_table(columns))
    path = _write_json(columns, threshold=threshold, generated_at=now)
    print(f"\nSnapshot written to {path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Assess ingested job data quality per source.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=relevance.DEFAULT_THRESHOLD,
        help="cosine similarity above which a posting counts as on-scope (default: %(default)s)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.threshold))


if __name__ == "__main__":
    main()
