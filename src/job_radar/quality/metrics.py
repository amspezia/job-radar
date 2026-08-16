import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

# Residual markup left behind when an HTML strip failed: a real tag, a named
# entity (&amp;), or a numeric entity (&#39;). Its presence in a stored
# description is a direct signal of a parsing bug in an adapter.
_HTML_MARKER_RE = re.compile(r"<[a-z/][^>]*>|&[a-z]+;|&#\d+;", re.IGNORECASE)

# A description shorter than this is almost certainly a truncated/failed parse
# rather than a genuinely terse posting.
_SHORT_DESC_CHARS = 200

# Keyword cross-check for developer/engineering scope. Deliberately broad — it
# is the cheap second opinion alongside the embedding score, not the verdict.
_DEV_TITLE_RE = re.compile(
    r"\b(engineer|engineering|developer|programmer|software|backend|back-end|"
    r"frontend|front-end|full[- ]?stack|devops|sre|data scientist|data engineer|"
    r"machine learning|platform|infrastructure|security engineer|qa|sdet|sysadmin|"
    r"mobile|ios|android|architect)\b",
    re.IGNORECASE,
)


def is_dev_title(title: str) -> bool:
    return bool(_DEV_TITLE_RE.search(title or ""))


@dataclass(frozen=True)
class JobRow:
    """The columns the assessment reads, decoupled from the ORM for testing.

    New fields are appended, not inserted, because `quality/cli.py::_load_rows`
    unpacks a SELECT tuple into this dataclass positionally — field order here
    must match that SELECT's column order exactly.
    """

    source: str
    title: str
    description: str
    salary_min: int | None
    salary_max: int | None
    currency: str | None
    location: str | None
    job_type: str | None
    published_at: datetime | None
    company: str
    requirements: str | None
    responsibilities: str | None
    content_hash: str


@dataclass
class SourceQuality:
    source: str
    count: int
    pct_of_corpus: float
    published_null_pct: float
    published_future_pct: float
    salary_pct: float
    currency_pct: float
    location_pct: float
    job_type_pct: float
    desc_median_len: float
    desc_short_pct: float
    desc_html_pct: float
    salary_invalid_pct: float
    dev_title_pct: float
    extraction_null_pct: float


def _pct(numerator: int, denom: int) -> float:
    return round(100.0 * numerator / denom, 1) if denom else 0.0


def quality_for(
    source: str, rows: Sequence[JobRow], corpus_total: int, now: datetime
) -> SourceQuality:
    n = len(rows)
    desc_lens = [len(r.description or "") for r in rows]
    return SourceQuality(
        source=source,
        count=n,
        pct_of_corpus=_pct(n, corpus_total),
        published_null_pct=_pct(sum(r.published_at is None for r in rows), n),
        published_future_pct=_pct(
            sum(r.published_at is not None and r.published_at > now for r in rows), n
        ),
        salary_pct=_pct(sum(r.salary_min is not None or r.salary_max is not None for r in rows), n),
        currency_pct=_pct(sum(r.currency is not None for r in rows), n),
        location_pct=_pct(sum(bool(r.location) for r in rows), n),
        job_type_pct=_pct(sum(bool(r.job_type) for r in rows), n),
        desc_median_len=round(statistics.median(desc_lens), 1) if desc_lens else 0.0,
        desc_short_pct=_pct(sum(length < _SHORT_DESC_CHARS for length in desc_lens), n),
        desc_html_pct=_pct(sum(bool(_HTML_MARKER_RE.search(r.description or "")) for r in rows), n),
        salary_invalid_pct=_pct(
            sum(
                r.salary_min is not None
                and r.salary_max is not None
                and r.salary_min > r.salary_max
                for r in rows
            ),
            n,
        ),
        dev_title_pct=_pct(sum(is_dev_title(r.title) for r in rows), n),
        # Both fields null means ingest-time LLM extraction produced nothing for this
        # posting — BM25 then has only the title to index, since it covers
        # title/requirements/responsibilities, not the raw description (retrieval/bm25.py).
        # Either field present still gives BM25 some signal, so only "both null" counts.
        extraction_null_pct=_pct(
            sum(not r.requirements and not r.responsibilities for r in rows), n
        ),
    )


def compute(rows: Sequence[JobRow], *, now: datetime | None = None) -> list[SourceQuality]:
    """Per-source quality metrics plus an aggregate 'ALL' row, by source name."""
    now = now or datetime.now(UTC)
    by_source: dict[str, list[JobRow]] = defaultdict(list)
    for row in rows:
        by_source[row.source].append(row)

    total = len(rows)
    per_source = [quality_for(src, grp, total, now) for src, grp in sorted(by_source.items())]
    if total:
        per_source.append(quality_for("ALL", list(rows), total, now))
    return per_source


def as_dict(quality: SourceQuality) -> dict:
    return asdict(quality)


def duplicate_rate(rows: Sequence[JobRow]) -> float:
    """Corpus-wide % of rows sitting in a likely-duplicate cluster content_hash missed.

    `content_hash` (ingest/dedup.py) keys on normalized company+title+location, so two
    postings of the same role from different sources collapse into one row only when
    their location strings match exactly. Groups rows by normalized (company, title)
    instead — a group with more than one distinct content_hash is the same role listed
    under different-enough location text (e.g. "Remote" vs "Remote - US") to have
    dodged the hash. Reports the % of the corpus inside such a cluster, not the number
    of clusters, so the number reads as "how much of the corpus is affected."
    """
    groups: dict[tuple[str, str], list[JobRow]] = defaultdict(list)
    for row in rows:
        key = (" ".join(row.company.split()).lower(), " ".join(row.title.split()).lower())
        groups[key].append(row)

    affected = sum(
        len(group) for group in groups.values() if len({r.content_hash for r in group}) > 1
    )
    return _pct(affected, len(rows))
