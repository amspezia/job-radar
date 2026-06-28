from datetime import UTC, datetime, timedelta

import pytest

from job_radar.quality import cli, metrics, relevance

_NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _row(**over: object) -> metrics.JobRow:
    base: dict = {
        "source": "greenhouse",
        "title": "Senior Backend Engineer",
        "description": "x" * 300,
        "salary_min": 100,
        "salary_max": 200,
        "currency": "USD",
        "location": "Remote",
        "job_type": "Full-Time",
        "published_at": _NOW - timedelta(days=10),
    }
    base.update(over)
    return metrics.JobRow(**base)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Senior Backend Engineer", True),
        ("Data Scientist", True),
        ("DevOps Engineer", True),
        ("Sales Manager", False),
        ("Account Executive", False),
        ("", False),
    ],
)
def test_is_dev_title(title: str, expected: bool) -> None:
    assert metrics.is_dev_title(title) is expected


def test_compute_per_source_metrics() -> None:
    rows = [
        _row(),  # complete, valid, dev, past
        _row(
            title="Sales Manager",
            description="<p>html</p>",
            salary_min=300,
            salary_max=200,  # min > max -> invalid
            currency=None,
            location=None,
            job_type=None,
            published_at=_NOW + timedelta(days=1),  # future
        ),
        _row(
            source="remotive",
            title="Data Scientist",
            description="short",
            salary_min=None,
            salary_max=None,
            published_at=None,
        ),
    ]

    by_source = {q.source: q for q in metrics.compute(rows, now=_NOW)}

    gh = by_source["greenhouse"]
    assert gh.count == 2
    assert gh.pct_of_corpus == round(100 * 2 / 3, 1)
    assert gh.salary_pct == 100.0
    assert gh.currency_pct == 50.0
    assert gh.location_pct == 50.0
    assert gh.job_type_pct == 50.0
    assert gh.desc_html_pct == 50.0  # only the <p>html</p> row
    assert gh.desc_short_pct == 50.0  # the 11-char row
    assert gh.salary_invalid_pct == 50.0  # min>max row
    assert gh.published_future_pct == 50.0
    assert gh.published_null_pct == 0.0
    assert gh.dev_title_pct == 50.0  # engineer yes, sales no

    rem = by_source["remotive"]
    assert rem.count == 1
    assert rem.salary_pct == 0.0
    assert rem.published_null_pct == 100.0
    assert rem.dev_title_pct == 100.0

    assert by_source["ALL"].count == 3


def test_compute_empty_rows_yields_no_all_row() -> None:
    assert metrics.compute([]) == []


async def test_build_centroid_averages_anchor_vectors() -> None:
    vectors = {"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]}

    async def fake_embed(text: str) -> list[float]:
        return vectors[text]

    centroid = await relevance.build_centroid(fake_embed, anchors=["a", "b"])
    assert centroid == [2.0, 3.0, 4.0]


def _quality(source: str, count: int) -> metrics.SourceQuality:
    return metrics.SourceQuality(
        source=source,
        count=count,
        pct_of_corpus=0.0,
        published_null_pct=0.0,
        published_future_pct=0.0,
        salary_pct=0.0,
        currency_pct=0.0,
        location_pct=0.0,
        job_type_pct=0.0,
        desc_median_len=0.0,
        desc_short_pct=0.0,
        desc_html_pct=0.0,
        salary_invalid_pct=0.0,
        dev_title_pct=0.0,
    )


def test_overall_relevance_is_count_weighted() -> None:
    quals = [_quality("x", 1), _quality("y", 3), _quality("ALL", 4)]
    rel = {
        "x": relevance.SourceRelevance(0.4, 10.0),
        "y": relevance.SourceRelevance(0.8, 50.0),
    }
    overall = cli._overall_relevance(quals, rel)
    assert overall is not None
    assert overall.relevance_mean == 0.7  # (1*0.4 + 3*0.8) / 4
    assert overall.dev_embed_pct == 40.0  # (1*10 + 3*50) / 4


def test_merge_attaches_relevance_and_renders() -> None:
    quals = [_quality("greenhouse", 2), _quality("ALL", 2)]
    rel = {"greenhouse": relevance.SourceRelevance(0.6, 75.0)}

    columns = cli._merge(quals, rel)
    gh = next(c for c in columns if c["source"] == "greenhouse")
    assert gh["relevance_mean"] == 0.6
    assert gh["dev_embed_pct"] == 75.0

    table = cli.render_table(columns)
    assert "greenhouse" in table
    assert "relevance mean" in table
