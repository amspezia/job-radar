import pytest

from job_radar.adapters.sources.base import NormalizedJob
from job_radar.ingest.dedup import content_hash


def _job(
    company="Acme", title="Senior Engineer", location="Worldwide", remote=True
) -> NormalizedJob:
    return NormalizedJob(
        source="remotive",
        source_type="aggregator",
        source_id="1",
        url="https://example.com/job/1",
        title=title,
        company=company,
        description="desc",
        salary_min=None,
        salary_max=None,
        currency=None,
        location=location,
        job_type="full_time",
        remote=remote,
        published_at=None,
    )


class TestContentHash:
    def test_is_a_64_char_hex_digest(self) -> None:
        h = content_hash(_job())
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_identical_jobs_hash_the_same(self) -> None:
        assert content_hash(_job()) == content_hash(_job())

    def test_whitespace_noise_does_not_change_the_hash(self) -> None:
        assert content_hash(_job()) == content_hash(_job(title="  Senior   Engineer "))

    def test_case_noise_does_not_change_the_hash(self) -> None:
        assert content_hash(_job()) == content_hash(_job(title="SENIOR ENGINEER"))

    def test_missing_location_matches_empty_string_location(self) -> None:
        assert content_hash(_job(location=None, remote=False)) == content_hash(
            _job(location="", remote=False)
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"title": "Junior Engineer"},
            {"company": "Globex"},
        ],
    )
    def test_different_identity_fields_change_the_hash(self, overrides: dict) -> None:
        assert content_hash(_job()) != content_hash(_job(**overrides))

    def test_location_still_distinguishes_non_remote_postings(self) -> None:
        # On-site roles in different cities are genuinely different reqs.
        assert content_hash(_job(location="Berlin", remote=False)) != content_hash(
            _job(location="London", remote=False)
        )

    def test_location_is_ignored_for_remote_postings(self) -> None:
        # One remote role listed once per eligible country is ONE posting; without
        # this, aggregators that fan a role out across countries never dedupe.
        assert content_hash(_job(location="Portugal")) == content_hash(_job(location="Brazil"))

    def test_remote_and_non_remote_postings_do_not_collide(self) -> None:
        # The remote flag changes which fields are hashed, so a remote and an
        # on-site posting of the same role must not land on the same identity.
        assert content_hash(_job(location="Berlin")) != content_hash(
            _job(location="Berlin", remote=False)
        )
