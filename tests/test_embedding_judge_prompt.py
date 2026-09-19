"""The judge prompt: a stable prefix and a posting view that mirrors the labeler's.

The reference strings below are written by hand from `eval/label.py::_show_job`
(minus its leading blank line and 2-space indent). If this file and that function
ever disagree, judge grades stop being comparable with the user's blind ones.
"""

import json

import pytest

from eval.embedding.judge.base import CandidateBrief, JobView
from eval.embedding.judge.prompt import (
    CV_CLIP,
    JOB_HEADER,
    JUDGMENT_SCHEMA,
    LABEL_VIEW_CLIP,
    PROMPT_VERSION,
    RUBRIC,
    build_static_prefix,
    prompt_sha,
    render_label_view,
)

BRIEF = CandidateBrief(
    seniority="senior",
    years_experience=8.0,
    target_titles=["Backend Engineer", "Python Engineer"],
    tech_stack=["python", "postgres"],
    domains=["fintech"],
    remote_required=True,
    cv_text="Built backend services.",
)


def job(**overrides) -> JobView:
    fields = {
        "title": "Senior Python Engineer",
        "company": "Acme",
        "source": "himalayas",
        "seniority": "senior",
        "requirements": "5+ years of Python.",
        "responsibilities": "Own the ingestion pipeline.",
        "description": "Full posting text.",
    }
    return JobView(**{**fields, **overrides})


class TestRenderLabelView:
    def test_extracted_branch_clips_both_fields_at_400(self) -> None:
        rendered = render_label_view(
            job(requirements="R" * 500, responsibilities="S" * 401, description="ignored")
        )
        assert rendered == (
            "Title   : Senior Python Engineer\n"
            "Company : Acme\n"
            "Source  : himalayas  |  Seniority: senior\n"
            f"Requires: {'R' * 400}…\n"
            f"Resp    : {'S' * 400}…"
        )

    def test_only_requirements_present(self) -> None:
        rendered = render_label_view(job(responsibilities=None))
        assert rendered == (
            "Title   : Senior Python Engineer\n"
            "Company : Acme\n"
            "Source  : himalayas  |  Seniority: senior\n"
            "Requires: 5+ years of Python."
        )

    def test_only_responsibilities_present(self) -> None:
        rendered = render_label_view(job(requirements=""))
        assert rendered == (
            "Title   : Senior Python Engineer\n"
            "Company : Acme\n"
            "Source  : himalayas  |  Seniority: senior\n"
            "Resp    : Own the ingestion pipeline."
        )

    def test_falls_back_to_description_when_neither_extracted_field_exists(self) -> None:
        rendered = render_label_view(job(requirements=None, responsibilities=""))
        assert rendered == (
            "Title   : Senior Python Engineer\n"
            "Company : Acme\n"
            "Source  : himalayas  |  Seniority: senior\n"
            "Desc    : Full posting text."
        )

    def test_missing_seniority_reads_unknown(self) -> None:
        assert "Seniority: unknown" in render_label_view(job(seniority=None))

    def test_empty_description_fallback_reads_none(self) -> None:
        rendered = render_label_view(job(requirements=None, responsibilities=None, description=""))
        assert rendered.endswith("Desc    : (none)")

    def test_fields_are_stripped_before_clipping(self) -> None:
        rendered = render_label_view(job(requirements="  5+ years of Python.  "))
        assert "Requires: 5+ years of Python.\n" in rendered

    def test_location_never_appears(self) -> None:
        # The labelers saw source, not location (F21) — a location in the text
        # the view drops must not leak in through another branch.
        rendered = render_label_view(job(description="Onsite in Lisbon, Portugal."))
        assert "Lisbon" not in rendered
        assert "Location" not in rendered
        assert not hasattr(job(), "location")

    @pytest.mark.parametrize(
        ("length", "clipped"), [(LABEL_VIEW_CLIP - 1, False), (LABEL_VIEW_CLIP, False), (401, True)]
    )
    def test_clip_boundary(self, length: int, clipped: bool) -> None:
        rendered = render_label_view(job(requirements="x" * length))
        expected = "x" * min(length, LABEL_VIEW_CLIP) + ("…" if clipped else "")
        assert rendered.splitlines()[3] == f"Requires: {expected}"


class TestStaticPrefix:
    def test_is_identical_across_jobs_and_calls(self) -> None:
        fewshot = [(job(title="Exemplar A"), 3), (job(title="Exemplar B"), 0)]
        first = build_static_prefix(BRIEF, fewshot)
        assert first == build_static_prefix(BRIEF, fewshot)
        # Nothing about the job being graded reaches the prefix, by construction.
        for n in range(20):
            assert build_static_prefix(BRIEF, fewshot) == first
            assert f"Job {n}" not in first

    def test_orders_rubric_brief_examples_then_the_job_header(self) -> None:
        prefix = build_static_prefix(BRIEF, [(job(title="Exemplar"), 2)])
        positions = [
            prefix.index(RUBRIC),
            prefix.index("# Candidate brief"),
            prefix.index("# Graded examples"),
            prefix.index(JOB_HEADER),
        ]
        assert positions == sorted(positions)
        assert prefix.endswith(JOB_HEADER)

    def test_examples_carry_the_rendered_view_and_a_json_grade(self) -> None:
        exemplar = job(title="Exemplar", requirements="Rust only.", responsibilities=None)
        prefix = build_static_prefix(BRIEF, [(exemplar, 0)])
        assert render_label_view(exemplar) in prefix
        assert '"grade": 0' in prefix

    def test_no_examples_section_without_fewshot(self) -> None:
        prefix = build_static_prefix(BRIEF, [])
        assert "# Graded examples" not in prefix
        assert prefix.endswith(JOB_HEADER)

    def test_brief_carries_the_candidate_and_clips_the_cv(self) -> None:
        brief = CandidateBrief(
            seniority="senior",
            years_experience=None,
            target_titles=["Backend Engineer"],
            tech_stack=["python"],
            domains=[],
            remote_required=True,
            cv_text="c" * (CV_CLIP + 50),
        )
        prefix = build_static_prefix(brief, [])
        assert "target_titles: Backend Engineer" in prefix
        assert "years_experience: None" in prefix
        assert "c" * CV_CLIP in prefix
        assert "c" * (CV_CLIP + 1) not in prefix


class TestCandidateBriefFromSnapshot:
    def test_reads_the_profile_snapshot_keys(self) -> None:
        brief = CandidateBrief.from_snapshot(
            {
                "seniority": "senior",
                "years_experience": 8.0,
                "target_titles": ["Backend Engineer", "Python Engineer"],
                "tech_stack": ["python", "postgres"],
                "domains": ["fintech"],
                "remote_required": True,
                "cv_text": "Built backend services.",
                "salary_floor": 90000,  # extra snapshot keys are ignored
            }
        )
        assert brief == BRIEF

    def test_tolerates_missing_and_null_keys(self) -> None:
        brief = CandidateBrief.from_snapshot({"seniority": None, "tech_stack": None})
        assert brief == CandidateBrief(
            seniority="unknown",
            years_experience=None,
            target_titles=[],
            tech_stack=[],
            domains=[],
            remote_required=False,
            cv_text="",
        )


class TestSchemaAndSha:
    def test_reason_comes_before_grade_and_both_are_required(self) -> None:
        assert list(JUDGMENT_SCHEMA["properties"]) == ["reason", "grade"]
        assert JUDGMENT_SCHEMA["type"] == "object"
        assert JUDGMENT_SCHEMA["properties"]["reason"] == {"type": "string"}
        assert JUDGMENT_SCHEMA["properties"]["grade"] == {"type": "integer", "enum": [0, 1, 2, 3]}
        assert JUDGMENT_SCHEMA["required"] == ["reason", "grade"]

    def test_schema_is_json_serializable(self) -> None:
        assert json.loads(json.dumps(JUDGMENT_SCHEMA)) == JUDGMENT_SCHEMA

    def test_prompt_sha_is_stable_and_candidate_independent(self) -> None:
        sha = prompt_sha()
        assert len(sha) == 64 and int(sha, 16) >= 0
        assert sha == prompt_sha()
        assert PROMPT_VERSION not in ("", None)
