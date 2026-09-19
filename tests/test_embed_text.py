import pytest

from job_radar.ingest.embed_text import build_embed_text


def _legacy(
    title: str, description: str, requirements: str | None, responsibilities: str | None
) -> str:
    """The inline expression `_prepare` used before build_embed_text — the oracle."""
    if requirements or responsibilities:
        return "\n".join(filter(None, [title, requirements, responsibilities]))
    return f"{title}\n{description}"


def test_both_extracted_fields_join_title_requirements_responsibilities() -> None:
    assert build_embed_text("Dev", "full text", "req", "resp") == "Dev\nreq\nresp"


def test_extracted_fields_replace_the_description() -> None:
    assert "full text" not in build_embed_text("Dev", "full text", "req", "resp")


@pytest.mark.parametrize(
    ("requirements", "responsibilities", "expected"),
    [("req", None, "Dev\nreq"), (None, "resp", "Dev\nresp"), ("req", "", "Dev\nreq")],
)
def test_single_extracted_field_leaves_no_blank_line(
    requirements: str | None, responsibilities: str | None, expected: str
) -> None:
    assert build_embed_text("Dev", "full text", requirements, responsibilities) == expected


@pytest.mark.parametrize(
    ("requirements", "responsibilities"), [(None, None), ("", ""), (None, ""), ("", None)]
)
def test_no_extracted_fields_falls_back_to_title_and_description(
    requirements: str | None, responsibilities: str | None
) -> None:
    assert build_embed_text("Dev", "full text", requirements, responsibilities) == "Dev\nfull text"


def test_fallback_keeps_empty_description_separator() -> None:
    assert build_embed_text("Dev", "", None, None) == "Dev\n"


def test_empty_title_is_dropped_in_the_extracted_branch() -> None:
    assert build_embed_text("", "full text", "req", "resp") == "req\nresp"


def test_empty_title_is_kept_in_the_fallback_branch() -> None:
    assert build_embed_text("", "full text", None, None) == "\nfull text"


@pytest.mark.parametrize("title", ["", "Senior Python Engineer"])
@pytest.mark.parametrize("description", ["", "Full posting body.\nSecond line."])
@pytest.mark.parametrize("requirements", [None, "", "3+ years of Python", "a\nb"])
@pytest.mark.parametrize("responsibilities", [None, "", "Own the ingest pipeline", "c\nd"])
def test_matches_legacy_inline_expression(
    title: str, description: str, requirements: str | None, responsibilities: str | None
) -> None:
    assert build_embed_text(title, description, requirements, responsibilities) == _legacy(
        title, description, requirements, responsibilities
    )
