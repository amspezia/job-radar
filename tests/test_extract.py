from pathlib import Path

import pytest

from job_radar.profile.extract import extract_text

_FIXTURES = Path(__file__).parent / "fixtures"


def test_extracts_text_from_pdf() -> None:
    text = extract_text(_FIXTURES / "cv_sample.pdf")
    assert "Jane Doe" in text
    assert "Senior Backend Engineer" in text
    assert "Python, PostgreSQL, Kubernetes" in text


def test_passes_through_text_files(tmp_path: Path) -> None:
    cv = tmp_path / "cv.txt"
    cv.write_text("Senior Backend Engineer\nPython, Postgres", encoding="utf-8")
    assert extract_text(cv) == "Senior Backend Engineer\nPython, Postgres"


def test_passes_through_markdown(tmp_path: Path) -> None:
    cv = tmp_path / "cv.md"
    cv.write_text("# Jane Doe\nEngineer", encoding="utf-8")
    assert extract_text(cv) == "# Jane Doe\nEngineer"


def test_image_only_pdf_raises_rather_than_returning_empty() -> None:
    # A PDF with no text layer (~scanned) must fail loudly, not yield "".
    with pytest.raises(ValueError, match="no extractable text"):
        extract_text(_FIXTURES / "cv_blank.pdf")


def test_whitespace_only_text_file_raises(tmp_path: Path) -> None:
    cv = tmp_path / "cv.txt"
    cv.write_text("   \n\t  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no extractable text"):
        extract_text(cv)


def test_unsupported_file_type_raises(tmp_path: Path) -> None:
    cv = tmp_path / "cv.docx"
    cv.write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="unsupported CV file type"):
        extract_text(cv)
