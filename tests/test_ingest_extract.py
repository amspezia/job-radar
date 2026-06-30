import pytest

from job_radar.ingest import extract as extract_mod
from job_radar.ingest.extract import _TRUNCATE_AT, _Extraction, extract_fields


async def test_extract_fields_returns_empty_for_blank_description() -> None:
    assert await extract_fields("") == ("", "")
    assert await extract_fields("   ") == ("", "")


async def test_extract_fields_returns_extracted_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_generate(prompt: str, schema: type, *, model: str | None = None) -> _Extraction:
        return _Extraction(
            requirements="Python 3.12, FastAPI, 3+ years backend experience",
            responsibilities="Build and maintain REST APIs, review pull requests",
        )

    monkeypatch.setattr(extract_mod, "generate", fake_generate)

    req, resp = await extract_fields("Some job description text here.")

    assert "Python" in req
    assert "APIs" in resp


async def test_extract_fields_returns_empty_on_llm_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(prompt: str, schema: type, *, model: str | None = None) -> None:
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(extract_mod, "generate", boom)

    assert await extract_fields("Some job description.") == ("", "")


async def test_extract_fields_truncates_long_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[str] = []

    async def capture(prompt: str, schema: type, *, model: str | None = None) -> _Extraction:
        received.append(prompt)
        return _Extraction(requirements="Python", responsibilities="Build things")

    monkeypatch.setattr(extract_mod, "generate", capture)

    long_desc = "x" * 10_000
    await extract_fields(long_desc)

    assert long_desc[:_TRUNCATE_AT] in received[0]
    assert long_desc[_TRUNCATE_AT:] not in received[0]
