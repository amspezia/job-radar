"""OllamaJudge against a fake chat client — no Ollama, no network."""

from dataclasses import dataclass, field

import pytest

from eval.embedding.judge.base import CandidateBrief, JobView, JudgeError
from eval.embedding.judge.ollama_judge import OllamaJudge
from eval.embedding.judge.prompt import (
    JUDGMENT_SCHEMA,
    LABEL_VIEW_CLIP,
    PROMPT_VERSION,
    build_static_prefix,
    prompt_sha,
    render_label_view,
)

BRIEF = CandidateBrief(
    seniority="senior",
    years_experience=8.0,
    target_titles=["Backend Engineer"],
    tech_stack=["python"],
    domains=["fintech"],
    remote_required=True,
    cv_text="Built backend services.",
)


def make_job(n: int = 0) -> JobView:
    return JobView(
        title=f"Backend Engineer {n}",
        company=f"Company {n}",
        source="himalayas",
        seniority="senior",
        requirements="Python and Postgres.",
        responsibilities=None,
        description="Full posting text.",
    )


@dataclass
class FakeResult:
    data: object
    prompt_eval_seconds: float | None = None


@dataclass
class FakeChatClient:
    """Replays scripted results and records every call."""

    results: list
    calls: list[dict] = field(default_factory=list)

    async def chat_json(self, model, messages, schema, *, options):
        self.calls.append(
            {"model": model, "messages": messages, "schema": schema, "options": options}
        )
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


def make_judge(results: list, **kwargs) -> tuple[OllamaJudge, FakeChatClient]:
    client = FakeChatClient(results=results)
    fewshot = [(make_job(90), 3), (make_job(91), 0)]
    return OllamaJudge(client, "Gemma3:12b", BRIEF, fewshot, **kwargs), client


class TestGrade:
    async def test_returns_the_judgment(self) -> None:
        judge, _ = make_judge([FakeResult({"reason": "Right stack.", "grade": 3}, 0.42)])
        judgment = await judge.grade(make_job())
        assert (judgment.grade, judgment.rationale) == (3, "Right stack.")
        assert judgment.prompt_eval_seconds == 0.42

    async def test_sends_the_prefix_the_posting_the_schema_and_the_options(self) -> None:
        judge, client = make_judge([FakeResult({"reason": "ok", "grade": 2})], seed=5, num_ctx=4096)
        job = make_job()
        await judge.grade(job)

        call = client.calls[0]
        assert call["model"] == "Gemma3:12b"
        assert call["schema"] is JUDGMENT_SCHEMA
        assert call["options"] == {
            "temperature": 0,
            "seed": 5,
            "num_ctx": 4096,
            "num_predict": 300,
        }
        system, user = call["messages"]
        assert system["role"] == "system"
        assert system["content"] == build_static_prefix(
            BRIEF, [(make_job(90), 3), (make_job(91), 0)]
        )
        assert user == {"role": "user", "content": render_label_view(job)}

    async def test_the_static_prefix_is_byte_identical_across_jobs(self) -> None:
        judge, client = make_judge([FakeResult({"reason": "ok", "grade": 1})])
        for n in range(25):
            await judge.grade(make_job(n))

        assert len({call["messages"][0]["content"] for call in client.calls}) == 1
        assert len({call["messages"][1]["content"] for call in client.calls}) == 25
        prefix = client.calls[0]["messages"][0]["content"]
        assert "Backend Engineer 7" not in prefix

    @pytest.mark.parametrize(
        "data",
        [
            {"reason": "ok", "grade": 4},
            {"reason": "ok", "grade": -1},
            {"reason": "ok", "grade": "2"},
            {"reason": "ok", "grade": 2.0},
            {"reason": "ok", "grade": True},
            {"reason": "ok"},
            {"grade": 2},
            {"reason": None, "grade": 2},
            "not an object",
        ],
    )
    async def test_an_unusable_answer_raises_instead_of_defaulting(self, data) -> None:
        judge, _ = make_judge([FakeResult(data)])
        with pytest.raises(JudgeError):
            await judge.grade(make_job())


class TestDescribe:
    async def test_records_what_reproduces_the_run(self) -> None:
        judge, _ = make_judge([], seed=5, num_ctx=4096)
        assert judge.describe() == {
            "model": "Gemma3:12b",
            "prompt_version": PROMPT_VERSION,
            "prompt_sha": prompt_sha(),
            "seed": 5,
            "num_ctx": 4096,
            "clip": LABEL_VIEW_CLIP,
        }
        assert judge.name == "ollama:Gemma3:12b"
