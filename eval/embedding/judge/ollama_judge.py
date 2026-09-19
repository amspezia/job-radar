"""A `Judge` backed by a local Ollama chat model with structured output.

The static prefix is built once, at construction, and sent as the system message
of every call; only the posting varies. An unusable answer raises `JudgeError` —
the runner retries and counts it, and no job ever gets an invented grade.
"""

from collections.abc import Sequence
from typing import Protocol

from eval.embedding.judge.base import CandidateBrief, JobView, JudgeError, Judgment
from eval.embedding.judge.prompt import (
    JUDGMENT_SCHEMA,
    LABEL_VIEW_CLIP,
    PROMPT_VERSION,
    build_static_prefix,
    prompt_sha,
    render_label_view,
)


class ChatResultLike(Protocol):
    data: dict
    prompt_eval_seconds: float | None


class ChatClient(Protocol):
    async def chat_json(
        self, model: str, messages: list[dict], schema: dict, *, options: dict
    ) -> ChatResultLike: ...


class OllamaJudge:
    def __init__(
        self,
        client: ChatClient,
        model: str,
        brief: CandidateBrief,
        fewshot: Sequence[tuple[JobView, int]],
        seed: int = 1,
        num_ctx: int = 8192,
        num_predict: int = 300,
    ) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self._client = client
        self._prefix = build_static_prefix(brief, fewshot)
        self._seed = seed
        self._num_ctx = num_ctx
        self._num_predict = num_predict

    def describe(self) -> dict:
        return {
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha": prompt_sha(),
            "seed": self._seed,
            "num_ctx": self._num_ctx,
            "clip": LABEL_VIEW_CLIP,
        }

    async def grade(self, job: JobView) -> Judgment:
        result = await self._client.chat_json(
            self.model,
            [
                {"role": "system", "content": self._prefix},
                {"role": "user", "content": render_label_view(job)},
            ],
            JUDGMENT_SCHEMA,
            options={
                "temperature": 0,
                "seed": self._seed,
                "num_ctx": self._num_ctx,
                "num_predict": self._num_predict,
            },
        )
        data = result.data
        if not isinstance(data, dict):
            raise JudgeError(f"{self.model} returned {type(data).__name__}, expected an object")
        grade = data.get("grade")
        reason = data.get("reason")
        if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 3:
            raise JudgeError(f"{self.model} returned grade {grade!r}, expected an integer 0-3")
        if not isinstance(reason, str):
            raise JudgeError(f"{self.model} returned reason {reason!r}, expected a string")
        return Judgment(
            grade=grade, rationale=reason, prompt_eval_seconds=result.prompt_eval_seconds
        )
