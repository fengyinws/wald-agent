import math
from time import perf_counter
from typing import Annotated, Literal

import httpx
from pydantic import Field, TypeAdapter, ValidationError

from wald_agent.config import Settings
from wald_agent.errors import InvalidProviderResponse, InvalidRequest
from wald_agent.probability import make_answer
from wald_agent.schemas import (
    BooleanQuestion,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    JsonValue,
    Probability,
    ScoreQuestion,
    StrictModel,
    Usage,
    provider_questions,
)
from wald_agent.transport import JsonTransport


class JevChoice(StrictModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class JevScore(StrictModel):
    type: Literal["score"]
    score: float
    probabilities: dict[str, Probability]
    confidence: Probability
    legend: dict[str, JsonValue]


class JevNoul(StrictModel):
    type: Literal["noul"]
    noul: Probability


JEV_ANSWER = TypeAdapter(Annotated[JevChoice | JevScore | JevNoul, Field(discriminator="type")])


class JevClient(JsonTransport):
    """Official TypeSafe HTTP protocol, normalized to Wald's result types."""

    def __init__(
        self, settings: Settings | None = None, http_client: httpx.AsyncClient | None = None
    ):
        self.settings = settings or Settings()
        super().__init__(self.settings.api_timeout_seconds, http_client, settings=self.settings)

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        request = DecisionRequest.model_validate(request.model_dump())
        if len(request.model_dump_json().encode("utf-8")) > self.settings.max_state_bytes:
            raise InvalidRequest("Decision request exceeds MAX_STATE_BYTES.")
        started = perf_counter()
        raw = await self.post(
            f"{self.settings.jev_base_url}/systemone",
            self.settings.key_for("jev"),
            {
                "model": self.settings.jev_model,
                "state": request.state,
                "questions": provider_questions(request),
            },
            "jev",
        )
        try:
            if set(raw["answers"]) != set(request.questions):
                raise ValueError("question IDs differ")
            answers = {}
            for name, question in request.questions.items():
                native = JEV_ANSWER.validate_python(raw["answers"][name])
                if isinstance(question, BooleanQuestion) and isinstance(native, JevNoul):
                    answers[name] = make_answer(
                        question,
                        {"false": 1 - native.noul, "true": native.noul},
                        request.review_threshold,
                    )
                elif isinstance(question, ChoiceQuestion) and isinstance(native, JevChoice):
                    answers[name] = make_answer(
                        question,
                        native.probabilities,
                        request.review_threshold,
                        provider_confidence=native.confidence,
                        provider_choice=native.choice,
                    )
                elif isinstance(question, ScoreQuestion) and isinstance(native, JevScore):
                    if not 0 <= native.score <= len(question.criteria) - 1:
                        raise ValueError("score outside scale")
                    if set(native.legend) != set(native.probabilities):
                        raise ValueError("score legend mismatch")
                    answers[name] = make_answer(
                        question,
                        native.probabilities,
                        request.review_threshold,
                        provider_confidence=native.confidence,
                        provider_score=native.score + question.min_value,
                    )
                    # Jev rounds its native score/probabilities; preserve its score separately.
                    if not math.isclose(
                        answers[name].score, native.score + question.min_value, abs_tol=0.05
                    ):
                        raise ValueError("score inconsistent with distribution")
                else:
                    raise ValueError("question/answer types differ")
            return DecisionResponse(
                provider="jev",
                model=raw["model"],
                answers=answers,
                needs_review=any(answer.needs_review for answer in answers.values()),
                latency_ms=(perf_counter() - started) * 1000,
                usage=Usage.model_validate(raw.get("usage", {})),
                probability_source="jev",
                attempts=raw.attempts,
                upstream_request_id=raw.upstream_request_id,
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise InvalidProviderResponse("Jev returned an invalid decision response.") from exc
