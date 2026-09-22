import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Name = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r"\S")]
Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Description = Text | dict[str, JsonValue] | list[JsonValue]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Description
    criteria: dict[Name, Description | None] = Field(min_length=2, max_length=255)


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Description
    criteria: list[Description] = Field(min_length=2, max_length=10)
    min_value: int = Field(default=0, ge=-1000, le=1000)


class BooleanQuestion(StrictModel):
    type: Literal["boolean", "noul"]
    instructions: Description
    criteria: dict[Literal["true", "false"], Description] | None = None

    @model_validator(mode="after")
    def check_criteria(self) -> "BooleanQuestion":
        if self.criteria is not None and set(self.criteria) != {"true", "false"}:
            raise ValueError("boolean criteria must contain both true and false")
        return self


Question = Annotated[ChoiceQuestion | ScoreQuestion | BooleanQuestion, Field(discriminator="type")]


class DecisionRequest(StrictModel):
    state: Text | dict[str, JsonValue] | list[JsonValue]
    questions: dict[Name, Question] = Field(min_length=1, max_length=64)
    review_threshold: Probability = 0.5

    @model_validator(mode="after")
    def finite_and_bounded(self) -> "DecisionRequest":
        try:
            encoded = json.dumps(self.model_dump(), ensure_ascii=False, allow_nan=False)
        except ValueError as exc:
            raise ValueError("request contains non-finite numbers") from exc
        if len(encoded.encode("utf-8")) > 4 * 1024 * 1024:
            raise ValueError("request exceeds the 4 MiB schema limit")
        return self


class AnswerBase(StrictModel):
    probabilities: dict[str, Probability] = Field(min_length=2)
    confidence: Probability
    needs_review: bool
    provider_confidence: Probability | None = None

    @model_validator(mode="after")
    def valid_distribution(self) -> "AnswerBase":
        if not math.isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to 1")
        return self


class ChoiceAnswer(AnswerBase):
    type: Literal["choice"] = "choice"
    choice: str

    @model_validator(mode="after")
    def valid_choice(self) -> "ChoiceAnswer":
        if self.choice not in self.probabilities or not math.isclose(
            self.probabilities[self.choice], max(self.probabilities.values()), abs_tol=1e-9
        ):
            raise ValueError("choice must have the highest probability")
        return self


class ScoreAnswer(AnswerBase):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, Description]
    provider_score: float | None = None

    @model_validator(mode="after")
    def valid_score(self) -> "ScoreAnswer":
        if set(self.legend) != set(self.probabilities):
            raise ValueError("score legend must match probabilities")
        try:
            expected = sum(int(k) * v for k, v in self.probabilities.items())
        except ValueError as exc:
            raise ValueError("score keys must be integer levels") from exc
        if not math.isclose(self.score, expected, abs_tol=1e-6):
            raise ValueError("score must equal its probability-weighted expectation")
        return self


class BooleanAnswer(AnswerBase):
    type: Literal["boolean"] = "boolean"
    boolean: bool
    probability_true: Probability

    @model_validator(mode="after")
    def valid_boolean(self) -> "BooleanAnswer":
        if set(self.probabilities) != {"true", "false"}:
            raise ValueError("boolean probability keys must be true and false")
        if not math.isclose(self.probability_true, self.probabilities["true"], abs_tol=1e-9):
            raise ValueError("probability_true must match probabilities.true")
        if self.boolean != (self.probability_true >= 0.5):
            raise ValueError("boolean must use the 0.5 threshold")
        return self


Answer = Annotated[ChoiceAnswer | ScoreAnswer | BooleanAnswer, Field(discriminator="type")]


class Usage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class DecisionResponse(StrictModel):
    provider: Literal["llm", "jev"]
    model: str
    answers: dict[str, Answer]
    needs_review: bool
    latency_ms: float = Field(ge=0)
    usage: Usage = Field(default_factory=Usage)
    probability_source: Literal["llm_self_report", "jev"]
    confidence_method: Literal["normalized_entropy"] = "normalized_entropy"
    calibration_status: Literal["not_validated_on_your_data"] = "not_validated_on_your_data"
    attempts: int = Field(default=1, ge=1)
    upstream_request_id: str | None = None
    decision_id: str | None = None
    request_id: str | None = None


class BatchItem(StrictModel):
    id: Name
    request: DecisionRequest


class BatchDecisionRequest(StrictModel):
    items: list[BatchItem] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def unique_ids(self) -> "BatchDecisionRequest":
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("batch item IDs must be unique")
        return self


class ErrorDetail(StrictModel):
    code: str
    message: str
    status_code: int
    retry_after: float | None = None
    decision_id: str | None = None


class BatchResult(StrictModel):
    id: str
    result: DecisionResponse | None = None
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def exactly_one_outcome(self) -> "BatchResult":
        if (self.result is None) == (self.error is None):
            raise ValueError("batch result must contain exactly one of result or error")
        return self


class BatchDecisionResponse(StrictModel):
    results: list[BatchResult]
    request_id: str | None = None
    decision_id: str | None = None
    needs_review: bool


class DecisionRecord(StrictModel):
    id: str
    kind: Literal["decision", "image", "batch"]
    status: Literal["processing", "succeeded", "failed"]
    created_at: float
    updated_at: float
    request: dict[str, JsonValue]
    result: dict[str, JsonValue] | None
    error: ErrorDetail | None
    needs_review: bool
    resolution: dict[str, JsonValue] | None
    revision: int


class ReviewPage(StrictModel):
    items: list[DecisionRecord]
    next_cursor: str | None


class ReviewResolution(StrictModel):
    answers: dict[Name, str | bool | int]
    reviewer: Text = Field(max_length=128)
    notes: str = Field(default="", max_length=2000)
    expected_revision: int = Field(default=0, ge=0)


def validate_labels(
    questions: dict[str, Question], labels: dict, *, complete: bool = False
) -> None:
    if complete and set(questions) != set(labels):
        raise ValueError("review must answer exactly the original question IDs")
    for name, label in labels.items():
        question = questions.get(name)
        if question is None:
            raise ValueError("label refers to an unknown question")
        if isinstance(question, BooleanQuestion):
            if type(label) is not bool:
                raise ValueError("boolean labels must be JSON booleans")
        elif isinstance(question, ChoiceQuestion):
            if not isinstance(label, str) or label not in question.criteria:
                raise ValueError("choice label must be a candidate key")
        elif type(label) is not int or str(label) not in probability_keys(question):
            raise ValueError("score label must be an integer on the question's scale")


def probability_keys(question: Question, *, native: bool = False) -> list[str]:
    if isinstance(question, ChoiceQuestion):
        return list(question.criteria)
    if isinstance(question, ScoreQuestion):
        start = 0 if native else question.min_value
        return [str(start + i) for i in range(len(question.criteria))]
    return ["false", "true"]


def provider_questions(request: DecisionRequest) -> dict[str, dict]:
    """Same business rubric for both providers; offsets/policy are applied locally."""
    result = {}
    for name, question in request.questions.items():
        item = question.model_dump(exclude={"min_value"}, exclude_none=True)
        # Keep null Choice descriptions; Jev explicitly supports them.
        if isinstance(question, ChoiceQuestion):
            item["criteria"] = question.criteria
        if isinstance(question, BooleanQuestion):
            item["type"] = "noul"
        result[name] = item
    return result
