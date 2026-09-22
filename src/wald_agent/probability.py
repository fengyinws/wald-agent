import math

from wald_agent.errors import InvalidProviderResponse
from wald_agent.schemas import (
    Answer,
    BooleanAnswer,
    ChoiceAnswer,
    ChoiceQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    probability_keys,
)


def validate_probabilities(values: object, keys: list[str]) -> dict[str, float]:
    if not isinstance(values, dict) or set(values) != set(keys):
        raise InvalidProviderResponse("Provider returned missing or unexpected probability keys.")
    if any(
        type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
        for v in values.values()
    ):
        raise InvalidProviderResponse("Provider probabilities must be finite numbers in [0, 1].")
    total = sum(values.values())
    if total <= 0 or not math.isclose(total, 1.0, rel_tol=0, abs_tol=0.001):
        raise InvalidProviderResponse("Provider probabilities must sum to 1 (tolerance 0.001).")
    # Only compensate for rounding. This is NOT statistical calibration.
    return {key: float(values[key] / total) for key in keys}


def concentration(probabilities: dict[str, float]) -> float:
    entropy = -sum(p * math.log(p) for p in probabilities.values() if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(probabilities))))


def make_answer(
    question: Question,
    values: object,
    review_threshold: float,
    *,
    provider_confidence: float | None = None,
    provider_choice: str | None = None,
    provider_score: float | None = None,
) -> Answer:
    native = validate_probabilities(values, probability_keys(question, native=True))
    probabilities = dict(zip(probability_keys(question), native.values(), strict=True))
    confidence = concentration(probabilities)
    common = {
        "probabilities": probabilities,
        "confidence": confidence,
        "needs_review": confidence < review_threshold,
        "provider_confidence": provider_confidence,
    }
    if isinstance(question, ChoiceQuestion):
        choice = (
            provider_choice
            if provider_choice is not None
            else max(probabilities, key=probabilities.__getitem__)
        )
        return ChoiceAnswer(choice=choice, **common)
    if isinstance(question, ScoreQuestion):
        return ScoreAnswer(
            score=sum(int(k) * p for k, p in probabilities.items()),
            legend={str(question.min_value + i): v for i, v in enumerate(question.criteria)},
            provider_score=provider_score,
            **common,
        )
    return BooleanAnswer(
        boolean=probabilities["true"] >= 0.5,
        probability_true=probabilities["true"],
        **common,
    )
