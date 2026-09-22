import math
from collections import Counter
from statistics import mean

from wald_agent.schemas import Answer, BooleanAnswer, ChoiceAnswer, ScoreAnswer


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def answer_value(answer: Answer) -> str | bool | float:
    if isinstance(answer, ChoiceAnswer):
        return answer.choice
    if isinstance(answer, BooleanAnswer):
        return answer.boolean
    return answer.score


def answer_label(answer: Answer) -> str:
    if isinstance(answer, ChoiceAnswer):
        return answer.choice
    if isinstance(answer, BooleanAnswer):
        return str(answer.boolean).lower()
    return max(answer.probabilities, key=answer.probabilities.__getitem__)


def compare_answers(left: Answer, right: Answer, score_tolerance: float) -> dict:
    if left.type != right.type or set(left.probabilities) != set(right.probabilities):
        raise ValueError("cannot compare answers on different scales")
    lv, rv = answer_value(left), answer_value(right)
    delta = abs(lv - rv) if isinstance(left, ScoreAnswer) else None
    return {
        "type": left.type,
        "wald_value": lv,
        "jev_value": rv,
        "agrees": delta <= score_tolerance if delta is not None else lv == rv,
        "score_absolute_difference": delta,
        "probability_true_absolute_difference": (
            abs(left.probabilities["true"] - right.probabilities["true"])
            if isinstance(left, BooleanAnswer)
            else None
        ),
        "total_variation_distance": 0.5
        * sum(abs(left.probabilities[k] - right.probabilities[k]) for k in left.probabilities),
        "probability_deltas_wald_minus_jev": {
            k: left.probabilities[k] - right.probabilities[k] for k in left.probabilities
        },
        "confidence_absolute_difference": abs(left.confidence - right.confidence),
        "review_agrees": left.needs_review == right.needs_review,
    }


def labeled_metrics(samples: list[tuple[Answer, str]], bins: int = 10) -> dict:
    """Per-question metrics. Brier = multiclass sum of squared errors (range 0..2)."""
    if not samples:
        return {"labeled_observations": 0}
    predictions = [answer_label(answer) for answer, _ in samples]
    targets = [label for _, label in samples]
    correct = [prediction == label for prediction, label in zip(predictions, targets, strict=True)]
    # Exclude wholly absent classes, as in common macro-F1 implementations.
    classes = set(predictions) | set(targets)
    f1 = []
    for label in classes:
        tp = sum(p == label and t == label for p, t in zip(predictions, targets, strict=True))
        fp = sum(p == label and t != label for p, t in zip(predictions, targets, strict=True))
        fn = sum(p != label and t == label for p, t in zip(predictions, targets, strict=True))
        f1.append(2 * tp / (2 * tp + fp + fn))
    brier = mean(
        sum((p - float(k == label)) ** 2 for k, p in answer.probabilities.items())
        for answer, label in samples
    )
    # ECE uses predicted-class probability, not Wald's entropy concentration.
    confidences = [
        answer.probabilities[p] for (answer, _), p in zip(samples, predictions, strict=True)
    ]
    ece = 0.0
    for bin_id in range(bins):
        indices = [i for i, p in enumerate(confidences) if min(int(p * bins), bins - 1) == bin_id]
        if indices:
            ece += (
                len(indices)
                / len(samples)
                * abs(mean(confidences[i] for i in indices) - mean(correct[i] for i in indices))
            )
    metrics = {
        "labeled_observations": len(samples),
        "accuracy": mean(correct),
        "macro_f1": mean(f1),
        "brier_multiclass": brier,
        "ece_10_bins": ece,
        "observed_classes": sorted(classes),
    }
    score_errors = [
        abs(answer.score - int(label))
        for answer, label in samples
        if isinstance(answer, ScoreAnswer)
    ]
    if score_errors:
        metrics["score_mae"] = mean(score_errors)
    return metrics


def repeat_agreement(labels: list[str]) -> float | None:
    """Fraction matching the modal label; unavailable for a single observation."""
    return max(Counter(labels).values()) / len(labels) if len(labels) >= 2 else None
