import json

import httpx
import pytest
from conftest import chat_response, jev_response
from pydantic import ValidationError

from wald_agent.benchmark import BenchmarkCase, load_cases, render_summary, run_benchmark, summarize
from wald_agent.engine import DecisionEngine
from wald_agent.errors import ProviderError
from wald_agent.jev import JevClient
from wald_agent.metrics import compare_answers, labeled_metrics, percentile
from wald_agent.probability import make_answer
from wald_agent.schemas import ChoiceQuestion, DecisionRequest


def test_known_metrics():
    question = ChoiceQuestion(type="choice", instructions="Pick", criteria={"a": "A", "b": "B"})
    answer = make_answer(question, {"a": 0.8, "b": 0.2}, 0.5)
    result = labeled_metrics([(answer, "a"), (answer, "b")])
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["brier_multiclass"] == pytest.approx(0.68)
    assert result["ece_10_bins"] == pytest.approx(0.3)
    assert percentile([10, 20, 30], 0.5) == 20
    assert percentile([10, 20, 30], 0.95) == 29
    assert percentile([], 0.5) is None


def test_score_and_probability_difference(decision_request):
    question = decision_request.questions["urgency"]
    left = make_answer(question, {"0": 1, "1": 0, "2": 0}, 0.5)
    right = make_answer(question, {"0": 0, "1": 0, "2": 1}, 0.5)
    difference = compare_answers(left, right, score_tolerance=0.25)
    assert difference["wald_value"] == 1
    assert difference["jev_value"] == 3
    assert difference["score_absolute_difference"] == 2
    assert difference["total_variation_distance"] == 1
    assert difference["agrees"] is False
    assert labeled_metrics([(left, "3")])["score_mae"] == 2


async def test_benchmark_warmup_and_paired_results(settings, decision_request, distributions):
    counts = {"wald": 0, "jev": 0}

    def handler(request):
        name = "wald" if request.url.host == "llm.example" else "jev"
        counts[name] += 1
        body = chat_response(distributions) if name == "wald" else jev_response(distributions)
        return httpx.Response(200, json=body)

    case = BenchmarkCase(
        id="same-input",
        request=decision_request,
        expected={"department": "billing", "refund_requested": True, "urgency": 3},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await run_benchmark(
            [case], DecisionEngine(settings, http), JevClient(settings, http), runs=3, warmup=1
        )
    assert counts == {"wald": 4, "jev": 4}
    assert len(report["warmups"]) == 1 and len(report["records"]) == 3
    assert report["summary"]["comparison"]["paired_successful_requests"] == 3
    assert report["summary"]["comparison"]["decision_agreement_rate"] == 1
    assert report["summary"]["wald"]["metrics_by_question"]["department"]["accuracy"] == 1
    assert report["summary"]["wald"]["metrics_by_question"]["urgency"][
        "score_mae"
    ] == pytest.approx(0.4)
    assert (
        report["summary"]["wald"]["repeat_modal_label_agreement"]["same-input"]["department"] == 1
    )
    assert "P50(ms)" in render_summary(report)
    # The report can be saved without custom JSON encoders, NaN or Infinity.
    json.dumps(report, allow_nan=False)


async def test_failures_not_counted_as_fast_correct_responses(
    settings, decision_request, distributions
):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=chat_response(distributions))

    class BrokenJev:
        async def decide(self, request):
            raise ProviderError("jev failed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await run_benchmark(
            [BenchmarkCase(id="a", request=decision_request)],
            DecisionEngine(settings, http),
            BrokenJev(),
            runs=2,
            warmup=0,
        )
    assert report["summary"]["wald"]["successes"] == 1
    assert report["summary"]["wald"]["error_rate"] == 0.5
    assert report["summary"]["jev"]["successful_latency_ms"]["p50"] is None
    assert report["summary"]["comparison"]["paired_p50_wald_over_jev"] is None
    assert report["summary"]["comparison"]["decision_agreement_rate"] is None
    assert all("differences" not in row for row in report["records"])


def test_paired_ratio_ignores_unpaired_fast_success(settings, decision_request, distributions):
    # A successful but unpaired 1 ms request must not make Wald's speed ratio look better.
    from wald_agent.schemas import DecisionResponse

    answers = {
        name: make_answer(question, distributions[name], 0.5)
        for name, question in decision_request.questions.items()
    }
    response = DecisionResponse(
        provider="llm",
        model="fake",
        answers=answers,
        needs_review=True,
        latency_ms=1,
        probability_source="llm_self_report",
    ).model_dump()
    rows = [
        {
            "case_id": "a",
            "wald": {"ok": True, "elapsed_ms": 100, "response": response},
            "jev": {"ok": True, "elapsed_ms": 20, "response": response},
        },
        {
            "case_id": "a",
            "wald": {"ok": True, "elapsed_ms": 1, "response": response},
            "jev": {"ok": False, "elapsed_ms": 3},
        },
    ]
    summary = summarize(rows, [BenchmarkCase(id="a", request=decision_request)])
    assert summary["comparison"]["paired_p50_wald_over_jev"] == 5


@pytest.mark.parametrize(
    "expected",
    [
        {"unknown": True},
        {"department": "invalid"},
        {"refund_requested": "true"},
        {"urgency": 0},
        {"urgency": True},
    ],
)
def test_ground_truth_validation(decision_request, expected):
    with pytest.raises(ValidationError):
        BenchmarkCase(id="a", request=decision_request, expected=expected)


def test_dataset_loader_rejects_duplicate_ids_and_mixed_rubrics(tmp_path, decision_request):
    path = tmp_path / "cases.jsonl"
    case = BenchmarkCase(id="a", request=decision_request)
    path.write_text(case.model_dump_json() + "\n" + case.model_dump_json())
    with pytest.raises(ValueError, match="unique"):
        load_cases(None, path)
    modified = decision_request.model_dump()
    modified["questions"]["urgency"]["min_value"] = 0
    case2 = BenchmarkCase(id="b", request=DecisionRequest.model_validate(modified))
    path.write_text(case.model_dump_json() + "\n" + case2.model_dump_json())
    with pytest.raises(ValueError, match="different definitions"):
        load_cases(None, path)


def test_example_datasets_validate():
    from conftest import ROOT

    cases = load_cases(None, ROOT / "examples/customer_service_cases.jsonl")
    assert len(cases) == 6
    assert all(len(case.expected) == 3 for case in cases)
