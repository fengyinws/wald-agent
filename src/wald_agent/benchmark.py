import argparse
import asyncio
import random
import sys
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Protocol

from pydantic import Field, ValidationError, model_validator

from wald_agent.cli_errors import safe_cli_error
from wald_agent.config import Settings
from wald_agent.engine import DecisionEngine
from wald_agent.errors import WaldError
from wald_agent.files import example_text, write_json
from wald_agent.jev import JevClient
from wald_agent.metrics import (
    answer_label,
    compare_answers,
    labeled_metrics,
    percentile,
    repeat_agreement,
)
from wald_agent.schemas import (
    BooleanQuestion,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    StrictModel,
    probability_keys,
)
from wald_agent.sdk import WaldClient


class DecisionProvider(Protocol):
    async def decide(self, request: DecisionRequest) -> DecisionResponse: ...


class BenchmarkCase(StrictModel):
    id: str = Field(min_length=1)
    request: DecisionRequest
    expected: dict[str, str | bool | int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_labels(self) -> "BenchmarkCase":
        for name, label in self.expected.items():
            if name not in self.request.questions:
                raise ValueError("expected labels must reference existing questions")
            question = self.request.questions[name]
            if isinstance(question, BooleanQuestion):
                if type(label) is not bool:
                    raise ValueError("boolean ground truth must be a JSON boolean")
            elif isinstance(question, ChoiceQuestion):
                if not isinstance(label, str) or label not in question.criteria:
                    raise ValueError("choice ground truth must be a candidate key")
            elif type(label) is not int or str(label) not in probability_keys(question):
                raise ValueError("score ground truth must be an integer on the requested scale")
        return self


def load_cases(input_path: Path | None, dataset_path: Path | None) -> list[BenchmarkCase]:
    if dataset_path:
        cases = [
            BenchmarkCase.model_validate_json(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        content = (
            input_path.read_text(encoding="utf-8")
            if input_path
            else example_text("customer_service.json")
        )
        cases = [
            BenchmarkCase(
                id=input_path.stem if input_path else "customer_service",
                request=DecisionRequest.model_validate_json(content),
            )
        ]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("dataset must be nonempty and case IDs must be unique")
    # Metrics are per question ID: disallow accidentally mixing different rubrics/scales.
    rubrics = {}
    for case in cases:
        for name, question in case.request.questions.items():
            serialized = question.model_dump()
            if name in rubrics and rubrics[name] != serialized:
                raise ValueError(f"question {name!r} has different definitions across cases")
            rubrics[name] = serialized
    return cases


async def measured_call(provider: DecisionProvider, request: DecisionRequest) -> dict:
    started = perf_counter()
    try:
        result = await provider.decide(request)
        elapsed = (perf_counter() - started) * 1000
        return {"ok": True, "elapsed_ms": elapsed, "response": result.model_dump(mode="json")}
    except WaldError as exc:
        return {
            "ok": False,
            "elapsed_ms": (perf_counter() - started) * 1000,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def summarize(records: list[dict], cases: list[BenchmarkCase]) -> dict:
    summary = {}
    case_map = {case.id: case for case in cases}
    for provider in ("wald", "jev"):
        calls = [record[provider] for record in records]
        successes = [call for call in calls if call["ok"]]
        latency = [call["elapsed_ms"] for call in successes]
        active_ms = sum(call["elapsed_ms"] for call in calls)
        response_objects = [DecisionResponse.model_validate(call["response"]) for call in successes]
        answer_count = sum(len(response.answers) for response in response_objects)
        review_count = sum(
            answer.needs_review
            for response in response_objects
            for answer in response.answers.values()
        )
        labeled: dict[str, list] = {}
        repeated: dict[str, dict[str, list[str]]] = {}
        for record in records:
            if not record[provider]["ok"]:
                continue
            response = DecisionResponse.model_validate(record[provider]["response"])
            case = case_map[record["case_id"]]
            for name, answer in response.answers.items():
                repeated.setdefault(case.id, {}).setdefault(name, []).append(answer_label(answer))
                if name in case.expected:
                    label = case.expected[name]
                    label = str(label).lower() if type(label) is bool else str(label)
                    labeled.setdefault(name, []).append((answer, label))
        summary[provider] = {
            "attempts": len(calls),
            "successes": len(successes),
            "failures": len(calls) - len(successes),
            "error_rate": (len(calls) - len(successes)) / len(calls) if calls else None,
            "successful_latency_ms": {
                "mean": mean(latency) if latency else None,
                "p50": percentile(latency, 0.5),
                "p95": percentile(latency, 0.95),
            },
            "serial_successful_requests_per_active_second": (
                len(successes) * 1000 / active_ms if active_ms else None
            ),
            "question_review_rate": review_count / answer_count if answer_count else None,
            "request_review_rate": (
                mean(response.needs_review for response in response_objects)
                if response_objects
                else None
            ),
            "models_observed": sorted({response.model for response in response_objects}),
            "metrics_by_question": {
                name: labeled_metrics(items) for name, items in labeled.items()
            },
            "repeat_modal_label_agreement": {
                case_id: {name: repeat_agreement(labels) for name, labels in questions.items()}
                for case_id, questions in repeated.items()
            },
        }
    differences = [item for record in records for item in record.get("differences", {}).values()]
    # Speed ratios use only paired successes so failures cannot skew the comparison populations.
    pairs = [record for record in records if record["wald"]["ok"] and record["jev"]["ok"]]
    wald_ms = [record["wald"]["elapsed_ms"] for record in pairs]
    jev_ms = [record["jev"]["elapsed_ms"] for record in pairs]
    summary["comparison"] = {
        "paired_successful_requests": len(pairs),
        "compared_questions": len(differences),
        "decision_agreement_rate": mean(d["agrees"] for d in differences) if differences else None,
        "mean_total_variation_distance": (
            mean(d["total_variation_distance"] for d in differences) if differences else None
        ),
        "paired_p50_wald_over_jev": (
            percentile(wald_ms, 0.5) / percentile(jev_ms, 0.5)
            if pairs and percentile(jev_ms, 0.5) > 0
            else None
        ),
    }
    return summary


async def run_benchmark(
    cases: list[BenchmarkCase],
    wald: DecisionProvider,
    jev: DecisionProvider,
    *,
    runs: int = 3,
    warmup: int = 1,
    seed: int = 42,
    score_tolerance: float = 0.25,
    progress: bool = False,
    retry_policy: int | None = 0,
) -> dict:
    if not cases or runs < 1 or warmup < 0 or not 0 <= score_tolerance < float("inf"):
        raise ValueError("need cases, runs >= 1, warmup >= 0, and finite score tolerance >= 0")
    providers = {"wald": wald, "jev": jev}
    rng = random.Random(seed)
    warmups, records = [], []
    # Warm each case/schema. Warmups are reported separately and excluded from measurements.
    for iteration in range(warmup):
        for case in cases:
            order = list(providers)
            rng.shuffle(order)
            row = {"case_id": case.id, "iteration": iteration, "order": order}
            for name in order:
                row[name] = await measured_call(providers[name], case.request)
            warmups.append(row)
    started = perf_counter()
    for iteration in range(runs):
        case_order = list(cases)
        rng.shuffle(case_order)
        for case in case_order:
            order = list(providers)
            rng.shuffle(order)
            row = {"case_id": case.id, "iteration": iteration, "order": order}
            for name in order:
                row[name] = await measured_call(providers[name], case.request)
                if progress:
                    call = row[name]
                    status = "ok" if call["ok"] else call["error"]["type"]
                    print(
                        f"[{iteration + 1}/{runs}] {case.id} {name}: "
                        f"{call['elapsed_ms']:.1f} ms ({status})",
                        file=sys.stderr,
                    )
            if row["wald"]["ok"] and row["jev"]["ok"]:
                left = DecisionResponse.model_validate(row["wald"]["response"])
                right = DecisionResponse.model_validate(row["jev"]["response"])
                row["differences"] = {
                    name: compare_answers(left.answers[name], right.answers[name], score_tolerance)
                    for name in case.request.questions
                }
            records.append(row)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "runs_per_case": runs,
            "warmups_per_case": warmup,
            "seed": seed,
            "score_tolerance": score_tolerance,
            "concurrency": 1,
            "ordering": "seeded random provider order within each pair; shuffled cases per run",
            "retries": retry_policy,
            "measured_wall_seconds": perf_counter() - started,
            "latency_scope": (
                "client-observed full request + network + validation; successful calls"
            ),
            "throughput_scope": (
                "serial requests / provider active time including failures; not load capacity"
            ),
            "caveats": [
                "Agreement is not accuracy. Accuracy metrics require expected labels.",
                "Repeated calls are correlated; small samples cannot establish calibration or P95.",
                "Provider prompt/schema caches are not controlled; warmups are excluded.",
                "Both use identical state and rubrics; model prompts and wire formats differ.",
                "Wald confidence is entropy concentration, not Jev's proprietary confidence.",
                "Macro-F1 uses per-question observed classes. Brier uses all probabilities.",
                "ECE uses predicted-class probability and 10 equal-width bins, not entropy.",
            ],
        },
        "cases": [case.model_dump(mode="json") for case in cases],
        "warmups": warmups,
        "records": records,
        "summary": summarize(records, cases),
    }


def render_summary(report: dict) -> str:
    def number(value):
        return "n/a" if value is None else f"{value:.2f}"

    lines = ["provider  success/attempts  P50(ms)  P95(ms)  error_rate"]
    for provider in ("wald", "jev"):
        summary = report["summary"][provider]
        latency = summary["successful_latency_ms"]
        lines.append(
            f"{provider:8}  {summary['successes']}/{summary['attempts']}"
            f"  {number(latency['p50'])}  {number(latency['p95'])}"
            f"  {number(summary['error_rate'])}"
        )
    comparison = report["summary"]["comparison"]
    lines.append(f"Decision agreement: {number(comparison['decision_agreement_rate'])}")
    lines.append(f"Paired P50 Wald/Jev ratio: {number(comparison['paired_p50_wald_over_jev'])}")
    for row in report["records"]:
        for name, delta in row.get("differences", {}).items():
            lines.append(
                f"{row['case_id']} run={row['iteration'] + 1} {name}: "
                f"Wald={delta['wald_value']} Jev={delta['jev_value']} "
                f"agree={delta['agrees']} TV={delta['total_variation_distance']:.4f}"
            )
    return "\n".join(lines)


async def run_from_args(args) -> dict:
    cases = load_cases(args.input, args.dataset)
    settings = Settings(api_max_retries=args.retries)
    # Fail before any paid calls if a required credential is absent.
    settings.key_for("jev")
    if not args.wald_url:
        settings.key_for("llm")
    async with AsyncExitStack() as stack:
        if args.wald_url:
            key = settings.wald_api_key.get_secret_value() if settings.wald_api_key else ""
            wald = await stack.enter_async_context(
                WaldClient(args.wald_url, key, timeout=settings.request_timeout_seconds + 5)
            )
        else:
            wald = await stack.enter_async_context(DecisionEngine(settings))
        jev = await stack.enter_async_context(JevClient(settings))
        report = await run_benchmark(
            cases,
            wald,
            jev,
            runs=args.runs,
            warmup=args.warmup,
            seed=args.seed,
            score_tolerance=args.score_tolerance,
            progress=True,
            retry_policy=None if args.wald_url else args.retries,
        )
    report["protocol"]["wald_mode"] = "http" if args.wald_url else "in_process"
    report["protocol"]["requested_models"] = {
        "wald": None if args.wald_url else settings.llm_model,
        "jev": settings.jev_model,
    }
    report["protocol"]["llm_response_format"] = (
        None if args.wald_url else settings.llm_response_format
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Wald and Jev on the SAME state and questions."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input", type=Path, help="DecisionRequest JSON (default: example request)"
    )
    source.add_argument("--dataset", type=Path, help="JSONL: id, request, optional expected labels")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--warmup", type=int, default=1, help="Excluded calls per provider per case"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        choices=range(6),
        help="Direct provider retries; default 0",
    )
    parser.add_argument("--score-tolerance", type=float, default=0.25)
    parser.add_argument(
        "--wald-url", help="Measure a running Wald HTTP API instead of in-process SDK"
    )
    parser.add_argument("--output", type=Path, default=Path("reports/comparison.json"))
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or not 0 <= args.score_tolerance < float("inf"):
        parser.error("runs >= 1, warmup >= 0, and finite score-tolerance >= 0 are required")
    try:
        report = asyncio.run(run_from_args(args))
        write_json(args.output, report)
    except (WaldError, OSError, ValueError, ValidationError) as exc:
        parser.exit(2, f"Benchmark failed: {safe_cli_error(exc)}\n")
    print(render_summary(report))
    print(f"Full report: {args.output.resolve()}")
    if any(report["summary"][p]["failures"] for p in ("wald", "jev")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
