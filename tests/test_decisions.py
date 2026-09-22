import json

import httpx
import pytest
from conftest import chat_response, jev_response
from pydantic import ValidationError

from wald_agent.config import Settings
from wald_agent.engine import DecisionEngine
from wald_agent.errors import (
    ConfigurationError,
    InvalidProviderResponse,
    ProviderError,
    ProviderTimeout,
)
from wald_agent.jev import JevClient
from wald_agent.probability import concentration, validate_probabilities
from wald_agent.schemas import DecisionRequest, provider_questions


async def test_same_state_rubrics_and_all_three_types(settings, decision_request, distributions):
    observed = {}

    def handler(request):
        observed[request.url.host] = json.loads(request.content)
        if request.url.host == "llm.example":
            assert request.headers["authorization"] == "Bearer test-llm-secret"
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(200, json=chat_response(distributions))
        assert request.headers["authorization"] == "Bearer test-jev-secret"
        assert request.url.path == "/v1/systemone"
        return httpx.Response(200, json=jev_response(distributions))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        wald = await DecisionEngine(settings, http).decide(decision_request)
        jev = await JevClient(settings, http).decide(decision_request)
    sent_wald = json.loads(observed["llm.example"]["messages"][1]["content"])
    sent_jev = observed["jev.example"]
    assert sent_wald["state"] == sent_jev["state"] == decision_request.state
    assert sent_wald["questions"] == sent_jev["questions"]
    assert sent_jev["questions"]["refund_requested"]["type"] == "noul"
    assert "min_value" not in sent_jev["questions"]["urgency"]
    assert len(observed) == 2  # one call per provider, all questions together
    schema = observed["llm.example"]["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    for result in (wald, jev):
        assert result.answers["department"].choice == "billing"
        assert result.answers["refund_requested"].boolean is True
        assert result.answers["refund_requested"].probability_true == pytest.approx(0.9)
        assert result.answers["urgency"].score == pytest.approx(2.6)
        assert list(result.answers["urgency"].probabilities) == ["1", "2", "3"]
        assert result.latency_ms > 0
        assert result.calibration_status == "not_validated_on_your_data"
        assert result.usage.input_tokens == 100
    assert jev.answers["department"].provider_confidence == 0.75
    assert jev.answers["refund_requested"].provider_confidence is None
    assert jev.answers["urgency"].provider_score == pytest.approx(2.6)
    assert wald.answers["department"].confidence == jev.answers["department"].confidence
    assert wald.probability_source == "llm_self_report"


@pytest.mark.parametrize(
    "bad",
    [
        {"a": 0.8, "b": 0.8},
        {"a": -0.1, "b": 1.1},
        {"a": 0, "b": 0},
        {"a": float("nan"), "b": 1},
        {"a": float("inf"), "b": 0},
        {"a": True, "b": False},
        {"a": "0.5", "b": "0.5"},
        {"a": 0.5},
        {"a": 0.5, "c": 0.5},
        {"a": 0.5, "b": 0.5, "c": 0},
        [],
    ],
)
def test_bad_probabilities_rejected(bad):
    with pytest.raises(InvalidProviderResponse):
        validate_probabilities(bad, ["a", "b"])


def test_rounding_and_entropy():
    p = validate_probabilities({"a": 0.3333, "b": 0.3333, "c": 0.3333}, ["a", "b", "c"])
    assert sum(p.values()) == pytest.approx(1)
    assert concentration(p) == pytest.approx(0, abs=1e-12)
    assert concentration({"a": 1, "b": 0}) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(extra="forbidden"),
        lambda p: p.update(questions={}),
        lambda p: p["questions"]["department"].update(criteria={"one": "only"}),
        lambda p: p["questions"]["department"].update(unknown=True),
        lambda p: p["questions"]["refund_requested"].update(criteria={"true": "yes"}),
        lambda p: p["questions"]["urgency"].update(criteria=["one"]),
        lambda p: p["questions"]["urgency"].update(min_value="1"),
        lambda p: p.update(review_threshold=1.1),
        lambda p: p.update(state={"bad": float("nan")}),
    ],
)
def test_request_validation(request_data, mutation):
    mutation(request_data)
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(request_data)


def test_noul_alias_null_criteria_and_unicode(request_data):
    request_data["questions"]["refund_requested"]["type"] = "noul"
    request_data["questions"]["department"]["criteria"]["其他"] = None
    request = DecisionRequest.model_validate(request_data)
    assert provider_questions(request)["department"]["criteria"]["其他"] is None
    assert "信用卡" in request.state["message"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda body: body["choices"][0]["message"].update(refusal="refused"),
        lambda body: body["choices"][0].update(finish_reason="length"),
        lambda body: body["choices"][0]["message"].update(content="not JSON"),
        lambda body: body["choices"][0]["message"].update(content='{"answers": {}}'),
        lambda body: body["choices"][0].update(message="invalid"),
        lambda body: body.update(choices=[]),
        lambda body: body.update(usage="invalid"),
    ],
)
async def test_llm_failure_shapes(settings, decision_request, distributions, mutation):
    body = chat_response(distributions)
    mutation(body)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    ) as http:
        with pytest.raises(InvalidProviderResponse):
            await DecisionEngine(settings, http).decide(decision_request)


async def test_model_extra_output_and_wrong_sum_rejected(settings, decision_request, distributions):
    for bad in ("extra", "sum"):
        body = chat_response(distributions)
        content = json.loads(body["choices"][0]["message"]["content"])
        if bad == "extra":
            content["explanation"] = "not permitted"
        else:
            content["answers"]["refund_requested"]["probabilities"] = {"false": 0.9, "true": 0.9}
        body["choices"][0]["message"]["content"] = json.dumps(content)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req, body=body: httpx.Response(200, json=body))
        ) as http:
            with pytest.raises(InvalidProviderResponse):
                await DecisionEngine(settings, http).decide(decision_request)


async def test_json_mode_is_explicit(settings, decision_request, distributions):
    settings.llm_response_format = "json_object"
    settings.llm_token_limit_field = "max_tokens"

    def handler(request):
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert "Return JSON matching" in body["messages"][-1]["content"]
        assert body["max_tokens"] == 2048
        return httpx.Response(200, json=chat_response(distributions))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await DecisionEngine(settings, http).decide(decision_request)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw["answers"].pop("urgency"),
        lambda raw: raw["answers"]["department"].update(choice="technical"),
        lambda raw: raw["answers"]["department"].update(choice=""),
        lambda raw: raw["answers"]["refund_requested"].update(noul=1.1),
        lambda raw: raw["answers"]["urgency"].update(score=9),
        lambda raw: raw["answers"]["urgency"].update(score=0),
        lambda raw: raw["answers"].update(extra={"type": "noul", "noul": 1}),
    ],
)
async def test_invalid_jev_results(settings, decision_request, distributions, mutation):
    body = jev_response(distributions)
    mutation(body)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    ) as http:
        with pytest.raises(InvalidProviderResponse):
            await JevClient(settings, http).decide(decision_request)


@pytest.mark.parametrize("status", [401, 429, 500, 529])
async def test_upstream_errors_sanitized_no_retry(settings, decision_request, status):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="private raw body test-llm-secret")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ProviderError) as error:
            await DecisionEngine(settings, http).decide(decision_request)
    assert str(status) in str(error.value)
    assert "secret" not in str(error.value) and "private" not in str(error.value)
    assert calls == 1


async def test_timeout_and_missing_key(settings, decision_request):
    def handler(request):
        raise httpx.ReadTimeout("private URL", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ProviderTimeout):
            await DecisionEngine(settings, http).decide(decision_request)
        settings.llm_api_key = None
        with pytest.raises(ConfigurationError):
            await DecisionEngine(settings, http).decide(decision_request)


def test_config_key_fallback_and_redaction(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "hidden-value")
    monkeypatch.setenv("LLM_API_KEY", "")
    settings = Settings(_env_file=None)
    assert settings.key_for("llm") == "hidden-value"
    assert "hidden-value" not in repr(settings)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_base_url="https://user:password@example.com")
