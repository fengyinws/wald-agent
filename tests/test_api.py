import httpx
import pytest
from conftest import chat_response

from wald_agent.api import create_app
from wald_agent.config import Settings
from wald_agent.engine import DecisionEngine
from wald_agent.sdk import WaldClient


async def test_http_api_and_sdk(settings, distributions, decision_request):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=chat_response(distributions))
        )
    ) as upstream:
        app = create_app(settings, DecisionEngine(settings, upstream))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http:
                result = await WaldClient("http://test", http_client=http).decide(decision_request)
                assert result.answers["department"].choice == "billing"
                assert (await http.get("/healthz")).json() == {"status": "ok"}
                assert "/v1/decide" in (await http.get("/openapi.json")).json()["paths"]
                invalid = decision_request.model_dump() | {"unrecognized": True}
                assert (await http.post("/v1/decide", json=invalid)).status_code == 422


async def test_missing_credentials_503(decision_request):
    app = create_app(Settings(_env_file=None))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            assert (await http.get("/healthz")).status_code == 200
            response = await http.post("/v1/decide", json=decision_request.model_dump())
            assert response.status_code == 503
            assert "LLM_API_KEY" in response.json()["detail"]


@pytest.mark.parametrize("simulate_timeout,expected", [(False, 502), (True, 504)])
async def test_gateway_errors(settings, decision_request, simulate_timeout, expected):
    def handler(request):
        if simulate_timeout:
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(429, json={"sensitive": "do not leak"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        app = create_app(settings, DecisionEngine(settings, upstream))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http:
                response = await http.post("/v1/decide", json=decision_request.model_dump())
                assert response.status_code == expected
                assert "sensitive" not in response.text


async def test_service_auth(settings, distributions, decision_request):
    from pydantic import SecretStr

    settings.wald_api_key = SecretStr("local-secret")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=chat_response(distributions))
        )
    ) as upstream:
        app = create_app(settings, DecisionEngine(settings, upstream))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http:
                body = decision_request.model_dump()
                assert (await http.post("/v1/decide", json=body)).status_code == 401
                response = await http.post(
                    "/v1/decide", json=body, headers={"Authorization": "Bearer local-secret"}
                )
                assert response.status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        '{"state":{"secret":NaN},"questions":{"cat":{"type":"boolean","instructions":"Cat?"}}}',
        '{"state":"test","questions":{},"review_threshold":Infinity}',
        '{"broken":',
    ],
)
async def test_invalid_json_and_nonfinite_input_returns_422(body):
    app = create_app(Settings(_env_file=None))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            response = await http.post(
                "/v1/decide", content=body, headers={"Content-Type": "application/json"}
            )
    assert response.status_code == 422
    assert "input" not in response.json()["detail"][0]


async def test_sdk_rejects_different_candidate_set(settings, distributions, decision_request):
    from wald_agent.errors import InvalidProviderResponse

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=chat_response(distributions))
        )
    ) as upstream:
        result = await DecisionEngine(settings, upstream).decide(decision_request)
    raw = result.model_dump()
    probabilities = raw["answers"]["department"]["probabilities"]
    probabilities["unrequested_option"] = probabilities.pop("technical")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=raw))
    ) as http:
        with pytest.raises(InvalidProviderResponse):
            await WaldClient(http_client=http).decide(decision_request)
