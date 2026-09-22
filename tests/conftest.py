import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from wald_agent.api import create_app
from wald_agent.config import Settings
from wald_agent.engine import DecisionEngine
from wald_agent.jev import JevClient
from wald_agent.llm import ChatClient
from wald_agent.schemas import DecisionRequest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "wald.sqlite3"))


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        llm_api_key="test-llm-secret",
        typesafe_api_key="test-jev-secret",
        llm_base_url="https://llm.example/v1",
        jev_base_url="https://jev.example/v1",
        api_max_retries=0,
        enabled_providers=["llm", "jev"],
    )


@pytest.fixture
def request_data():
    return json.loads((ROOT / "examples/customer_service.json").read_text())


@pytest.fixture
def decision_request(request_data):
    return DecisionRequest.model_validate(request_data)


@pytest.fixture
def distributions():
    return {
        "department": {"billing": 0.8, "technical": 0.1, "sales": 0.05, "other": 0.05},
        "refund_requested": {"false": 0.1, "true": 0.9},
        "urgency": {"0": 0.1, "1": 0.2, "2": 0.7},
    }


@pytest.fixture
def api_factory(settings, distributions):
    @asynccontextmanager
    async def running(handler=None):
        handler = handler or (lambda req: httpx.Response(200, json=chat_response(distributions)))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
            app = create_app(
                settings,
                DecisionEngine(settings, upstream),
                JevClient(settings, upstream),
                ChatClient(settings, upstream, vision=True),
            )
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://wald.test"
                ) as client:
                    yield app, client

    return running


def chat_response(distributions, **overrides):
    body = {
        "model": "test-llm-version",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "answers": {
                                name: {"probabilities": p} for name, p in distributions.items()
                            }
                        }
                    )
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 60},
    }
    body.update(overrides)
    return body


def jev_response(distributions):
    return {
        "model": "jev-test-version",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.75,
                "probabilities": distributions["department"],
            },
            "refund_requested": {"type": "noul", "noul": distributions["refund_requested"]["true"]},
            "urgency": {
                "type": "score",
                "score": 1.6,
                "confidence": 0.65,
                "probabilities": distributions["urgency"],
                "legend": {"0": "routine", "1": "soon", "2": "today"},
            },
        },
        "usage": {"input_tokens": 100, "output_tokens": 60},
    }
