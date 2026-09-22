import base64
import json

import httpx
import pytest

from wald_agent.errors import InvalidProviderResponse
from wald_agent.jev import JevClient
from wald_agent.llm import ChatClient
from wald_agent.vision import VisualFacts, judge_image, local_image_url, remote_image_url

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)


def facts_dict(uncertainties=None):
    return {
        "summary": "A cat on a sofa",
        "observations": ["Cat with orange fur"],
        "visible_text": [],
        "uncertainties": uncertainties or [],
        "image_quality": "clear",
    }


def image_jev_response():
    return {
        "model": "jev-test",
        "usage": {"input_tokens": 123, "output_tokens": 20},
        "answers": {
            "main_subject": {
                "type": "choice",
                "choice": "cat",
                "confidence": 1.0,
                "probabilities": {"cat": 1.0, "dog": 0.0, "other": 0.0, "unclear": 0.0},
            },
            "contains_cat": {"type": "noul", "noul": 1.0},
        },
    }


async def test_vision_then_real_jev_protocol(settings, tmp_path):
    path = tmp_path / "test.png"
    path.write_bytes(PNG)
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(request.url.path)
        if request.url.path.endswith("chat/completions"):
            image = body["messages"][1]["content"][1]
            assert image["type"] == "image_url"
            assert image["image_url"]["url"].startswith("data:image/png;base64,")
            return httpx.Response(
                200,
                json={
                    "model": "vision-test",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    facts_dict(["Small part of the scene is hidden"])
                                )
                            },
                        }
                    ],
                },
            )
        assert body["state"]["visual_facts"]["summary"] == "A cat on a sofa"
        assert "base64" not in json.dumps(body)
        assert body["questions"]["contains_cat"]["type"] == "noul"
        return httpx.Response(200, json=image_jev_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await judge_image(
            JevClient(settings, http),
            image_url=local_image_url(path),
            vision=ChatClient(settings, http, vision=True),
        )
    assert calls == ["/v1/chat/completions", "/v1/systemone"]
    assert result["native_jev_vision"] is False
    assert result["mode"] == "vision_then_jev"
    assert result["jev"]["answers"]["contains_cat"]["boolean"] is True
    assert result["needs_review"] is True  # visual ambiguity survives a confident Jev result
    assert "uncertain_visual_evidence" in result["review_reasons"]
    assert result["timing_ms"]["pipeline_total"] >= (
        result["timing_ms"]["jev"] + result["timing_ms"]["vision"]
    )


async def test_description_mode_uses_only_jev(settings):
    def handler(request):
        assert request.url.path == "/v1/systemone"
        return httpx.Response(200, json=image_jev_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await judge_image(JevClient(settings, http), facts=VisualFacts(**facts_dict()))
    assert result["vision"] is None
    assert result["timing_ms"]["vision"] == 0
    assert result["needs_review"] is False


async def test_failed_vision_never_calls_jev(settings):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "invalid JSON"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(InvalidProviderResponse):
            await judge_image(
                JevClient(settings, http),
                image_url="https://example.com/cat.png",
                vision=ChatClient(settings, http, vision=True),
            )
    assert calls == ["/v1/chat/completions"]


def test_image_inputs(tmp_path):
    image = tmp_path / "test.png"
    image.write_bytes(PNG)
    encoded = local_image_url(image)
    assert base64.b64decode(encoded.split(",", 1)[1]) == PNG
    image.write_text("this is not a picture")
    with pytest.raises(ValueError, match="PNG"):
        local_image_url(image)
    with pytest.raises(ValueError):
        remote_image_url("file:///private/file.png")
    with pytest.raises(ValueError):
        remote_image_url("https://user:password@example.com/picture.jpg")


async def test_image_arguments_validated_before_network(settings):
    def handler(request):
        pytest.fail("invalid image args must not make API calls")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        jev = JevClient(settings, http)
        with pytest.raises(ValueError):
            await judge_image(jev)
        with pytest.raises(ValueError):
            await judge_image(
                jev, facts=VisualFacts(**facts_dict()), image_url="https://x.test/a.png"
            )
        with pytest.raises(ValueError):
            await judge_image(jev, facts=VisualFacts(**facts_dict()), review_threshold=1.1)
