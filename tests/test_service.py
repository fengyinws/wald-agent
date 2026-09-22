import asyncio
import json

import httpx
import pytest
from conftest import chat_response
from pydantic import SecretStr, ValidationError

from wald_agent.config import Settings
from wald_agent.errors import ConfigurationError, RemoteServiceError, WaldError
from wald_agent.schemas import BatchDecisionRequest, ReviewResolution
from wald_agent.sdk import WaldClient
from wald_agent.storage import DecisionStore
from wald_agent.vision import ImageDecisionRequest


async def test_idempotency_survives_restart_and_rejects_changed_input(
    api_factory, distributions, request_data
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=chat_response(distributions))

    headers = {"Idempotency-Key": "order-123", "X-Request-ID": "first-request"}
    async with api_factory(handler) as (_, http):
        first = await http.post("/v1/decide", json=request_data, headers=headers)
        assert first.status_code == 200
        assert first.headers["x-request-id"] == "first-request"
    async with api_factory(handler) as (_, http):
        replay = await http.post(
            "/v1/decide", json=request_data, headers=headers | {"X-Request-ID": "replay"}
        )
        assert replay.json()["decision_id"] == first.json()["decision_id"]
        assert replay.json()["request_id"] == "replay"
        record = await http.get(f"/v1/decisions/{first.json()['decision_id']}")
        assert record.json()["result"]["request_id"] == "first-request"
        assert record.json()["status"] == "succeeded"
        conflict = await http.post(
            "/v1/decide", json=request_data | {"state": "changed"}, headers=headers
        )
        assert conflict.status_code == 409
    assert len(calls) == 1


async def test_simultaneous_duplicate_does_not_call_provider_twice(
    api_factory, distributions, request_data
):
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def handler(request):
        calls.append(request)
        started.set()
        await release.wait()
        return httpx.Response(200, json=chat_response(distributions))

    async with api_factory(handler) as (_, http):
        first = asyncio.create_task(
            http.post("/v1/decide", json=request_data, headers={"Idempotency-Key": "same"})
        )
        await started.wait()
        conflict = await http.post(
            "/v1/decide", json=request_data, headers={"Idempotency-Key": "same"}
        )
        release.set()
        assert (await first).status_code == 200
        assert conflict.status_code == 409
    assert len(calls) == 1


async def test_failed_request_is_durable_and_sdk_preserves_error(api_factory, decision_request):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="PRIVATE RAW ERROR", headers={"Retry-After": "1"})

    async with api_factory(handler) as (_, http):
        sdk = WaldClient("http://wald.test", http_client=http)
        errors = []
        for _ in range(2):
            with pytest.raises(RemoteServiceError) as failure:
                await sdk.decide(decision_request, idempotency_key="failed-call")
            errors.append(failure.value)
        assert errors[0].decision_id == errors[1].decision_id
        assert errors[0].code == "upstream_error"
        assert errors[0].status_code == 502
        assert errors[0].request_id
        assert errors[0].retry_after == 1
        record = await sdk.get_decision(errors[0].decision_id)
        assert record.status == "failed"
        assert "PRIVATE" not in str(record)
    assert len(calls) == 1


async def test_clients_have_isolated_records_keys_and_review_queues(
    api_factory, settings, decision_request
):
    settings.wald_api_keys = {"alice": SecretStr("alice-key"), "bob": SecretStr("bob-key")}
    async with api_factory() as (_, http):
        alice = WaldClient("http://wald.test", "alice-key", http_client=http)
        bob = WaldClient("http://wald.test", "bob-key", http_client=http)
        first = await alice.decide(decision_request, idempotency_key="same-key")
        second = await bob.decide(decision_request, idempotency_key="same-key")
        assert first.decision_id != second.decision_id
        with pytest.raises(RemoteServiceError) as failure:
            await bob.get_decision(first.decision_id)
        assert failure.value.status_code == 404
        assert [item.id for item in (await alice.list_reviews()).items] == [first.decision_id]
        assert [item.id for item in (await bob.list_reviews()).items] == [second.decision_id]


async def test_review_validation_pagination_and_concurrent_updates(api_factory, decision_request):
    async with api_factory() as (_, http):
        sdk = WaldClient("http://wald.test", http_client=http)
        decisions = [await sdk.decide(decision_request) for _ in range(3)]
        first = await sdk.list_reviews(limit=2)
        second = await sdk.list_reviews(limit=2, after=first.next_cursor)
        assert [item.id for item in first.items + second.items] == [
            item.decision_id for item in decisions
        ]
        assert second.next_cursor is None
        identifier = decisions[0].decision_id
        bad = {"department": "billing", "refund_requested": "true", "urgency": 1}
        with pytest.raises(RemoteServiceError) as failure:
            await sdk.resolve_review(identifier, ReviewResolution(answers=bad, reviewer="reviewer"))
        assert failure.value.status_code == 422
        correction = ReviewResolution(
            answers=bad | {"refund_requested": True},
            reviewer="reviewer",
            notes="Checked original invoice",
        )
        outcomes = await asyncio.gather(
            sdk.resolve_review(identifier, correction),
            sdk.resolve_review(identifier, correction),
            return_exceptions=True,
        )
        assert sum(isinstance(item, RemoteServiceError) for item in outcomes) == 1
        conflict = next(item for item in outcomes if isinstance(item, RemoteServiceError))
        assert conflict.status_code == 409
        reviewed = (await sdk.list_reviews(resolved=True)).items[0]
        assert reviewed.revision == 1
        assert reviewed.resolution["answers"]["refund_requested"] is True
        assert len((await sdk.list_reviews()).items) == 2


async def test_input_retention_can_be_disabled(api_factory, settings, decision_request):
    settings.audit_store_inputs = False
    async with api_factory() as (_, http):
        sdk = WaldClient("http://wald.test", http_client=http)
        result = await sdk.decide(decision_request)
        record = await sdk.get_decision(result.decision_id)
        assert "state" not in record.request
        assert "questions" in record.request


async def test_batch_partial_failures_order_and_individual_review(
    api_factory, decision_request, distributions
):
    def handler(request):
        state = json.loads(json.loads(request.content)["messages"][1]["content"])["state"]
        if state == "provider failure":
            return httpx.Response(503)
        return httpx.Response(200, json=chat_response(distributions))

    batch = BatchDecisionRequest.model_validate(
        {
            "items": [
                {"id": "good", "request": decision_request.model_dump()},
                {
                    "id": "bad",
                    "request": decision_request.model_dump() | {"state": "provider failure"},
                },
            ]
        }
    )
    async with api_factory(handler) as (_, http):
        sdk = WaldClient("http://wald.test", http_client=http)
        result = await sdk.decide_batch(batch, idempotency_key="batch")
        assert [item.id for item in result.results] == ["good", "bad"]
        assert result.results[0].result.answers["department"].choice == "billing"
        assert result.results[1].error.code == "upstream_error"
        assert (await sdk.list_reviews()).items[0].id == result.results[0].result.decision_id
        assert len((await sdk.list_reviews()).items) == 1  # No duplicate batch parent in the queue.
        replay = await sdk.decide_batch(batch, idempotency_key="batch")
        assert replay.decision_id == result.decision_id


async def test_image_api_calls_jev_and_keeps_visual_uncertainty(api_factory):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.url.path == "/v1/systemone"
        return httpx.Response(
            200,
            json={
                "model": "jev-test",
                "answers": {"cat": {"type": "noul", "noul": 0.99}},
                "usage": {},
            },
        )

    request = ImageDecisionRequest.model_validate(
        {
            "facts": {
                "summary": "A cat is visible",
                "observations": ["pointed ears"],
                "visible_text": [],
                "uncertainties": ["Partly hidden by a chair"],
                "image_quality": "limited",
            },
            "questions": {"cat": {"type": "boolean", "instructions": "Is a cat visible?"}},
        }
    )
    async with api_factory(handler) as (_, http):
        sdk = WaldClient("http://wald.test", http_client=http)
        result = await sdk.decide_image(request, idempotency_key="image-one")
        assert result.native_jev_vision is False
        assert result.jev.answers["cat"].boolean is True
        assert result.needs_review is True
        assert result.review_reasons
        assert (await sdk.get_decision(result.decision_id)).kind == "image"
    assert len(calls) == 1


async def test_cancelled_provider_request_settles_record_and_frees_capacity(
    api_factory, decision_request
):
    entered = asyncio.Event()

    async def handler(request):
        entered.set()
        await asyncio.Event().wait()

    async with api_factory(handler) as (app, _):
        service = app.state.service
        task = asyncio.create_task(
            service.execute(decision_request, "anonymous", idempotency_key="cancel")
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(WaldError) as failure:
            await service.execute(decision_request, "anonymous", idempotency_key="cancel")
        assert failure.value.code == "service_busy"
        assert (await app.state.store.get(failure.value.decision_id, "anonymous"))[
            "status"
        ] == "failed"
        assert service.gate.outstanding == 0


async def test_cancellation_during_reservation_does_not_strand_record(
    api_factory, decision_request, monkeypatch
):
    entered, release = asyncio.Event(), asyncio.Event()
    async with api_factory() as (app, _):
        reserve = app.state.store.reserve

        async def delayed_reservation(*args):
            result = await reserve(*args)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(app.state.store, "reserve", delayed_reservation)
        task = asyncio.create_task(
            app.state.service.execute(decision_request, "anonymous", idempotency_key="reserve")
        )
        await entered.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(WaldError) as failure:
            await app.state.service.execute(
                decision_request, "anonymous", idempotency_key="reserve"
            )
        assert "before execution" in str(failure.value)


async def test_whole_service_deadline_is_persisted(api_factory, settings, request_data):
    settings.request_timeout_seconds = 0.02

    async def handler(request):
        await asyncio.sleep(0.1)
        pytest.fail("provider should have been cancelled")

    async with api_factory(handler) as (_, http):
        result = await http.post("/v1/decide", json=request_data)
        assert result.status_code == 504
        record = await http.get(f"/v1/decisions/{result.json()['decision_id']}")
        assert record.json()["error"]["code"] == "upstream_timeout"


async def test_disconnect_during_commit_preserves_success(
    api_factory, decision_request, monkeypatch
):
    entered, release = asyncio.Event(), asyncio.Event()
    async with api_factory() as (app, _):
        finish = app.state.store.finish

        async def delayed_commit(identifier, result, error):
            if result is not None:
                entered.set()
                await release.wait()
            await finish(identifier, result, error)

        monkeypatch.setattr(app.state.store, "finish", delayed_commit)
        task = asyncio.create_task(
            app.state.service.execute(decision_request, "anonymous", idempotency_key="commit")
        )
        await entered.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        replay = await app.state.service.execute(
            decision_request, "anonymous", idempotency_key="commit"
        )
        assert replay["answers"]["department"]["choice"] == "billing"
        record = await app.state.store.get(replay["decision_id"], "anonymous")
        assert record["status"] == "succeeded"


async def test_readiness_metrics_and_rate_limit(api_factory, settings, request_data):
    settings.rate_limit_per_minute = 1
    settings.wald_api_key = SecretStr("service-secret")
    async with api_factory() as (_, http):
        assert (await http.get("/readyz")).status_code == 200
        assert (await http.get("/metrics")).status_code == 401
        headers = {"Authorization": "Bearer service-secret"}
        assert (
            await http.post("/v1/decide", json=request_data, headers=headers)
        ).status_code == 200
        rate = await http.post("/v1/decide", json=request_data, headers=headers)
        assert rate.status_code == 429
        assert int(rate.headers["retry-after"]) > 0
        metrics = await http.get("/metrics", headers=headers)
        assert 'wald_http_requests_total{route="/v1/decide",status="429"}' in metrics.text
        assert "service-secret" not in metrics.text


async def test_chunked_body_is_bounded(api_factory, settings):
    settings.max_body_bytes = 1024

    async def chunks():
        yield b"a" * 600
        yield b"b" * 600

    async with api_factory() as (_, http):
        result = await http.post("/v1/decide", content=chunks())
        assert result.status_code == 413


async def test_duplicate_json_keys_are_rejected(api_factory):
    async with api_factory() as (_, http):
        result = await http.post(
            "/v1/decide",
            content='{"state":"one","state":"two"}',
            headers={"Content-Type": "application/json"},
        )
        assert result.status_code == 422
        assert result.json()["error"]["code"] == "invalid_request"
        assert "two" not in result.text


async def test_http_image_pipeline_validates_and_omits_raw_image(api_factory):
    import base64

    from test_vision import PNG, facts_dict, image_jev_response

    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(
                200,
                json={
                    "model": "vision-test",
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": json.dumps(facts_dict())}}
                    ],
                },
            )
        return httpx.Response(200, json=image_jev_response())

    image_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    async with api_factory(handler) as (_, http):
        sdk = WaldClient("http://wald.test", http_client=http)
        result = await sdk.decide_image(ImageDecisionRequest(image_url=image_url))
        record = await sdk.get_decision(result.decision_id)
        assert record.result["mode"] == "vision_then_jev"
        assert record.result["vision"]["attempts"] == 1
        assert "image_url" not in record.request
        assert len(record.request["image_sha256"]) == 64
        assert image_url not in record.model_dump_json()
        invalid = await http.post(
            "/v1/images/decide", json={"image_url": "data:image/png;base64,YQ=="}
        )
        assert invalid.status_code == 422
    assert calls == ["/v1/chat/completions", "/v1/systemone"]


async def test_auth_is_checked_before_consuming_body(api_factory, settings):
    settings.wald_api_key = SecretStr("key")

    async def never_read():
        pytest.fail("unauthorized body must not be consumed")
        yield b"never"

    async with api_factory() as (_, http):
        assert (await http.post("/v1/decide", content=never_read())).status_code == 401


async def test_slow_body_has_deadline(api_factory, settings):
    settings.body_timeout_seconds = 0.01

    async def slow():
        await asyncio.sleep(0.1)
        yield b"{}"

    async with api_factory() as (_, http):
        assert (await http.post("/v1/decide", content=slow())).status_code == 408


async def test_exclusive_storage_lock_and_startup_recovery(tmp_path):
    path = tmp_path / "recovery.sqlite3"
    original, contender = DecisionStore(path), DecisionStore(path)
    await original.start()
    try:
        with pytest.raises(ConfigurationError):
            await contender.start()
        await original.reserve(
            "unfinished", "alice", "decision", "fingerprint", "hash", {"state": "input"}
        )
    finally:
        await original.close()
    await contender.start()
    try:
        record = await contender.get("unfinished", "alice")
        assert record["status"] == "failed"
        assert record["error"]["code"] == "request_interrupted"
    finally:
        await contender.close()


def test_production_requires_strong_service_auth_and_provider_credentials():
    for key in (None, "short-key"):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, wald_env="production", wald_api_key=key)
    valid_auth = Settings(_env_file=None, wald_env="production", wald_api_key="x" * 32)
    with pytest.raises(ConfigurationError):
        valid_auth.validate_startup()


async def test_production_hides_interactive_docs(api_factory, settings):
    settings.wald_env = "production"
    settings.wald_api_key = SecretStr("x" * 32)
    async with api_factory() as (_, http):
        headers = {"Authorization": "Bearer " + "x" * 32}
        assert (await http.get("/docs", headers=headers)).status_code == 404
        assert (await http.get("/openapi.json", headers=headers)).status_code == 404
