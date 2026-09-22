import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import pytest

from wald_agent.errors import (
    CircuitOpen,
    InvalidProviderResponse,
    ProviderError,
    ProviderTimeout,
    RateLimited,
    ServiceBusy,
)
from wald_agent.resilience import CircuitBreaker, retry_after_seconds
from wald_agent.runtime import AdmissionGate, RateLimiter
from wald_agent.transport import JsonTransport


async def test_retry_after_and_attempt_metadata(settings):
    settings.api_max_retries = 2
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(
                429, headers={"Retry-After": "0"}, json={"error": {"code": "rate_limit"}}
            )
        return httpx.Response(200, headers={"x-request-id": "upstream-123"}, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await JsonTransport(1, http, settings=settings).post(
            "https://test", "key", {}, "llm"
        )
    assert result == {"ok": True}
    assert result.attempts == len(calls) == 3
    assert result.upstream_request_id == "upstream-123"


@pytest.mark.parametrize(
    "status,code", [(401, "bad_key"), (400, "invalid_request"), (429, "insufficient_quota")]
)
async def test_permanent_errors_never_retry_or_open_circuit(settings, status, code):
    settings.api_max_retries = 2
    settings.circuit_failure_threshold = 1
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status, json={"error": {"code": code, "message": "PRIVATE_UPSTREAM_BODY"}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        transport = JsonTransport(1, http, settings=settings)
        for _ in range(2):
            with pytest.raises(ProviderError) as failure:
                await transport.post("https://test", "key", {}, "llm")
            assert not isinstance(failure.value, CircuitOpen)
            assert "PRIVATE" not in str(failure.value)
    assert len(calls) == 2


async def test_retry_after_beyond_deadline_does_not_retry_early(settings):
    settings.api_max_retries = 2
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, headers={"Retry-After": "60"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ProviderError) as error:
            await JsonTransport(0.1, http, settings=settings).post("https://test", "key", {}, "llm")
    assert len(calls) == 1
    assert error.value.retry_after == 60


@pytest.mark.parametrize(
    "error_class,expected_calls",
    [(httpx.ConnectError, 3), (httpx.ReadTimeout, 1), (httpx.WriteError, 1)],
)
async def test_only_pretransmission_network_errors_retry(settings, error_class, expected_calls):
    settings.api_max_retries = 2
    settings.api_retry_base_seconds = 0
    calls = []

    def handler(request):
        calls.append(request)
        raise error_class("private-network-detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ProviderError):
            await JsonTransport(1, http, settings=settings).post("https://test", "key", {}, "llm")
    assert len(calls) == expected_calls


async def test_whole_stream_deadline(settings):
    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"ok":'
            await asyncio.sleep(0.1)
            yield b"true}"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=SlowStream()))
    ) as http:
        with pytest.raises(ProviderTimeout):
            await JsonTransport(0.02, http, settings=settings).post(
                "https://test", "key", {}, "llm"
            )


@pytest.mark.parametrize("content", [b'{"ok":true,"ok":false}', b'{"value":NaN}', b"x" * 1025])
async def test_response_strict_json_and_size_limit(settings, content):
    settings.api_max_response_bytes = 1024
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content))
    ) as http:
        with pytest.raises(InvalidProviderResponse):
            await JsonTransport(1, http, settings=settings).post("https://test", "key", {}, "llm")


async def test_circuit_opens_allows_one_probe_and_ignores_old_completions(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("wald_agent.resilience.time", SimpleNamespace(monotonic=lambda: clock[0]))
    circuit = CircuitBreaker(threshold=1, cooldown=10)
    failing = await circuit.acquire()
    earlier = await circuit.acquire()
    await circuit.finish(failing, False)
    await circuit.finish(earlier, True)
    with pytest.raises(CircuitOpen):
        await circuit.acquire()
    clock[0] = 11
    probe = await circuit.acquire()
    with pytest.raises(CircuitOpen):
        await circuit.acquire()
    await circuit.finish(probe, None)  # A cancelled probe must not leave probing stuck.
    clock[0] = 22
    recovered = await circuit.acquire()
    await circuit.finish(recovered, True)
    assert not (await circuit.acquire())[1]


def test_retry_after_parser_handles_dates_and_invalid_values():
    assert retry_after_seconds("-1") == 0
    for invalid in (None, "NaN", "Infinity", "not a date"):
        assert retry_after_seconds(invalid) is None
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=60))
    assert 58 < retry_after_seconds(future) <= 60


async def test_admission_rejects_overflow_and_releases_after_cancellation():
    gate = AdmissionGate(1, 1, 1)
    entered = asyncio.Event()

    async def hold():
        async with gate.slot():
            entered.set()
            await asyncio.Event().wait()

    first = asyncio.create_task(hold())
    await entered.wait()
    second = asyncio.create_task(hold())
    await asyncio.sleep(0)
    with pytest.raises(ServiceBusy):
        async with gate.slot():
            pytest.fail("overloaded request was admitted")
    for task in (first, second):
        task.cancel()
    await asyncio.gather(first, second, return_exceptions=True)
    async with gate.slot():
        pass
    assert gate.outstanding == 0


async def test_queue_timeout_releases_capacity():
    gate = AdmissionGate(1, 1, 0.01)
    async with gate.slot():
        with pytest.raises(ServiceBusy):
            async with gate.slot():
                pytest.fail("should time out waiting")
    assert gate.outstanding == 0


def test_rate_limits_are_per_client_and_refill(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("wald_agent.runtime.time", SimpleNamespace(monotonic=lambda: clock[0]))
    limiter = RateLimiter(1)
    limiter.consume("alice")
    limiter.consume("bob")
    with pytest.raises(RateLimited) as failure:
        limiter.consume("alice")
    assert failure.value.retry_after == 60
    clock[0] = 60
    limiter.consume("alice")
