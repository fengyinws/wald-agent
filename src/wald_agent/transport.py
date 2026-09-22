import asyncio
import json
import random
import time

import httpx

from wald_agent.config import Settings
from wald_agent.errors import InvalidProviderResponse, ProviderError, ProviderTimeout
from wald_agent.observability import Metrics, logger, request_id_context
from wald_agent.resilience import CircuitBreaker, retry_after_seconds

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}
QUOTA_ERRORS = {
    "insufficient_quota",
    "credit_balance_exhausted",
    "billing_hard_limit_reached",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
}


def strict_json_loads(content: str | bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("non-finite JSON number")

    return json.loads(content, object_pairs_hook=unique, parse_constant=reject_constant)


class JsonPayload(dict):
    def __init__(self, body: dict, attempts: int, upstream_request_id: str | None):
        super().__init__(body)
        self.attempts = attempts
        self.upstream_request_id = upstream_request_id


class JsonTransport:
    """Bounded responses, total deadline, explicit retries and a circuit per pooled client."""

    def __init__(
        self,
        timeout: float,
        http_client: httpx.AsyncClient | None = None,
        *,
        settings: Settings | None = None,
    ):
        self.timeout = timeout
        self.settings = settings
        self.max_retries = settings.api_max_retries if settings else 0
        self.max_response_bytes = settings.api_max_response_bytes if settings else 2 * 1024 * 1024
        connect = min(timeout, settings.api_connect_timeout_seconds if settings else 5)
        self.http_timeout = httpx.Timeout(timeout, connect=connect, pool=connect)
        self._owns_client = http_client is None
        self.http = http_client or httpx.AsyncClient(
            timeout=self.http_timeout,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=32),
        )
        self.circuit = CircuitBreaker(
            settings.circuit_failure_threshold if settings else 5,
            settings.circuit_cooldown_seconds if settings else 30,
        )
        self.metrics: Metrics | None = None

    async def post(
        self,
        url: str,
        key: str,
        body: dict,
        provider: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> JsonPayload:
        return await self.request("POST", url, key, body, provider, headers=headers)

    async def request(
        self,
        method: str,
        url: str,
        key: str,
        body: dict | None,
        provider: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> JsonPayload:
        ticket = await self.circuit.acquire()
        outcome = None
        try:
            async with asyncio.timeout(self.timeout):
                result = await self._post(method, url, key, body, provider, headers)
            outcome = True
            return result
        except TimeoutError as exc:
            outcome = False
            raise ProviderTimeout(f"{provider} exceeded its total request deadline.") from exc
        except ProviderError as exc:
            # Invalid credentials/requests are not a transient outage and must not open the circuit.
            outcome = getattr(exc, "circuit_failure", False) is False
            raise
        finally:
            await self.circuit.finish(ticket, outcome)

    async def _post(self, method, url, key, body, provider, extra_headers) -> JsonPayload:
        deadline = time.monotonic() + self.timeout
        headers = dict(extra_headers or {})
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request_id = request_id_context.get()
        if request_id:
            headers["X-Client-Request-Id"] = request_id
        for attempt in range(1, self.max_retries + 2):
            delay = None
            retryable = False
            metric_outcome = "cancelled"
            try:
                async with self.http.stream(
                    method,
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.http_timeout,
                ) as response:
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > self.max_response_bytes:
                            raise InvalidProviderResponse(
                                f"{provider} response exceeds configured size limit."
                            )
                    status = response.status_code
                    metric_outcome = str(status)
                    if 200 <= status < 300:
                        try:
                            data = strict_json_loads(bytes(content))
                        except (ValueError, UnicodeError) as exc:
                            raise InvalidProviderResponse(
                                f"{provider} returned invalid JSON."
                            ) from exc
                        if not isinstance(data, dict):
                            raise InvalidProviderResponse(
                                f"{provider} response must be a JSON object."
                            )
                        return JsonPayload(data, attempt, response.headers.get("x-request-id"))
                    delay = retry_after_seconds(response.headers.get("retry-after"))
                    retryable = status in RETRYABLE_STATUS
                    if status == 429:
                        try:
                            error = strict_json_loads(bytes(content)).get("error", {})
                            if isinstance(error, dict) and str(error.get("code")) in QUOTA_ERRORS:
                                retryable = False
                        except (ValueError, AttributeError):
                            pass
                    failure = self.http_error(status, bytes(content), provider, delay)
                    failure.circuit_failure = retryable
            except InvalidProviderResponse:
                metric_outcome = "invalid_response"
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                metric_outcome = "connection_error"
                failure = ProviderError(f"{provider} could not establish a connection.")
                failure.__cause__ = exc
                failure.circuit_failure = True
                retryable = True
            except httpx.TimeoutException as exc:
                metric_outcome = "timeout"
                # The provider may have received/billed the POST; don't replay a read timeout.
                failure = ProviderTimeout(f"{provider} request timed out.")
                failure.__cause__ = exc
                failure.circuit_failure = True
            except httpx.RequestError as exc:
                metric_outcome = "network_error"
                failure = ProviderError(f"{provider} failed at the network layer.")
                failure.__cause__ = exc
                failure.circuit_failure = True
            finally:
                if self.metrics:
                    self.metrics.upstream.labels(provider, metric_outcome).inc()
            failure.attempts = attempt
            if not retryable or attempt > self.max_retries:
                raise failure
            if delay is None:
                base = self.settings.api_retry_base_seconds if self.settings else 0.5
                cap = self.settings.api_retry_max_seconds if self.settings else 10
                delay = random.uniform(0, min(cap, base * 2 ** (attempt - 1)))
            if delay >= deadline - time.monotonic():
                raise failure
            logger.info("upstream_retry", extra={"provider": provider, "attempt": attempt})
            await asyncio.sleep(delay)
        raise AssertionError("unreachable retry state")

    def http_error(
        self, status: int, content: bytes, provider: str, delay: float | None
    ) -> ProviderError:
        return ProviderError(
            f"{provider} returned HTTP {status}; check credentials, model, quota and rate limits.",
            retry_after=delay,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
