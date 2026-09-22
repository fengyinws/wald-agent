import asyncio
import hashlib
import json
import re
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import TypeAdapter

from wald_agent.config import Settings
from wald_agent.errors import (
    ConfigurationError,
    InvalidRequest,
    ProviderTimeout,
    RateLimited,
    ServiceBusy,
    WaldError,
)
from wald_agent.observability import Metrics, logger, request_id_context
from wald_agent.schemas import (
    BatchDecisionRequest,
    BatchResult,
    DecisionRequest,
    DecisionResponse,
    ErrorDetail,
    Question,
    ReviewResolution,
    validate_labels,
)
from wald_agent.storage import DecisionStore, encode
from wald_agent.vision import ImageDecisionRequest, judge_image


class RateLimiter:
    def __init__(self, per_minute: int):
        self.capacity = per_minute
        self.buckets: dict[str, tuple[float, float]] = {}

    def consume(self, principal: str) -> None:
        now = time.monotonic()
        tokens, previous = self.buckets.get(principal, (float(self.capacity), now))
        tokens = min(self.capacity, tokens + (now - previous) * self.capacity / 60)
        if tokens < 1:
            self.buckets[principal] = (tokens, now)
            raise RateLimited(
                "Client request rate exceeded.", retry_after=(1 - tokens) * 60 / self.capacity
            )
        self.buckets[principal] = (tokens - 1, now)


class AdmissionGate:
    def __init__(self, concurrent: int, queued: int, queue_timeout: float):
        self.semaphore = asyncio.Semaphore(concurrent)
        self.limit = concurrent + queued
        self.queue_timeout = queue_timeout
        self.outstanding = 0

    @asynccontextmanager
    async def slot(self):
        if self.outstanding >= self.limit:
            raise ServiceBusy("Request queue is full.", retry_after=1)
        self.outstanding += 1
        acquired = False
        try:
            try:
                async with asyncio.timeout(self.queue_timeout):
                    await self.semaphore.acquire()
                    acquired = True
            except TimeoutError as exc:
                raise ServiceBusy("Request queue deadline exceeded.", retry_after=1) from exc
            yield
        finally:
            if acquired:
                self.semaphore.release()
            self.outstanding -= 1


def error_detail(error: WaldError) -> dict:
    return ErrorDetail(
        code=error.code,
        message=str(error),
        status_code=error.status_code,
        retry_after=error.retry_after,
        decision_id=error.decision_id,
    ).model_dump()


class DecisionService:
    def __init__(
        self, settings: Settings, store: DecisionStore, engine, jev, vision, metrics: Metrics
    ):
        self.settings, self.store = settings, store
        self.engine, self.jev, self.vision, self.metrics = engine, jev, vision, metrics
        self.limiter = RateLimiter(settings.rate_limit_per_minute)
        self.gate = AdmissionGate(
            settings.max_concurrent_requests,
            settings.max_queued_requests,
            settings.queue_timeout_seconds,
        )
        self.accepting = True

    async def execute(
        self,
        payload: DecisionRequest | ImageDecisionRequest | BatchDecisionRequest,
        principal: str,
        *,
        provider: str = "llm",
        idempotency_key: str | None = None,
    ) -> dict:
        if not self.accepting:
            raise ServiceBusy("Service is shutting down.", retry_after=1)
        kind = (
            "image"
            if isinstance(payload, ImageDecisionRequest)
            else ("batch" if isinstance(payload, BatchDecisionRequest) else "decision")
        )
        if kind == "image":
            provider = "jev"
        if provider not in self.settings.enabled_providers:
            raise ConfigurationError(f"Provider {provider} is not enabled in ENABLED_PROVIDERS.")
        self.settings.key_for(provider)
        if kind == "image" and payload.image_url is not None:
            self.settings.key_for("vision")
        if idempotency_key is not None and not re.fullmatch(
            r"[A-Za-z0-9._:-]{1,128}", idempotency_key
        ):
            raise InvalidRequest(
                "Idempotency-Key must contain 1-128 ASCII letters, digits, . _ : or -."
            )
        if kind != "batch":
            self.limiter.consume(principal)
        request = payload.model_dump(mode="json")
        if kind == "decision" and len(encode(request).encode()) > self.settings.max_state_bytes:
            raise InvalidRequest("Decision request exceeds MAX_STATE_BYTES.")
        # Include output-affecting configuration. Changing models with the same key is a conflict.
        configuration = {
            "version": 1,
            "provider": provider,
            "kind": kind,
            "model": self.settings.llm_model if provider == "llm" else self.settings.jev_model,
            "base_url": self.settings.llm_base_url
            if provider == "llm"
            else self.settings.jev_base_url,
            "vision_model": self.settings.vision_model if kind == "image" else None,
            "vision_base_url": (self.settings.vision_base_url or self.settings.llm_base_url)
            if kind == "image"
            else None,
            "temperature": self.settings.llm_temperature,
            "response_format": self.settings.llm_response_format,
            "max_output_tokens": self.settings.llm_max_output_tokens,
            "token_limit_field": self.settings.llm_token_limit_field,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                {"configuration": configuration, "request": request},
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        key_hash = hashlib.sha256(idempotency_key.encode()).hexdigest() if idempotency_key else None
        identifier = str(uuid4())
        audit = self._audit_request(request, kind)
        reservation = asyncio.create_task(
            self.store.reserve(identifier, principal, kind, fingerprint, key_hash, audit)
        )
        try:
            previous = await asyncio.shield(reservation)
        except asyncio.CancelledError:
            # SQLite may commit in the worker thread after its caller is cancelled.
            previous = await reservation
            if previous is None:
                await self.store.finish(
                    identifier,
                    None,
                    error_detail(ServiceBusy("Request was interrupted before execution.")),
                )
            raise
        if previous:
            if previous["error"]:
                saved = previous["error"]
                failure = WaldError(saved["message"], retry_after=saved.get("retry_after"))
                failure.code, failure.status_code = saved["code"], saved["status_code"]
                failure.decision_id = previous["id"]
                raise failure
            replay = previous["result"]
            replay["request_id"] = request_id_context.get()
            return replay
        try:
            async with asyncio.timeout(self.settings.request_timeout_seconds):
                if kind == "batch":
                    result = await self._batch(payload, principal, provider, identifier)
                else:
                    async with self.gate.slot():
                        if kind == "image":
                            result = await judge_image(
                                self.jev,
                                questions=payload.questions,
                                facts=payload.facts,
                                image_url=payload.image_url,
                                vision=self.vision,
                                review_threshold=payload.review_threshold,
                            )
                        else:
                            client = self.engine if provider == "llm" else self.jev
                            response = await client.decide(payload)
                            result = response.model_dump(mode="json")
            result["decision_id"] = identifier
            result["request_id"] = request_id_context.get()
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                failure = ServiceBusy("Request was interrupted before completion.")
            elif isinstance(exc, TimeoutError):
                failure = ProviderTimeout("Decision exceeded REQUEST_TIMEOUT_SECONDS.")
            elif isinstance(exc, WaldError):
                failure = exc
            else:
                failure = WaldError(
                    "Internal service error; inspect the request ID in server logs."
                )
                logger.error("decision_failed", extra={"error_code": type(exc).__name__})
            failure.decision_id = identifier
            # Cancellation must not strand a durable idempotency record as processing.
            cleanup = asyncio.create_task(
                self.store.finish(identifier, None, error_detail(failure))
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            if isinstance(exc, asyncio.CancelledError):
                raise
            if failure is exc:
                raise
            raise failure from exc

        # A disconnect during commit must not race a second write of a failure result.
        completion = asyncio.create_task(self.store.finish(identifier, result, None))
        try:
            await asyncio.shield(completion)
        except asyncio.CancelledError:
            await completion
            raise
        except WaldError as exc:
            exc.decision_id = identifier
            raise
        self.metrics.decisions.labels(kind, str(result["needs_review"]).lower()).inc()
        return result

    def _audit_request(self, request: dict, kind: str) -> dict:
        # Persist questions for validating later corrections, even when input retention is off.
        audit = json.loads(encode(request))
        if kind == "image" and audit.get("image_url"):
            audit["image_sha256"] = hashlib.sha256(audit.pop("image_url").encode()).hexdigest()
        if not self.settings.audit_store_inputs:
            audit.pop("state", None)
            audit.pop("facts", None)
            if kind == "batch":
                for item in audit["items"]:
                    item["request"].pop("state", None)
        return audit

    async def _batch(self, payload, principal, provider, identifier) -> dict:
        async def one(item):
            try:
                result = await self.execute(
                    item.request,
                    principal,
                    provider=provider,
                    idempotency_key=hashlib.sha256(f"{identifier}:{item.id}".encode()).hexdigest(),
                )
                return BatchResult(id=item.id, result=DecisionResponse.model_validate(result))
            except WaldError as exc:
                return BatchResult(id=item.id, error=ErrorDetail(**error_detail(exc)))

        # TaskGroup cancels AND awaits siblings on interruption before the parent is settled.
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(one(item)) for item in payload.items]
        results = [task.result() for task in tasks]
        return {
            "results": [result.model_dump(mode="json") for result in results],
            "needs_review": any(result.result and result.result.needs_review for result in results),
        }

    async def resolve(self, identifier: str, principal: str, resolution: ReviewResolution) -> dict:
        record = await self.store.get(identifier, principal)
        if record["kind"] == "batch":
            raise InvalidRequest("Review the individual decisions inside a batch.")
        questions = TypeAdapter(dict[str, Question]).validate_python(record["request"]["questions"])
        try:
            validate_labels(questions, resolution.answers, complete=True)
        except ValueError as exc:
            raise InvalidRequest(str(exc)) from exc
        data = resolution.model_dump(exclude={"expected_revision"})
        data["reviewed_at"] = datetime.now(UTC).isoformat()
        return await self.store.resolve(identifier, principal, resolution.expected_revision, data)
