import asyncio
import threading
from urllib.parse import urlencode

import httpx
from pydantic import ValidationError

from wald_agent.errors import InvalidProviderResponse, RemoteServiceError
from wald_agent.probability import validate_probabilities
from wald_agent.schemas import (
    BatchDecisionRequest,
    BatchDecisionResponse,
    BooleanQuestion,
    DecisionRecord,
    DecisionRequest,
    DecisionResponse,
    ReviewPage,
    ReviewResolution,
    probability_keys,
)
from wald_agent.transport import JsonTransport, strict_json_loads
from wald_agent.vision import ImageDecisionRequest, ImageDecisionResponse


def validate_response(raw: dict, request: DecisionRequest) -> DecisionResponse:
    try:
        result = DecisionResponse.model_validate(raw)
        if set(result.answers) != set(request.questions):
            raise ValueError("question IDs differ")
        for name, question in request.questions.items():
            answer = result.answers[name]
            expected_type = "boolean" if isinstance(question, BooleanQuestion) else question.type
            if answer.type != expected_type:
                raise ValueError("question/answer types differ")
            validate_probabilities(answer.probabilities, probability_keys(question))
        return result
    except (ValueError, ValidationError) as exc:
        raise InvalidProviderResponse("Wald returned an invalid decision response.") from exc


class WaldClient(JsonTransport):
    """Async Python SDK for a running Wald HTTP service."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "",
        timeout: float = 155,
        http_client: httpx.AsyncClient | None = None,
        max_response_bytes: int = 16 * 1024 * 1024,
    ):
        if max_response_bytes < 1024:
            raise ValueError("max_response_bytes must be at least 1024")
        super().__init__(timeout, http_client)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_response_bytes = max_response_bytes

    def http_error(self, status, content, provider, delay) -> RemoteServiceError:
        error = RemoteServiceError(f"Wald returned HTTP {status}.", retry_after=delay)
        error.status_code = status
        try:
            raw = strict_json_loads(content)
            detail = raw["error"]
            if isinstance(detail["message"], str) and len(detail["message"]) <= 2000:
                error.args = (detail["message"],)
            if isinstance(detail["code"], str) and len(detail["code"]) <= 128:
                error.code = detail["code"]
            for name in ("request_id", "decision_id"):
                value = raw.get(name)
                if isinstance(value, str) and len(value) <= 128:
                    setattr(error, name, value)
        except (ValueError, KeyError, TypeError):
            pass
        return error

    async def decide(
        self,
        request: DecisionRequest,
        *,
        provider: str = "llm",
        idempotency_key: str | None = None,
    ) -> DecisionResponse:
        if provider not in {"llm", "jev"}:
            raise ValueError("provider must be llm or jev")
        raw = await self.post(
            f"{self.base_url}/v1/decide?provider={provider}",
            self.api_key,
            request.model_dump(),
            "wald",
            headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        return validate_response(raw, request)

    async def decide_batch(
        self,
        request: BatchDecisionRequest,
        *,
        provider: str = "llm",
        idempotency_key: str | None = None,
    ) -> BatchDecisionResponse:
        if provider not in {"llm", "jev"}:
            raise ValueError("provider must be llm or jev")
        raw = await self.post(
            f"{self.base_url}/v1/decide/batch?provider={provider}",
            self.api_key,
            request.model_dump(),
            "wald",
            headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        result = self._parse(BatchDecisionResponse, raw)
        if [item.id for item in result.results] != [item.id for item in request.items]:
            raise InvalidProviderResponse("Wald batch returned different item IDs or order.")
        for item, requested in zip(result.results, request.items, strict=True):
            if item.result:
                validate_response(item.result.model_dump(), requested.request)
        return result

    async def decide_image(
        self,
        request: ImageDecisionRequest,
        *,
        idempotency_key: str | None = None,
    ) -> ImageDecisionResponse:
        raw = await self.post(
            f"{self.base_url}/v1/images/decide",
            self.api_key,
            request.model_dump(),
            "wald",
            headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        result = self._parse(ImageDecisionResponse, raw)
        validate_response(
            result.jev.model_dump(), DecisionRequest(state="image", questions=request.questions)
        )
        return result

    async def get_decision(self, identifier: str) -> DecisionRecord:
        from uuid import UUID

        identifier = str(UUID(identifier))
        raw = await self.request(
            "GET", f"{self.base_url}/v1/decisions/{identifier}", self.api_key, None, "wald"
        )
        return self._parse(DecisionRecord, raw)

    async def list_reviews(
        self, *, resolved: bool = False, limit: int = 25, after: str | None = None
    ) -> ReviewPage:
        query = {"resolved": str(resolved).lower(), "limit": limit}
        if after:
            query["after"] = after
        raw = await self.request(
            "GET", f"{self.base_url}/v1/reviews?{urlencode(query)}", self.api_key, None, "wald"
        )
        return self._parse(ReviewPage, raw)

    async def resolve_review(self, identifier: str, resolution: ReviewResolution) -> DecisionRecord:
        from uuid import UUID

        identifier = str(UUID(identifier))
        raw = await self.post(
            f"{self.base_url}/v1/reviews/{identifier}",
            self.api_key,
            resolution.model_dump(),
            "wald",
        )
        return self._parse(DecisionRecord, raw)

    @staticmethod
    def _parse(model, raw):
        try:
            return model.model_validate(raw)
        except ValidationError as exc:
            raise InvalidProviderResponse("Wald returned an invalid response.") from exc


class SyncWaldClient:
    """Synchronous SDK with a persistent, thread-owned event loop and connection pool."""

    def __init__(self, *args, **kwargs):
        self._client = WaldClient(*args, **kwargs)
        self._runner = asyncio.Runner()
        self._thread = threading.get_ident()
        self._closed = False

    def _run(self, method, *args, **kwargs):
        if self._closed or threading.get_ident() != self._thread:
            raise RuntimeError("SyncWaldClient must be open and used on its owning thread.")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Use the async WaldClient inside an async event loop.")
        return self._runner.run(method(*args, **kwargs))

    def decide(self, request, **kwargs) -> DecisionResponse:
        return self._run(self._client.decide, request, **kwargs)

    def decide_batch(self, request, **kwargs) -> BatchDecisionResponse:
        return self._run(self._client.decide_batch, request, **kwargs)

    def decide_image(self, request, **kwargs) -> ImageDecisionResponse:
        return self._run(self._client.decide_image, request, **kwargs)

    def get_decision(self, identifier) -> DecisionRecord:
        return self._run(self._client.get_decision, identifier)

    def list_reviews(self, **kwargs) -> ReviewPage:
        return self._run(self._client.list_reviews, **kwargs)

    def resolve_review(self, identifier, resolution) -> DecisionRecord:
        return self._run(self._client.resolve_review, identifier, resolution)

    def close(self):
        if not self._closed:
            self._run(self._client.aclose)
            self._runner.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
