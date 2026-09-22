import asyncio
import math
import re
import secrets
import time
from uuid import uuid4

from starlette.responses import JSONResponse

from wald_agent.errors import WaldError
from wald_agent.observability import logger, request_id_context
from wald_agent.transport import strict_json_loads


def failure_response(
    status: int, code: str, message: str, request_id: str, retry_after=None, decision_id=None
):
    headers = (
        {"Retry-After": str(max(1, math.ceil(retry_after)))} if retry_after is not None else {}
    )
    return JSONResponse(
        status_code=status,
        content={
            "error": {"code": code, "message": message},
            "detail": message,
            "request_id": request_id,
            "decision_id": decision_id,
        },
        headers=headers,
    )


def route_label(path: str) -> str:
    if path in {
        "/healthz",
        "/readyz",
        "/metrics",
        "/v1/decide",
        "/v1/decide/batch",
        "/v1/images/decide",
        "/v1/reviews",
    }:
        return path
    if path.startswith("/v1/decisions/"):
        return "/v1/decisions/{id}"
    if path.startswith("/v1/reviews/"):
        return "/v1/reviews/{id}"
    if path in {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}:
        return "documentation"
    return "unmatched"


class ServiceMiddleware:
    """Authenticate before reading input; bound even chunked bodies; attach request IDs."""

    def __init__(self, app, owner):
        self.app, self.owner = app, owner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        settings = self.owner.state.settings
        metrics = self.owner.state.metrics
        started = time.monotonic()
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        provided = headers.get(b"x-request-id", b"").decode("ascii", errors="ignore")
        request_id = provided if re.fullmatch(r"[A-Za-z0-9._-]{1,128}", provided) else str(uuid4())
        token = request_id_context.set(request_id)
        scope.setdefault("state", {})["request_id"] = request_id
        path = scope["path"]
        route = route_label(path)
        status = 500
        response_started = False

        async def traced_send(message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status, response_started = message["status"], True
                message["headers"] = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ] + [
                    (b"x-request-id", request_id.encode()),
                    (b"x-content-type-options", b"nosniff"),
                ]
            await send(message)

        async def fail(status_code, code, text):
            await failure_response(status_code, code, text, request_id)(scope, receive, traced_send)

        try:
            principal = "anonymous"
            keys = settings.service_keys()
            if path not in {"/healthz", "/readyz"} and keys:
                supplied = headers.get(b"authorization", b"")
                principal = None
                for name, secret in keys.items():
                    if secrets.compare_digest(supplied, f"Bearer {secret}".encode()):
                        principal = name
                if principal is None:
                    await fail(401, "unauthorized", "Invalid or missing Wald API key.")
                    return
            scope["state"]["principal"] = principal
            if scope["method"] == "GET" and path.startswith("/v1/"):
                self.owner.state.service.limiter.consume(principal)
            content_length = headers.get(b"content-length")
            if content_length is not None:
                try:
                    length = int(content_length)
                    if length < 0:
                        raise ValueError
                except ValueError:
                    await fail(400, "invalid_content_length", "Invalid Content-Length.")
                    return
                if length > settings.max_body_bytes:
                    await fail(413, "body_too_large", "Request body exceeds MAX_BODY_BYTES.")
                    return
            chunks = []
            size = 0
            try:
                async with asyncio.timeout(settings.body_timeout_seconds):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            status = 499
                            return
                        chunk = message.get("body", b"")
                        size += len(chunk)
                        if size > settings.max_body_bytes:
                            await fail(
                                413, "body_too_large", "Request body exceeds MAX_BODY_BYTES."
                            )
                            return
                        chunks.append(chunk)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                await fail(408, "body_timeout", "Request body was not received in time.")
                return
            body = b"".join(chunks)
            content_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
            if body and (
                not content_type
                or content_type == b"application/json"
                or content_type.endswith(b"+json")
            ):
                try:
                    strict_json_loads(body)
                except (ValueError, UnicodeError, RecursionError):
                    await JSONResponse(
                        status_code=422,
                        content={
                            "error": {"code": "invalid_request", "message": "Invalid JSON body."},
                            "detail": [
                                {
                                    "loc": ["body"],
                                    "msg": "Invalid, duplicate or non-finite JSON.",
                                    "type": "json_invalid",
                                }
                            ],
                            "request_id": request_id,
                        },
                    )(scope, receive, traced_send)
                    return
            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, bounded_receive, traced_send)
        except WaldError as exc:
            await failure_response(
                exc.status_code, exc.code, str(exc), request_id, exc.retry_after, exc.decision_id
            )(scope, receive, traced_send)
        except asyncio.CancelledError:
            status = 499
            raise
        except Exception:
            logger.error("http_failed", extra={"route": route, "error_code": "internal_error"})
            if response_started:
                raise
            await fail(500, "internal_error", "Internal service error; inspect the request ID.")
        finally:
            elapsed = time.monotonic() - started
            metrics.requests.labels(route, str(status)).inc()
            metrics.latency.labels(route).observe(elapsed)
            if path not in {"/healthz", "/readyz", "/metrics"}:
                logger.info(
                    "http_request",
                    extra={
                        "route": route,
                        "status": status,
                        "elapsed_ms": round(elapsed * 1000, 2),
                    },
                )
            request_id_context.reset(token)
