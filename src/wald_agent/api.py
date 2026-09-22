import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from typing import Annotated, Literal

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST

from wald_agent.config import Settings
from wald_agent.engine import DecisionEngine
from wald_agent.errors import WaldError
from wald_agent.jev import JevClient
from wald_agent.llm import ChatClient
from wald_agent.middleware import ServiceMiddleware, failure_response
from wald_agent.observability import Metrics, configure_logging, logger
from wald_agent.runtime import DecisionService
from wald_agent.schemas import (
    BatchDecisionRequest,
    BatchDecisionResponse,
    DecisionRecord,
    DecisionRequest,
    DecisionResponse,
    ReviewPage,
    ReviewResolution,
)
from wald_agent.storage import DecisionStore
from wald_agent.vision import ImageDecisionRequest, ImageDecisionResponse


def create_app(settings: Settings | None = None, engine=None, jev=None, vision=None) -> FastAPI:
    configured = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configured.validate_startup()
        configure_logging(configured.log_level)
        app.state.settings = configured
        app.state.metrics = Metrics()
        store = DecisionStore(configured.database_path, configured.retention_days)
        await store.start()
        try:
            async with AsyncExitStack() as stack:
                llm_client = engine or await stack.enter_async_context(DecisionEngine(configured))
                jev_client = jev or await stack.enter_async_context(JevClient(configured))
                vision_client = vision or await stack.enter_async_context(
                    ChatClient(configured, vision=True)
                )
                for transport in (getattr(llm_client, "chat", None), jev_client, vision_client):
                    if transport is not None:
                        transport.metrics = app.state.metrics
                service = DecisionService(
                    configured, store, llm_client, jev_client, vision_client, app.state.metrics
                )
                app.state.service, app.state.store = service, store

                async def maintain():
                    while True:
                        await asyncio.sleep(60)
                        try:
                            await store.prune()
                        except WaldError:
                            logger.error(
                                "storage_maintenance_failed",
                                extra={"error_code": "storage_unavailable"},
                            )

                maintenance = asyncio.create_task(maintain())
                try:
                    yield
                finally:
                    service.accepting = False
                    maintenance.cancel()
                    with suppress(asyncio.CancelledError):
                        await maintenance
        finally:
            await store.close()

    app = FastAPI(
        title="Wald Decision API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if configured.wald_env == "development" else None,
        redoc_url=None,
        openapi_url="/openapi.json" if configured.wald_env == "development" else None,
    )
    app.add_middleware(ServiceMiddleware, owner=app)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        errors = [
            {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {"code": "invalid_request", "message": "Request validation failed."},
                "detail": errors,
                "request_id": request.state.request_id,
            },
        )

    @app.exception_handler(WaldError)
    async def service_error(request: Request, exc: WaldError):
        return failure_response(
            exc.status_code,
            exc.code,
            str(exc),
            request.state.request_id,
            exc.retry_after,
            exc.decision_id,
        )

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready(request: Request):
        try:
            for provider in configured.enabled_providers:
                configured.key_for(provider)
            await request.app.state.store.ping()
            if not request.app.state.service.accepting:
                raise WaldError("shutting down")
        except WaldError:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready", "checks": "configuration_and_storage"}

    @app.get("/metrics")
    async def metrics(request: Request):
        return Response(
            request.app.state.metrics.render(), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    @app.post("/v1/decide", response_model=DecisionResponse)
    async def decide(
        payload: DecisionRequest,
        request: Request,
        provider: Literal["llm", "jev"] = "llm",
        idempotency_key: Annotated[str | None, Header()] = None,
    ):
        return await request.app.state.service.execute(
            payload, request.state.principal, provider=provider, idempotency_key=idempotency_key
        )

    @app.post("/v1/decide/batch", response_model=BatchDecisionResponse)
    async def batch(
        payload: BatchDecisionRequest,
        request: Request,
        provider: Literal["llm", "jev"] = "llm",
        idempotency_key: Annotated[str | None, Header()] = None,
    ):
        return await request.app.state.service.execute(
            payload, request.state.principal, provider=provider, idempotency_key=idempotency_key
        )

    @app.post("/v1/images/decide", response_model=ImageDecisionResponse)
    async def image_decide(
        payload: ImageDecisionRequest,
        request: Request,
        idempotency_key: Annotated[str | None, Header()] = None,
    ):
        return await request.app.state.service.execute(
            payload, request.state.principal, idempotency_key=idempotency_key
        )

    @app.get("/v1/decisions/{identifier}", response_model=DecisionRecord)
    async def record(identifier: str, request: Request):
        return await request.app.state.store.get(identifier, request.state.principal)

    @app.get("/v1/reviews", response_model=ReviewPage)
    async def reviews(
        request: Request,
        resolved: bool = False,
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
        after: Annotated[str | None, Query(max_length=36)] = None,
    ):
        return await request.app.state.store.list_reviews(
            request.state.principal, resolved, limit, after
        )

    @app.post("/v1/reviews/{identifier}", response_model=DecisionRecord)
    async def resolve(identifier: str, payload: ReviewResolution, request: Request):
        return await request.app.state.service.resolve(identifier, request.state.principal, payload)

    return app


app = create_app()
