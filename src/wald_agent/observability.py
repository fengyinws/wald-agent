import contextvars
import json
import logging
from datetime import UTC, datetime

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

request_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
logger = logging.getLogger("wald")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": request_id_context.get(),
        }
        # Only explicitly approved scalar metadata, never arbitrary extras, inputs or exceptions.
        for key in (
            "route",
            "status",
            "elapsed_ms",
            "provider",
            "attempt",
            "error_code",
            "decision_id",
        ):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        return json.dumps(data, ensure_ascii=False, allow_nan=False)


def configure_logging(level: str) -> None:
    if not any(getattr(handler, "_wald_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler._wald_handler = True
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "wald_http_requests_total", "HTTP requests", ["route", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "wald_http_duration_seconds",
            "HTTP duration",
            ["route"],
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 150),
        )
        self.upstream = Counter(
            "wald_upstream_attempts_total",
            "Upstream attempts",
            ["provider", "outcome"],
            registry=self.registry,
        )
        self.decisions = Counter(
            "wald_decisions_total",
            "Completed decisions",
            ["kind", "review"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
