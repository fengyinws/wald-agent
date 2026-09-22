class WaldError(Exception):
    """Safe to display: never include credentials or raw upstream response bodies."""

    code = "wald_error"
    status_code = 500

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.decision_id: str | None = None
        self.request_id: str | None = None


class ConfigurationError(WaldError):
    code = "configuration_error"
    status_code = 503


class ProviderError(WaldError):
    code = "upstream_error"
    status_code = 502


class ProviderTimeout(ProviderError):
    code = "upstream_timeout"
    status_code = 504


class RemoteServiceError(ProviderError):
    """An error from a Wald HTTP service, with its original status and correlation IDs."""

    code = "remote_service_error"


class InvalidProviderResponse(ProviderError):
    code = "invalid_upstream_response"


class CircuitOpen(ProviderError):
    code = "upstream_circuit_open"
    status_code = 503


class ServiceBusy(WaldError):
    code = "service_busy"
    status_code = 503


class RateLimited(WaldError):
    code = "rate_limited"
    status_code = 429


class IdempotencyConflict(WaldError):
    code = "idempotency_conflict"
    status_code = 409


class RecordNotFound(WaldError):
    code = "record_not_found"
    status_code = 404


class StorageError(WaldError):
    code = "storage_unavailable"
    status_code = 503


class InvalidRequest(WaldError):
    code = "invalid_request"
    status_code = 422
