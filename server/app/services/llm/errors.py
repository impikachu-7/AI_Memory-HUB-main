"""Safe, normalized errors raised by cloud LLM providers."""
from __future__ import annotations

from fastapi import HTTPException


_CODE_BY_STATUS = {
    401: "PROVIDER_AUTH_ERROR",
    402: "PROVIDER_BILLING_OR_CREDITS",
    403: "PROVIDER_FORBIDDEN",
    404: "MODEL_NOT_FOUND",
    408: "PROVIDER_TIMEOUT",
    409: "PROVIDER_CONFLICT",
    429: "RATE_LIMITED",
}


class ProviderError(HTTPException):
    """An error safe to pass from an adapter to the generation route."""

    def __init__(
        self,
        provider: str,
        model: str | None,
        status_code: int,
        code: str,
        error_type: str,
        detail: str = "Provider request failed",
    ) -> None:
        self.provider = provider
        self.model = model
        self.provider_status = status_code
        self.code = code
        self.error_type = error_type
        super().__init__(status_code, detail)


def provider_error(
    exc: Exception,
    provider: str,
    model: str | None = None,
    status_code: int | None = None,
) -> ProviderError:
    """Convert an SDK exception without retaining its potentially sensitive body."""
    status = status_code if status_code is not None else getattr(exc, "status_code", None)
    if not isinstance(status, int):
        name = type(exc).__name__.lower()
        if "timeout" in name:
            status = 408
        elif "connection" in name or "unavailable" in name:
            status = 503
        else:
            status = 502

    if status >= 500:
        code = "PROVIDER_UNAVAILABLE"
    else:
        code = _CODE_BY_STATUS.get(status, "PROVIDER_ERROR")
    return ProviderError(provider, model, status, code, type(exc).__name__)
