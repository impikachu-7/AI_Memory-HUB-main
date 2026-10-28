"""Safe, normalized errors raised by cloud LLM providers."""
from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

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
        retry_after_seconds: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.provider_status = status_code
        self.code = code
        self.error_type = error_type
        self.retry_after_seconds = retry_after_seconds
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
    retry_after = _retry_after_seconds(exc) if provider == "openrouter" and status == 429 else None
    return ProviderError(provider, model, status, code, type(exc).__name__, retry_after_seconds=retry_after)


def _retry_after_seconds(exc: Exception) -> int | None:
    """Extract only a bounded, numeric retry hint from response headers."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value:
        try:
            return max(0, min(86400, int(float(value))))
        except (TypeError, ValueError):
            try:
                reset_at = parsedate_to_datetime(str(value))
                if reset_at.tzinfo is None:
                    reset_at = reset_at.replace(tzinfo=UTC)
                return max(0, min(86400, int((reset_at - datetime.now(UTC)).total_seconds())))
            except (TypeError, ValueError, OverflowError):
                pass
    # OpenRouter/provider limits may expose a Unix reset timestamp instead.
    value = headers.get("x-ratelimit-reset") or headers.get("X-RateLimit-Reset")
    try:
        return max(0, min(86400, int(float(value) - datetime.now(UTC).timestamp()))) if value else None
    except (TypeError, ValueError, OverflowError):
        return None
