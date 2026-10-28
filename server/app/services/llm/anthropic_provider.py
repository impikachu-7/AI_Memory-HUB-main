"""Anthropic Claude provider using Anthropic's live Models API."""
import logging
from collections.abc import Iterator
from fastapi import HTTPException

from app.services.llm.base import LLMProvider
from app.services.llm.errors import provider_error

log = logging.getLogger(__name__)


def _client(api_key: str):
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key, timeout=45.0, max_retries=0)
    except ImportError as exc:
        raise HTTPException(503, "anthropic SDK is not installed") from exc


def _safe_raise(exc: Exception, model: str | None = None) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    raise provider_error(exc, "anthropic", model) from exc


def _build_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split system message out; return (system_text, user/assistant messages)."""
    system = None
    out = []
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content", "")
        elif m.get("role") in ("user", "assistant") and m.get("content"):
            out.append({"role": m["role"], "content": m["content"]})
    return system, out


class AnthropicProvider(LLMProvider):
    def validate_credentials(self, api_key: str | None) -> None:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            _client(api_key).models.list(limit=1)
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc)

    def list_models(self, api_key: str | None) -> list[dict]:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            page = _client(api_key).models.list(limit=1000)
            result = []
            for model in page.data:
                model_key = getattr(model, "id", None)
                if not model_key:
                    continue
                lifecycle = getattr(model, "lifecycle", None)
                if lifecycle and lifecycle != "active":
                    continue
                result.append({
                    "model_key": model_key,
                    "display_name": getattr(model, "display_name", None) or model_key,
                    "context_length": getattr(model, "max_input_tokens", None),
                    "is_local": False,
                })
            return result
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc)

    def generate(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> str:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        system, msgs = _build_anthropic_messages(messages)
        if not msgs:
            msgs = [{"role": "user", "content": "Hello"}]
        try:
            kwargs: dict = {"model": model_key, "max_tokens": max_output_tokens or 8096, "messages": msgs}
            if system:
                kwargs["system"] = system
            resp = _client(api_key).messages.create(**kwargs)
            return resp.content[0].text if resp.content else ""
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc, model_key)
            return ""

    def stream(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> Iterator[str]:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        system, msgs = _build_anthropic_messages(messages)
        if not msgs:
            msgs = [{"role": "user", "content": "Hello"}]
        try:
            kwargs: dict = {"model": model_key, "max_tokens": max_output_tokens or 8096, "messages": msgs}
            if system:
                kwargs["system"] = system
            with _client(api_key).messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc, model_key)
