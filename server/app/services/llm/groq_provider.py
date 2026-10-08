"""Groq provider — OpenAI-compatible API at https://api.groq.com/openai/v1."""
import logging
from collections.abc import Iterator
from fastapi import HTTPException

from app.services.llm.base import LLMProvider
from app.services.llm.errors import provider_error

log = logging.getLogger(__name__)

_BASE_URL = "https://api.groq.com/openai/v1"

_NON_CHAT_MODEL_MARKERS = (
    "whisper", "distil-whisper", "orpheus", "text-to-speech", "speech-to-text",
    "transcription", "transcribe", "embedding", "rerank", "audio", "tts",
)


def _supports_chat_completions(model) -> bool:
    """Groq's /models catalog also includes speech and other task models."""
    model_id = getattr(model, "id", "")
    if not isinstance(model_id, str) or not model_id:
        return False
    lowered = model_id.lower()
    if any(marker in lowered for marker in _NON_CHAT_MODEL_MARKERS):
        return False

    metadata = model.model_dump() if hasattr(model, "model_dump") else {}
    endpoints = metadata.get("supported_endpoints")
    if isinstance(endpoints, list):
        return any("chat/completions" in str(endpoint).lower() for endpoint in endpoints)
    modalities = metadata.get("modalities")
    if isinstance(modalities, dict):
        inputs = modalities.get("input") or modalities.get("input_modalities")
        outputs = modalities.get("output") or modalities.get("output_modalities")
        if isinstance(inputs, list) and isinstance(outputs, list):
            return "text" in inputs and "text" in outputs
    return True


def _client(api_key: str):
    try:
        import openai
        return openai.OpenAI(api_key=api_key, base_url=_BASE_URL)
    except ImportError as exc:
        raise HTTPException(503, "openai SDK is not installed") from exc


def _safe_raise(exc: Exception, model: str | None = None) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    raise provider_error(exc, "groq", model) from exc


class GroqProvider(LLMProvider):
    def validate_credentials(self, api_key: str | None) -> None:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            _client(api_key).models.list()
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc)

    def list_models(self, api_key: str | None) -> list[dict]:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            models = _client(api_key).models.list()
            return [
                {"model_key": m.id, "display_name": m.id, "is_local": False,
                 "context_length": getattr(m, "context_window", None), "is_free": None}
                for m in models.data if getattr(m, "active", True) and _supports_chat_completions(m)
            ]
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc)
            return []

    def generate(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> str:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            kwargs = {"model": model_key, "messages": messages}
            if max_output_tokens is not None:
                kwargs["max_tokens"] = max_output_tokens
            resp = _client(api_key).chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc, model_key)
            return ""

    def stream(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> Iterator[str]:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            kwargs = {"model": model_key, "messages": messages, "stream": True}
            if max_output_tokens is not None:
                kwargs["max_tokens"] = max_output_tokens
            with _client(api_key).chat.completions.create(**kwargs) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc, model_key)
