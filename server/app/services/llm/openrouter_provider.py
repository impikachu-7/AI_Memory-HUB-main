"""OpenRouter provider — OpenAI-compatible API at https://openrouter.ai/api/v1.

OpenRouter requires HTTP-Referer and X-Title headers for routing.
These are non-sensitive application identifiers, not credentials.
"""
import logging
from collections.abc import Iterator
from fastapi import HTTPException

from app.services.llm.base import LLMProvider
from app.services.llm.errors import provider_error

log = logging.getLogger(__name__)

_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://ai-memory-hub.app",
    "X-Title": "AI Memory Hub",
}
_NON_CHAT_MODEL_MARKERS = (
    "content-safety", "safety-guard", "guardrail", "moderation", "classifier",
    "embedding", "embed-", "rerank", "whisper", "transcrib", "text-to-speech",
    "image-generation", "video-generation",
)


def _supports_text_chat(model_id: str, metadata: dict) -> bool:
    """Keep specialized safety/task models out of the general chat picker."""
    lowered_id = model_id.lower()
    if any(marker in lowered_id for marker in _NON_CHAT_MODEL_MARKERS):
        return False
    name = str(metadata.get("name") or "").lower()
    description = str(metadata.get("description") or "").lower()
    if any(marker in name for marker in _NON_CHAT_MODEL_MARKERS):
        return False
    if any(marker in description for marker in (
        "moderates both inputs", "safety classification", "content moderation",
        "safety classifier", "embedding model", "reranking model",
        "speech recognition model", "text-to-speech model",
    )):
        return False

    architecture = metadata.get("architecture")
    if not isinstance(architecture, dict):
        # Unknown modality is not sufficient evidence for a chat picker.
        return False
    input_modalities = architecture.get("input_modalities")
    output_modalities = architecture.get("output_modalities")
    if isinstance(input_modalities, list) and isinstance(output_modalities, list):
        return "text" in input_modalities and "text" in output_modalities
    if isinstance(input_modalities, list) and "text" not in input_modalities:
        return False
    if isinstance(output_modalities, list) and "text" not in output_modalities:
        return False
    modality = architecture.get("modality")
    if isinstance(modality, str) and "->" in modality:
        input_mode, output_mode = modality.split("->", 1)
        return "text" in input_mode.split("+") and "text" in output_mode.split("+")
    return False


def _client(api_key: str):
    try:
        import openai
        return openai.OpenAI(
            api_key=api_key,
            base_url=_BASE_URL,
            default_headers=_DEFAULT_HEADERS,
            timeout=45.0,
            max_retries=0,
        )
    except ImportError as exc:
        raise HTTPException(503, "openai SDK is not installed") from exc


def _safe_raise(exc: Exception, model: str | None = None) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    raise provider_error(exc, "openrouter", model) from exc


class OpenRouterProvider(LLMProvider):
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
            discovered = []
            for model in models.data:
                metadata = model.model_dump() if hasattr(model, "model_dump") else {}
                if not _supports_text_chat(model.id, metadata):
                    continue
                pricing = metadata.get("pricing") or getattr(model, "pricing", None) or {}
                if not isinstance(pricing, dict):
                    pricing = getattr(pricing, "model_dump", lambda: {})()
                prompt = pricing.get("prompt")
                completion = pricing.get("completion")
                try:
                    is_free = prompt is not None and completion is not None and float(prompt) == 0 and float(completion) == 0
                except (TypeError, ValueError):
                    # Incomplete or nonnumeric prices are unknown, not free.
                    is_free = False
                # This integration was added for free-model discovery. Paid
                # catalog entries can be listed by OpenRouter while a user's
                # account has no credits, then fail at generation with HTTP 402.
                if not is_free:
                    continue
                discovered.append({"model_key": model.id, "display_name": getattr(model, "name", model.id) or model.id,
                                   "is_local": False, "context_length": metadata.get("context_length"), "is_free": is_free})
            # The provider's free model router is a supported route in its own right.
            has_free_chat_model = any(item.get("is_free") is True for item in discovered)
            if has_free_chat_model and not any(item["model_key"] == "openrouter/free" for item in discovered):
                discovered.insert(0, {"model_key": "openrouter/free", "display_name": "Free Models Router", "is_local": False,
                                      "context_length": 200000, "is_free": True})
            return discovered
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

