"""NVIDIA NIM provider using NVIDIA's OpenAI-compatible API."""
from collections.abc import Iterator

from fastapi import HTTPException

from app.services.llm.base import LLMProvider
from app.services.llm.errors import provider_error

_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _client(api_key: str):
    try:
        import openai
        return openai.OpenAI(api_key=api_key, base_url=_BASE_URL, timeout=30)
    except ImportError as exc:
        raise HTTPException(503, "openai SDK is not installed") from exc


def _require_key(api_key: str | None) -> str:
    if not api_key:
        raise HTTPException(401, "Provider authentication failed")
    return api_key


def _raise(exc: Exception, model: str | None = None):
    if isinstance(exc, HTTPException):
        raise exc
    raise provider_error(exc, "nvidia", model) from exc


def _free_status(metadata: dict) -> bool | None:
    for field in ("is_free", "free"):
        if isinstance(metadata.get(field), bool):
            return metadata[field]
    labels = metadata.get("labels") or metadata.get("tags") or []
    if isinstance(labels, list) and any(isinstance(label, str) and "free endpoint" in label.lower() for label in labels):
        return True
    pricing = metadata.get("pricing")
    if isinstance(pricing, dict):
        prompt, completion = pricing.get("prompt"), pricing.get("completion")
        try:
            if prompt is not None and completion is not None:
                return float(prompt) == 0 and float(completion) == 0
        except (TypeError, ValueError):
            return None
    return None


class NvidiaProvider(LLMProvider):
    def validate_credentials(self, api_key: str | None) -> None:
        try:
            _client(_require_key(api_key)).models.list()
        except Exception as exc:
            _raise(exc)

    def list_models(self, api_key: str | None) -> list[dict]:
        try:
            models = _client(_require_key(api_key)).models.list()
            result = []
            for model in models.data:
                raw = model.model_dump() if hasattr(model, "model_dump") else {}
                model_key = getattr(model, "id", None)
                if not model_key:
                    continue
                result.append({"model_key": model_key, "display_name": raw.get("name") or model_key,
                               "is_local": False, "context_length": raw.get("context_length"),
                               "is_free": _free_status(raw)})
            return result
        except Exception as exc:
            _raise(exc)

    def generate(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> str:
        try:
            kwargs = {"model": model_key, "messages": messages}
            if max_output_tokens is not None:
                kwargs["max_tokens"] = max_output_tokens
            response = _client(_require_key(api_key)).chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            _raise(exc, model_key)

    def stream(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> Iterator[str]:
        try:
            kwargs = {"model": model_key, "messages": messages, "stream": True}
            if max_output_tokens is not None:
                kwargs["max_tokens"] = max_output_tokens
            with _client(_require_key(api_key)).chat.completions.create(**kwargs) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
        except Exception as exc:
            _raise(exc, model_key)
