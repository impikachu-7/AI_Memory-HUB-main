"""NVIDIA NIM provider using NVIDIA's OpenAI-compatible API."""
from collections.abc import Iterator

from fastapi import HTTPException

from app.services.llm.base import LLMProvider
from app.services.llm.errors import provider_error

_BASE_URL = "https://integrate.api.nvidia.com/v1"
# NVIDIA's shared endpoints can stall. Keep one attempt bounded so automatic
# SDK retries cannot turn a single timeout into several minutes of silence.
_REQUEST_TIMEOUT_SECONDS = 90.0
_NON_CHAT_MODEL_MARKERS = (
    "embed", "rerank", "whisper", "speech", "transcrib", "tts", "flux",
    "image-generation", "video-generation", "ising-calibration",
)


def _supports_chat_completions(model_id: str, metadata: dict) -> bool:
    lowered = model_id.lower()
    if any(marker in lowered for marker in _NON_CHAT_MODEL_MARKERS):
        return False
    endpoints = metadata.get("supported_endpoints")
    if isinstance(endpoints, list):
        # When NVIDIA supplies endpoint metadata, respect it. Some versions of
        # its OpenAI-compatible /models endpoint omit these optional fields,
        # however; absence is not evidence that a valid LLM is non-chat.
        return any("chat/completions" in str(endpoint).lower() for endpoint in endpoints)
    modalities = metadata.get("modalities")
    if isinstance(modalities, dict):
        inputs = modalities.get("input") or modalities.get("input_modalities")
        outputs = modalities.get("output") or modalities.get("output_modalities")
        if isinstance(inputs, list) and isinstance(outputs, list):
            return "text" in inputs and "text" in outputs
        if isinstance(inputs, list) and "text" not in inputs:
            return False
        if isinstance(outputs, list) and "text" not in outputs:
            return False
    # NVIDIA's OpenAI-compatible /models response commonly contains only
    # standard model fields (id/object/owned_by) rather than modality metadata.
    # Keep unknown IDs because this endpoint is the chat API's own catalog;
    # known non-chat model families and explicit negative modality metadata
    # were rejected above. Free status remains a separate, strict check.
    return True


def _client(api_key: str):
    try:
        import openai
        return openai.OpenAI(
            api_key=api_key,
            base_url=_BASE_URL,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
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


def _catalog_model_id(api_key: str, selected_model: str) -> str | None:
    """Resolve a short NVIDIA model alias only when the catalog match is unique.

    NVIDIA documents canonical IDs such as ``meta/llama-3.1-8b-instruct``.
    Some catalog clients or previously saved selections can omit the publisher
    prefix. Never guess between duplicate suffixes and never rewrite a valid
    exact ID.
    """
    try:
        models = _client(api_key).models.list()
    except Exception:
        return None

    ids = [getattr(model, "id", None) for model in models.data]
    ids = [model_id for model_id in ids if isinstance(model_id, str) and model_id]
    if selected_model in ids:
        return selected_model

    matches = [model_id for model_id in ids if model_id.rsplit("/", 1)[-1] == selected_model]
    return matches[0] if len(matches) == 1 else None


def _is_not_found(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 404


def _free_status(metadata: dict) -> bool | None:
    for field in ("is_free", "free", "is_free_endpoint", "free_endpoint"):
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
                if not _supports_chat_completions(model_key, raw):
                    continue
                result.append({"model_key": model_key, "display_name": raw.get("name") or model_key,
                               "is_local": False, "context_length": raw.get("context_length"),
                               "is_free": _free_status(raw)})
            return result
        except Exception as exc:
            _raise(exc)

    def generate(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> str:
        try:
            key = _require_key(api_key)
            kwargs = {"model": model_key, "messages": messages}
            if max_output_tokens is not None:
                kwargs["max_tokens"] = max_output_tokens
            client = _client(key)
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                canonical_id = _catalog_model_id(key, model_key) if _is_not_found(exc) else None
                if not canonical_id or canonical_id == model_key:
                    raise
                kwargs["model"] = canonical_id
                response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            _raise(exc, model_key)

    def stream(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> Iterator[str]:
        try:
            key = _require_key(api_key)
            kwargs = {"model": model_key, "messages": messages, "stream": True}
            if max_output_tokens is not None:
                kwargs["max_tokens"] = max_output_tokens
            client = _client(key)
            yielded = False
            try:
                stream = client.chat.completions.create(**kwargs)
                with stream:
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            yielded = True
                            yield delta
            except Exception as exc:
                canonical_id = _catalog_model_id(key, model_key) if _is_not_found(exc) and not yielded else None
                if not canonical_id or canonical_id == model_key:
                    raise
                kwargs["model"] = canonical_id
                with client.chat.completions.create(**kwargs) as stream:
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            yield delta
        except Exception as exc:
            _raise(exc, model_key)
