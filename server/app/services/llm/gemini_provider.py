"""Google Gemini provider — uses google-generativeai SDK."""
import logging
from collections.abc import Iterator
from fastapi import HTTPException

from app.services.llm.base import LLMProvider
from app.services.llm.errors import provider_error

log = logging.getLogger(__name__)


def _configure(api_key: str):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    except ImportError as exc:
        raise HTTPException(503, "google-generativeai SDK is not installed") from exc


def _safe_raise(exc: Exception, model: str | None = None) -> None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        raise provider_error(exc, "gemini", model, status) from exc
    name = type(exc).__name__
    msg = str(exc)
    if "API_KEY_INVALID" in msg or "PERMISSION_DENIED" in msg or "invalid" in msg.lower() and "key" in msg.lower():
        raise provider_error(exc, "gemini", model, 401) from exc
    if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower() or "rate" in msg.lower():
        raise provider_error(exc, "gemini", model, 429) from exc
    if "NOT_FOUND" in msg or "not found" in msg.lower():
        raise provider_error(exc, "gemini", model, 404) from exc
    if "UNAVAILABLE" in msg or "ServiceUnavailable" in name:
        raise provider_error(exc, "gemini", model, 503) from exc
    raise provider_error(exc, "gemini", model) from exc


class GeminiProvider(LLMProvider):
    def validate_credentials(self, api_key: str | None) -> None:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            genai = _configure(api_key)
            list(genai.list_models())
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc)

    def list_models(self, api_key: str | None) -> list[dict]:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            genai = _configure(api_key)
            return [
                {"model_key": m.name, "display_name": m.display_name or m.name, "is_local": False}
                for m in genai.list_models()
                if "generateContent" in (m.supported_generation_methods or [])
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
            genai = _configure(api_key)
            # Convert OpenAI-style messages to Gemini format
            contents = _to_gemini_contents(messages)
            system_instruction = _extract_system(messages)
            model = genai.GenerativeModel(model_key, system_instruction=system_instruction)
            kwargs = {"generation_config": {"max_output_tokens": max_output_tokens}} if max_output_tokens is not None else {}
            response = model.generate_content(contents, **kwargs)
            return response.text or ""
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc, model_key)
            return ""

    def stream(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> Iterator[str]:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            genai = _configure(api_key)
            contents = _to_gemini_contents(messages)
            system_instruction = _extract_system(messages)
            model = genai.GenerativeModel(model_key, system_instruction=system_instruction)
            kwargs = {"generation_config": {"max_output_tokens": max_output_tokens}} if max_output_tokens is not None else {}
            for chunk in model.generate_content(contents, stream=True, **kwargs):
                if chunk.text:
                    yield chunk.text
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc, model_key)


def _extract_system(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") == "system":
            return m.get("content")
    return None


def _to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style messages to Gemini content format, skipping system messages."""
    role_map = {"user": "user", "assistant": "model"}
    return [
        {"role": role_map.get(m["role"], "user"), "parts": [{"text": m["content"]}]}
        for m in messages
        if m.get("role") in role_map and m.get("content")
    ]
