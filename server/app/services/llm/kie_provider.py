"""Kie Responses API provider.

The Kie key is deployment-managed (KIE_API_KEY) and never reaches the browser.
The API is non-streaming, so ``stream`` yields the completed text once while
preserving the application's existing NDJSON response contract.
"""
from collections.abc import Iterator
import logging

import httpx
from fastapi import HTTPException

from app.core.config import get_settings
from app.services.llm.base import LLMProvider

log = logging.getLogger(__name__)
_URL = "https://api.kie.ai/codex/v1/responses"
# The requested Kie Chat catalog.  These are intentionally limited to chat
# models; Kie's image/video/audio task models use different asynchronous APIs.
_MODELS = [
    {"model_key": "gpt-6-astra", "display_name": "GPT-6 Astra", "is_local": False},
    {"model_key": "gpt-5.6", "display_name": "GPT-5.6", "is_local": False},
    {"model_key": "gpt-5.5", "display_name": "GPT-5.5", "is_local": False},
    {"model_key": "gpt-5.2", "display_name": "GPT-5.2", "is_local": False},
    {"model_key": "gemini-3.8-flash", "display_name": "Gemini 3.8 Flash", "is_local": False},
    {"model_key": "gemini-3.7-flash", "display_name": "Gemini 3.7 Flash", "is_local": False},
    {"model_key": "gemini-3.6-flash", "display_name": "Gemini 3.6 Flash", "is_local": False},
    {"model_key": "claude-sonnet", "display_name": "Claude Sonnet", "is_local": False},
    {"model_key": "claude-opus", "display_name": "Claude Opus", "is_local": False},
    {"model_key": "grok-4.6", "display_name": "Grok 4.6 (via Kie)", "is_local": False},
    {"model_key": "grok-4.5", "display_name": "Grok 4.5 (via Kie)", "is_local": False},
    {"model_key": "grok-4.3", "display_name": "Grok 4.3 (via Kie)", "is_local": False},
    {"model_key": "openai-codex", "display_name": "OpenAI Codex (via Kie)", "is_local": False},
]


def _key() -> str:
    key = get_settings().kie_api_key
    if not key:
        raise HTTPException(503, "Kie is not configured")
    return key


def _input(messages: list[dict]) -> list[dict]:
    result = []
    for message in messages:
        content = [{"type": "input_text", "text": message["content"]}]
        # Image URLs are accepted only for the current user turn and only by Kie.
        if message.get("image_url"):
            content.append({"type": "input_image", "image_url": message["image_url"]})
        result.append({"role": message["role"], "content": content})
    return result


def _text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                parts.append(content.get("text") or content.get("value") or "")
    return "".join(parts)


class KieProvider(LLMProvider):
    def validate_credentials(self, api_key: str | None) -> None:
        _key()

    def list_models(self, api_key: str | None) -> list[dict]:
        _key()
        return _MODELS

    def generate(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> str:
        if model_key not in {model["model_key"] for model in _MODELS}:
            raise HTTPException(400, "Model not available from provider")
        try:
            response = httpx.post(
                _URL,
                headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
                json={"model": model_key, "input": _input(messages)},
                timeout=60,
            )
        except httpx.RequestError as exc:
            raise HTTPException(503, "Kie is temporarily unavailable") from exc
        if response.status_code in {401, 403}:
            raise HTTPException(401, "Provider authentication failed")
        if response.status_code == 429:
            raise HTTPException(429, "Provider rate limit reached")
        if response.status_code == 404:
            raise HTTPException(400, "Model not available from provider")
        if response.status_code >= 500:
            raise HTTPException(503, "Kie is temporarily unavailable")
        if response.status_code >= 400:
            raise HTTPException(400, "Invalid request to provider")
        try:
            text = _text(response.json())
        except ValueError as exc:
            raise HTTPException(502, "Provider returned an unexpected response") from exc
        if not text:
            raise HTTPException(502, "Provider returned an empty response")
        return text

    def stream(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> Iterator[str]:
        yield self.generate(messages, api_key, model_key, max_output_tokens)
