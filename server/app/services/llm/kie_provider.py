"""Kie Responses API provider.

Users provide a Kie API key through their encrypted provider configuration.
The API is non-streaming, so ``stream`` yields the completed text once while
preserving the application's existing NDJSON response contract.
"""
from collections.abc import Iterator
import logging

import httpx
from fastapi import HTTPException

from app.services.llm.base import LLMProvider

log = logging.getLogger(__name__)
_URL = "https://api.kie.ai/codex/v1/responses"
_MODELS_URL = "https://api.kie.ai/openai/v1/models"


def _key(api_key: str | None) -> str:
    if not api_key:
        raise HTTPException(401, "Provider authentication failed")
    return api_key


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
    if not isinstance(payload, dict):
        raise ValueError("Unexpected response shape")
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts = []
    output = payload.get("output", [])
    if not isinstance(output, list):
        raise ValueError("Unexpected response shape")
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                parts.append(content.get("text") or content.get("value") or "")
    return "".join(parts)


class KieProvider(LLMProvider):
    def validate_credentials(self, api_key: str | None) -> None:
        self.list_models(_key(api_key))

    def list_models(self, api_key: str | None) -> list[dict]:
        key = _key(api_key)
        try:
            response = httpx.get(_MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30)
        except httpx.RequestError as exc:
            raise HTTPException(503, "Kie is temporarily unavailable") from exc
        if response.status_code in {401, 403}:
            raise HTTPException(401, "Provider authentication failed")
        if response.status_code == 429:
            raise HTTPException(429, "Provider rate limit reached")
        if response.status_code >= 500:
            raise HTTPException(503, "Kie is temporarily unavailable")
        if response.status_code >= 400:
            raise HTTPException(400, "Kie model discovery failed")
        try:
            payload = response.json()
        except ValueError as exc:
            raise HTTPException(502, "Provider returned an unexpected response") from exc
        if not isinstance(payload, dict):
            raise HTTPException(502, "Provider returned an unexpected response")
        entries = payload.get("models") or payload.get("data") or []
        if not isinstance(entries, list):
            raise HTTPException(502, "Provider returned an unexpected response")
        result = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            model_key = item.get("slug") or item.get("id")
            if model_key:
                result.append({"model_key": str(model_key), "display_name": str(item.get("name") or model_key),
                               "context_length": item.get("context_window"), "is_local": False})
        return result

    def generate(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> str:
        try:
            response = httpx.post(
                _URL,
                headers={"Authorization": f"Bearer {_key(api_key)}", "Content-Type": "application/json"},
                json={"model": model_key, "input": _input(messages), **({"max_output_tokens": max_output_tokens} if max_output_tokens is not None else {})},
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
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(502, "Provider returned an unexpected response") from exc
        if not text:
            raise HTTPException(502, "Provider returned an empty response")
        return text

    def stream(self, messages: list[dict], api_key: str | None, model_key: str, max_output_tokens: int | None = None) -> Iterator[str]:
        yield self.generate(messages, api_key, model_key, max_output_tokens)
