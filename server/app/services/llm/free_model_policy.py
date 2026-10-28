"""Allowlist provider catalogs to models with verifiable free inference.

Provider model-list APIs generally prove that a model exists, not that it is
free. Unknown pricing is intentionally rejected. Gemini and Groq currently
publish free-plan availability separately from their model-list APIs, so those
catalogs are intersected with the verified IDs below and the live API list.
Last reviewed against official pricing/rate-limit docs on 2026-10-29.
"""
from __future__ import annotations


# Gemini Developer API pricing explicitly marks Standard text inference as
# free for these models. Use model-list IDs as returned, stripping only the SDK
# resource prefix for comparison (the original ID is still sent to Gemini).
# Source: https://ai.google.dev/gemini-api/docs/pricing
_GEMINI_FREE_CHAT_MODELS = frozenset({
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
})

# Groq's Free Plan Limits table lists these currently available chat models.
# Non-chat rows (speech, prompt guard, safeguard) are deliberately omitted.
# Source: https://console.groq.com/docs/rate-limits
_GROQ_FREE_CHAT_MODELS = frozenset({
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
})


def _normalized_id(provider: str, model_key: str) -> str:
    value = model_key.strip()
    if provider == "gemini" and value.startswith("models/"):
        return value[len("models/"):]
    return value


def is_verified_free_model(provider: str, model: dict) -> bool:
    """Return true only for local inference or a positively verified free ID."""
    model_key = model.get("model_key")
    if not isinstance(model_key, str) or not model_key.strip():
        return False
    if provider == "ollama":
        return model.get("is_local") is True
    if provider == "gemini":
        return _normalized_id(provider, model_key) in _GEMINI_FREE_CHAT_MODELS
    if provider == "groq":
        return model_key in _GROQ_FREE_CHAT_MODELS
    # OpenRouter/NVIDIA report model-level price/free-endpoint evidence.
    # Other APIs list paid and free models together, or have no permanent
    # verifiable free tier; a missing value must never be treated as free.
    return model.get("is_free") is True


def verified_catalog(provider: str, models: list[dict]) -> list[dict]:
    """Return a normalized catalog containing only verified free chat models."""
    result = []
    seen: set[str] = set()
    for model in models:
        if not is_verified_free_model(provider, model):
            continue
        model_key = model["model_key"].strip()
        if model_key in seen:
            continue
        seen.add(model_key)
        result.append({**model, "model_key": model_key, "is_free": provider != "ollama", "is_local": provider == "ollama"})
    return result

