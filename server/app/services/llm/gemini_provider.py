"""Google Gemini provider using the supported Google GenAI SDK."""
from collections.abc import Iterator
from fastapi import HTTPException

from app.services.llm.base import LLMProvider
from app.services.llm.errors import provider_error


def _client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise HTTPException(503, "google-genai SDK is not installed") from exc
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        raise provider_error(exc, "gemini") from exc


def _model_name(model_key: str) -> str:
    value = (model_key or "").strip()
    if value.startswith("models/"):
        return value[len("models/"):]
    return value


def _safe_raise(exc: Exception, model: str | None = None) -> None:
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "code", None)
    if isinstance(status, int):
        raise provider_error(exc, "gemini", model, status) from exc

    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "api key" in msg or "api_key" in msg or "unauthorized" in msg or "permission" in msg:
        raise provider_error(exc, "gemini", model, 401) from exc
    if "quota" in msg or "resource exhausted" in msg or "rate limit" in msg:
        raise provider_error(exc, "gemini", model, 429) from exc
    if "not found" in msg:
        raise provider_error(exc, "gemini", model, 404) from exc
    if "timeout" in name or "deadline" in msg:
        raise provider_error(exc, "gemini", model, 408) from exc
    if "unavailable" in msg or "connection" in msg:
        raise provider_error(exc, "gemini", model, 503) from exc
    raise provider_error(exc, "gemini", model) from exc


class GeminiProvider(LLMProvider):
    def validate_credentials(self, api_key: str | None) -> None:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            client = _client(api_key)
            next(iter(client.models.list()), None)
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc)

    def list_models(self, api_key: str | None) -> list[dict]:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            client = _client(api_key)
            result: list[dict] = []
            for model in client.models.list():
                supported = getattr(model, "supported_actions", None) or []
                if "generateContent" not in supported:
                    continue
                name = getattr(model, "name", None)
                if not name:
                    continue
                display = getattr(model, "display_name", None) or name
                result.append(
                    {
                        "model_key": name,
                        "display_name": display,
                        "is_local": False,
                    }
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc)
            return []

    def _contents_and_system(self, messages: list[dict]):
        from google.genai import types

        system_parts: list[str] = []
        contents = []
        role_map = {"user": "user", "assistant": "model"}

        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
                continue
            mapped_role = role_map.get(role)
            if not mapped_role:
                continue
            contents.append(
                types.Content(
                    role=mapped_role,
                    parts=[types.Part(text=content)],
                )
            )

        return contents, "\n\n".join(system_parts) or None

    def _config(self, system_instruction: str | None, max_output_tokens: int | None):
        from google.genai import types

        kwargs = {}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        return types.GenerateContentConfig(**kwargs) if kwargs else None

    def generate(
        self,
        messages: list[dict],
        api_key: str | None,
        model_key: str,
        max_output_tokens: int | None = None,
    ) -> str:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            client = _client(api_key)
            contents, system_instruction = self._contents_and_system(messages)
            response = client.models.generate_content(
                model=_model_name(model_key),
                contents=contents,
                config=self._config(system_instruction, max_output_tokens),
            )
            return response.text or ""
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc, model_key)
            return ""

    def stream(
        self,
        messages: list[dict],
        api_key: str | None,
        model_key: str,
        max_output_tokens: int | None = None,
    ) -> Iterator[str]:
        if not api_key:
            raise HTTPException(401, "Provider authentication failed")
        try:
            client = _client(api_key)
            contents, system_instruction = self._contents_and_system(messages)
            stream = client.models.generate_content_stream(
                model=_model_name(model_key),
                contents=contents,
                config=self._config(system_instruction, max_output_tokens),
            )
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except HTTPException:
            raise
        except Exception as exc:
            _safe_raise(exc, model_key)
