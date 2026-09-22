"""LLM provider abstraction base class.

All concrete providers must implement validate_credentials, list_models,
generate, and stream.  Error contract:
    - Provider failures are normalized to ProviderError with a stable code.

SECURITY: exception messages must NEVER contain the API key string.
"""
from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.services.llm.errors import ProviderError


class LLMProvider(ABC):
    """Common interface for every LLM backend."""

    @abstractmethod
    def validate_credentials(self, api_key: str | None) -> None:
        """Raise HTTPException if the key is invalid or the provider is unreachable."""

    @abstractmethod
    def list_models(self, api_key: str | None) -> list[dict]:
        """Return a list of dicts with at minimum {model_key, display_name}.

        Raise HTTPException(503) if the provider endpoint is unreachable.
        """

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        api_key: str | None,
        model_key: str,
        max_output_tokens: int | None = None,
    ) -> str:
        """Non-streaming generation.  Return the complete response string."""

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        api_key: str | None,
        model_key: str,
        max_output_tokens: int | None = None,
    ) -> Iterator[str]:
        """Streaming generation.  Yield text chunks as they arrive.

        Raise ProviderError on provider error.
        """
