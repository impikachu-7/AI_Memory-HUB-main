"""Provider registration and current model catalog metadata tests."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.llm.registry import get_provider


def _sdk_model(model_id, **metadata):
    attributes = {key: value for key, value in metadata.items() if key != "name"}
    return SimpleNamespace(id=model_id, name=metadata.get("name", model_id), model_dump=lambda: metadata, **attributes)


def test_nvidia_provider_registered_and_discovers_models():
    from app.services.llm.nvidia_provider import NvidiaProvider
    assert isinstance(get_provider("nvidia"), NvidiaProvider)
    client = MagicMock()
    client.models.list.return_value.data = [_sdk_model("nvidia/model-a", context_length=128000, labels=["Free Endpoint"])]
    with patch("openai.OpenAI", return_value=client):
        models = NvidiaProvider().list_models("user-nvidia-key")
    assert models[0]["model_key"] == "nvidia/model-a"
    assert models[0]["context_length"] == 128000
    assert models[0]["is_free"] is True


def test_nvidia_discovery_excludes_non_chat_models():
    from app.services.llm.nvidia_provider import NvidiaProvider
    client = MagicMock()
    client.models.list.return_value.data = [
        _sdk_model("meta/llama-3.1-8b-instruct"),
        _sdk_model("nvidia/nv-embedqa-e5-v5"),
        _sdk_model("nvidia/whisper-large-v3"),
    ]
    with patch("openai.OpenAI", return_value=client):
        models = NvidiaProvider().list_models("user-nvidia-key")
    assert [model["model_key"] for model in models] == ["meta/llama-3.1-8b-instruct"]


def test_nvidia_test_connection_checks_models_endpoint():
    from app.services.llm.nvidia_provider import NvidiaProvider
    client = MagicMock()
    with patch("openai.OpenAI", return_value=client):
        NvidiaProvider().validate_credentials("user-nvidia-key")
    client.models.list.assert_called_once_with()


def test_nvidia_client_uses_long_timeout_for_slow_hosted_models():
    from app.services.llm.nvidia_provider import _BASE_URL, _REQUEST_TIMEOUT_SECONDS, _client
    client = MagicMock()
    with patch("openai.OpenAI", return_value=client) as openai_client:
        assert _client("user-nvidia-key") is client

    openai_client.assert_called_once_with(
        api_key="user-nvidia-key",
        base_url=_BASE_URL,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    assert _REQUEST_TIMEOUT_SECONDS >= 180


def test_nvidia_chat_uses_openai_compatible_client():
    from app.services.llm.nvidia_provider import NvidiaProvider
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [SimpleNamespace(message=SimpleNamespace(content="Hello from NIM"))]
    with patch("openai.OpenAI", return_value=client):
        response = NvidiaProvider().generate([{"role": "user", "content": "Hi"}], "user-nvidia-key", "nvidia/model-a")
    assert response == "Hello from NIM"
    client.chat.completions.create.assert_called_once_with(model="nvidia/model-a", messages=[{"role": "user", "content": "Hi"}])


def test_nvidia_generation_resolves_unique_catalog_alias_after_404():
    from app.services.llm.nvidia_provider import NvidiaProvider

    class ModelNotFound(Exception):
        status_code = 404

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        ModelNotFound("private provider detail"),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Resolved"))]),
    ]
    client.models.list.return_value.data = [_sdk_model("meta/llama-3.1-8b-instruct")]
    with patch("openai.OpenAI", return_value=client):
        response = NvidiaProvider().generate(
            [{"role": "user", "content": "Hi"}],
            "user-nvidia-key",
            "llama-3.1-8b-instruct",
        )

    assert response == "Resolved"
    assert client.chat.completions.create.call_args_list[0].kwargs["model"] == "llama-3.1-8b-instruct"
    assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "meta/llama-3.1-8b-instruct"
    client.models.list.assert_called_once_with()


def test_nvidia_generation_does_not_guess_between_duplicate_aliases():
    from app.services.llm.nvidia_provider import NvidiaProvider
    from app.services.llm.errors import ProviderError
    import pytest

    class ModelNotFound(Exception):
        status_code = 404

    client = MagicMock()
    client.chat.completions.create.side_effect = ModelNotFound("private provider detail")
    client.models.list.return_value.data = [
        _sdk_model("publisher-a/shared-model"),
        _sdk_model("publisher-b/shared-model"),
    ]
    with patch("openai.OpenAI", return_value=client), pytest.raises(ProviderError) as error:
        NvidiaProvider().generate([{"role": "user", "content": "Hi"}], "user-nvidia-key", "shared-model")

    assert error.value.code == "MODEL_NOT_FOUND"
    assert client.chat.completions.create.call_count == 1


def test_openrouter_discovery_marks_current_zero_pricing_and_includes_router():
    from app.services.llm.openrouter_provider import OpenRouterProvider
    client = MagicMock()
    client.models.list.return_value.data = [
        _sdk_model("vendor/free-model", pricing={"prompt": "0", "completion": "0.0"}, architecture={"input_modalities": ["text"], "output_modalities": ["text"], "modality": "text->text"}),
        _sdk_model("vendor/paid-model", pricing={"prompt": "0.1", "completion": "0.2"}, architecture={"input_modalities": ["text"], "output_modalities": ["text"], "modality": "text->text"}),
        _sdk_model("vendor/image-model", pricing={"prompt": "0", "completion": "0"}, architecture={"input_modalities": ["text"], "output_modalities": ["image"], "modality": "text->image"}),
        _sdk_model("nvidia/nemotron-3.5-content-safety:free", name="Nemotron 3.5 Content Safety (free)", description="Moderates both inputs and responses; returns a safety classification.", pricing={"prompt": "0", "completion": "0"}, architecture={"input_modalities": ["text", "image"], "output_modalities": ["text"], "modality": "text+image->text"}),
    ]
    with patch("openai.OpenAI", return_value=client):
        models = OpenRouterProvider().list_models("user-openrouter-key")
    by_id = {model["model_key"]: model for model in models}
    assert by_id["vendor/free-model"]["is_free"] is True
    assert "vendor/paid-model" not in by_id
    assert "vendor/image-model" not in by_id
    assert "nvidia/nemotron-3.5-content-safety:free" not in by_id
    assert by_id["openrouter/free"]["is_free"] is True


def test_openrouter_does_not_offer_free_router_without_a_free_chat_model():
    from app.services.llm.openrouter_provider import OpenRouterProvider
    client = MagicMock()
    client.models.list.return_value.data = [
        _sdk_model("vendor/paid-model", pricing={"prompt": "0.1", "completion": "0.2"}, architecture={"input_modalities": ["text"], "output_modalities": ["text"], "modality": "text->text"}),
        _sdk_model("vendor/image-model", pricing={"prompt": "0", "completion": "0"}, architecture={"input_modalities": ["text"], "output_modalities": ["image"], "modality": "text->image"}),
    ]
    with patch("openai.OpenAI", return_value=client):
        models = OpenRouterProvider().list_models("user-openrouter-key")
    assert models == []


def test_openrouter_client_uses_bounded_timeout():
    from app.services.llm.openrouter_provider import _client
    client = MagicMock()
    with patch("openai.OpenAI", return_value=client) as openai_client:
        _client("user-openrouter-key")
    assert openai_client.call_args.kwargs["timeout"] == 45.0


def test_groq_discovery_uses_live_catalog():
    from app.services.llm.groq_provider import GroqProvider
    client = MagicMock()
    client.models.list.return_value.data = [
        _sdk_model("openai/gpt-oss-20b", context_window=131072),
        _sdk_model("whisper-large-v3", context_window=448),
        _sdk_model("canopylabs/orpheus-v1-english", context_window=4000),
        _sdk_model("llama-3.3-70b-versatile", active=False, context_window=131072),
    ]
    with patch("openai.OpenAI", return_value=client):
        models = GroqProvider().list_models("user-groq-key")
    assert [model["model_key"] for model in models] == ["openai/gpt-oss-20b"]


def test_openai_discovery_excludes_non_chat_models():
    from app.services.llm.openai_provider import OpenAIProvider
    client = MagicMock()
    client.models.list.return_value.data = [
        _sdk_model("gpt-4o-mini"),
        _sdk_model("gpt-4o-mini-transcribe"),
        _sdk_model("gpt-image-1"),
        _sdk_model("o3-mini"),
    ]
    with patch("openai.OpenAI", return_value=client):
        models = OpenAIProvider().list_models("user-openai-key")
    assert [model["model_key"] for model in models] == ["gpt-4o-mini", "o3-mini"]
