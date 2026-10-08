"""Provider registration and current model catalog metadata tests."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.llm.registry import get_provider


def _sdk_model(model_id, **metadata):
    return SimpleNamespace(id=model_id, name=metadata.get("name", model_id), model_dump=lambda: metadata)


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


def test_nvidia_test_connection_checks_models_endpoint():
    from app.services.llm.nvidia_provider import NvidiaProvider
    client = MagicMock()
    with patch("openai.OpenAI", return_value=client):
        NvidiaProvider().validate_credentials("user-nvidia-key")
    client.models.list.assert_called_once_with()


def test_nvidia_chat_uses_openai_compatible_client():
    from app.services.llm.nvidia_provider import NvidiaProvider
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [SimpleNamespace(message=SimpleNamespace(content="Hello from NIM"))]
    with patch("openai.OpenAI", return_value=client):
        response = NvidiaProvider().generate([{"role": "user", "content": "Hi"}], "user-nvidia-key", "nvidia/model-a")
    assert response == "Hello from NIM"
    client.chat.completions.create.assert_called_once_with(model="nvidia/model-a", messages=[{"role": "user", "content": "Hi"}])


def test_openrouter_discovery_marks_current_zero_pricing_and_includes_router():
    from app.services.llm.openrouter_provider import OpenRouterProvider
    client = MagicMock()
    client.models.list.return_value.data = [
        _sdk_model("vendor/free-model", pricing={"prompt": "0", "completion": "0"}),
        _sdk_model("vendor/paid-model", pricing={"prompt": "0.1", "completion": "0.2"}),
    ]
    with patch("openai.OpenAI", return_value=client):
        models = OpenRouterProvider().list_models("user-openrouter-key")
    by_id = {model["model_key"]: model for model in models}
    assert by_id["vendor/free-model"]["is_free"] is True
    assert by_id["vendor/paid-model"]["is_free"] is False
    assert by_id["openrouter/free"]["is_free"] is True


def test_groq_discovery_uses_live_catalog():
    from app.services.llm.groq_provider import GroqProvider
    client = MagicMock()
    client.models.list.return_value.data = [_sdk_model("openai/gpt-oss-20b", context_window=131072)]
    with patch("openai.OpenAI", return_value=client):
        models = GroqProvider().list_models("user-groq-key")
    assert models[0]["model_key"] == "openai/gpt-oss-20b"
