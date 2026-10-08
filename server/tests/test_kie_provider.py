import pytest
from fastapi import HTTPException
from unittest.mock import Mock, patch

from app.services.llm.kie_provider import KieProvider, _input, _key, _text


def test_kie_response_input_keeps_image_urls_server_side():
    payload = _input([
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "What is in this image?", "image_url": "https://example.com/image.png"},
    ])
    assert payload[0]["content"] == [{"type": "input_text", "text": "Be helpful."}]
    assert payload[1]["content"][1] == {"type": "input_image", "image_url": "https://example.com/image.png"}


def test_kie_response_text_handles_standard_responses_payload():
    assert _text({"output": [{"content": [{"type": "output_text", "text": "A teal logo."}]}]}) == "A teal logo."


def test_kie_requires_user_credential():
    with pytest.raises(HTTPException) as error:
        KieProvider().list_models(None)
    assert error.value.status_code == 401
    assert "KIE_API_KEY" not in str(error.value.detail)


def test_kie_test_connection_uses_supplied_user_credential():
    response = Mock(status_code=200, json=lambda: {"models": [{"slug": "gpt-5-5", "name": "GPT 5.5"}]})
    with patch("app.services.llm.kie_provider.httpx.get", return_value=response) as get:
        KieProvider().validate_credentials("user-owned-kie-secret")
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer user-owned-kie-secret"
    assert _key("user-owned-kie-secret") == "user-owned-kie-secret"


def test_kie_invalid_key_returns_safe_error():
    with patch("app.services.llm.kie_provider.httpx.get", return_value=Mock(status_code=401)):
        with pytest.raises(HTTPException) as error:
            KieProvider().validate_credentials("private-invalid-key")
    assert error.value.status_code == 401
    assert "private-invalid-key" not in str(error.value.detail)


def test_kie_model_discovery_uses_live_catalog():
    response = Mock(status_code=200, json=lambda: {"models": [{"slug": "gpt-5-5", "name": "GPT 5.5", "context_window": 272000}]})
    with patch("app.services.llm.kie_provider.httpx.get", return_value=response):
        models = KieProvider().list_models("user-owned-kie-secret")
    assert models == [{"model_key": "gpt-5-5", "display_name": "GPT 5.5", "context_length": 272000, "is_local": False}]


def test_kie_rejects_malformed_model_catalog_safely():
    response = Mock(status_code=200, json=lambda: ["not", "an", "object"])
    with patch("app.services.llm.kie_provider.httpx.get", return_value=response):
        with pytest.raises(HTTPException) as error:
            KieProvider().list_models("user-owned-kie-secret")
    assert error.value.status_code == 502
    assert "user-owned-kie-secret" not in str(error.value.detail)


def test_kie_generation_uses_user_supplied_credential():
    response = Mock(status_code=200, json=lambda: {"output_text": "Hello"})
    with patch("app.services.llm.kie_provider.httpx.post", return_value=response) as post:
        result = KieProvider().generate([{"role": "user", "content": "Hi"}], "user-owned-kie-secret", "gpt-5-5")
    assert result == "Hello"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer user-owned-kie-secret"
