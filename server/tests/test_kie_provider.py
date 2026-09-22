from app.services.llm.kie_provider import _input, _text


def test_kie_response_input_keeps_image_urls_server_side():
    payload = _input([
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "What is in this image?", "image_url": "https://example.com/image.png"},
    ])
    assert payload[0]["content"] == [{"type": "input_text", "text": "Be helpful."}]
    assert payload[1]["content"][1] == {"type": "input_image", "image_url": "https://example.com/image.png"}


def test_kie_response_text_handles_standard_responses_payload():
    assert _text({"output": [{"content": [{"type": "output_text", "text": "A teal logo."}]}]}) == "A teal logo."
