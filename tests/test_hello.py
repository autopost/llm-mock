# tests/test_hello.py
import anthropic
import pytest

client = anthropic.Anthropic(api_key="fake-key")

@pytest.mark.llm_replay(fixture="hello")
def test_hello():
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    assert message.content[0].text  # replayed from fixtures/hello.json