import json
from unittest.mock import patch

import httpx
import pytest
import respx

from llm_mock import llm_mock, FixtureNotFoundError
from llm_mock.fixture_store import Interaction, save

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

ANTHROPIC_REQUEST = {
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 10,
}

OPENAI_REQUEST = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Say hi"}],
}

FAKE_ANTHROPIC_RESPONSE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi there!"}],
    "model": "claude-sonnet-4-6",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}

FAKE_OPENAI_RESPONSE = {
    "id": "chatcmpl-01",
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
}


def _post(url: str, body: dict) -> httpx.Response:
    return httpx.post(url, json=body, headers={"x-api-key": "test"})


def _make_fake_forward(response_body: dict, status: int = 200):
    def fake_forward(request, body):
        return httpx.Response(status, json=response_body)
    return fake_forward


# --- Anthropic record ---

def test_anthropic_record_saves_fixture(tmp_path):
    fixture = tmp_path / "anth.json"
    with patch(
        "llm_mock.providers.anthropic._forward_request",
        side_effect=_make_fake_forward(FAKE_ANTHROPIC_RESPONSE),
    ):
        with llm_mock(mode="record", fixture=str(fixture), provider="anthropic"):
            _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    assert fixture.exists()
    data = json.loads(fixture.read_text())
    assert data["provider"] == "anthropic"
    assert len(data["interactions"]) == 1


def test_anthropic_record_returns_real_response(tmp_path):
    fixture = tmp_path / "anth.json"
    with patch(
        "llm_mock.providers.anthropic._forward_request",
        side_effect=_make_fake_forward(FAKE_ANTHROPIC_RESPONSE),
    ):
        with llm_mock(mode="record", fixture=str(fixture), provider="anthropic"):
            response = _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    assert response.status_code == 200
    assert response.json() == FAKE_ANTHROPIC_RESPONSE


# --- Anthropic replay ---

def test_anthropic_replay_returns_fixture(tmp_path):
    fixture = tmp_path / "anth.json"
    with patch(
        "llm_mock.providers.anthropic._forward_request",
        side_effect=_make_fake_forward(FAKE_ANTHROPIC_RESPONSE),
    ):
        with llm_mock(mode="record", fixture=str(fixture), provider="anthropic"):
            _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    with llm_mock(mode="replay", fixture=str(fixture), provider="anthropic"):
        response = _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    assert response.status_code == 200
    assert response.json() == FAKE_ANTHROPIC_RESPONSE


def test_anthropic_replay_no_real_call(tmp_path):
    fixture = tmp_path / "anth.json"
    with patch(
        "llm_mock.providers.anthropic._forward_request",
        side_effect=_make_fake_forward(FAKE_ANTHROPIC_RESPONSE),
    ):
        with llm_mock(mode="record", fixture=str(fixture), provider="anthropic"):
            _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    with patch("llm_mock.providers.anthropic._forward_request") as mock_fwd:
        with llm_mock(mode="replay", fixture=str(fixture), provider="anthropic"):
            _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)
        mock_fwd.assert_not_called()


# --- Replay error cases ---

def test_replay_raises_when_no_fixture(tmp_path):
    fixture = tmp_path / "empty.json"
    with pytest.raises(FixtureNotFoundError):
        with llm_mock(mode="replay", fixture=str(fixture), provider="anthropic"):
            _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)


# --- OpenAI record ---

def test_openai_record_saves_fixture(tmp_path):
    fixture = tmp_path / "oai.json"
    with patch(
        "llm_mock.providers.openai._forward_request",
        side_effect=_make_fake_forward(FAKE_OPENAI_RESPONSE),
    ):
        with llm_mock(mode="record", fixture=str(fixture), provider="openai"):
            _post(OPENAI_URL, OPENAI_REQUEST)

    assert fixture.exists()
    data = json.loads(fixture.read_text())
    assert data["provider"] == "openai"


# --- OpenAI replay ---

def test_openai_replay_returns_fixture(tmp_path):
    fixture = tmp_path / "oai.json"
    with patch(
        "llm_mock.providers.openai._forward_request",
        side_effect=_make_fake_forward(FAKE_OPENAI_RESPONSE),
    ):
        with llm_mock(mode="record", fixture=str(fixture), provider="openai"):
            _post(OPENAI_URL, OPENAI_REQUEST)

    with llm_mock(mode="replay", fixture=str(fixture), provider="openai"):
        response = _post(OPENAI_URL, OPENAI_REQUEST)

    assert response.status_code == 200
    assert response.json() == FAKE_OPENAI_RESPONSE


# --- FEAT-5: edge cases in record mode ---

def test_record_does_not_save_on_4xx(tmp_path):
    fixture = tmp_path / "anth.json"
    error_body = {"type": "error", "error": {"type": "authentication_error"}}
    with patch(
        "llm_mock.providers.anthropic._forward_request",
        side_effect=_make_fake_forward(error_body, status=401),
    ):
        with llm_mock(mode="record", fixture=str(fixture), provider="anthropic"):
            response = _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    assert response.status_code == 401
    assert not fixture.exists()


def test_record_does_not_save_on_5xx(tmp_path):
    fixture = tmp_path / "anth.json"
    with patch(
        "llm_mock.providers.anthropic._forward_request",
        side_effect=_make_fake_forward({"error": "server error"}, status=500),
    ):
        with llm_mock(mode="record", fixture=str(fixture), provider="anthropic"):
            response = _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    assert response.status_code == 500
    assert not fixture.exists()


def test_record_propagates_network_error(tmp_path):
    fixture = tmp_path / "anth.json"
    with patch(
        "llm_mock.providers.anthropic._forward_request",
        side_effect=httpx.ConnectError("timeout"),
    ):
        with pytest.raises(httpx.ConnectError):
            with llm_mock(mode="record", fixture=str(fixture), provider="anthropic"):
                _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    assert not fixture.exists()


# --- provider="all" ---

def test_all_provider_intercepts_both(tmp_path):
    fixture = tmp_path / "all.json"
    with patch("llm_mock.providers.anthropic._forward_request",
               side_effect=_make_fake_forward(FAKE_ANTHROPIC_RESPONSE)), \
         patch("llm_mock.providers.openai._forward_request",
               side_effect=_make_fake_forward(FAKE_OPENAI_RESPONSE)):
        with llm_mock(mode="record", fixture=str(fixture), provider="all"):
            _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)
            _post(OPENAI_URL, OPENAI_REQUEST)

    data = json.loads(fixture.read_text())
    assert len(data["interactions"]) == 2


# --- auto mode ---

def test_auto_replays_when_fixture_exists(tmp_path):
    fixture = tmp_path / "auto.json"
    with patch("llm_mock.providers.anthropic._forward_request",
               side_effect=_make_fake_forward(FAKE_ANTHROPIC_RESPONSE)):
        with llm_mock(mode="record", fixture=str(fixture), provider="anthropic"):
            _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    with patch("llm_mock.providers.anthropic._forward_request") as mock_fwd:
        with llm_mock(mode="auto", fixture=str(fixture), provider="anthropic"):
            response = _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)
        mock_fwd.assert_not_called()

    assert response.status_code == 200
    assert response.json() == FAKE_ANTHROPIC_RESPONSE


def test_auto_records_when_no_fixture(tmp_path):
    fixture = tmp_path / "auto.json"
    with patch("llm_mock.providers.anthropic._forward_request",
               side_effect=_make_fake_forward(FAKE_ANTHROPIC_RESPONSE)):
        with llm_mock(mode="auto", fixture=str(fixture), provider="anthropic"):
            response = _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    assert response.status_code == 200
    assert fixture.exists()
    data = json.loads(fixture.read_text())
    assert len(data["interactions"]) == 1


def test_auto_records_when_hash_missing(tmp_path):
    fixture = tmp_path / "auto.json"
    other_request = {**ANTHROPIC_REQUEST, "messages": [{"role": "user", "content": "Different"}]}

    with patch("llm_mock.providers.anthropic._forward_request",
               side_effect=_make_fake_forward(FAKE_ANTHROPIC_RESPONSE)):
        with llm_mock(mode="record", fixture=str(fixture), provider="anthropic"):
            _post(ANTHROPIC_URL, other_request)

    with patch("llm_mock.providers.anthropic._forward_request",
               side_effect=_make_fake_forward(FAKE_ANTHROPIC_RESPONSE)):
        with llm_mock(mode="auto", fixture=str(fixture), provider="anthropic"):
            _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    data = json.loads(fixture.read_text())
    assert len(data["interactions"]) == 2


# --- match_on ---

def test_match_on_excludes_temperature(tmp_path):
    fixture = tmp_path / "anth.json"
    request_hot = {**ANTHROPIC_REQUEST, "temperature": 0.9}

    with patch("llm_mock.providers.anthropic._forward_request",
               side_effect=_make_fake_forward(FAKE_ANTHROPIC_RESPONSE)):
        with llm_mock(mode="record", fixture=str(fixture), provider="anthropic",
                      match_on=["model", "messages"]):
            _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    # replay with different temperature — should still hit the fixture
    with llm_mock(mode="replay", fixture=str(fixture), provider="anthropic",
                  match_on=["model", "messages"]):
        response = _post(ANTHROPIC_URL, request_hot)

    assert response.status_code == 200
    assert response.json() == FAKE_ANTHROPIC_RESPONSE


# --- LLM_MOCK_DISABLED ---

def test_disabled_bypasses_interception(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MOCK_DISABLED", "1")
    fixture = tmp_path / "never.json"

    with respx.mock() as outer:
        outer.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=FAKE_ANTHROPIC_RESPONSE))
        with llm_mock(mode="replay", fixture=str(fixture), provider="anthropic"):
            response = _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    assert response.status_code == 200
    assert not fixture.exists()


def test_disabled_does_not_raise_on_missing_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MOCK_DISABLED", "1")
    fixture = tmp_path / "never.json"

    with respx.mock() as outer:
        outer.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=FAKE_ANTHROPIC_RESPONSE))
        with llm_mock(mode="replay", fixture=str(fixture), provider="anthropic"):
            response = _post(ANTHROPIC_URL, ANTHROPIC_REQUEST)

    assert response.status_code == 200
