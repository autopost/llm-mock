import json
import pytest
from pathlib import Path

from llm_mock.fixture_store import (
    Interaction,
    load,
    save,
    FixtureNotFoundError,
    FixtureParseError,
)

INTERACTION = Interaction(
    hash="abc123",
    request={"model": "claude-sonnet-4-6", "messages": []},
    response={"content": "hello"},
    recorded_at="2026-04-23T00:00:00+00:00",
)


def test_round_trip(tmp_path):
    fixture = tmp_path / "test.json"
    save(fixture, "anthropic", INTERACTION)
    result = load(fixture, "abc123")
    assert result.hash == "abc123"
    assert result.response == {"content": "hello"}


def test_save_creates_directories(tmp_path):
    fixture = tmp_path / "deep" / "nested" / "fixture.json"
    save(fixture, "anthropic", INTERACTION)
    assert fixture.exists()


def test_save_overwrites_same_hash(tmp_path):
    fixture = tmp_path / "test.json"
    save(fixture, "anthropic", INTERACTION)
    updated = INTERACTION.model_copy(update={"response": {"content": "updated"}})
    save(fixture, "anthropic", updated)
    result = load(fixture, "abc123")
    assert result.response == {"content": "updated"}
    data = json.loads(fixture.read_text())
    assert len(data["interactions"]) == 1


def test_load_missing_file(tmp_path):
    with pytest.raises(FixtureNotFoundError):
        load(tmp_path / "missing.json", "abc123")


def test_load_missing_hash(tmp_path):
    fixture = tmp_path / "test.json"
    save(fixture, "anthropic", INTERACTION)
    with pytest.raises(FixtureNotFoundError, match="Hint"):
        load(fixture, "nonexistent")


def test_load_corrupted_json(tmp_path):
    fixture = tmp_path / "bad.json"
    fixture.write_text("not valid json{{{")
    with pytest.raises(FixtureParseError):
        load(fixture, "abc123")


def test_fixture_path_adds_json_extension(tmp_path):
    fixture_no_ext = tmp_path / "myfixture"
    save(fixture_no_ext, "anthropic", INTERACTION)
    assert (tmp_path / "myfixture.json").exists()


def test_new_fixture_has_version_2(tmp_path):
    fixture = tmp_path / "test.json"
    save(fixture, "anthropic", INTERACTION)
    data = json.loads(fixture.read_text())
    assert data["version"] == "2.0"


def test_v1_fixture_loads_and_upgrades_on_write(tmp_path):
    fixture = tmp_path / "old.json"
    fixture.write_text(json.dumps({
        "version": "1.0",
        "provider": "anthropic",
        "interactions": [{
            "hash": "abc123",
            "request": {"model": "claude-sonnet-4-6", "messages": []},
            "response": {"content": "hello"},
            "recorded_at": "2026-04-23T00:00:00+00:00",
        }]
    }))

    # v1.0 fixture loads fine
    result = load(fixture, "abc123")
    assert result.hash == "abc123"
    assert result.streaming is False
    assert result.stream_events == []

    # on next write, version is bumped to 2.0
    save(fixture, "anthropic", INTERACTION)
    data = json.loads(fixture.read_text())
    assert data["version"] == "2.0"
