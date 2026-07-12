import json
import sys
from pathlib import Path

import pytest

from llm_mock.cli import main
from llm_mock.fixture_store import Interaction, save

INTERACTION_1 = Interaction(
    hash="aabbcc112233",
    request={
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Summarize this document about climate change."}],
    },
    response={"content": "Summary here."},
    recorded_at="2026-05-01T10:00:00+00:00",
)

INTERACTION_STREAM = Interaction(
    hash="ff00aa112233",
    request={
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Stream this."}],
        "stream": True,
    },
    streaming=True,
    stream_events=[{"event": "message_stop", "data": {"type": "message_stop"}}],
    recorded_at="2026-05-03T12:00:00+00:00",
)

INTERACTION_2 = Interaction(
    hash="ddeeff445566",
    request={
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    },
    response={"content": "Paris."},
    recorded_at="2026-05-02T11:00:00+00:00",
)


def _make_fixture(tmp_path, *interactions):
    path = tmp_path / "test"
    for interaction in interactions:
        save(path, "anthropic", interaction)
    return path


# --- list ---

def test_list_shows_interactions(tmp_path, capsys):
    path = _make_fixture(tmp_path, INTERACTION_1, INTERACTION_2)
    sys.argv = ["llm-mock", "list", str(path)]
    main()
    out = capsys.readouterr().out
    assert "aabbcc112233" in out
    assert "ddeeff445566" in out
    assert "claude-sonnet-4-6" in out
    assert "Summarize this document" in out
    assert "Interactions: 2" in out


def test_list_shows_streaming_flag(tmp_path, capsys):
    path = _make_fixture(tmp_path, INTERACTION_1, INTERACTION_STREAM)
    sys.argv = ["llm-mock", "list", str(path)]
    main()
    out = capsys.readouterr().out
    assert "[streaming]" in out
    assert out.count("[streaming]") == 1  # only the streaming interaction


def test_list_truncates_long_message(tmp_path, capsys):
    long_msg = "A" * 100
    interaction = INTERACTION_1.model_copy(
        update={"request": {**INTERACTION_1.request, "messages": [{"role": "user", "content": long_msg}]}}
    )
    path = _make_fixture(tmp_path, interaction)
    sys.argv = ["llm-mock", "list", str(path)]
    main()
    out = capsys.readouterr().out
    assert "..." in out


def test_list_empty_fixture(tmp_path, capsys):
    from llm_mock.fixture_store import Fixture
    path = tmp_path / "empty.json"
    path.write_text(Fixture(provider="anthropic").model_dump_json())
    sys.argv = ["llm-mock", "list", str(path)]
    main()
    out = capsys.readouterr().out
    assert "(empty)" in out


def test_list_missing_file_exits(tmp_path):
    sys.argv = ["llm-mock", "list", str(tmp_path / "missing")]
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


# --- clear (whole file) ---

def test_clear_deletes_fixture_file(tmp_path):
    path = _make_fixture(tmp_path, INTERACTION_1)
    json_path = path.with_suffix(".json")
    assert json_path.exists()
    sys.argv = ["llm-mock", "clear", str(path)]
    main()
    assert not json_path.exists()


def test_clear_missing_file_exits(tmp_path):
    sys.argv = ["llm-mock", "clear", str(tmp_path / "missing")]
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


# --- clear --hash ---

def test_clear_hash_removes_single_interaction(tmp_path):
    path = _make_fixture(tmp_path, INTERACTION_1, INTERACTION_2)
    sys.argv = ["llm-mock", "clear", str(path), "--hash", "aabbcc112233"]
    main()
    data = json.loads(path.with_suffix(".json").read_text())
    assert len(data["interactions"]) == 1
    assert data["interactions"][0]["hash"] == "ddeeff445566"


def test_clear_hash_not_found_exits(tmp_path):
    path = _make_fixture(tmp_path, INTERACTION_1)
    sys.argv = ["llm-mock", "clear", str(path), "--hash", "nonexistent"]
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


# --- init ---

def test_init_creates_fixtures_dir_and_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.argv = ["llm-mock", "init"]
    main()
    assert (tmp_path / "tests" / "fixtures").is_dir()
    assert (tmp_path / "record_fixtures.py").exists()
    assert (tmp_path / "tests" / "test_example.py").exists()


def test_init_does_not_overwrite_existing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "record_fixtures.py"
    existing.write_text("# my custom script")
    sys.argv = ["llm-mock", "init"]
    main()
    assert existing.read_text() == "# my custom script"


def test_init_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.argv = ["llm-mock", "init"]
    main()
    main()  # second run should not raise
