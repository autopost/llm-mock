import pytest
from llm_mock.fixture_store import Interaction, save
from llm_mock.matcher import make_hash

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

REQUEST = {
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 10,
}

RESPONSE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi!"}],
    "model": "claude-sonnet-4-6",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 2},
}


def _make_fixture(directory, name):
    interaction = Interaction(
        hash=make_hash(REQUEST),
        request=REQUEST,
        response=RESPONSE,
        recorded_at="2026-05-16T00:00:00+00:00",
    )
    save(directory / name, "anthropic", interaction)


def test_marker_replays_fixture(pytester):
    fixtures_dir = pytester.path / "fixtures"
    fixtures_dir.mkdir()
    _make_fixture(fixtures_dir, "greet")

    pytester.makepyfile(f"""
        import httpx
        import pytest

        @pytest.mark.llm_replay(fixture="greet")
        def test_example():
            response = httpx.post(
                "{ANTHROPIC_URL}",
                json={REQUEST},
                headers={{"x-api-key": "test"}},
            )
            assert response.status_code == 200
            assert response.json()["content"][0]["text"] == "Hi!"
    """)

    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_marker_with_explicit_path(pytester):
    fixtures_dir = pytester.path / "custom"
    fixtures_dir.mkdir()
    _make_fixture(fixtures_dir, "greet")

    pytester.makepyfile(f"""
        import httpx
        import pytest

        @pytest.mark.llm_replay(fixture="custom/greet")
        def test_example():
            response = httpx.post(
                "{ANTHROPIC_URL}",
                json={REQUEST},
                headers={{"x-api-key": "test"}},
            )
            assert response.status_code == 200
    """)

    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_marker_fails_when_fixture_missing(pytester):
    pytester.makepyfile(f"""
        import httpx
        import pytest

        @pytest.mark.llm_replay(fixture="nonexistent")
        def test_example():
            httpx.post(
                "{ANTHROPIC_URL}",
                json={REQUEST},
                headers={{"x-api-key": "test"}},
            )
    """)

    result = pytester.runpytest("-v")
    result.assert_outcomes(failed=1)


def test_marker_without_fixture_arg_raises(pytester):
    pytester.makepyfile("""
        import pytest

        @pytest.mark.llm_replay
        def test_example():
            pass
    """)

    result = pytester.runpytest("-v")
    result.assert_outcomes(failed=1)


def test_marker_auto_mode_replays_existing_fixture(pytester):
    fixtures_dir = pytester.path / "fixtures"
    fixtures_dir.mkdir()
    _make_fixture(fixtures_dir, "greet")

    pytester.makepyfile(f"""
        import httpx
        import pytest

        @pytest.mark.llm_replay(fixture="greet", mode="auto")
        def test_example():
            response = httpx.post(
                "{ANTHROPIC_URL}",
                json={REQUEST},
                headers={{"x-api-key": "test"}},
            )
            assert response.status_code == 200
            assert response.json()["content"][0]["text"] == "Hi!"
    """)

    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_llm_mock_live_flag_bypasses_interception(pytester):
    pytester.makepyfile(f"""
        import httpx
        import pytest
        import respx

        @pytest.mark.llm_replay(fixture="nonexistent")
        def test_example():
            with respx.mock() as m:
                m.post("{ANTHROPIC_URL}").mock(return_value=httpx.Response(200, json={RESPONSE}))
                response = httpx.post("{ANTHROPIC_URL}", json={REQUEST}, headers={{"x-api-key": "test"}})
            assert response.status_code == 200
    """)

    result = pytester.runpytest("--llm-mock-live", "-v")
    result.assert_outcomes(passed=1)


def test_unmarked_test_unaffected(pytester):
    pytester.makepyfile("""
        def test_plain():
            assert 1 + 1 == 2
    """)

    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)
