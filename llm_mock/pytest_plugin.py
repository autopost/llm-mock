from __future__ import annotations

from pathlib import Path

import pytest

from llm_mock.context_manager import llm_mock


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--llm-mock-disabled",
        action="store_true",
        default=False,
        help="Bypass llm-mock and hit the real LLM API (same as LLM_MOCK_DISABLED=1)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "llm_replay(fixture, mode='replay', provider='all'): mock LLM API calls from a fixture file",
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    marker = item.get_closest_marker("llm_replay")
    if marker is None:
        return (yield)

    if item.config.getoption("--llm-mock-disabled", default=False):
        return (yield)

    fixture_name = marker.kwargs.get("fixture") or (marker.args[0] if marker.args else None)
    if fixture_name is None:
        raise ValueError("@pytest.mark.llm_replay requires a 'fixture' argument")

    mode = marker.kwargs.get("mode", "replay")
    provider = marker.kwargs.get("provider", "all")
    fixture_path = _resolve_fixture_path(fixture_name, item)

    with llm_mock(mode=mode, fixture=str(fixture_path), provider=provider):
        return (yield)


def _resolve_fixture_path(fixture_name: str, item: pytest.Item) -> Path:
    # If it already looks like a path, use it as-is
    if "/" in fixture_name or "\\" in fixture_name:
        return Path(fixture_name)
    # Auto-discover: look for fixtures/ next to the test file
    test_dir = Path(str(item.fspath)).parent
    return test_dir / "fixtures" / fixture_name
