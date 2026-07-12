from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_mock.fixture_store import FixtureNotFoundError, FixtureParseError, _fixture_path, load, save


def _load_fixture_or_exit(path: Path):
    from llm_mock.fixture_store import Fixture
    if not path.exists():
        print(f"Error: fixture file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return Fixture.model_validate_json(path.read_text())
    except Exception as exc:
        print(f"Error: cannot parse fixture {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    path = _fixture_path(args.fixture)
    fixture = _load_fixture_or_exit(path)

    print(f"Fixture : {path}")
    print(f"Provider: {fixture.provider}")
    print(f"Interactions: {len(fixture.interactions)}")

    if not fixture.interactions:
        print("  (empty)")
        return

    print()
    for i, interaction in enumerate(fixture.interactions, 1):
        model = interaction.request.get("model", "?")
        recorded_at = interaction.recorded_at[:19]  # trim microseconds
        messages = interaction.request.get("messages", [])
        preview = messages[0].get("content", "") if messages else ""
        if len(preview) > 60:
            preview = preview[:57] + "..."
        stream_flag = "  [streaming]" if interaction.streaming else ""
        print(f"  {i}. {interaction.hash[:12]}…  {model}  {recorded_at}{stream_flag}")
        print(f"       \"{preview}\"")


def cmd_clear(args: argparse.Namespace) -> None:
    path = _fixture_path(args.fixture)
    fixture = _load_fixture_or_exit(path)

    if args.hash:
        before = len(fixture.interactions)
        fixture.interactions = [i for i in fixture.interactions if i.hash != args.hash]
        if len(fixture.interactions) == before:
            print(f"Error: no interaction with hash {args.hash} in {path}", file=sys.stderr)
            sys.exit(1)
        path.write_text(json.dumps(fixture.model_dump(), indent=2))
        print(f"Removed interaction {args.hash[:12]}… from {path}")
    else:
        path.unlink()
        print(f"Deleted {path}")


_RECORD_SCRIPT = '''\
"""
Record LLM API responses for use in tests.

Run once with a real API key:
    ANTHROPIC_API_KEY=sk-ant-... python record_fixtures.py

After running, commit the generated fixture files:
    git add tests/fixtures/
    git commit -m "add llm-mock fixtures"

Tests can then replay without an API key:
    pytest
"""

import anthropic
from llm_mock import llm_mock

client = anthropic.Anthropic()

with llm_mock(mode="record", fixture="tests/fixtures/example"):
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    print("Recorded:", message.content[0].text)
'''

_TEST_EXAMPLE = '''\
import anthropic
import pytest

client = anthropic.Anthropic(api_key="fake-key")


@pytest.mark.llm_replay(fixture="example")
def test_example():
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    assert message.content[0].text
'''


def cmd_init(args: argparse.Namespace) -> None:
    fixtures_dir = Path("tests/fixtures")
    record_script = Path("record_fixtures.py")
    test_file = Path("tests/test_example.py")

    created = []

    fixtures_dir.mkdir(parents=True, exist_ok=True)
    created.append(str(fixtures_dir))

    if not record_script.exists():
        record_script.write_text(_RECORD_SCRIPT)
        created.append(str(record_script))

    if not test_file.exists():
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(_TEST_EXAMPLE)
        created.append(str(test_file))

    print("llm-mock init")
    print()
    for path in created:
        print(f"  created  {path}")
    print()
    print("Next steps:")
    print("  1. Record fixtures (needs API key, run once):")
    print("       ANTHROPIC_API_KEY=sk-ant-... python record_fixtures.py")
    print("  2. Commit the fixtures:")
    print("       git add tests/fixtures/ && git commit -m 'add llm-mock fixtures'")
    print("  3. Run tests (no API key needed):")
    print("       pytest")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-mock",
        description="Manage llm-mock fixture files",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    sub.add_parser("init", help="Set up llm-mock in the current project")

    # list
    p_list = sub.add_parser("list", help="Show all recorded interactions in a fixture file")
    p_list.add_argument("fixture", help="Path to fixture file (e.g. tests/fixtures/summarize)")

    # clear
    p_clear = sub.add_parser("clear", help="Delete a fixture file or a single interaction")
    p_clear.add_argument("fixture", help="Path to fixture file")
    p_clear.add_argument("--hash", metavar="HASH", help="Remove only the interaction with this hash")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "clear":
        cmd_clear(args)
