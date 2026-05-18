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
        print(f"  {i}. {interaction.hash[:12]}…  {model}  {recorded_at}")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-mock",
        description="Manage llm-mock fixture files",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="Show all recorded interactions in a fixture file")
    p_list.add_argument("fixture", help="Path to fixture file (e.g. tests/fixtures/summarize)")

    # clear
    p_clear = sub.add_parser("clear", help="Delete a fixture file or a single interaction")
    p_clear.add_argument("fixture", help="Path to fixture file")
    p_clear.add_argument("--hash", metavar="HASH", help="Remove only the interaction with this hash")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "clear":
        cmd_clear(args)
