from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Interaction(BaseModel):
    hash: str
    request: dict
    response: dict
    recorded_at: str


class Fixture(BaseModel):
    version: str = "1.0"
    provider: str
    interactions: list[Interaction] = []


class FixtureNotFoundError(Exception):
    pass


class FixtureParseError(Exception):
    pass


def _fixture_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix != ".json":
        p = p.with_suffix(".json")
    return p


def load(fixture_path: str | Path, interaction_hash: str) -> Interaction:
    p = _fixture_path(fixture_path)
    if not p.exists():
        raise FixtureNotFoundError(f"Fixture file not found: {p}")
    try:
        fixture = Fixture.model_validate_json(p.read_text())
    except Exception as exc:
        raise FixtureParseError(f"Cannot parse fixture {p}: {exc}") from exc
    for interaction in fixture.interactions:
        if interaction.hash == interaction_hash:
            return interaction
    raise FixtureNotFoundError(
        f"No recorded interaction for hash {interaction_hash[:12]}… in {p}\n"
        "Hint: run with mode='record' first to capture a real response."
    )


def save(fixture_path: str | Path, provider: str, interaction: Interaction) -> None:
    p = _fixture_path(fixture_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            fixture = Fixture.model_validate_json(p.read_text())
        except Exception as exc:
            raise FixtureParseError(f"Cannot parse existing fixture {p}: {exc}") from exc
        fixture.interactions = [i for i in fixture.interactions if i.hash != interaction.hash]
        fixture.interactions.append(interaction)
    else:
        fixture = Fixture(provider=provider, interactions=[interaction])
    p.write_text(json.dumps(fixture.model_dump(), indent=2))
