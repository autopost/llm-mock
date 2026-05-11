from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

import respx

from llm_mock.providers import anthropic as anthropic_provider
from llm_mock.providers import openai as openai_provider

Provider = Literal["anthropic", "openai", "all"]
Mode = Literal["record", "replay"]


@contextmanager
def llm_mock(
    mode: Mode,
    fixture: str,
    provider: Provider = "all",
):
    with respx.mock(assert_all_called=False) as router:
        if provider in ("anthropic", "all"):
            anthropic_provider._build_route(router, mode, fixture)
        if provider in ("openai", "all"):
            openai_provider._build_route(router, mode, fixture)
        yield router
