from __future__ import annotations

import urllib.request as urlreq
from datetime import datetime, timezone

import httpx
import respx

from llm_mock import fixture_store, matcher

ANTHROPIC_BASE_URL = "https://api.anthropic.com"

# Headers that urllib sets itself or that are connection-specific — copying them causes errors.
_SKIP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}


def _forward_request(request: httpx.Request, body: bytes) -> httpx.Response:
    # Uses urllib (stdlib) to bypass respx, which patches httpx globally.
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_HEADERS}
    req = urlreq.Request(url=str(request.url), data=body, headers=headers, method=request.method)
    try:
        with urlreq.urlopen(req) as resp:
            return httpx.Response(resp.status, content=resp.read())
    except urlreq.HTTPError as e:
        return httpx.Response(e.code, content=e.read())
    except urlreq.URLError as e:
        # Network error or timeout — re-raise so the SDK raises its normal exception.
        raise httpx.ConnectError(str(e)) from e


def _build_route(mock_router: respx.MockRouter, mode: str, fixture_path: str, match_on: tuple | list = matcher.DEFAULT_MATCH_ON) -> None:
    @mock_router.post(f"{ANTHROPIC_BASE_URL}/v1/messages")
    def handle(request: httpx.Request) -> httpx.Response:
        import json
        body = request.read()
        req_data = json.loads(body)
        h = matcher.make_hash(req_data, match_on)

        if mode == "replay":
            interaction = fixture_store.load(fixture_path, h)
            return httpx.Response(200, json=interaction.response)

        if mode == "auto":
            try:
                interaction = fixture_store.load(fixture_path, h)
                return httpx.Response(200, json=interaction.response)
            except fixture_store.FixtureNotFoundError:
                pass  # fixture missing or hash not recorded yet — fall through to record

        real_response = _forward_request(request, body)

        # Do not save error responses — let the SDK raise its normal exception.
        if real_response.status_code >= 400:
            return real_response

        resp_data = real_response.json()
        fixture_store.save(
            fixture_path,
            provider="anthropic",
            interaction=fixture_store.Interaction(
                hash=h,
                request=req_data,
                response=resp_data,
                recorded_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
        return httpx.Response(real_response.status_code, json=resp_data)
