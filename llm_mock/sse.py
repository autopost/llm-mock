from __future__ import annotations

import json

import httpx


def parse_sse(content: bytes) -> list[dict]:
    """Parse raw SSE bytes into a list of event dicts."""
    events = []
    for chunk in content.decode("utf-8").split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        event_type = None
        data_str = None
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()
        if data_str is None:
            continue
        entry: dict = {}
        if event_type:
            entry["event"] = event_type
        if data_str == "[DONE]":
            entry["data"] = "[DONE]"
        else:
            entry["data"] = json.loads(data_str)
        events.append(entry)
    return events


def build_sse(events: list[dict]) -> bytes:
    """Reconstruct SSE bytes from a list of event dicts."""
    parts = []
    for e in events:
        if "event" in e:
            parts.append(f"event: {e['event']}")
        data = e["data"]
        if data == "[DONE]":
            parts.append("data: [DONE]")
        else:
            parts.append(f"data: {json.dumps(data)}")
        parts.append("")
    return "\n".join(parts).encode("utf-8")


def make_sse_response(events: list[dict]) -> httpx.Response:
    return httpx.Response(
        200,
        content=build_sse(events),
        headers={"content-type": "text/event-stream"},
    )
