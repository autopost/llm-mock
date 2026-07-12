from llm_mock.sse import build_sse, parse_sse

ANTHROPIC_SSE = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_01"}}\n'
    "\n"
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","delta":{"text":"Hello"}}\n'
    "\n"
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n'
    "\n"
).encode()

OPENAI_SSE = (
    'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hi"}}]}\n'
    "\n"
    "data: [DONE]\n"
    "\n"
).encode()


def test_parse_anthropic_sse():
    events = parse_sse(ANTHROPIC_SSE)
    assert len(events) == 3
    assert events[0]["event"] == "message_start"
    assert events[0]["data"]["type"] == "message_start"
    assert events[1]["event"] == "content_block_delta"
    assert events[1]["data"]["delta"]["text"] == "Hello"
    assert events[2]["event"] == "message_stop"


def test_parse_openai_sse():
    events = parse_sse(OPENAI_SSE)
    assert len(events) == 2
    assert events[0]["data"]["choices"][0]["delta"]["content"] == "Hi"
    assert events[1]["data"] == "[DONE]"


def test_roundtrip_anthropic():
    events = parse_sse(ANTHROPIC_SSE)
    rebuilt = parse_sse(build_sse(events))
    assert rebuilt == events


def test_roundtrip_openai():
    events = parse_sse(OPENAI_SSE)
    rebuilt = parse_sse(build_sse(events))
    assert rebuilt == events


def test_parse_empty_content():
    assert parse_sse(b"") == []


def test_parse_ignores_blank_chunks():
    content = b"\n\n\n\ndata: {\"type\": \"ping\"}\n\n"
    events = parse_sse(content)
    assert len(events) == 1
