# llm-mock

Record real LLM responses once, replay them in tests forever — no API key required, no cost, no non-determinism.

```python
# Record once against the real API
with llm_mock(mode="record", fixture="tests/fixtures/summarize"):
    result = my_pipeline("Summarize this document...")

# Replay in CI — no API calls, deterministic, instant
with llm_mock(mode="replay", fixture="tests/fixtures/summarize"):
    result = my_pipeline("Summarize this document...")
    assert "key points" in result
```

---

## Why

- **API calls during tests are expensive.** A CI run hitting real LLM APIs can cost dollars per run at scale.
- **LLM outputs are non-deterministic.** Even at `temperature=0`, responses can vary across model versions.
- **Your production code stays untouched.** llm-mock intercepts at the HTTP transport layer — no changes to application code required.

llm-mock records and replays at the structured request level (model + messages + temperature), stores human-readable JSON fixtures, and integrates natively with pytest.

---

## Installation

```bash
pip install llm-mock
```

**Runtime dependencies:** `httpx`, `respx`, `pydantic`

---

## How to use

### Your production code — untouched

```python
# my_app/pipeline.py
import anthropic

client = anthropic.Anthropic()

def summarize(text: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
    return message.content[0].text
```

`pipeline.py` has zero knowledge of llm-mock. No imports, no changes needed.

### Step 1 — Record (run once, locally)

Create a small script or a dedicated test that runs with `mode="record"`. You need a real API key for this step.

```python
# record_fixtures.py
from llm_mock import llm_mock
from my_app.pipeline import summarize

with llm_mock(mode="record", fixture="tests/fixtures/summarize"):
    result = summarize("Long article about climate change...")
    print(result)  # real response from the API
```

```bash
ANTHROPIC_API_KEY=sk-... python record_fixtures.py
```

This creates `tests/fixtures/summarize.json`. **Commit this file to git.**

### Step 2 — Replay (in tests, forever)

```python
# tests/test_pipeline.py
from llm_mock import llm_mock
from my_app.pipeline import summarize

def test_summarize():
    with llm_mock(mode="replay", fixture="tests/fixtures/summarize"):
        result = summarize("Long article about climate change...")
        assert "climate" in result
```

```bash
pytest  # no API key needed, runs offline, instant
```

llm-mock intercepts the httpx call the Anthropic SDK makes internally and returns the saved response — your test code calls `summarize()` exactly as it would in production.

### Step 3 — Re-record when things change

If you change the prompt, update the model, or want to refresh fixtures:

```bash
ANTHROPIC_API_KEY=sk-... python record_fixtures.py  # overwrites old fixture
git add tests/fixtures/summarize.json
git commit -m "refresh summarize fixture"
```

---

## Quick start (direct API usage)

A complete working example from scratch.

### 1. Install

```bash
git clone https://github.com/yourname/llm-mock
cd llm-mock
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Save your API key

```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-api03-...' > .env
echo '.env' >> .gitignore
```

### 3. Create a record script

Create `try_record.py`:

```python
import anthropic
from llm_mock import llm_mock

client = anthropic.Anthropic()

with llm_mock(mode="record", fixture="fixtures/hello"):
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    print("Response:", message.content[0].text)
    print("Fixture saved to fixtures/hello.json")
```

### 4. Run it

```bash
source .env && .venv/bin/python try_record.py
```

You should see the real response printed and `fixtures/hello.json` created.

### 5. Verify the fixture

```bash
cat fixtures/hello.json
```

### 6. Replay without an API key

Create `try_replay.py`:

```python
import anthropic
from llm_mock import llm_mock

client = anthropic.Anthropic(api_key="fake-key")  # key is irrelevant in replay

with llm_mock(mode="replay", fixture="fixtures/hello"):
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    print("Replayed:", message.content[0].text)
```

```bash
.venv/bin/python try_replay.py
```

The exact same response is returned instantly — no network call made.

### pytest decorator (coming in v0.2)

```python
@pytest.mark.llm_replay(fixture="greet")
def test_greeting():
    ...
```

---

## How it works

```
Record mode:
  Your code → Anthropic/OpenAI SDK → httpx
    → llm-mock intercepts → forwards to real API
    → saves response to fixture JSON
    → returns response to your code

Replay mode:
  Your code → Anthropic/OpenAI SDK → httpx
    → llm-mock intercepts → looks up fixture by SHA256(model + messages + temperature)
    → returns saved response — no network call made
```

**Request matching** uses SHA256 of `(model, messages, temperature)`. Same request always hits the same fixture entry. Different temperature or different message content → different fixture entry.

---

## API reference

### `llm_mock(mode, fixture, provider="all")`

Context manager that activates record or replay mode.

| Parameter | Type | Description |
|---|---|---|
| `mode` | `"record"` \| `"replay"` | Whether to hit the real API and save, or return from fixture |
| `fixture` | `str` | Path to the fixture file. `.json` extension added automatically if omitted |
| `provider` | `"anthropic"` \| `"openai"` \| `"all"` | Which provider(s) to intercept. Default: `"all"` |

```python
from llm_mock import llm_mock

with llm_mock(mode="replay", fixture="tests/fixtures/my_test", provider="anthropic"):
    ...
```

### Exceptions

| Exception | When raised |
|---|---|
| `FixtureNotFoundError` | Replay mode: fixture file missing, or no matching hash in file |
| `FixtureParseError` | Fixture file exists but contains invalid JSON |

```python
from llm_mock import llm_mock, FixtureNotFoundError

try:
    with llm_mock(mode="replay", fixture="tests/fixtures/missing"):
        client.messages.create(...)
except FixtureNotFoundError as e:
    print(e)  # includes hint to run in record mode first
```

---

## Fixture file format

Fixture files are plain JSON — readable, diffable, committable.

```json
{
  "version": "1.0",
  "provider": "anthropic",
  "interactions": [
    {
      "hash": "a3f2c1...",
      "request": {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64
      },
      "response": {
        "id": "msg_01XYZ",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help you today?"}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 9}
      },
      "recorded_at": "2026-04-23T10:00:00+00:00"
    }
  ]
}
```

Multiple interactions (from different requests) are stored in the same file. Re-recording an existing hash overwrites only that entry.

---

## Supported providers

| Provider | Intercepted endpoint | Status |
|---|---|---|
| Anthropic | `api.anthropic.com/v1/messages` | Supported |
| OpenAI | `api.openai.com/v1/chat/completions` | Supported |
| Streaming (`stream=True`) | — | v1.1 |

---

## Comparison

| Tool | Record mode | Native SDK support | In-process |
|---|---|---|---|
| **llm-mock** | yes | yes (Anthropic + OpenAI) | yes |
| [llm_recorder](https://github.com/zby/llm_recorder) | yes | no (LiteLLM only) | yes |
| [AIMock](https://github.com/CopilotKit/aimock) | no | yes | no (HTTP server) |
| [vcr-langchain](https://github.com/amosjyng/vcr-langchain) | yes | no (LangChain only) | yes |

---

## Development

```bash
git clone https://github.com/yourname/llm-mock
cd llm-mock
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## Roadmap

- **v0.2** — pytest plugin (`@pytest.mark.llm_replay`), CLI (`llm-mock list / clear`), `auto` mode, disable via env var
- **v1.1** — streaming support
- **v2** — shared fixtures for teams, semantic matching, web dashboard
