# llm-mock

**pytest plugin to mock OpenAI and Anthropic API calls** — record real responses once, replay them in tests forever. No API key needed in CI, no cost per run, no flaky non-determinism.

```python
# Record once against the real API (run locally with your API key)
with llm_mock(mode="record", fixture="tests/fixtures/summarize"):
    result = my_pipeline("Summarize this document...")

# Replay in tests — no API key, no cost, deterministic
@pytest.mark.llm_replay(fixture="summarize")
def test_summarize():
    result = my_pipeline("Summarize this document...")
    assert "key points" in result
```

Works with the **Anthropic SDK** (`claude-*` models) and the **OpenAI SDK** (`gpt-*` models) out of the box — no changes to your application code required.

---

## Why mock LLM calls in tests?

- **Cost.** A CI pipeline hitting real LLM APIs can cost dollars per run at scale.
- **Flakiness.** LLM outputs are non-deterministic — even `temperature=0` varies across model versions.
- **Speed.** Replayed fixtures return instantly; no network round-trip.
- **Offline.** Tests run without credentials in CI, on a plane, in a container.

llm-mock intercepts at the **HTTP transport layer** (via `httpx`/`respx`) — your production code is never touched. Fixtures are plain JSON files you commit to git, diff in PRs, and refresh on demand.

---

## Installation

```bash
pip install llm-mock
```

Or install from source:

```bash
git clone https://github.com/autopost/llm-mock.git
cd llm-mock
pip install -e .
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

Use the pytest decorator — no `with` block needed inside the test:

```python
# tests/test_pipeline.py
import pytest
from my_app.pipeline import summarize

@pytest.mark.llm_replay(fixture="summarize")
def test_summarize():
    result = summarize("Long article about climate change...")
    assert "climate" in result
```

```bash
pytest  # no API key needed, runs offline, instant
```

The decorator auto-discovers the fixture path relative to the test file — `fixture="summarize"` looks for `tests/fixtures/summarize.json` when the test lives in `tests/`.

llm-mock intercepts the httpx call the Anthropic SDK makes internally and returns the saved response — your test code calls `summarize()` exactly as it would in production.

**Alternative:** use the context manager directly if you need more control:

```python
from llm_mock import llm_mock

def test_summarize():
    with llm_mock(mode="replay", fixture="tests/fixtures/summarize"):
        result = summarize("Long article about climate change...")
        assert "climate" in result
```

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
pip install llm-mock
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
llm-mock list tests/fixtures/hello
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

### 7. Write a test with the pytest decorator

```python
# tests/test_hello.py
import anthropic
import pytest

client = anthropic.Anthropic(api_key="fake-key")

@pytest.mark.llm_replay(fixture="hello")
def test_hello():
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    assert message.content[0].text  # replayed from fixtures/hello.json
```

```bash
.venv/bin/pytest tests/test_hello.py -v
```

---

## CLI

Inspect and manage fixture files from the terminal.

> **Note:** activate your virtual environment first so `llm-mock` is on your PATH:
> ```bash
> source .venv/bin/activate
> ```
> Or run it directly with `.venv/bin/llm-mock <command>`.

### `llm-mock list <fixture>`

Show all recorded interactions in a fixture file:

```bash
$ llm-mock list tests/fixtures/summarize

Fixture : tests/fixtures/summarize.json
Provider: anthropic
Interactions: 2

  1. a3f2c1d4e5b6…  claude-sonnet-4-6        2026-04-23T10:00:00
       "Summarize this document about climate change..."
  2. b4g3d2e5f6c7…  claude-haiku-4-5-20251001  2026-04-24T11:00:00
       "What is the capital of France?"
```

### `llm-mock clear <fixture>`

Delete an entire fixture file:

```bash
llm-mock clear tests/fixtures/summarize
```

Delete a single interaction by hash:

```bash
llm-mock clear tests/fixtures/summarize --hash a3f2c1d4e5b6
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

### `llm_mock(mode, fixture, provider="all", match_on=...)`

Context manager that activates record, replay, or auto mode.

| Parameter | Type | Description |
|---|---|---|
| `mode` | `"record"` \| `"replay"` \| `"auto"` | `record` hits the real API and saves; `replay` returns from fixture; `auto` replays if fixture exists, records if not |
| `fixture` | `str` | Path to the fixture file. `.json` extension added automatically if omitted |
| `provider` | `"anthropic"` \| `"openai"` \| `"all"` | Which provider(s) to intercept. Default: `"all"` |
| `match_on` | `list[str]` | Fields used to match requests to fixtures. Default: `["model", "messages", "temperature"]` |

**`auto` mode** is the recommended default for most projects — it self-heals when new requests appear without manual mode switches:

```python
@pytest.mark.llm_replay(fixture="summarize", mode="auto")
def test_summarize():
    ...
```

```python
from llm_mock import llm_mock

with llm_mock(mode="replay", fixture="tests/fixtures/my_test", provider="anthropic"):
    ...
```

#### Configurable match keys

By default requests are matched by `model + messages + temperature`. You can customise this with `match_on`:

```python
# Ignore temperature — different temperature values hit the same fixture
with llm_mock(mode="replay", fixture="tests/fixtures/summary",
              match_on=["model", "messages"]):
    ...

# Include system prompt in matching — different system prompts get separate fixture entries
with llm_mock(mode="replay", fixture="tests/fixtures/summary",
              match_on=["model", "messages", "system"]):
    ...
```

**Supported fields:**

| Field | Default | Description |
|---|---|---|
| `"model"` | included | The model name, e.g. `"claude-sonnet-4-6"`, `"gpt-4o"` |
| `"messages"` | included | The full messages array — role + content |
| `"temperature"` | included | Sampling temperature. Remove from `match_on` to make tests temperature-agnostic |
| `"system"` | excluded | Top-level system prompt. Add to `match_on` when different system prompts should produce separate fixture entries |

**When to change the defaults:**

- **Exclude `temperature`** — your app varies temperature between environments (dev vs prod) but you want a single fixture to cover both
- **Include `system`** — your app uses system prompts and you need separate fixtures per system prompt (e.g. different personas or instructions)

### Environment variables & CLI flags

| Method | Effect |
|---|---|
| `LLM_MOCK_DISABLED=1` | Disables all interception — LLM calls go to the real API as normal |
| `pytest --llm-mock-disabled` | Same as above, but as a pytest flag — no env var needed |

Useful for refreshing all fixtures in one shot without touching test code:

```bash
# via env var
LLM_MOCK_DISABLED=1 ANTHROPIC_API_KEY=sk-... pytest

# via pytest flag
pytest --llm-mock-disabled
```

Or in a weekly CI job that validates against the live model.

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

- **v0.2** — `auto` mode, `LLM_MOCK_DISABLED` env var ✓
- **v0.3** — `match_on` configurable match keys, `--llm-mock-disabled` pytest flag ✓
- **v1.1** — streaming support (`stream=True` for Anthropic and OpenAI)
- **v2** — shared fixtures for teams, semantic matching, web dashboard

---

## Related

`pytest mock openai` · `pytest mock anthropic` · `mock LLM calls python` · `record replay LLM` · `vcr cassette openai` · `fake openai response pytest` · `test without API key` · `offline LLM testing` · `deterministic LLM tests`
