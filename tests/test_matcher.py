from llm_mock.matcher import make_hash

BASE = {
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "hello"}],
}


def test_same_input_same_hash():
    assert make_hash(BASE) == make_hash(BASE)


def test_different_model_different_hash():
    other = {**BASE, "model": "claude-opus-4-7"}
    assert make_hash(BASE) != make_hash(other)


def test_different_message_different_hash():
    other = {**BASE, "messages": [{"role": "user", "content": "bye"}]}
    assert make_hash(BASE) != make_hash(other)


def test_default_temperature_matches_explicit():
    with_temp = {**BASE, "temperature": 1.0}
    assert make_hash(BASE) == make_hash(with_temp)


def test_different_temperature_different_hash():
    other = {**BASE, "temperature": 0.5}
    assert make_hash(BASE) != make_hash(other)


def test_hash_is_hex_string():
    h = make_hash(BASE)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# --- configurable match_on ---

def test_exclude_temperature_different_temps_same_hash():
    a = {**BASE, "temperature": 0.2}
    b = {**BASE, "temperature": 0.9}
    assert make_hash(a, match_on=["model", "messages"]) == make_hash(b, match_on=["model", "messages"])


def test_include_system_prompt_affects_hash():
    with_system = {**BASE, "system": "You are a helpful assistant."}
    without_system = {**BASE}
    match_on = ["model", "messages", "system"]
    assert make_hash(with_system, match_on=match_on) != make_hash(without_system, match_on=match_on)


def test_same_system_prompt_same_hash():
    a = {**BASE, "system": "You are a helpful assistant."}
    b = {**BASE, "system": "You are a helpful assistant."}
    match_on = ["model", "messages", "system"]
    assert make_hash(a, match_on=match_on) == make_hash(b, match_on=match_on)


def test_match_on_model_only():
    a = {**BASE, "messages": [{"role": "user", "content": "hello"}]}
    b = {**BASE, "messages": [{"role": "user", "content": "completely different"}]}
    assert make_hash(a, match_on=["model"]) == make_hash(b, match_on=["model"])
