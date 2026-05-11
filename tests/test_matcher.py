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
