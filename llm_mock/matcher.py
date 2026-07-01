import hashlib
import json

DEFAULT_MATCH_ON = ("model", "messages", "temperature")

_DEFAULTS = {
    "temperature": 1.0,
    "system": None,
}


def make_hash(request: dict, match_on: tuple | list = DEFAULT_MATCH_ON) -> str:
    key = {}
    for field in match_on:
        if field in request:
            key[field] = request[field]
        elif field in _DEFAULTS:
            key[field] = _DEFAULTS[field]
        else:
            key[field] = request.get(field)
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
