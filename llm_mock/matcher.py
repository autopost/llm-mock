import hashlib
import json


def make_hash(request: dict) -> str:
    key = {
        "model": request["model"],
        "messages": request["messages"],
        "temperature": request.get("temperature", 1.0),
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
