"""Read-only, scoped access to persistent ASR and reply records."""
from .live_runtime import post_json


def search_history(config, query):
    def value(key, default=""):
        return str(query.get(key, [default])[0]).strip()
    kind = value("kind", "all")
    kinds = ["heard_utterance", "dialogue_turn"] if kind == "all" else [kind]
    if any(item not in {"heard_utterance", "dialogue_turn", "robot_echo"} for item in kinds):
        raise ValueError("unsupported history kind")
    session = value("session", config.session_id)
    if not session or len(session) > 200:
        raise ValueError("session is required")
    payload = {"scope": {"subject_id": config.subject_id, "robot_id": config.robot_id, "session_id": session},
               "kinds": kinds, "query_text": value("q")[:1000], "strict_text": True,
               "limit": 25, "offset": max(0, min(1000000, int(value("offset", "0"))))}
    if value("person"):
        payload["entity_id"] = value("person")[:200]
    if value("as_of_ms"):
        payload["as_of_ms"] = int(value("as_of_ms"))
    return post_json(config.memory_url + "/v1/search", payload, timeout=2.)


def trace_history(config, query):
    turn = str(query.get("turn", [""])[0]).strip()
    session = str(query.get("session", [config.session_id])[0]).strip()
    if not turn or not session:
        raise ValueError("turn and session are required")
    return post_json(config.memory_url + "/v1/search", {
        "scope": {"subject_id": config.subject_id, "robot_id": config.robot_id, "session_id": session},
        "kind": "brain_trace", "entity_id": turn[:300], "limit": 100}, timeout=2.)
