"""Cross-session familiarity: how much this person has really talked with us before.

The perception loop runs at 20 Hz and must never block on the memory service, so
a single background worker refreshes a bounded cache and the loop only reads it.

Familiarity is a weak prior on how to read a silent gaze. It is never evidence
that somebody is speaking to the robot, it cannot resolve an identity, and an
unreachable memory service is reported as unknown rather than as "stranger" —
"no record" and "could not ask" are different states and stay different.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

ANONYMOUS = {"", "unknown", "robot", "none", "environment"}
# Four separate prior sessions is treated as fully familiar. An engineering
# saturation point, not a measured threshold for human familiarity.
FULL_FAMILIARITY_SESSIONS = 4


class FamiliarityIndex:
    def __init__(self, memory_url, subject_id, robot_id, session_id, search=None,
                 refresh_ms=60000, ttl_ms=900000, max_people=32):
        self.memory_url = memory_url
        self.scope = {"subject_id": subject_id, "robot_id": robot_id, "session_id": session_id}
        self.session_id = session_id
        self.refresh_ms = refresh_ms
        self.ttl_ms = ttl_ms
        self.max_people = max_people
        self._search = search or self._http_search
        self.lock = threading.Lock()
        self.records = {}
        self.pending = set()
        self.last_error = ""
        self.queries = 0
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="familiarity")
        self.closed = False

    def close(self):
        self.closed = True
        self.worker.shutdown(wait=False, cancel_futures=True)

    def _http_search(self, person_id):
        from .live_runtime import post_json
        return post_json(self.memory_url + "/v1/search",
                         {"scope": self.scope, "all_sessions": True, "entity_id": person_id,
                          "kinds": ["heard_utterance", "dialogue_turn"], "limit": 200},
                         timeout=2.0)

    def request(self, stamp, person_ids):
        """Queue a refresh for people we have no fresh answer about."""
        if self.closed:
            return
        with self.lock:
            wanted = []
            for person in person_ids:
                if person in ANONYMOUS or person in self.pending:
                    continue
                record = self.records.get(person)
                if record and 0 <= stamp - record["checked_ms"] < self.refresh_ms:
                    continue
                wanted.append(person)
            if not wanted:
                return
            for person in wanted[:4]:
                self.pending.add(person)
        for person in wanted[:4]:
            try:
                # The caller's clock is the only clock: cache ages are compared
                # against the same stamps the perception loop uses.
                self.worker.submit(self._refresh, person, stamp)
            except RuntimeError:
                # A tick can race with shutdown; drop the request rather than
                # taking the perception loop down with it.
                self.closed = True
                with self.lock:
                    self.pending.difference_update(wanted)
                return

    def _refresh(self, person_id, stamp):
        try:
            response = self._search(person_id)
            sessions, last_ms, turns = set(), 0, 0
            for hit in response.get("hits", []):
                record = hit.get("record") or {}
                entity = record.get("entity") or {}
                if entity.get("person_id") != person_id:
                    continue
                session = (record.get("scope") or {}).get("session_id")
                if not session or session == self.session_id:
                    continue  # the current session is already covered by working memory
                sessions.add(session)
                turns += 1
                last_ms = max(last_ms, int(record.get("observed_at_ms") or 0))
            score = min(1.0, len(sessions) / float(FULL_FAMILIARITY_SESSIONS))
            entry = {"familiarity": score, "prior_sessions": len(sessions), "prior_turns": turns,
                     "last_seen_ms": last_ms or None, "checked_ms": stamp, "known": True}
            error = ""
        except Exception as exc:
            entry = None
            error = str(exc)
        with self.lock:
            self.pending.discard(person_id)
            self.queries += 1
            self.last_error = error
            if entry is not None:
                self.records[person_id] = entry
                if len(self.records) > self.max_people:
                    oldest = min(self.records, key=lambda key: self.records[key]["checked_ms"])
                    self.records.pop(oldest, None)

    def scores(self, stamp):
        """Familiarity per person; people we could not ask simply do not appear."""
        with self.lock:
            return {person: dict(record) for person, record in self.records.items()
                    if 0 <= stamp - record["checked_ms"] < self.ttl_ms}

    def status(self, stamp):
        with self.lock:
            fresh = sum(1 for record in self.records.values() if 0 <= stamp - record["checked_ms"] < self.ttl_ms)
            return {"cached_people": fresh, "pending": len(self.pending), "queries": self.queries,
                    "error": self.last_error}
