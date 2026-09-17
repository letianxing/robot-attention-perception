"""Endogenous attention: memory proposing something worth thinking about.

Everywhere else memory only biases candidates that the senses already produced.
Here it can raise a candidate of its own, which is what "suddenly remembering"
means — the one place attention is driven by something nobody is doing right now.

That makes it the easiest thing in the system to turn into a nuisance, so this is
deliberately conservative: only genuinely unfinished business counts, one item at
a time, rate limited, and never while the room is busy. Nothing here decides to
speak; it puts a candidate into the workspace competition and the Brain still
owns what happens next. No topic association, no "that reminds me" — those would
need a relevance model this does not have.
"""
import threading

# Engineering limits, not measured human behaviour.
QUIET_MS = 3000            # nobody has spoken for this long
MIN_INTERVAL_MS = 300000   # at most one endogenous candidate every five minutes
MAX_AGE_MS = 1800000       # older than half an hour is no longer worth raising
SALIENCE = 0.55            # below a live directed call, above idle background


class MemoryCandidates:
    """Turns Brain's unfinished business into workspace candidates."""

    def __init__(self, session_id, quiet_ms=QUIET_MS, min_interval_ms=MIN_INTERVAL_MS, enabled=True):
        self.session_id = session_id
        self.quiet_ms = quiet_ms
        self.min_interval_ms = min_interval_ms
        self.enabled = enabled
        self.lock = threading.Lock()
        self.raised = set()
        self.last_raised_ms = 0
        self.last_reason = "no_cycle_yet"

    def status(self):
        with self.lock:
            return {"enabled": self.enabled, "raised": len(self.raised),
                    "last_raised_ms": self.last_raised_ms or None, "reason": self.last_reason}

    def candidates(self, stamp, brain, people, last_human_ms):
        """Returns at most one candidate. Every rejection records why."""
        with self.lock:
            if not self.enabled:
                self.last_reason = "disabled"
                return []
            if not brain or brain.get("session_id") != self.session_id:
                self.last_reason = "no_current_session_snapshot"
                return []
            if not 0 <= stamp - int(brain.get("stamp_ms") or 0) < 2000:
                self.last_reason = "brain_snapshot_stale"
                return []
            # Nobody to raise it with, so there is nothing to attend to.
            visible = {person.person_id for person in people
                       if person.face_visible and 0 <= stamp - person.stamp_ms < 600}
            if not visible:
                self.last_reason = "nobody_present"
                return []
            if last_human_ms and stamp - last_human_ms < self.quiet_ms:
                self.last_reason = "room_is_busy"
                return []
            if self.last_raised_ms and stamp - self.last_raised_ms < self.min_interval_ms:
                self.last_reason = "rate_limited"
                return []

            summary = brain.get("attention_memory") or {}
            for item in (summary.get("unfinished") or [])[:8]:
                key = str(item.get("id") or "")
                person = item.get("person_id")
                age = stamp - int(item.get("stamp_ms") or 0)
                if not key or key in self.raised:
                    continue
                if person not in visible:
                    continue  # raise it with the person it concerns, not at whoever is here
                if not 0 <= age < MAX_AGE_MS:
                    continue
                self.raised.add(key)
                if len(self.raised) > 256:
                    self.raised = {key}
                self.last_raised_ms = stamp
                self.last_reason = "raised"
                return [{"candidate_id": "memory:" + key,
                         "kind": "memory_event",
                         "person_id": person,
                         "stamp_ms": stamp,
                         "salience": SALIENCE,
                         "urgency": 0.0,
                         "summary": str(item.get("summary") or "")[:200],
                         "detail": {key: item[key] for key in ("kind", "question", "turn_id") if key in item},
                         "age_ms": age,
                         "reasons": ["unfinished_business", "person_present", "room_quiet"],
                         "limits": "由未完成的事提出，不含话题联想；不构成说话许可。"}]
            self.last_reason = "nothing_unfinished"
            return []
