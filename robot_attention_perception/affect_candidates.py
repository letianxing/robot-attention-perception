"""Somebody present, silent for a while, and looking persistently unhappy.

This answers a different question from the engagement layer. Engagement asks "is
this person addressing me", where an expression is weak evidence and would make
the robot approach anyone who happens to look cheerful. This asks "does somebody
here seem to want me to open", which is exactly what an expression is about. So
it produces a workspace candidate and competes there, like a memory event.

What it is not: a diagnosis. The classifier gives a label per frame and is noisy,
so nothing here fires on one frame, on an invalid frame, or on a weak margin. The
payload reports how many samples over how long, and says "looks low" rather than
claiming anything about how the person feels. Brain still decides whether to say
anything, and what.
"""
import threading
from collections import deque

NEGATIVE = {"sad", "angry", "fear", "disgust", "sadness", "anger"}
WINDOW_MS = 20000          # how far back the expression samples are read
MIN_SILENCE_MS = 20000     # nobody has spoken for this long
MIN_SAMPLES = 12           # a handful of frames is not a mood
MIN_SPAN_MS = 8000         # and neither is a frown: the samples must span real time.
                           # Vision runs at 20 fps, so a sample count alone would let
                           # six tenths of a second qualify as "persistent".
MIN_NEGATIVE_SHARE = 0.6   # most of the valid samples, not a passing frown
MIN_INTERVAL_MS = 600000   # at most once every ten minutes per person
SALIENCE = 0.5             # below a live directed call, above idle background


class AffectCandidates:
    """Turns sustained low expression during silence into a workspace candidate."""

    def __init__(self, enabled=True, min_silence_ms=MIN_SILENCE_MS, min_interval_ms=MIN_INTERVAL_MS):
        self.enabled = enabled
        self.min_silence_ms = min_silence_ms
        self.min_interval_ms = min_interval_ms
        self.lock = threading.Lock()
        self.samples = {}
        self.last_raised = {}
        self.last_reason = "no_cycle_yet"

    def status(self):
        with self.lock:
            return {"enabled": self.enabled, "tracked_people": len(self.samples),
                    "reason": self.last_reason, "min_silence_ms": self.min_silence_ms}

    def observe(self, stamp, people):
        """Record only frames the classifier itself marked valid."""
        with self.lock:
            for person in people:
                if not person.face_visible or not person.emotion_valid:
                    continue
                label = str(person.emotion_label or "").lower()
                if label in {"", "unknown"}:
                    continue
                window = self.samples.setdefault(person.person_id, deque(maxlen=512))
                window.append((person.stamp_ms, label, float(person.emotion_valence)))
            for window in self.samples.values():
                while window and stamp - window[0][0] > WINDOW_MS:
                    window.popleft()
            for key in [key for key, window in self.samples.items() if not window]:
                self.samples.pop(key, None)
            if len(self.samples) > 16:
                self.samples = dict(list(self.samples.items())[-16:])

    def candidates(self, stamp, people, last_heard_ms):
        """At most one candidate. Every rejection records why."""
        with self.lock:
            if not self.enabled:
                self.last_reason = "disabled"
                return []
            if last_heard_ms and stamp - last_heard_ms < self.min_silence_ms:
                self.last_reason = "not_silent_long_enough"
                return []
            visible = {p.person_id: p for p in people
                       if p.face_visible and 0 <= stamp - p.stamp_ms < 600
                       and p.identity_confidence >= .5}
            if not visible:
                self.last_reason = "nobody_identified_present"
                return []

            for person_id, person in visible.items():
                window = self.samples.get(person_id) or ()
                if len(window) < MIN_SAMPLES:
                    continue
                if window[-1][0] - window[0][0] < MIN_SPAN_MS:
                    continue
                negative = sum(1 for _, label, _ in window if label in NEGATIVE)
                share = negative / float(len(window))
                if share < MIN_NEGATIVE_SHARE:
                    continue
                previous = self.last_raised.get(person_id)
                if previous and stamp - previous < self.min_interval_ms:
                    self.last_reason = "rate_limited"
                    return []
                self.last_raised[person_id] = stamp
                self.last_reason = "raised"
                span = window[-1][0] - window[0][0]
                valence = sum(value for _, _, value in window) / float(len(window))
                return [{"candidate_id": "affect:" + person_id + ":" + str(stamp),
                         "kind": "affect_event",
                         "person_id": person_id,
                         "stamp_ms": stamp,
                         "salience": SALIENCE,
                         "urgency": 0.0,
                         "summary": "表情持续偏低，且已安静一段时间",
                         "detail": {"samples": len(window), "negative_share": round(share, 3),
                                    "mean_valence": round(valence, 3), "window_ms": span,
                                    "silent_ms": stamp - int(last_heard_ms or 0),
                                    "dominant_label": max({label for _, label, _ in window},
                                                          key=lambda name: sum(1 for _, l, _ in window if l == name))},
                         "reasons": ["sustained_low_expression", "room_quiet", "person_identified"],
                         "limits": "表情分类是有噪声的线索，不是情绪或健康诊断；只是提出一个候选，说不说由 Brain 决定。"}]
            self.last_reason = "no_sustained_low_expression"
            return []
