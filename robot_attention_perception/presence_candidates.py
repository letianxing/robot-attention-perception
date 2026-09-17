"""Somebody has been here a while and nothing has been said.

The affect producer next door only fires when the expression stays negative, so
a person sitting quietly with an ordinary face never reached the workspace at
all — in the field the owner was present and silent for minutes and the robot
never opened its mouth. Sitting with someone in silence is a reason to speak
long before they look unhappy.

Kept deliberately dull: it needs a recognised person who has actually been here
a while, a quiet room, a robot that has not just spoken, and it offers at most
one opening every few minutes. It carries whatever the expression happens to be
— including neutral — as a hint, not a diagnosis, and Brain still decides
whether to say anything and what.
"""
import threading

PRESENT_MS = 60000          # here this long before silence means anything
QUIET_MS = 45000            # nobody has said anything for this long
ROBOT_QUIET_MS = 60000      # and we did not just say something ourselves
MIN_INTERVAL_MS = 300000    # at most once every five minutes per person
ABSENCE_MS = 10000          # out of frame longer than this and the clock restarts
AWAY_MS = 90000             # gone at least this long and coming back is an arrival
ARRIVAL_QUIET_MS = 5000     # and the room has to have settled before we speak into it
ARRIVAL_INTERVAL_MS = 600000
SALIENCE = 0.35             # below an affect event, well below a live call


class PresenceCandidates:
    """Turns long quiet co-presence into one low-salience workspace candidate."""

    def __init__(self, enabled=True, present_ms=PRESENT_MS, quiet_ms=QUIET_MS,
                 min_interval_ms=MIN_INTERVAL_MS):
        self.enabled = enabled
        self.present_ms = present_ms
        self.quiet_ms = quiet_ms
        self.min_interval_ms = min_interval_ms
        self.lock = threading.Lock()
        self.since = {}
        self.last_seen = {}
        self.last_raised = {}
        self.last_arrival = {}
        # Somebody coming back after a while is its own reason to speak, and it
        # belongs here with the other internally generated candidates rather
        # than in Brain's own bookkeeping — where it used to live, opening turns
        # nobody else could see or arbitrate against.
        self.arrivals = {}
        self.last_reason = "no_cycle_yet"

    def status(self):
        with self.lock:
            return {"enabled": self.enabled, "tracked_people": len(self.since),
                    "reason": self.last_reason, "quiet_ms": self.quiet_ms}

    def observe(self, stamp, people):
        """Track how long each recognised person has been continuously present."""
        with self.lock:
            for person in people:
                if not person.face_visible or float(person.identity_confidence or 0) < .5:
                    continue
                key = str(person.person_id or "")
                if not key or key in {"unknown", "stranger"}:
                    continue
                gap = stamp - self.last_seen[key] if key in self.last_seen else None
                if key not in self.since or (gap is not None and gap > AWAY_MS):
                    if gap is None or gap > AWAY_MS:
                        self.arrivals[key] = stamp
                self.since.setdefault(key, stamp)
                self.last_seen[key] = stamp
            # Out of frame for a moment is not leaving — the person manager keeps
            # identity across that. Gone for ten seconds is, and the clock restarts.
            for key, last in list(self.last_seen.items()):
                if stamp - last > ABSENCE_MS:
                    self.since.pop(key, None)
                    self.last_seen.pop(key, None)

    def candidates(self, stamp, people, last_heard_ms, robot_last_spoke_ms=0):
        """At most one candidate, and every rejection records why."""
        with self.lock:
            if not self.enabled:
                self.last_reason = "disabled"
                return []
            arrival = self._arrival(stamp, people, last_heard_ms, robot_last_spoke_ms)
            if arrival:
                return arrival
            if last_heard_ms and stamp - last_heard_ms < self.quiet_ms:
                self.last_reason = "not_quiet_long_enough"
                return []
            if robot_last_spoke_ms and stamp - robot_last_spoke_ms < ROBOT_QUIET_MS:
                self.last_reason = "robot_spoke_recently"
                return []
            visible = [p for p in people
                       if p.face_visible and 0 <= stamp - p.stamp_ms < 600
                       and float(p.identity_confidence or 0) >= .5]
            if not visible:
                self.last_reason = "nobody_identified_present"
                return []
            for person in visible:
                key = str(person.person_id or "")
                arrived = self.since.get(key)
                if arrived is None or stamp - arrived < self.present_ms:
                    continue
                previous = self.last_raised.get(key)
                if previous and stamp - previous < self.min_interval_ms:
                    self.last_reason = "rate_limited"
                    return []
                self.last_raised[key] = stamp
                self.last_reason = "raised"
                expression = str(person.emotion_label or "") if person.emotion_valid else ""
                return [{"candidate_id": "presence:" + key + ":" + str(stamp),
                         "kind": "presence_event",
                         "person_id": key,
                         "stamp_ms": stamp,
                         "salience": SALIENCE,
                         "urgency": 0.0,
                         "summary": "他在这儿有一会儿了，房间一直很安静",
                         "detail": {"present_ms": stamp - arrived,
                                    "silent_ms": stamp - int(last_heard_ms or 0),
                                    "expression": expression or "未判定",
                                    "looking_at_robot": bool(float(person.gaze_score or 0) >= .65)},
                         "reasons": ["long_quiet_co_presence", "person_identified"],
                         "limits": "只是说这个人在场且很久没说话；表情是线索不是诊断，开不开口由 Brain 决定。"}]
            self.last_reason = "nobody_present_long_enough"
            return []

    def _arrival(self, stamp, people, last_heard_ms, robot_last_spoke_ms):
        """A recognised person who has been away is back, and the room is quiet."""
        if last_heard_ms and stamp - int(last_heard_ms) < ARRIVAL_QUIET_MS:
            self.last_reason = "arrival_needs_a_quiet_room"
            return []
        if robot_last_spoke_ms and stamp - robot_last_spoke_ms < ROBOT_QUIET_MS:
            self.last_reason = "robot_spoke_recently"
            return []
        for person in people:
            key = str(person.person_id or "")
            came_back = self.arrivals.get(key)
            if not came_back or stamp - came_back > 20000:
                continue
            if not person.face_visible or float(person.identity_confidence or 0) < .5:
                continue
            previous = self.last_arrival.get(key)
            if previous and stamp - previous < ARRIVAL_INTERVAL_MS:
                continue
            self.last_arrival[key] = stamp
            self.arrivals.pop(key, None)
            self.last_reason = "arrival_raised"
            return [{"candidate_id": "arrival:" + key + ":" + str(stamp),
                     "kind": "arrival_event",
                     "person_id": key,
                     "stamp_ms": stamp,
                     "salience": 0.45,
                     "urgency": 0.0,
                     "summary": "认识的人回来了，房间是安静的",
                     "detail": {"away_ms": None, "quiet_ms": stamp - int(last_heard_ms or 0),
                                "expression": str(person.emotion_label or "") if person.emotion_valid else "未判定"},
                     "reasons": ["returned_after_absence", "person_identified"],
                     "limits": "只说这个人回来了；说不说、说什么由 Brain 决定。"}]
        return []
