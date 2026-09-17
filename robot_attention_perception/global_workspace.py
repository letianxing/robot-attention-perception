"""One broadcast point for what attention selected, and why.

The perception loop is a cognitive cycle in the LIDA sense — perceive, compete,
broadcast, act, feed back — and this module owns the broadcast step. Two things
happen here that the algorithm plugin cannot do:

  * Candidates that are not sensory compete too. A memory event has no modality
    and no bearing, so it cannot enter the per-modality competition inside the
    plugin, but it can absolutely be the most worthwhile thing to think about.
    The plugin decides where to look and listen; the workspace decides what gets
    thought about.

  * Only a few winners get through. A soft winner-take-all shares a limited
    capacity among the strongest candidates and reports the rest as suppressed,
    which is the workspace-level counterpart of Selective Tuning's suppression
    of non-winners. Urgent items pre-empt that limit instead of queueing.

Nothing here adds evidence or grants permission. "Workspace contents" means
"what won this cycle's competition", not awareness. A consumer that falls behind
drops cycles and is told how many; the perception loop is never blocked.
"""
import json
import queue
import threading

SCHEMA_VERSION = 2
CAPACITY = 4          # how many candidates get through per cycle
SOFT_WTA_EXPONENT = 2.0
URGENT = 0.7          # at or above this, an item pre-empts the capacity limit
LIMITS = ("内容是本轮竞争的胜出者与依据，不是意识；容量、soft-WTA 指数与紧急阈值均为工程先验，未做现场标定；"
          "订阅者落后会丢帧并计数，不会阻塞感知循环。")


class GlobalWorkspace:
    """Latest broadcast plus a bounded replay buffer and push subscribers."""

    def __init__(self, session_id, history=64, queue_size=32):
        self.session_id = session_id
        self.queue_size = queue_size
        self.lock = threading.Lock()
        self.cycle = 0
        self.latest = {}
        self.history = []
        self.history_limit = history
        self.subscribers = {}
        self.next_subscriber = 0
        self.dropped = 0

    def _compete(self, entries):
        """Soft winner-take-all over every candidate, then a hard capacity limit.

        Shares are reported so a consumer can see how close the contest was;
        they are not probabilities and not a confidence.
        """
        strengths = [max(0.0, float(item["activation"])) ** SOFT_WTA_EXPONENT for item in entries]
        total = sum(strengths)
        for item, strength in zip(entries, strengths):
            item["share"] = round(strength / total, 5) if total > 0 else 0.0
        ranked = sorted(entries, key=lambda item: (float(item.get("urgency") or 0.0) >= URGENT,
                                                   item["share"]), reverse=True)
        admitted = 0
        for item in ranked:
            urgent = float(item.get("urgency") or 0.0) >= URGENT
            if urgent or admitted < CAPACITY:
                item["admitted"] = True
                item["admission"] = "urgent" if urgent and admitted >= CAPACITY else "within_capacity"
                admitted += 1
            else:
                item["admitted"] = False
                item["admission"] = "suppressed_over_capacity"
        return ranked

    def publish(self, stamp_ms, attention, distribution, sources, algorithm, provenance=None, events=()):
        """Called once per perception cycle, after the algorithm has decided.

        `events` are non-sensory candidates — memory, and later internal state —
        that compete here because they have no modality to compete in.
        """
        attention = attention if isinstance(attention, dict) else {}
        distribution = distribution or {}
        candidates = distribution.get("candidates") or {}
        entries = [{"candidate_id": key,
                    "kind": "sensory",
                    "modality": value.get("modality"),
                    "person_id": value.get("person_id"),
                    "activation": round(float(value.get("a") or 0.0), 5),
                    "engagement": round(float(value.get("engagement") or 0.0), 5),
                    "urgency": 0.0,
                    "focus": bool(value.get("focus")),
                    "reasons": list(value.get("reasons") or [])}
                   for key, value in candidates.items() if value.get("valid")]
        for event in events or ():
            entries.append({"candidate_id": str(event.get("candidate_id") or ""),
                            "kind": str(event.get("kind") or "event"),
                            "modality": None,
                            "person_id": event.get("person_id"),
                            "activation": round(float(event.get("salience") or 0.0), 5),
                            "engagement": 0.0,
                            "urgency": round(float(event.get("urgency") or 0.0), 5),
                            "focus": False,
                            "summary": event.get("summary"),
                            "detail": event.get("detail"),
                            "reasons": list(event.get("reasons") or [])})
        coalition = self._compete(entries)[:16]

        contents = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "stamp_ms": int(stamp_ms),
            "focus": distribution.get("focus") or {},
            "target": {key: attention.get(key) for key in
                       ("target_id", "target_kind", "person_id", "face_id", "azimuth_deg", "confidence",
                        "interaction_phase", "listen", "addressed_to_robot", "reasons", "focus_status")},
            "engagement": distribution.get("engagement") or attention.get("engagement") or {},
            "coalition": coalition,
            "capacity": {"limit": CAPACITY, "admitted": sum(1 for item in coalition if item.get("admitted")),
                         "suppressed": sum(1 for item in coalition if not item.get("admitted")),
                         "soft_wta_exponent": SOFT_WTA_EXPONENT, "urgent_threshold": URGENT},
            "sources": sources or {},
            "algorithm": algorithm or {},
            "provenance": provenance or {},
            "limits": LIMITS,
        }
        with self.lock:
            self.cycle += 1
            contents["cycle"] = self.cycle
            self.latest = contents
            self.history.append(contents)
            if len(self.history) > self.history_limit:
                del self.history[:-self.history_limit]
            targets = list(self.subscribers.values())
        for channel in targets:
            try:
                channel.put_nowait(contents)
            except queue.Full:
                # A slow consumer loses cycles; perception must not wait for it.
                with self.lock:
                    self.dropped += 1
        return contents

    def snapshot(self):
        with self.lock:
            return dict(self.latest) if self.latest else {
                "schema_version": SCHEMA_VERSION, "session_id": self.session_id, "cycle": 0,
                "reason": "no_cycle_broadcast_yet", "limits": LIMITS}

    def replay(self, since_cycle=0):
        with self.lock:
            return [item for item in self.history if item["cycle"] > int(since_cycle or 0)]

    def status(self):
        with self.lock:
            return {"cycle": self.cycle, "subscribers": len(self.subscribers), "dropped_cycles": self.dropped,
                    "history": len(self.history), "queue_size": self.queue_size}

    def subscribe(self):
        channel = queue.Queue(maxsize=self.queue_size)
        with self.lock:
            self.next_subscriber += 1
            key = self.next_subscriber
            self.subscribers[key] = channel
            current = dict(self.latest) if self.latest else None
        if current:
            channel.put_nowait(current)
        return key, channel

    def unsubscribe(self, key):
        with self.lock:
            self.subscribers.pop(key, None)

    def close(self):
        with self.lock:
            self.subscribers.clear()


def sse_event(contents):
    """Server-sent-event framing so a browser or service can just listen."""
    return ("event: workspace\ndata: " +
            json.dumps(contents, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode("utf-8")
