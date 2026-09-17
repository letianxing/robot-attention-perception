from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable


class RosbridgeTransport:
    """Small reconnecting rosbridge client for native macOS capture processes."""

    def __init__(self, url: str, attention_callback: Callable[[dict[str, Any]], None], brain_callback=None, internal_callback=None):
        self.url = url
        self.attention_callback = attention_callback
        self.brain_callback = brain_callback
        self.internal_callback=internal_callback
        self.connected = False
        self.last_error = ""
        self.last_message_ms = 0
        self._socket = None
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rosbridge-client", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        socket = self._socket
        if socket is not None:
            socket.close()
        self._thread.join(timeout=2.0)

    def publish_json(self, topic: str, payload: dict[str, Any]) -> bool:
        return self.publish(
            topic,
            {"data": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        )

    def publish(self, topic: str, message: dict[str, Any]) -> bool:
        return self._send({"op": "publish", "topic": topic, "msg": message})

    def _run(self) -> None:
        try:
            import websocket
        except Exception as exc:
            self.last_error = f"websocket-client unavailable: {exc}"
            return
        while not self._stop.is_set():
            app = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._socket = app
            app.run_forever(ping_interval=10, ping_timeout=4)
            self.connected = False
            if not self._stop.is_set():
                self._stop.wait(0.5)

    def _on_open(self, _socket) -> None:
        self.connected = True
        self.last_error = ""
        if self.internal_callback:self._send({"op":"subscribe","topic":"/brain/internal_state","type":"std_msgs/msg/String","queue_length":1})
        if self.brain_callback:self._send({"op":"subscribe","topic":"/brain/state","type":"std_msgs/msg/String","queue_length":1})
        for topic in (
            "/attention/brain_input",
            "/attention/utterance",
            "/vision/people_json",
            "/voice/acoustic_tracks",
            "/robot/playback_state",
            "/attention/person_hypotheses",
            "/attention/active_behaviors",
        ):
            self._send({"op": "advertise", "topic": topic, "type": "std_msgs/msg/String"})
        self._send(
            {
                "op": "advertise",
                "topic": "/perception/vision/people",
                "type": "perception_interfaces/msg/People",
            }
        )
        self._send(
            {
                "op": "advertise",
                "topic": "/perception/voice/tracks",
                "type": "perception_interfaces/msg/AcousticTracks",
            }
        )
        self._send(
            {
                "op": "subscribe",
                "id": "mac-console-attention",
                "topic": "/attention/state",
                "type": "std_msgs/msg/String",
                "throttle_rate": 20,
                "queue_length": 1,
            }
        )
        self._send(
            {
                "op": "subscribe",
                "id": "mac-console-typed-attention",
                "topic": "/perception/attention/state",
                "type": "perception_interfaces/msg/AttentionState",
                "throttle_rate": 20,
                "queue_length": 1,
            }
        )

    def _on_message(self, _socket, raw: str) -> None:
        try:
            envelope = json.loads(raw)
            if envelope.get("op") != "publish":
                return
            message = envelope.get("msg") or {}
            if envelope.get("topic")=="/brain/internal_state":
                if self.internal_callback:self.internal_callback(json.loads(message.get("data") or "{}"))
                return
            if envelope.get("topic")=="/brain/state":
                if self.brain_callback:self.brain_callback(json.loads(message.get("data") or "{}"))
                return
            if envelope.get("topic") == "/attention/state":
                payload = json.loads(message.get("data") or "{}")
            elif envelope.get("topic") == "/perception/attention/state":
                payload = _typed_attention(message)
            else:
                return
            self.last_message_ms = int(time.time() * 1000)
            self.attention_callback(payload)
        except Exception as exc:
            self.last_error = f"invalid rosbridge attention message: {exc}"

    def _on_error(self, _socket, error: Any) -> None:
        self.last_error = str(error)

    def _on_close(self, _socket, _status, _message) -> None:
        self.connected = False

    def _send(self, payload: dict[str, Any]) -> bool:
        socket = self._socket
        if not self.connected or socket is None:
            return False
        try:
            with self._send_lock:
                socket.send(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.connected = False
            return False


def _typed_attention(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "stamp_ms": _stamp_ms((message.get("header") or {}).get("stamp") or {}),
        "algorithm": str(message.get("algorithm") or "ros4hri_native"),
        "target_id": str(message.get("target_id") or "none"),
        "target_kind": str(message.get("target_kind") or "none"),
        "confidence": float(message.get("confidence", 0.0) or 0.0),
        "listen": bool(message.get("listen", False)),
        "addressed_to_robot": bool(message.get("addressed_to_robot", False)),
        "source_track_id": str(message.get("source_track_id") or "") or None,
        "person_id": str(message.get("person_id") or "") or None,
        "azimuth_deg": float(message.get("azimuth_deg", 0.0)) if message.get("has_azimuth") else None,
        "reasons": tuple(str(value) for value in message.get("reasons") or []),
        "transcripts": (),
    }


def _stamp_ms(stamp: dict[str, Any]) -> int:
    return int(stamp.get("sec", 0) or 0) * 1000 + int(int(stamp.get("nanosec", 0) or 0) / 1_000_000)
