#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_attention_perception.live_runtime import _ros_time, _typed_acoustic, _typed_vision
from robot_attention_perception.rosbridge_transport import RosbridgeTransport
from robot_attention_perception.types import AcousticTrack, VisionPerson


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:9091")
    parser.add_argument("--timeout", type=float, default=6.0)
    args = parser.parse_args()
    received = []
    transport = RosbridgeTransport(args.url, received.append)
    deadline = time.monotonic() + args.timeout
    while not transport.connected and time.monotonic() < deadline:
        time.sleep(0.05)
    if not transport.connected:
        raise SystemExit(f"rosbridge connection failed: {transport.last_error}")

    for _ in range(8):
        stamp_ms = int(time.time() * 1000)
        person = VisionPerson(
            person_id="roundtrip_person",
            stamp_ms=stamp_ms,
            face_id="roundtrip_face",
            body_id="roundtrip_body",
            azimuth_deg=8.0,
            distance_m=1.2,
            distance_confidence=0.95,
            depth_source="test_fixture",
            proxemic_space="social",
            engagement_status="engaged",
            face_visible=True,
            face_confidence=0.95,
            gaze_score=0.9,
            body_facing_score=0.85,
        )
        track = AcousticTrack(
            track_id="roundtrip_voice",
            stamp_ms=stamp_ms,
            voice_activity=True,
            speech_probability=0.95,
            clarity=0.9,
            azimuth_deg=10.0,
        )
        header = {"stamp": _ros_time(stamp_ms), "frame_id": "base_link"}
        transport.publish(
            "/perception/vision/people",
            {"header": header, "people": [_typed_vision(person)]},
        )
        transport.publish(
            "/perception/voice/tracks",
            {"header": header, "tracks": [_typed_acoustic(track)]},
        )
        transport.publish_json(
            "/vision/people_json",
            {"stamp_ms": stamp_ms, "people": [asdict(person)]},
        )
        transport.publish_json(
            "/voice/acoustic_tracks",
            {"stamp_ms": stamp_ms, "tracks": [asdict(track)]},
        )
        time.sleep(0.08)
    time.sleep(0.2)
    transport.close()
    matches = [
        state
        for state in received
        if state.get("target_id") == "person:roundtrip_person" and state.get("listen") is True
    ]
    result = {
        "ok": bool(matches),
        "received_states": len(received),
        "matched_states": len(matches),
        "last_match": matches[-1] if matches else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
