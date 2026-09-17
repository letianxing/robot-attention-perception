#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from robot_attention_perception.async_fusion import AsyncPerceptionBuffer
from robot_attention_perception.scenario import acoustic_track_from_dict, optional_float, vision_person_from_dict
from robot_attention_perception.types import RobotState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="ros4hri_native")
    parser.add_argument("--input-jsonl", default="-")
    args = parser.parse_args()

    buffer = AsyncPerceptionBuffer.with_algorithm(args.algorithm)
    source = sys.stdin if args.input_jsonl == "-" else open(args.input_jsonl, "r", encoding="utf-8")
    with source:
        for line in source:
            if not line.strip():
                continue
            event = json.loads(line)
            stamp_ms = int(event.get("stamp_ms", event.get("stamp", 0)))
            event_type = event.get("type", "")
            if event_type == "acoustic":
                tracks = [
                    acoustic_track_from_dict(item, stamp_ms)
                    for item in event.get("tracks", [])
                ]
                buffer.ingest_acoustic(tracks)
            elif event_type == "vision":
                people = [
                    vision_person_from_dict(item, stamp_ms)
                    for item in event.get("people", [])
                ]
                buffer.ingest_vision(people)
            elif event_type == "robot":
                buffer.ingest_robot(
                    RobotState(
                        stamp_ms=stamp_ms,
                        speaking=bool(event.get("speaking", False)),
                        moving=bool(event.get("moving", False)),
                        playback_rms_db=float(event.get("playback_rms_db", 0.0)),
                    )
                )
            else:
                print(json.dumps({"warning": f"unknown event type {event_type}"}, ensure_ascii=False), flush=True)
                continue

            state = buffer.update(stamp_ms)
            if state is not None:
                print(json.dumps(asdict(state), ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
