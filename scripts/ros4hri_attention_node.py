#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_attention_perception.async_fusion import AsyncPerceptionBuffer
from robot_attention_perception.scenario import acoustic_track_from_dict, vision_person_from_dict
from robot_attention_perception.types import RobotState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="ros4hri_native")
    parser.add_argument("--voice-topic", default="/voice/acoustic_tracks")
    parser.add_argument("--vision-json-topic", default="/vision/people_json")
    parser.add_argument("--vision-people-topic", default="/vision/people")
    parser.add_argument("--robot-playback-topic", default="/robot/playback_state")
    parser.add_argument("--attention-topic", default="/attention/state")
    parser.add_argument("--typed-voice-topic", default="/perception/voice/tracks")
    parser.add_argument("--typed-vision-topic", default="/perception/vision/people")
    parser.add_argument("--typed-attention-topic", default="/perception/attention/state")
    parser.add_argument("--target-voice-topic", default="/attention/target_voice_id")
    parser.add_argument("--look-at-topic", default="/attention/look_at")
    parser.add_argument("--queue-size", type=int, default=10)
    args = parser.parse_args()

    try:
        import rclpy
        from geometry_msgs.msg import PointStamped
        from perception_interfaces.msg import AcousticTracks, AttentionState as AttentionStateMsg, People
        from std_msgs.msg import String
    except Exception as exc:
        raise SystemExit(f"ROS2 attention node requires a sourced ROS2 environment: {exc}")

    try:
        from vision_detection.msg import PeopleSignals
    except Exception:
        PeopleSignals = None

    rclpy.init()
    node = rclpy.create_node("robot_attention_perception")
    buffer = AsyncPerceptionBuffer.with_algorithm(args.algorithm)
    attention_pub = node.create_publisher(String, args.attention_topic, args.queue_size)
    typed_attention_pub = node.create_publisher(AttentionStateMsg, args.typed_attention_topic, args.queue_size)
    target_voice_pub = node.create_publisher(String, args.target_voice_topic, args.queue_size)
    look_at_pub = node.create_publisher(PointStamped, args.look_at_topic, args.queue_size)

    def publish_state(stamp_ms: int) -> None:
        state = buffer.update(stamp_ms)
        if state is None:
            return
        payload = String()
        payload.data = json.dumps(asdict(state), ensure_ascii=False, separators=(",", ":"))
        attention_pub.publish(payload)
        typed_attention_pub.publish(attention_state_message(state, args.algorithm, node, AttentionStateMsg))

        target_voice = String()
        target_voice.data = state.source_track_id or ""
        target_voice_pub.publish(target_voice)

        if state.azimuth_deg is not None:
            look_at_pub.publish(point_from_azimuth(state.azimuth_deg, node))

    def on_voice(msg) -> None:
        stamp_ms = now_ms()
        tracks = parse_acoustic_payload(msg.data, stamp_ms)
        if tracks:
            buffer.ingest_acoustic(tracks)
            publish_state(stamp_ms)

    def on_vision_json(msg) -> None:
        stamp_ms = now_ms()
        people = parse_vision_payload(msg.data, stamp_ms)
        if people:
            buffer.ingest_vision(people)
            publish_state(stamp_ms)

    def on_robot_playback(msg) -> None:
        stamp_ms = now_ms()
        state = parse_robot_state(msg.data, stamp_ms)
        buffer.ingest_robot(state)
        publish_state(stamp_ms)

    def on_typed_voice(msg) -> None:
        tracks = [acoustic_track_from_dict(typed_acoustic_dict(item), stamp_to_ms(item.observed_at)) for item in msg.tracks]
        if tracks:
            buffer.ingest_acoustic(tracks)
        publish_state(stamp_to_ms(msg.header.stamp) or now_ms())

    def on_typed_vision(msg) -> None:
        people = [vision_person_from_dict(typed_vision_dict(item), stamp_to_ms(item.observed_at)) for item in msg.people]
        if people:
            buffer.ingest_vision(people)
        publish_state(stamp_to_ms(msg.header.stamp) or now_ms())

    node.create_subscription(String, args.voice_topic, on_voice, args.queue_size)
    node.create_subscription(String, args.vision_json_topic, on_vision_json, args.queue_size)
    node.create_subscription(String, args.robot_playback_topic, on_robot_playback, args.queue_size)
    node.create_subscription(AcousticTracks, args.typed_voice_topic, on_typed_voice, args.queue_size)
    node.create_subscription(People, args.typed_vision_topic, on_typed_vision, args.queue_size)
    if PeopleSignals is not None:
        node.create_subscription(PeopleSignals, args.vision_people_topic, lambda msg: on_vision_people(msg, buffer, publish_state), args.queue_size)

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_acoustic_payload(data: str, stamp_ms: int):
    payload = json.loads(data)
    if "engineering_track" in payload:
        return [acoustic_track_from_dict(payload["engineering_track"], int(payload.get("stamp_ms", stamp_ms)))]
    if "tracks" in payload:
        return [acoustic_track_from_dict(item, int(payload.get("stamp_ms", stamp_ms))) for item in payload["tracks"]]
    if "track_id" in payload:
        return [acoustic_track_from_dict(payload, int(payload.get("stamp_ms", stamp_ms)))]
    return []


def parse_vision_payload(data: str, stamp_ms: int):
    payload = json.loads(data)
    if isinstance(payload, list):
        people = payload
    elif isinstance(payload, dict):
        people = payload.get("people", [])
    else:
        people = []
    return [vision_person_from_dict(item, int(item.get("stamp_ms", stamp_ms))) for item in people]


def parse_robot_state(data: str, stamp_ms: int) -> RobotState:
    payload = json.loads(data)
    return RobotState(
        stamp_ms=int(payload.get("stamp_ms", stamp_ms)),
        speaking=bool(payload.get("speaking", payload.get("is_speaking", False))),
        moving=bool(payload.get("moving", False)),
        playback_rms_db=float(payload.get("playback_rms_db", 0.0)),
    )


def on_vision_people(msg, buffer: AsyncPerceptionBuffer, publish_state) -> None:
    stamp_ms = now_ms()
    people = []
    for person in msg.people:
        people.append(
            vision_person_from_dict(
                {
                    "person_id": person.person_id,
                    "role": person.role,
                    "face_id": person.face_id,
                    "body_id": person.body_id,
                    "voice_id": person.voice_id,
                    "azimuth_deg": person.azimuth_deg if person.has_azimuth else None,
                    "elevation_deg": person.elevation_deg if person.has_elevation else None,
                    "distance_m": person.distance_m if person.has_distance else None,
                    "proxemic_space": person.proxemic_space,
                    "engagement_status": person.engagement_status,
                    "face_visible": person.face_visible,
                    "face_confidence": person.face_confidence,
                    "gaze_score": person.gaze_score,
                    "body_facing_score": person.body_facing_score,
                    "bbox_area_ratio": person.bbox_area_ratio,
                    "gesture": person.gesture,
                    "gesture_score": person.gesture_score,
                    "identity_confidence": person.identity_confidence,
                    "emotion_valence": person.emotion_valence,
                    "emotion_arousal": person.emotion_arousal,
                    "emotion_valid": person.emotion_valid,
                    "emotion_label": person.emotion_label,
                },
                stamp_ms,
            )
        )
    if people:
        buffer.ingest_vision(people)
        publish_state(stamp_ms)


def point_from_azimuth(azimuth_deg: float, node):
    from geometry_msgs.msg import PointStamped
    import math

    point = PointStamped()
    point.header.stamp = node.get_clock().now().to_msg()
    point.header.frame_id = "base_link"
    radius = 1.0
    radians = math.radians(float(azimuth_deg))
    point.point.x = radius * math.cos(radians)
    point.point.y = radius * math.sin(radians)
    point.point.z = 0.0
    return point


def typed_acoustic_dict(message):
    return {
        "track_id": message.track_id,
        "voice_activity": message.voice_activity,
        "speech_probability": message.speech_probability,
        "clarity": message.clarity,
        "azimuth_deg": message.azimuth_deg if message.has_azimuth else None,
        "elevation_deg": message.elevation_deg if message.has_elevation else None,
        "distance_m": message.distance_m if message.has_distance else None,
        "overlap_probability": message.overlap_probability,
        "self_echo_probability": message.self_echo_probability,
        "speaker_label": message.speaker_label,
        "speaker_similarity": message.speaker_similarity,
        "target_speaker_probability": message.target_speaker_probability,
        "tse_enabled": message.tse_enabled,
        "tse_healthy": message.tse_healthy,
        "tse_latency_ms": message.tse_latency_ms,
        "target_speech_rejected": message.target_speech_rejected,
        "transcript": (
            {
                "track_id": message.track_id,
                "text": message.transcript_text,
                "is_final": message.transcript_is_final,
                "language": message.language,
            }
            if message.transcript_text
            else None
        ),
    }


def typed_vision_dict(message):
    return {
        "person_id": message.person_id,
        "face_id": message.face_id,
        "body_id": message.body_id,
        "voice_id": message.voice_id,
        "role": message.role,
        "azimuth_deg": message.azimuth_deg if message.has_azimuth else None,
        "elevation_deg": message.elevation_deg if message.has_elevation else None,
        "distance_m": message.distance_m if message.has_distance else None,
        "distance_confidence": message.distance_confidence,
        "depth_source": message.depth_source,
        "proxemic_space": message.proxemic_space,
        "engagement_status": message.engagement_status,
        "face_visible": message.face_visible,
        "face_confidence": message.face_confidence,
        "mouth_open_ratio": message.mouth_open_ratio,
        "lip_motion": message.lip_motion,
        "mouth_roi_features": list(message.mouth_roi_features),
        "gaze_score": message.gaze_score,
        "body_facing_score": message.body_facing_score,
        "bbox_area_ratio": message.bbox_area_ratio,
        "gesture": message.gesture,
        "gesture_score": message.gesture_score,
        "identity_confidence": message.identity_confidence,
        "emotion_valence": message.emotion_valence,
        "emotion_arousal": message.emotion_arousal,
        "emotion_valid": message.emotion_valid,
        "emotion_label": message.emotion_label,
    }


def attention_state_message(state, algorithm, node, message_class):
    message = message_class()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = "base_link"
    message.algorithm = algorithm
    message.target_id = state.target_id
    message.target_kind = state.target_kind
    message.confidence = state.confidence
    message.listen = state.listen
    message.addressed_to_robot = state.addressed_to_robot
    message.source_track_id = state.source_track_id or ""
    message.person_id = state.person_id or ""
    message.has_azimuth = state.azimuth_deg is not None
    message.azimuth_deg = float(state.azimuth_deg or 0.0)
    message.reasons = list(state.reasons)
    return message


def stamp_to_ms(stamp) -> int:
    return int(stamp.sec) * 1000 + int(stamp.nanosec) // 1_000_000


def now_ms() -> int:
    return int(time.time() * 1000)


if __name__ == "__main__":
    main()
