from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import (
    AcousticTrack,
    PerceptionFrame,
    RobotState,
    TranscriptRecord,
    VisionPerson,
)


def load_scenario(path: str | Path) -> list[PerceptionFrame]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [frame_from_dict(item) for item in data.get("frames", [])]


def frame_from_dict(data: dict[str, Any]) -> PerceptionFrame:
    stamp_ms = int(data["stamp_ms"])
    robot_data = data.get("robot", {})
    robot = RobotState(
        stamp_ms=stamp_ms,
        speaking=bool(robot_data.get("speaking", False)),
        moving=bool(robot_data.get("moving", False)),
        playback_rms_db=float(robot_data.get("playback_rms_db", 0.0)),
    )
    people = [vision_person_from_dict(item, stamp_ms) for item in data.get("vision_people", [])]
    tracks = [acoustic_track_from_dict(item, stamp_ms) for item in data.get("acoustic_tracks", [])]
    return PerceptionFrame(
        stamp_ms=stamp_ms,
        acoustic_tracks=tracks,
        vision_people=people,
        robot=robot,
    )


def acoustic_track_from_dict(data: dict[str, Any], stamp_ms: int) -> AcousticTrack:
    transcript = None
    transcript_data = data.get("transcript")
    if transcript_data:
        transcript = TranscriptRecord(
            track_id=str(data["track_id"]),
            text=str(transcript_data.get("text", "")),
            is_final=bool(transcript_data.get("is_final", False)),
            language=str(transcript_data.get("language", "unknown")),
            clarity=float(transcript_data.get("clarity", data.get("clarity", 0.0))),
            started_ms=optional_int(transcript_data.get("started_ms")),
            ended_ms=optional_int(transcript_data.get("ended_ms")),
            emitted_ms=optional_int(transcript_data.get("emitted_ms")),
        )
    return AcousticTrack(
        track_id=str(data["track_id"]),
        stamp_ms=stamp_ms,
        voice_activity=bool(data.get("voice_activity", False)),
        speech_probability=float(data.get("speech_probability", 0.0)),
        clarity=float(data.get("clarity", 0.0)),
        azimuth_deg=optional_float(data.get("azimuth_deg")),
        elevation_deg=optional_float(data.get("elevation_deg")),
        distance_m=optional_float(data.get("distance_m")),
        overlap_probability=float(data.get("overlap_probability", 0.0)),
        self_echo_probability=float(data.get("self_echo_probability", 0.0)),
        speaker_label=str(data.get("speaker_label", "unknown")),
        target_speaker_probability=float(data.get("target_speaker_probability", 1.0) or 1.0),
        tse_enabled=bool(data.get("tse_enabled", False)),
        tse_healthy=bool(data.get("tse_healthy", True)),
        tse_latency_ms=float(data.get("tse_latency_ms", 0.0) or 0.0),
        target_speech_rejected=bool(data.get("target_speech_rejected", False)),
        transcript=transcript,
    )


def vision_person_from_dict(data: dict[str, Any], stamp_ms: int) -> VisionPerson:
    return VisionPerson(
        person_id=str(data["person_id"]),
        stamp_ms=stamp_ms,
        face_id=str(data.get("face_id", "")),
        body_id=str(data.get("body_id", "")),
        voice_id=str(data.get("voice_id", "")),
        role=str(data.get("role", "unknown")),
        azimuth_deg=optional_float(data.get("azimuth_deg")),
        elevation_deg=optional_float(data.get("elevation_deg")),
        distance_m=optional_float(data.get("distance_m")),
        distance_confidence=float(data.get("distance_confidence", 0.0)),
        depth_source=str(data.get("depth_source", "none")),
        proxemic_space=str(data.get("proxemic_space", "unknown")),
        engagement_status=str(data.get("engagement_status", "unknown")),
        face_visible=bool(data.get("face_visible", False)),
        face_confidence=float(data.get("face_confidence", 0.0)),
        mouth_open_ratio=float(data.get("mouth_open_ratio", 0.0)),
        lip_motion=bool(data.get("lip_motion", False)),
        mouth_roi_features=tuple(float(value) for value in data.get("mouth_roi_features", ())),
        gaze_score=float(data.get("gaze_score", 0.0)),
        body_facing_score=float(data.get("body_facing_score", 0.0)),
        bbox_area_ratio=float(data.get("bbox_area_ratio", 0.0)),
        gesture=str(data.get("gesture", "")),
        gesture_score=float(data.get("gesture_score", 0.0)),
        identity_confidence=float(data.get("identity_confidence", 0.0)),
        emotion_valence=float(data.get("emotion_valence", 0.0)),
        emotion_arousal=float(data.get("emotion_arousal", 0.0)),
        emotion_valid=bool(data.get("emotion_valid", False)),
        emotion_label=str(data.get("emotion_label", "unknown")),
    )


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
