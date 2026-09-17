from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TranscriptRecord:
    """ASR output kept separate from intent, permission, and action decisions."""

    track_id: str
    text: str
    is_final: bool
    language: str = "unknown"
    clarity: float = 0.0
    started_ms: Optional[int] = None
    ended_ms: Optional[int] = None
    emitted_ms: Optional[int] = None

    @property
    def latency_ms(self) -> Optional[int]:
        if self.ended_ms is None or self.emitted_ms is None:
            return None
        return max(0, self.emitted_ms - self.ended_ms)


@dataclass(frozen=True)
class AcousticTrack:
    track_id: str
    stamp_ms: int
    voice_activity: bool
    speech_probability: float
    clarity: float
    azimuth_deg: Optional[float] = None
    elevation_deg: Optional[float] = None
    distance_m: Optional[float] = None
    overlap_probability: float = 0.0
    self_echo_probability: float = 0.0
    speaker_label: str = "unknown"
    speaker_similarity: float = 0.0
    target_speaker_probability: float = 1.0
    tse_enabled: bool = False
    tse_healthy: bool = True
    tse_latency_ms: float = 0.0
    target_speech_rejected: bool = False
    transcript: Optional[TranscriptRecord] = None

    def aged(self, age_ms: int, max_age_ms: int) -> "AcousticTrack":
        if age_ms <= 0:
            return self
        factor = freshness_factor(age_ms, max_age_ms)
        return AcousticTrack(
            track_id=self.track_id,
            stamp_ms=self.stamp_ms,
            voice_activity=self.voice_activity and factor > 0.0,
            speech_probability=self.speech_probability * factor,
            clarity=self.clarity * factor,
            azimuth_deg=self.azimuth_deg,
            elevation_deg=self.elevation_deg,
            distance_m=self.distance_m,
            overlap_probability=self.overlap_probability,
            self_echo_probability=self.self_echo_probability,
            speaker_label=self.speaker_label,
            speaker_similarity=self.speaker_similarity,
            transcript=self.transcript,
        )


@dataclass(frozen=True)
class VisionPerson:
    person_id: str
    stamp_ms: int
    face_id: str = ""
    body_id: str = ""
    voice_id: str = ""
    role: str = "unknown"
    azimuth_deg: Optional[float] = None
    elevation_deg: Optional[float] = None
    distance_m: Optional[float] = None
    distance_confidence: float = 0.0
    depth_source: str = "none"
    proxemic_space: str = "unknown"
    engagement_status: str = "unknown"
    face_visible: bool = False
    face_confidence: float = 0.0
    mouth_open_ratio: float = 0.0
    lip_motion: bool = False
    mouth_roi_features: tuple[float, ...] = ()
    gaze_score: float = 0.0
    body_facing_score: float = 0.0
    bbox_area_ratio: float = 0.0
    gesture: str = ""
    gesture_score: float = 0.0
    identity_confidence: float = 0.0
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.0
    emotion_valid: bool = False
    emotion_label: str = "unknown"

    def aged(self, age_ms: int, max_age_ms: int) -> "VisionPerson":
        if age_ms <= 0:
            return self
        factor = freshness_factor(age_ms, max_age_ms)
        return VisionPerson(
            person_id=self.person_id,
            stamp_ms=self.stamp_ms,
            face_id=self.face_id,
            body_id=self.body_id,
            voice_id=self.voice_id,
            role=self.role,
            azimuth_deg=self.azimuth_deg,
            elevation_deg=self.elevation_deg,
            distance_m=self.distance_m,
            distance_confidence=self.distance_confidence * factor,
            depth_source=self.depth_source,
            proxemic_space=self.proxemic_space,
            engagement_status=self.engagement_status if factor > 0.25 else "unknown",
            face_visible=self.face_visible and factor > 0.0,
            face_confidence=self.face_confidence * factor,
            mouth_open_ratio=self.mouth_open_ratio,
            lip_motion=self.lip_motion and factor > 0.4,
            mouth_roi_features=self.mouth_roi_features,
            gaze_score=self.gaze_score * factor,
            body_facing_score=self.body_facing_score * factor,
            bbox_area_ratio=self.bbox_area_ratio,
            gesture=self.gesture if factor > 0.45 else "",
            gesture_score=self.gesture_score * factor,
            identity_confidence=self.identity_confidence,
            emotion_valence=self.emotion_valence,
            emotion_arousal=self.emotion_arousal,
            emotion_valid=self.emotion_valid,
            emotion_label=self.emotion_label,
        )


@dataclass(frozen=True)
class RobotState:
    stamp_ms: int
    speaking: bool = False
    moving: bool = False
    playback_rms_db: float = 0.0


@dataclass(frozen=True)
class PerceptionFrame:
    stamp_ms: int
    acoustic_tracks: list[AcousticTrack] = field(default_factory=list)
    vision_people: list[VisionPerson] = field(default_factory=list)
    robot: RobotState = field(default_factory=lambda: RobotState(stamp_ms=0))


@dataclass(frozen=True)
class AttentionState:
    stamp_ms: int
    target_id: str
    target_kind: str
    confidence: float
    listen: bool
    addressed_to_robot: bool
    source_track_id: Optional[str] = None
    person_id: Optional[str] = None
    azimuth_deg: Optional[float] = None
    reasons: tuple[str, ...] = ()
    transcripts: tuple[TranscriptRecord, ...] = ()


def freshness_factor(age_ms: int, max_age_ms: int) -> float:
    if max_age_ms <= 0:
        return 0.0
    if age_ms <= 0:
        return 1.0
    if age_ms >= max_age_ms:
        return 0.0
    return 1.0 - (float(age_ms) / float(max_age_ms))
