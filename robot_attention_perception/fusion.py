from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import fabs
from typing import Optional

from .causal_inference import BayesianCausalInference, CausalInferenceConfig
from .types import AcousticTrack, AttentionState, PerceptionFrame, TranscriptRecord, VisionPerson


WAKE_WORDS = (
    "reachy",
    "hey reachy",
    "机器人",
    "小圆",
    "小园",   # 同音误识
    "小元",   # 同音误识
    "小袁",   # 同音误识（现场实际出现过）
)


@dataclass(frozen=True)
class FusionConfig:
    algorithm: str = "ros4hri_native"
    association_max_angle_deg: float = 35.0
    min_attention_score: float = 0.52
    min_listen_score: float = 0.62
    hysteresis_bonus: float = 0.10
    use_transcript_wake_words: bool = True
    audio_weight: float = 0.45
    clarity_weight: float = 0.20
    gaze_weight: float = 0.18
    facing_weight: float = 0.14
    identity_weight: float = 0.08
    gesture_weight: float = 0.08
    distance_weight: float = 0.05
    overlap_penalty: float = 0.12
    echo_penalty: float = 0.45
    robot_speaking_echo_penalty: float = 0.18
    self_echo_reject_threshold: float = 0.75
    common_source_prior: float = 0.62
    causal_association_threshold: float = 0.48
    causal_spatial_sigma_deg: float = 18.0
    causal_temporal_sigma_ms: float = 120.0


@dataclass
class AttentionMemory:
    last_target_id: Optional[str] = None


@dataclass(frozen=True)
class _Candidate:
    score: float
    acoustic: AcousticTrack
    vision: Optional[VisionPerson]
    reasons: tuple[str, ...]


class AttentionAlgorithm:
    """Base class for replaceable attention policies."""

    name = "base"
    description = "abstract attention algorithm"

    def update(
        self,
        frame: PerceptionFrame,
        config: FusionConfig,
        memory: AttentionMemory,
    ) -> AttentionState:
        raise NotImplementedError


class AttentionFusion:
    """Stateful attention engine with pluggable policies.

    The default policy is `ros4hri_native`, which follows ROS4HRI's
    face/body/voice/person association model. Custom algorithms can be loaded
    either from the built-in registry or by passing `module:ClassName`.
    """

    _registry: dict[str, type[AttentionAlgorithm]] = {}

    def __init__(self, config: FusionConfig | None = None, algorithm: str | None = None):
        base_config = config or FusionConfig()
        if algorithm is not None:
            base_config = FusionConfig(**{**base_config.__dict__, "algorithm": algorithm})
        self.config = base_config
        self.memory = AttentionMemory()
        self.algorithm = self.create_algorithm(self.config.algorithm)

    def update(self, frame: PerceptionFrame) -> AttentionState:
        return self.algorithm.update(frame, self.config, self.memory)

    @classmethod
    def register_algorithm(cls, algorithm_cls: type[AttentionAlgorithm]) -> None:
        cls._registry[algorithm_cls.name] = algorithm_cls

    @classmethod
    def available_algorithms(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry))

    @classmethod
    def create_algorithm(cls, name: str) -> AttentionAlgorithm:
        algorithm_cls = cls._registry.get(name)
        if algorithm_cls is None and ":" in name:
            module_name, class_name = name.split(":", 1)
            module = import_module(module_name)
            algorithm_cls = getattr(module, class_name)
        if algorithm_cls is None:
            choices = ", ".join(cls.available_algorithms())
            raise ValueError(f"unknown attention algorithm '{name}', available: {choices}")
        algorithm = algorithm_cls()
        if not isinstance(algorithm, AttentionAlgorithm):
            raise TypeError(f"{algorithm_cls!r} is not an AttentionAlgorithm")
        return algorithm


class WeightedAudioVisualAttention(AttentionAlgorithm):
    """Simple deterministic policy suitable for simulation and early robotics tests."""

    name = "weighted_audio_visual"
    description = "weighted VAD/DOA/clarity/gaze/facing/identity attention score"

    def update(
        self,
        frame: PerceptionFrame,
        config: FusionConfig,
        memory: AttentionMemory,
    ) -> AttentionState:
        candidates = [self._score_track(frame, track, config, memory) for track in frame.acoustic_tracks]
        candidates = [candidate for candidate in candidates if candidate.score >= config.min_attention_score]

        transcripts = tuple(
            track.transcript
            for track in frame.acoustic_tracks
            if track.transcript is not None
        )

        if not candidates:
            memory.last_target_id = None
            return AttentionState(
                stamp_ms=frame.stamp_ms,
                target_id="none",
                target_kind="none",
                confidence=0.0,
                listen=False,
                addressed_to_robot=False,
                reasons=("no_active_candidate",),
                transcripts=transcripts,
            )

        best = max(candidates, key=lambda candidate: candidate.score)
        target_id = self._target_id(best)
        memory.last_target_id = target_id
        addressed = self._is_addressed(best, config)

        return AttentionState(
            stamp_ms=frame.stamp_ms,
            target_id=target_id,
            target_kind="person" if best.vision is not None else "sound_source",
            confidence=round(min(1.0, best.score), 3),
            listen=best.score >= config.min_listen_score,
            addressed_to_robot=addressed,
            source_track_id=best.acoustic.track_id,
            person_id=best.vision.person_id if best.vision is not None else None,
            azimuth_deg=self._merged_azimuth(best),
            reasons=best.reasons,
            transcripts=transcripts,
        )

    def _score_track(
        self,
        frame: PerceptionFrame,
        track: AcousticTrack,
        config: FusionConfig,
        memory: AttentionMemory,
    ) -> _Candidate:
        reasons: list[str] = []
        if not track.voice_activity:
            return _Candidate(0.0, track, None, ("no_voice_activity",))
        if (
            frame.robot.speaking
            and track.self_echo_probability >= config.self_echo_reject_threshold
        ):
            return _Candidate(0.0, track, None, ("rejected_self_echo", "robot_playback_active"))

        vision, association_reasons = self._associate(track, frame.vision_people, config)
        reasons.extend(association_reasons)
        score = 0.0

        speech = clamp(track.speech_probability)
        clarity = clamp(track.clarity)
        score += config.audio_weight * speech
        score += config.clarity_weight * clarity
        if speech > 0.5:
            reasons.append("speech")
        if clarity > 0.65:
            reasons.append("clear_voice")

        if vision is not None:
            score += self._score_vision(vision, config, reasons)

        if config.use_transcript_wake_words and self._has_wake_word(track.transcript):
            score += 0.12
            reasons.append("wake_word")

        overlap = clamp(track.overlap_probability)
        if overlap > 0.4:
            score -= config.overlap_penalty * overlap
            reasons.append("overlap")

        echo = clamp(track.self_echo_probability)
        if echo > 0.2:
            score -= config.echo_penalty * echo
            reasons.append("self_echo")
        if frame.robot.speaking and echo > 0.1:
            score -= config.robot_speaking_echo_penalty
            reasons.append("robot_playback_active")

        candidate_target_id = self._target_id(_Candidate(score, track, vision, ()))
        if memory.last_target_id and candidate_target_id == memory.last_target_id:
            score += config.hysteresis_bonus
            reasons.append("hysteresis")

        return _Candidate(max(0.0, score), track, vision, tuple(reasons))

    def _associate(
        self,
        track: AcousticTrack,
        people: list[VisionPerson],
        config: FusionConfig,
    ) -> tuple[Optional[VisionPerson], list[str]]:
        if not people:
            return None, []

        if track.azimuth_deg is None:
            visible = [person for person in people if person.face_visible]
            if len(visible) == 1:
                result = self._causal_result(track, visible[0], config)
                if result.should_associate:
                    return visible[0], [
                        f"matched_single_visible:{visible[0].person_id}",
                        f"causal_common:{result.common_source_probability:.2f}",
                    ]
            return None, ["causal_separate:ambiguous_mono_audio"]

        scored = []
        for person in people:
            if person.azimuth_deg is None:
                continue
            delta = angular_distance_deg(track.azimuth_deg, person.azimuth_deg)
            result = self._causal_result(track, person, config)
            if delta <= config.association_max_angle_deg and result.should_associate:
                scored.append((result.common_source_probability, delta, person))
        if not scored:
            return None, ["causal_separate:spatiotemporal_conflict"]
        probability, delta, person = max(scored, key=lambda item: item[0])
        return person, [
            f"matched_bearing:{person.person_id}:{delta:.1f}deg",
            f"causal_common:{probability:.2f}",
        ]

    @staticmethod
    def _causal_result(track: AcousticTrack, person: VisionPerson, config: FusionConfig):
        inference = BayesianCausalInference(
            CausalInferenceConfig(
                common_source_prior=config.common_source_prior,
                spatial_sigma_deg=config.causal_spatial_sigma_deg,
                temporal_sigma_ms=config.causal_temporal_sigma_ms,
                association_threshold=config.causal_association_threshold,
            )
        )
        audio_reliability = 0.5 * clamp(track.speech_probability) + 0.5 * clamp(track.clarity)
        visual_reliability = (
            clamp(person.face_confidence) + clamp(person.gaze_score) + clamp(person.body_facing_score)
        ) / 3.0
        return inference.infer(
            track.azimuth_deg,
            person.azimuth_deg,
            track.stamp_ms,
            person.stamp_ms,
            audio_reliability,
            visual_reliability,
        )

    def _score_vision(
        self,
        vision: VisionPerson,
        config: FusionConfig,
        reasons: list[str],
    ) -> float:
        gaze = clamp(vision.gaze_score)
        facing = clamp(vision.body_facing_score)
        score = 0.0
        score += config.gaze_weight * gaze
        score += config.facing_weight * facing
        if vision.distance_m is not None and vision.distance_confidence > 0.0:
            distance_score = clamp((4.5 - vision.distance_m) / 4.0)
            score += config.distance_weight * distance_score * clamp(vision.distance_confidence)
            reasons.append(f"depth:{vision.depth_source}:{vision.distance_m:.2f}m")
        else:
            score += config.distance_weight * clamp(vision.bbox_area_ratio * 4.0)
        reasons.append(f"matched_vision:{vision.person_id}")
        if gaze > 0.55:
            reasons.append("gaze_to_robot")
        if facing > 0.55:
            reasons.append("body_facing_robot")
        if vision.role in {"owner", "known"}:
            score += config.identity_weight * max(0.5, clamp(vision.identity_confidence))
            reasons.append(f"role:{vision.role}")
        if vision.gesture in {"wave", "invite", "call"}:
            score += config.gesture_weight
            reasons.append(f"gesture:{vision.gesture}")
        return score

    @staticmethod
    def _has_wake_word(transcript: Optional[TranscriptRecord]) -> bool:
        if transcript is None:
            return False
        text = transcript.text.lower()
        return any(word in text for word in WAKE_WORDS)

    def _is_addressed(self, candidate: _Candidate, config: FusionConfig) -> bool:
        if candidate.score < config.min_listen_score:
            return False
        if config.use_transcript_wake_words and self._has_wake_word(candidate.acoustic.transcript):
            return True
        if candidate.vision is None:
            # In acoustic-only deployments wake words are optional. Once the
            # candidate already clears min_listen_score, treat a clear speech
            # track as addressed; applications may still require a wake word
            # at their own command/execution boundary.
            return not config.use_transcript_wake_words
        return (
            candidate.vision.gaze_score >= 0.58
            or candidate.vision.body_facing_score >= 0.68
            or candidate.vision.gesture in {"wave", "invite", "call"}
        )

    @staticmethod
    def _target_id(candidate: _Candidate) -> str:
        if candidate.vision is not None:
            return f"person:{candidate.vision.person_id}"
        return f"sound:{candidate.acoustic.track_id}"

    @staticmethod
    def _merged_azimuth(candidate: _Candidate) -> Optional[float]:
        if candidate.vision is not None and candidate.vision.azimuth_deg is not None:
            return candidate.vision.azimuth_deg
        return candidate.acoustic.azimuth_deg


class Ros4HriNativeAttention(WeightedAudioVisualAttention):
    """Default policy that follows ROS4HRI person/face/body/voice semantics."""

    name = "ros4hri_native"
    description = "ROS4HRI-style person/voice association with engagement and proxemics"

    def _associate(
        self,
        track: AcousticTrack,
        people: list[VisionPerson],
        config: FusionConfig,
    ) -> tuple[Optional[VisionPerson], list[str]]:
        for person in people:
            if person.voice_id and person.voice_id == track.track_id:
                return person, [f"ros4hri_voice_match:{person.person_id}"]
        return super()._associate(track, people, config)

    def _score_vision(
        self,
        vision: VisionPerson,
        config: FusionConfig,
        reasons: list[str],
    ) -> float:
        score = super()._score_vision(vision, config, reasons)
        if vision.engagement_status in {"engaged", "engaging"}:
            score += 0.10
            reasons.append(f"engagement:{vision.engagement_status}")
        if vision.proxemic_space in {"intimate", "personal", "social"}:
            score += 0.04
            reasons.append(f"proxemic:{vision.proxemic_space}")
        return score

    def _is_addressed(self, candidate: _Candidate, config: FusionConfig) -> bool:
        if candidate.vision is not None and candidate.vision.engagement_status == "engaged":
            return candidate.score >= config.min_listen_score
        return super()._is_addressed(candidate, config)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def angular_distance_deg(a: float, b: float) -> float:
    delta = fabs((a - b + 180.0) % 360.0 - 180.0)
    return min(delta, 360.0 - delta)


AttentionFusion.register_algorithm(Ros4HriNativeAttention)
AttentionFusion.register_algorithm(WeightedAudioVisualAttention)
