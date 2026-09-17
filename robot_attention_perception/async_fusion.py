from __future__ import annotations

from dataclasses import dataclass, field

from .fusion import AttentionFusion, FusionConfig
from .types import AcousticTrack, AttentionState, PerceptionFrame, RobotState, VisionPerson


@dataclass(frozen=True)
class AsyncFusionConfig:
    audio_ttl_ms: int = 180
    vision_ttl_ms: int = 450
    robot_ttl_ms: int = 250
    max_frame_rate_hz: float = 50.0


@dataclass
class AsyncPerceptionBuffer:
    """Fuse mixed-rate perception messages without waiting for exact timestamp matches."""

    fusion: AttentionFusion = field(default_factory=AttentionFusion)
    config: AsyncFusionConfig = field(default_factory=AsyncFusionConfig)
    acoustic_tracks: dict[str, AcousticTrack] = field(default_factory=dict)
    vision_people: dict[str, VisionPerson] = field(default_factory=dict)
    robot_state: RobotState = field(default_factory=lambda: RobotState(stamp_ms=0))
    last_update_ms: int = 0

    @classmethod
    def with_algorithm(
        cls,
        algorithm: str = "ros4hri_native",
        use_transcript_wake_words: bool = True,
    ) -> "AsyncPerceptionBuffer":
        return cls(
            fusion=AttentionFusion(
                FusionConfig(
                    algorithm=algorithm,
                    use_transcript_wake_words=use_transcript_wake_words,
                    # Without a wake word, a fresh final ASR transcript is
                    # the addressing signal. Use a slightly lower listen
                    # threshold so the transcript fallback (speech=1.0,
                    # clarity floor=0.7) can pass without fabricating a wake
                    # word bonus.
                    min_listen_score=0.52 if not use_transcript_wake_words else 0.62,
                )
            )
        )

    def ingest_acoustic(self, tracks: list[AcousticTrack] | tuple[AcousticTrack, ...]) -> None:
        for track in tracks:
            self.acoustic_tracks[track.track_id] = track

    def ingest_vision(self, people: list[VisionPerson] | tuple[VisionPerson, ...]) -> None:
        for person in people:
            self.vision_people[person.person_id] = person

    def ingest_robot(self, state: RobotState) -> None:
        self.robot_state = state

    def update(self, now_ms: int) -> AttentionState | None:
        min_period_ms = int(1000.0 / max(self.config.max_frame_rate_hz, 1.0))
        if self.last_update_ms and now_ms - self.last_update_ms < min_period_ms:
            return None
        self.last_update_ms = now_ms

        self._prune(now_ms)
        frame = PerceptionFrame(
            stamp_ms=now_ms,
            acoustic_tracks=[
                track.aged(now_ms - track.stamp_ms, self.config.audio_ttl_ms)
                for track in self.acoustic_tracks.values()
            ],
            vision_people=[
                person.aged(now_ms - person.stamp_ms, self.config.vision_ttl_ms)
                for person in self.vision_people.values()
            ],
            robot=(
                self.robot_state
                if now_ms - self.robot_state.stamp_ms <= self.config.robot_ttl_ms
                else RobotState(stamp_ms=now_ms)
            ),
        )
        state = self.fusion.update(frame)
        if any(now_ms - track.stamp_ms > 0 for track in self.acoustic_tracks.values()):
            state = self._append_reason(state, "async_audio_cache")
        if any(now_ms - person.stamp_ms > 0 for person in self.vision_people.values()):
            state = self._append_reason(state, "async_vision_cache")
        return state

    def _prune(self, now_ms: int) -> None:
        self.acoustic_tracks = {
            track_id: track
            for track_id, track in self.acoustic_tracks.items()
            if now_ms - track.stamp_ms <= self.config.audio_ttl_ms
        }
        self.vision_people = {
            person_id: person
            for person_id, person in self.vision_people.items()
            if now_ms - person.stamp_ms <= self.config.vision_ttl_ms
        }

    @staticmethod
    def _append_reason(state: AttentionState, reason: str) -> AttentionState:
        return AttentionState(
            stamp_ms=state.stamp_ms,
            target_id=state.target_id,
            target_kind=state.target_kind,
            confidence=state.confidence,
            listen=state.listen,
            addressed_to_robot=state.addressed_to_robot,
            source_track_id=state.source_track_id,
            person_id=state.person_id,
            azimuth_deg=state.azimuth_deg,
            reasons=tuple((*state.reasons, reason)),
            transcripts=state.transcripts,
        )
