from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import exp, log
from typing import Any

from .types import AcousticTrack, VisionPerson


@dataclass(frozen=True)
class PersonManagerConfig:
    observation_ttl_ms: int = 700
    reidentification_retention_ms: int = 5000
    # Vision keeps one id for one face even across a lost box, so an id that
    # comes back after a longer absence is the same person rather than a new
    # one. Box geometry cannot support that claim, which is why it has its own
    # much shorter window above.
    identity_retention_ms: int = 60000
    angle_sigma_deg: float = 18.0
    evidence_alpha: float = 0.35
    score_decay: float = 0.96
    voice_bind_threshold: float = 0.52
    identity_bind_threshold: float = 0.85
    identity_margin: float = 0.15
    ambiguity_threshold: float = 0.62
    ambiguity_hold_ms: int = 1800
    clarification_cooldown_ms: int = 10_000


@dataclass(frozen=True)
class IdentityEvidence:
    modality: str
    observation_id: str
    candidates: dict[str, float]
    stamp_ms: int


@dataclass
class _PersonTrack:
    temporary_id: str
    created_ms: int
    last_seen_ms: int
    observation: VisionPerson
    persistent_id: str = ""
    face_probabilities: dict[str, float] = field(default_factory=dict)
    body_probabilities: dict[str, float] = field(default_factory=dict)
    voice_probabilities: dict[str, float] = field(default_factory=dict)
    identity_probabilities: dict[str, float] = field(default_factory=dict)
    ambiguity_since_ms: int = 0
    clarification_emitted_ms: int = 0
    last_identity_leader: str = ""


class ProbabilisticPersonManager:
    """Builds stable anonymous persons from transient face/body/voice observations."""

    def __init__(self, config: PersonManagerConfig | None = None):
        self.config = config or PersonManagerConfig()
        self.tracks: dict[str, _PersonTrack] = {}
        self.next_id = 1
        self.events: list[dict[str, Any]] = []
        self.active_behaviors: list[dict[str, Any]] = []

    def update(
        self,
        vision_people: list[VisionPerson],
        acoustic_tracks: list[AcousticTrack],
        stamp_ms: int,
    ) -> list[VisionPerson]:
        self.events = []
        self.active_behaviors = []
        self._expire(stamp_ms)
        assigned: set[str] = set()
        for observation in vision_people:
            track = self._match_visual(observation, assigned, stamp_ms)
            if track is None:
                track = self._create(observation, stamp_ms)
            assigned.add(track.temporary_id)
            self._update_visual(track, observation, stamp_ms)
        self._associate_voices(acoustic_tracks, stamp_ms)
        self._evaluate_stability(stamp_ms)
        return self.active_people(stamp_ms)

    def ingest_identity_evidence(self, evidence: IdentityEvidence) -> bool:
        target = self._track_for_observation(evidence.modality, evidence.observation_id)
        if target is None:
            return False
        self._decay_map(target.identity_probabilities)
        alpha = self.config.evidence_alpha
        for person_id, confidence in evidence.candidates.items():
            value = max(0.0, min(1.0, float(confidence)))
            previous = target.identity_probabilities.get(person_id, 0.0)
            target.identity_probabilities[person_id] = previous + alpha * (value - previous)
        self._maybe_bind_identity(target, evidence.stamp_ms)
        return True

    def register(self, temporary_id: str, persistent_id: str, stamp_ms: int) -> bool:
        track = self.tracks.get(temporary_id)
        if track is None or not persistent_id.strip():
            return False
        track.persistent_id = persistent_id.strip()
        track.identity_probabilities[persistent_id.strip()] = 1.0
        self.events.append(
            {
                "kind": "person_registered",
                "stamp_ms": stamp_ms,
                "temporary_id": temporary_id,
                "persistent_id": persistent_id.strip(),
            }
        )
        return True

    def apply_av_binding(
        self,
        face_id: str,
        voice_id: str,
        confidence: float,
        stamp_ms: int,
    ) -> bool:
        """Reinforce a temporary face/voice association after conservative AV binding."""
        track = self._track_for_observation("face", face_id)
        if track is None or not voice_id:
            return False
        self._reinforce(track.voice_probabilities, voice_id, confidence)
        self.events.append(
            {
                "kind": "av_voice_bound",
                "stamp_ms": stamp_ms,
                "temporary_id": track.temporary_id,
                "face_id": face_id,
                "voice_id": voice_id,
                "confidence": round(float(confidence), 3),
            }
        )
        return True

    def active_people(self, stamp_ms: int) -> list[VisionPerson]:
        people = []
        for track in self.tracks.values():
            if stamp_ms - track.last_seen_ms > self.config.observation_ttl_ms:
                continue
            voice_id, voice_probability = _leader(track.voice_probabilities)
            resolved_id = track.persistent_id or track.temporary_id
            people.append(
                replace(
                    track.observation,
                    person_id=resolved_id,
                    role=(track.observation.role if track.persistent_id==track.observation.person_id and track.observation.role in {"owner","known"} else "known") if track.persistent_id else "anonymous",
                    voice_id=voice_id if voice_probability >= self.config.voice_bind_threshold else "",
                )
            )
        return people

    def snapshots(self, stamp_ms: int) -> list[dict[str, Any]]:
        output = []
        for track in self.tracks.values():
            identity_id, identity_probability = _leader(track.identity_probabilities)
            voice_id, voice_probability = _leader(track.voice_probabilities)
            output.append(
                {
                    "temporary_id": track.temporary_id,
                    "resolved_person_id": track.persistent_id or track.temporary_id,
                    "anonymous": not bool(track.persistent_id),
                    "role": track.observation.role,
                    "identity_confidence": track.observation.identity_confidence,
                    "gesture": track.observation.gesture,
                    "gaze_score": track.observation.gaze_score,
                    "lip_motion": track.observation.lip_motion,
                    "emotion_valid": track.observation.emotion_valid,
                    "emotion_label": track.observation.emotion_label,
                    "emotion_valence": track.observation.emotion_valence,
                    "emotion_arousal": track.observation.emotion_arousal,
                    "active": stamp_ms - track.last_seen_ms <= self.config.observation_ttl_ms,
                    "age_ms": max(0, stamp_ms - track.last_seen_ms),
                    "azimuth_deg": track.observation.azimuth_deg,
                    "face_probabilities": _sorted_scores(track.face_probabilities),
                    "body_probabilities": _sorted_scores(track.body_probabilities),
                    "voice_probabilities": _sorted_scores(track.voice_probabilities),
                    "identity_probabilities": _sorted_scores(track.identity_probabilities),
                    "best_voice_id": voice_id,
                    "best_voice_probability": round(voice_probability, 3),
                    "best_identity_id": identity_id,
                    "best_identity_probability": round(identity_probability, 3),
                }
            )
        return sorted(output, key=lambda item: (not item["active"], item["temporary_id"]))

    def _create(self, observation: VisionPerson, stamp_ms: int) -> _PersonTrack:
        temporary_id = f"anonymous_{self.next_id:04d}"
        self.next_id += 1
        persistent_id = (
            observation.person_id
            if observation.person_id
            and observation.role not in {"unknown", "stranger", "anonymous"}
            and not observation.person_id.startswith(("vision_", "anonymous_"))
            else ""
        )
        track = _PersonTrack(
            temporary_id,
            stamp_ms,
            stamp_ms,
            observation,
            persistent_id=persistent_id,
        )
        if persistent_id:
            track.identity_probabilities[persistent_id] = 1.0
        self.tracks[temporary_id] = track
        self.events.append(
            {"kind": "temporary_person_created", "stamp_ms": stamp_ms, "temporary_id": temporary_id}
        )
        return track

    def _match_visual(
        self,
        observation: VisionPerson,
        assigned: set[str],
        stamp_ms: int,
    ) -> _PersonTrack | None:
        # Same person id as a track we already have: that is the same person,
        # whatever the boxes did in between. Without this the field session made
        # a new stranger every few seconds out of one person in the room, and
        # nothing about them — what they said, that they had spoken at all —
        # survived long enough to be used.
        person_id = str(observation.person_id or "")
        if _carries_identity(person_id):
            for track in self.tracks.values():
                if track.temporary_id in assigned:
                    continue
                if (str(track.observation.person_id or "") == person_id
                        and stamp_ms - track.last_seen_ms <= self.config.identity_retention_ms):
                    return track
        best = None
        best_score = 0.0
        for track in self.tracks.values():
            if track.temporary_id in assigned:
                continue
            age_ms = stamp_ms - track.last_seen_ms
            if age_ms > self.config.reidentification_retention_ms:
                continue
            score = 0.0
            if observation.body_id:
                score = max(score, track.body_probabilities.get(observation.body_id, 0.0) + 0.55)
            if observation.face_id:
                score = max(score, track.face_probabilities.get(observation.face_id, 0.0) + 0.45)
            if observation.azimuth_deg is not None and track.observation.azimuth_deg is not None:
                delta = abs(observation.azimuth_deg - track.observation.azimuth_deg)
                score = max(score, 0.58 * exp(-0.5 * (delta / self.config.angle_sigma_deg) ** 2))
            score *= max(0.2, 1.0 - age_ms / self.config.reidentification_retention_ms)
            if score > best_score:
                best = track
                best_score = score
        return best if best_score >= 0.24 else None

    def _update_visual(self, track: _PersonTrack, observation: VisionPerson, stamp_ms: int) -> None:
        self._decay_map(track.face_probabilities)
        self._decay_map(track.body_probabilities)
        if observation.face_id:
            self._reinforce(track.face_probabilities, observation.face_id, observation.face_confidence)
        if observation.body_id:
            self._reinforce(track.body_probabilities, observation.body_id, 0.9)
        if (
            not track.persistent_id
            and observation.person_id
            and observation.role not in {"unknown", "stranger", "anonymous"}
            and not observation.person_id.startswith(("vision_", "anonymous_"))
        ):
            track.persistent_id = observation.person_id
            track.identity_probabilities[observation.person_id] = 1.0
        track.observation = observation
        track.last_seen_ms = stamp_ms

    def _associate_voices(self, acoustic_tracks: list[AcousticTrack], stamp_ms: int) -> None:
        active = [
            track
            for track in self.tracks.values()
            if stamp_ms - track.last_seen_ms <= self.config.observation_ttl_ms
        ]
        for person in active:
            self._decay_map(person.voice_probabilities)
        for voice in acoustic_tracks:
            if not voice.voice_activity or not active or not 0<=stamp_ms-voice.stamp_ms<700:
                continue
            # Audio evidence may suggest a transient source association, but it
            # must never rename the sole visible face. Final speech identity is
            # resolved separately with voice priority and explicit conflict data.
            likelihoods = []
            for person in active:
                if voice.azimuth_deg is None or person.observation.azimuth_deg is None:
                    likelihood = 0.82 if len(active) == 1 else 0.25
                else:
                    delta = abs(voice.azimuth_deg - person.observation.azimuth_deg)
                    likelihood = exp(-0.5 * (delta / self.config.angle_sigma_deg) ** 2)
                likelihood += 0.18 * person.voice_probabilities.get(voice.track_id, 0.0)
                likelihoods.append((person, likelihood))
            total = 0.20 + sum(value for _person, value in likelihoods)
            for person, likelihood in likelihoods:
                probability = likelihood / total
                self._reinforce(person.voice_probabilities, voice.track_id, probability)

    def _evaluate_stability(self, stamp_ms: int) -> None:
        for track in self.tracks.values():
            ambiguous = _entropy(track.voice_probabilities) >= self.config.ambiguity_threshold
            identity_leader, identity_probability = _leader(track.identity_probabilities)
            if (
                track.last_identity_leader
                and identity_leader
                and identity_leader != track.last_identity_leader
                and identity_probability >= 0.45
            ):
                ambiguous = True
                self.events.append(
                    {
                        "kind": "identity_jump_smoothed",
                        "stamp_ms": stamp_ms,
                        "temporary_id": track.temporary_id,
                        "from": track.last_identity_leader,
                        "to": identity_leader,
                    }
                )
            if identity_leader:
                track.last_identity_leader = identity_leader
            if not ambiguous:
                track.ambiguity_since_ms = 0
                continue
            if track.ambiguity_since_ms == 0:
                track.ambiguity_since_ms = stamp_ms
            held_ms = stamp_ms - track.ambiguity_since_ms
            if (
                held_ms >= self.config.ambiguity_hold_ms
                and stamp_ms - track.clarification_emitted_ms >= self.config.clarification_cooldown_ms
            ):
                behavior = {
                    "kind": "clarify_person_identity",
                    "stamp_ms": stamp_ms,
                    "temporary_id": track.temporary_id,
                    "reason": "unstable_face_body_voice_association",
                }
                self.active_behaviors.append(behavior)
                self.events.append(behavior)
                track.clarification_emitted_ms = stamp_ms

    def _maybe_bind_identity(self, track: _PersonTrack, stamp_ms: int) -> None:
        ranked = sorted(track.identity_probabilities.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return
        best_id, best_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        if (
            best_probability >= self.config.identity_bind_threshold
            and best_probability - second_probability >= self.config.identity_margin
        ):
            if track.persistent_id != best_id:
                track.persistent_id = best_id
                self.events.append(
                    {
                        "kind": "persistent_person_bound",
                        "stamp_ms": stamp_ms,
                        "temporary_id": track.temporary_id,
                        "persistent_id": best_id,
                        "probability": round(best_probability, 3),
                    }
                )

    def _track_for_observation(self, modality: str, observation_id: str) -> _PersonTrack | None:
        field_name = {
            "face": "face_probabilities",
            "body": "body_probabilities",
            "voice": "voice_probabilities",
        }.get(modality)
        if field_name is None:
            return None
        matches = [
            (getattr(track, field_name).get(observation_id, 0.0), track)
            for track in self.tracks.values()
        ]
        if not matches:
            return None
        score, track = max(matches, key=lambda item: item[0])
        return track if score >= 0.2 else None

    def _expire(self, stamp_ms: int) -> None:
        # A track that carries a real id is worth keeping longer than one held
        # together by box geometry: if that id walks back in we want the same
        # person, not a fresh stranger with no history.
        def retention(track):
            identified = bool(track.persistent_id) or _carries_identity(str(track.observation.person_id or ""))
            return self.config.identity_retention_ms if identified else self.config.reidentification_retention_ms
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if stamp_ms - track.last_seen_ms > retention(track)
        ]
        for track_id in expired:
            track = self.tracks.pop(track_id)
            self.events.append(
                {
                    "kind": "temporary_person_expired",
                    "stamp_ms": stamp_ms,
                    "temporary_id": track.temporary_id,
                    "persistent_id": track.persistent_id or None,
                }
            )

    def _reinforce(self, scores: dict[str, float], key: str, evidence: float) -> None:
        alpha = self.config.evidence_alpha
        previous = scores.get(key, 0.0)
        scores[key] = max(0.0, min(1.0, previous + alpha * (float(evidence) - previous)))

    def _decay_map(self, scores: dict[str, float]) -> None:
        for key in list(scores):
            scores[key] *= self.config.score_decay
            if scores[key] < 0.01:
                del scores[key]


def _carries_identity(person_id: str) -> bool:
    """Does this id mean a person, or only a box the detector happened to keep?

    A name means a person. So does a guest id, which vision issues from the face
    template and reuses when the same face comes back. A face_track id is the
    detector's bookkeeping for overlapping boxes and says nothing once the boxes
    stop overlapping, so it gets no credit here — that is the naming contract
    with the vision service.
    """
    return bool(person_id) and person_id not in {"unknown", "stranger"} and not person_id.startswith(
        ("anonymous_", "vision_face_track"))


def _leader(scores: dict[str, float]) -> tuple[str, float]:
    return max(scores.items(), key=lambda item: item[1]) if scores else ("", 0.0)


def _sorted_scores(scores: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 3) for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)}


def _entropy(scores: dict[str, float]) -> float:
    values = [value for value in scores.values() if value > 0.0]
    if len(values) < 2:
        return 0.0
    total = sum(values)
    probabilities = [value / total for value in values]
    return -sum(value * log(value) for value in probabilities) / log(len(probabilities))
