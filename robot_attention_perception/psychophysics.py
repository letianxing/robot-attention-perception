from __future__ import annotations

from dataclasses import asdict
from math import exp
from typing import Any

from .causal_inference import BayesianCausalInference
from .fusion import AttentionFusion
from .types import AcousticTrack, PerceptionFrame, RobotState, VisionPerson


def double_flash_curve(intervals_ms: list[int], binding_window_ms: int = 100) -> list[dict[str, float]]:
    """Return the engineering fission response over beep intervals."""
    window = max(1, int(binding_window_ms))
    return [
        {
            "beep_interval_ms": float(interval),
            "fission_strength": exp(-0.5 * (float(interval) / window) ** 2),
        }
        for interval in intervals_ms
    ]


def mcgurk_consistency_benchmark() -> dict[str, float]:
    """Compare aligned congruent and incongruent AV speech confidence."""
    causal = BayesianCausalInference().infer(5.0, 6.0, 1000, 1020, 0.9, 0.9)
    common = causal.common_source_probability
    congruent = common * 0.96 + (1.0 - common) * 0.72
    incongruent = common * 0.42 + (1.0 - common) * 0.72
    return {
        "common_source_probability": common,
        "congruent_confidence": congruent,
        "incongruent_confidence": incongruent,
        "consistency_gap": congruent - incongruent,
    }


def cocktail_attention_benchmark(frame_period_ms: int = 20) -> dict[str, Any]:
    """Measure target lock rate and switching delay for two visual speakers."""
    fusion = AttentionFusion(algorithm="ros4hri_native")
    states = []
    switch_at_ms = 1400
    first_new_target_ms = None
    for stamp_ms in range(1000, 2001, frame_period_ms):
        target_left = stamp_ms < switch_at_ms
        people = [
            VisionPerson(
                person_id="left",
                stamp_ms=stamp_ms,
                voice_id="voice_left",
                azimuth_deg=-35.0,
                face_visible=True,
                face_confidence=0.9,
                gaze_score=0.85 if target_left else 0.2,
                body_facing_score=0.8 if target_left else 0.3,
                engagement_status="engaged" if target_left else "unengaged",
                proxemic_space="social",
            ),
            VisionPerson(
                person_id="right",
                stamp_ms=stamp_ms,
                voice_id="voice_right",
                azimuth_deg=35.0,
                face_visible=True,
                face_confidence=0.9,
                gaze_score=0.2 if target_left else 0.85,
                body_facing_score=0.3 if target_left else 0.8,
                engagement_status="unengaged" if target_left else "engaged",
                proxemic_space="social",
            ),
        ]
        active_id = "voice_left" if target_left else "voice_right"
        active_az = -35.0 if target_left else 35.0
        tracks = [
            AcousticTrack(active_id, stamp_ms, True, 0.95, 0.9, azimuth_deg=active_az),
            AcousticTrack(
                "voice_right" if target_left else "voice_left",
                stamp_ms,
                True,
                0.45,
                0.5,
                azimuth_deg=-active_az,
                overlap_probability=0.8,
            ),
        ]
        state = fusion.update(
            PerceptionFrame(stamp_ms, tracks, people, RobotState(stamp_ms))
        )
        states.append(state)
        if stamp_ms >= switch_at_ms and state.target_id == "person:right" and first_new_target_ms is None:
            first_new_target_ms = stamp_ms
    expected = ["person:left" if state.stamp_ms < switch_at_ms else "person:right" for state in states]
    lock_count = sum(state.target_id == target for state, target in zip(states, expected))
    return {
        "frames": len(states),
        "target_lock_rate": lock_count / len(states),
        "switch_delay_ms": None if first_new_target_ms is None else first_new_target_ms - switch_at_ms,
        "last_state": asdict(states[-1]),
    }


def full_benchmark() -> dict[str, Any]:
    """Run all three psychophysical engineering analogues."""
    return {
        "double_flash": double_flash_curve([25, 50, 75, 100, 150, 250]),
        "mcgurk": mcgurk_consistency_benchmark(),
        "cocktail_party": cocktail_attention_benchmark(),
    }
