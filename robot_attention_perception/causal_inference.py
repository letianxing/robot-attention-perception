from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class CausalInferenceConfig:
    common_source_prior: float = 0.62
    spatial_sigma_deg: float = 18.0
    temporal_sigma_ms: float = 120.0
    separate_likelihood: float = 0.22
    association_threshold: float = 0.48


@dataclass(frozen=True)
class CausalInferenceResult:
    common_source_probability: float
    spatial_likelihood: float
    temporal_likelihood: float
    integrated_azimuth_deg: float | None
    model_averaged_azimuth_deg: float | None
    should_associate: bool


class BayesianCausalInference:
    """Infer whether asynchronous visual and acoustic cues share one cause."""

    def __init__(self, config: CausalInferenceConfig | None = None):
        self.config = config or CausalInferenceConfig()

    def infer(
        self,
        audio_azimuth_deg: float | None,
        visual_azimuth_deg: float | None,
        audio_stamp_ms: int,
        visual_stamp_ms: int,
        audio_reliability: float,
        visual_reliability: float,
    ) -> CausalInferenceResult:
        temporal_delta = abs(int(audio_stamp_ms) - int(visual_stamp_ms))
        temporal = exp(-0.5 * (temporal_delta / max(1.0, self.config.temporal_sigma_ms)) ** 2)
        if audio_azimuth_deg is None or visual_azimuth_deg is None:
            spatial = 0.72
        else:
            delta = angular_distance_deg(audio_azimuth_deg, visual_azimuth_deg)
            spatial = exp(-0.5 * (delta / max(1.0, self.config.spatial_sigma_deg)) ** 2)
        reliability = max(0.05, min(1.0, audio_reliability)) * max(
            0.05, min(1.0, visual_reliability)
        )
        common_likelihood = spatial * temporal * (0.35 + 0.65 * reliability)
        prior = max(0.01, min(0.99, self.config.common_source_prior))
        numerator = prior * common_likelihood
        denominator = numerator + (1.0 - prior) * self.config.separate_likelihood
        probability = numerator / max(denominator, 1e-9)
        integrated = weighted_azimuth(
            audio_azimuth_deg,
            visual_azimuth_deg,
            audio_reliability,
            visual_reliability,
        )
        averaged = model_average(audio_azimuth_deg, integrated, probability)
        return CausalInferenceResult(
            common_source_probability=probability,
            spatial_likelihood=spatial,
            temporal_likelihood=temporal,
            integrated_azimuth_deg=integrated,
            model_averaged_azimuth_deg=averaged,
            should_associate=probability >= self.config.association_threshold,
        )


def angular_distance_deg(left: float, right: float) -> float:
    delta = abs((left - right + 180.0) % 360.0 - 180.0)
    return min(delta, 360.0 - delta)


def weighted_azimuth(
    audio: float | None,
    visual: float | None,
    audio_reliability: float,
    visual_reliability: float,
) -> float | None:
    if audio is None:
        return visual
    if visual is None:
        return audio
    audio_weight = max(0.01, float(audio_reliability))
    visual_weight = max(0.01, float(visual_reliability))
    return (audio * audio_weight + visual * visual_weight) / (audio_weight + visual_weight)


def model_average(
    segregated_audio: float | None,
    integrated: float | None,
    common_probability: float,
) -> float | None:
    if segregated_audio is None:
        return integrated
    if integrated is None:
        return segregated_audio
    probability = max(0.0, min(1.0, common_probability))
    return probability * integrated + (1.0 - probability) * segregated_audio
