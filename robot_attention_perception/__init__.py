"""Audio-visual attention fusion simulation package."""

from .fusion import AttentionFusion, FusionConfig
from .types import (
    AcousticTrack,
    AttentionState,
    PerceptionFrame,
    RobotState,
    TranscriptRecord,
    VisionPerson,
)

__all__ = [
    "AcousticTrack",
    "AttentionFusion",
    "AttentionState",
    "FusionConfig",
    "PerceptionFrame",
    "RobotState",
    "TranscriptRecord",
    "VisionPerson",
]

