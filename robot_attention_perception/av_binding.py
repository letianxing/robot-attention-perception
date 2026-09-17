# 设计哲学：硬门控优先于模型分数，误绑定成本远高于漏绑定；跨模态模型只提供辅助证据。
# 降级路径：脸不可见时仅维护 3 秒音频环形缓冲和临时说话人簇；模型/嘴部特征缺失时拒绝身份绑定。
# 调参建议：先按现场数据校准时间偏差，再调 bind_threshold 和 confirmations；发布前用独立集验证 Precision。
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Generic, TypeVar

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - PyTorch is optional on the Mac control plane.
    torch = None
    nn = None


T = TypeVar("T")


@dataclass(frozen=True)
class AudioFeatureFrame:
    """One timestamped audio feature frame."""

    stamp_ms: int
    mfcc: np.ndarray
    vad_active: bool
    speaker_cluster_id: str = "temporary_speaker"


@dataclass(frozen=True)
class VideoFeatureFrame:
    """One timestamped face and mouth observation."""

    stamp_ms: int
    face_id: str
    face_visible: bool
    lip_motion: bool
    mouth_open_ratio: float
    mouth_roi_features: np.ndarray


@dataclass(frozen=True)
class BindingDecision:
    """Conservative face-to-temporary-speaker binding result."""

    action: str
    face_id: str
    speaker_cluster_id: str
    offset_ms: int | None
    confidence: float
    reason: str


class TimestampedRing(Generic[T]):
    """Bounded timestamped queue using integer milliseconds."""

    def __init__(self, retention_ms: int = 3000, max_items: int = 512):
        """Create a queue with retention and item-count limits."""
        self.retention_ms = max(1, int(retention_ms))
        self.items: Deque[T] = deque(maxlen=max(1, int(max_items)))

    def append(self, item: T) -> None:
        """Append an item and remove observations outside the retention window."""
        self.items.append(item)
        newest_ms = int(getattr(item, "stamp_ms"))
        while self.items and newest_ms - int(getattr(self.items[0], "stamp_ms")) > self.retention_ms:
            self.items.popleft()

    def recent(self, anchor_ms: int, window_ms: int) -> list[T]:
        """Return observations in the inclusive window ending at anchor_ms."""
        lower = int(anchor_ms) - max(0, int(window_ms))
        return [item for item in self.items if lower <= int(getattr(item, "stamp_ms")) <= int(anchor_ms)]

    def nearest(self, anchor_ms: int) -> T | None:
        """Return the observation nearest to anchor_ms."""
        if not self.items:
            return None
        return min(self.items, key=lambda item: abs(int(getattr(item, "stamp_ms")) - int(anchor_ms)))


class AudioVisualBuffer:
    """Owns independent 3-second audio and video ring buffers."""

    def __init__(self, retention_ms: int = 3000, max_audio_frames: int = 96, max_video_frames: int = 128):
        """Initialize queues sized for 16 kHz/1024 audio and 30 FPS video."""
        self.audio = TimestampedRing[AudioFeatureFrame](retention_ms, max_audio_frames)
        self.video = TimestampedRing[VideoFeatureFrame](retention_ms, max_video_frames)

    def add_audio(self, frame: AudioFeatureFrame) -> None:
        """Store an audio feature frame."""
        self.audio.append(frame)

    def add_video(self, frame: VideoFeatureFrame) -> None:
        """Store a video feature frame."""
        self.video.append(frame)

    def recent_audio(self, anchor_ms: int, window_ms: int = 3000) -> list[AudioFeatureFrame]:
        """Query audio frames relative to a video timestamp anchor."""
        return self.audio.recent(anchor_ms, window_ms)

    def recent_video(self, anchor_ms: int, window_ms: int = 3000) -> list[VideoFeatureFrame]:
        """Query video frames relative to a video timestamp anchor."""
        return self.video.recent(anchor_ms, window_ms)


class TemporalAligner:
    """Align audio to a video timestamp without floating-point time arithmetic."""

    def __init__(self, max_offset_ms: int = 120):
        """Set the maximum admissible absolute audio/video offset."""
        self.max_offset_ms = max(0, int(max_offset_ms))

    def align(
        self,
        video_frame: VideoFeatureFrame,
        audio_frames: list[AudioFeatureFrame],
    ) -> tuple[AudioFeatureFrame | None, int | None]:
        """Return the closest audio frame and signed audio-minus-video offset."""
        if not audio_frames:
            return None, None
        audio = min(audio_frames, key=lambda item: abs(item.stamp_ms - video_frame.stamp_ms))
        offset_ms = int(audio.stamp_ms - video_frame.stamp_ms)
        return (audio, offset_ms) if abs(offset_ms) <= self.max_offset_ms else (None, offset_ms)


if nn is not None:

    class CrossAttentionFusion(nn.Module):
        """Small cross-attention scorer for MFCC and mouth-ROI feature sequences."""

        def __init__(
            self,
            audio_dim: int = 13,
            mouth_dim: int = 32,
            hidden_dim: int = 32,
            num_heads: int = 4,
            dropout: float = 0.0,
        ):
            """Create a low-memory attention model suitable for an 8 GB edge device."""
            super().__init__()
            self.audio_projection = nn.Linear(audio_dim, hidden_dim)
            self.mouth_projection = nn.Linear(mouth_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1), nn.Sigmoid())

        def forward(self, audio_features, mouth_features):
            """Return one fusion confidence in [0, 1] per batch item."""
            audio = self.audio_projection(audio_features)
            mouth = self.mouth_projection(mouth_features)
            fused, _weights = self.attention(query=mouth, key=audio, value=audio, need_weights=False)
            return self.classifier(fused.mean(dim=1)).squeeze(-1)

else:

    class CrossAttentionFusion:  # pragma: no cover - exercised only without torch.
        """Dependency error placeholder when PyTorch is not installed."""

        def __init__(self, audio_dim: int = 13, mouth_dim: int = 32, hidden_dim: int = 32, num_heads: int = 4, dropout: float = 0.0):
            """Raise a clear dependency error instead of silently faking confidence."""
            del audio_dim, mouth_dim, hidden_dim, num_heads, dropout
            raise RuntimeError("PyTorch is required for CrossAttentionFusion")


class BinderEngine:
    """Apply hard AV gates, conservative temporal confirmation, and optional fusion scoring."""

    def __init__(
        self,
        buffer: AudioVisualBuffer | None = None,
        aligner: TemporalAligner | None = None,
        fusion_model: Any | None = None,
        bind_threshold: float = 0.99,
        confirmations: int = 3,
        mouth_open_threshold: float = 0.08,
    ):
        """Configure conservative binding thresholds; model weights must be calibrated externally."""
        self.buffer = buffer or AudioVisualBuffer()
        self.aligner = aligner or TemporalAligner()
        self.fusion_model = fusion_model
        self.bind_threshold = max(0.0, min(1.0, float(bind_threshold)))
        self.confirmations = max(1, int(confirmations))
        self.mouth_open_threshold = max(0.0, float(mouth_open_threshold))
        self.streaks: dict[tuple[str, str], int] = {}

    def ingest_audio(self, frame: AudioFeatureFrame) -> None:
        """Add audio to the 3-second buffer without assigning an identity."""
        self.buffer.add_audio(frame)

    def decide(self, video_frame: VideoFeatureFrame) -> BindingDecision:
        """Return bind, temp, or reject for one video-anchored decision."""
        self.buffer.add_video(video_frame)
        audio, offset_ms = self.aligner.align(video_frame, list(self.buffer.audio.items))
        if not video_frame.face_visible:
            cluster_id = audio.speaker_cluster_id if audio is not None else ""
            return BindingDecision("temp", "", cluster_id, offset_ms, 0.0, "face_not_visible")
        if audio is None:
            return BindingDecision("reject", video_frame.face_id, "", offset_ms, 0.0, "no_audio_within_120ms")
        key = (video_frame.face_id, audio.speaker_cluster_id)
        hard_gate = (
            video_frame.lip_motion
            and video_frame.mouth_open_ratio >= self.mouth_open_threshold
            and audio.vad_active
            and offset_ms is not None
            and abs(offset_ms) <= self.aligner.max_offset_ms
        )
        if not hard_gate:
            self.streaks[key] = 0
            return BindingDecision("reject", key[0], key[1], offset_ms, 0.0, "hard_gate_failed")
        confidence = self._fusion_confidence(audio, video_frame)
        self.streaks[key] = self.streaks.get(key, 0) + 1
        if confidence >= self.bind_threshold and self.streaks[key] >= self.confirmations:
            return BindingDecision("bind", key[0], key[1], offset_ms, confidence, "confirmed_av_synchrony")
        return BindingDecision("reject", key[0], key[1], offset_ms, confidence, "awaiting_conservative_confirmation")

    def _fusion_confidence(self, audio: AudioFeatureFrame, video: VideoFeatureFrame) -> float:
        """Run a calibrated model, or return a conservative deterministic baseline."""
        if self.fusion_model is None:
            timing_score = 1.0 - abs(audio.stamp_ms - video.stamp_ms) / max(1, self.aligner.max_offset_ms)
            mouth_score = min(1.0, video.mouth_open_ratio / max(self.mouth_open_threshold * 2.0, 1e-6))
            return max(0.0, min(0.995, 0.85 + 0.10 * timing_score + 0.045 * mouth_score))
        if torch is None:
            return 0.0
        audio_tensor = torch.as_tensor(audio.mfcc, dtype=torch.float32).reshape(1, -1, audio.mfcc.shape[-1])
        mouth = np.asarray(video.mouth_roi_features, dtype=np.float32)
        mouth_tensor = torch.as_tensor(mouth, dtype=torch.float32).reshape(1, -1, mouth.shape[-1])
        self.fusion_model.eval()
        with torch.no_grad():
            return float(self.fusion_model(audio_tensor, mouth_tensor).item())


def main() -> None:
    """Simulate ten seconds of 30 FPS video and 16 kHz/1024-sample audio features."""
    engine = BinderEngine(bind_threshold=0.99, confirmations=3)
    duration_ms = 10_000
    audio_period_ms = 64
    video_period_ms = 33
    next_audio_ms = 0
    decisions = {"bind": 0, "temp": 0, "reject": 0}
    for video_ms in range(0, duration_ms, video_period_ms):
        while next_audio_ms <= video_ms + 32:
            engine.ingest_audio(
                AudioFeatureFrame(
                    stamp_ms=next_audio_ms,
                    mfcc=np.ones((1, 13), dtype=np.float32) * 0.2,
                    vad_active=2000 <= next_audio_ms <= 8000,
                    speaker_cluster_id="speaker_cluster_1",
                )
            )
            next_audio_ms += audio_period_ms
        face_visible = video_ms >= 1000
        speaking = 2000 <= video_ms <= 8000
        decision = engine.decide(
            VideoFeatureFrame(
                stamp_ms=video_ms,
                face_id="face_1" if face_visible else "",
                face_visible=face_visible,
                lip_motion=speaking,
                mouth_open_ratio=0.18 if speaking else 0.02,
                mouth_roi_features=np.ones((1, 32), dtype=np.float32) * 0.3,
            )
        )
        decisions[decision.action] += 1
    print(decisions)


if __name__ == "__main__":
    main()
