import unittest

import numpy as np

from robot_attention_perception.av_binding import (
    AudioFeatureFrame,
    AudioVisualBuffer,
    BinderEngine,
    TemporalAligner,
    VideoFeatureFrame,
)


def audio(stamp_ms, active=True):
    return AudioFeatureFrame(
        stamp_ms=stamp_ms,
        mfcc=np.ones((1, 13), dtype=np.float32),
        vad_active=active,
        speaker_cluster_id="voice_1",
    )


def video(stamp_ms, visible=True, moving=True):
    return VideoFeatureFrame(
        stamp_ms=stamp_ms,
        face_id="face_1" if visible else "",
        face_visible=visible,
        lip_motion=moving,
        mouth_open_ratio=0.2 if moving else 0.01,
        mouth_roi_features=np.ones((1, 32), dtype=np.float32),
    )


class AudioVisualBindingTest(unittest.TestCase):
    def test_audio_buffer_keeps_only_retention_window(self):
        buffer = AudioVisualBuffer(retention_ms=3000)
        buffer.add_audio(audio(0))
        buffer.add_audio(audio(4000))

        self.assertEqual([item.stamp_ms for item in buffer.audio.items], [4000])

    def test_aligner_returns_signed_offset_and_rejects_out_of_window(self):
        aligner = TemporalAligner(max_offset_ms=120)

        matched, offset = aligner.align(video(1000), [audio(1064)])
        rejected, rejected_offset = aligner.align(video(1000), [audio(1201)])

        self.assertEqual(matched.stamp_ms, 1064)
        self.assertEqual(offset, 64)
        self.assertIsNone(rejected)
        self.assertEqual(rejected_offset, 201)

    def test_face_hidden_never_binds_identity(self):
        engine = BinderEngine()
        engine.ingest_audio(audio(1000))

        decision = engine.decide(video(1000, visible=False))

        self.assertEqual(decision.action, "temp")

    def test_requires_repeated_hard_gate_confirmations(self):
        engine = BinderEngine(confirmations=3, bind_threshold=0.99)
        actions = []
        for stamp in (1000, 1033, 1066):
            engine.ingest_audio(audio(stamp))
            actions.append(engine.decide(video(stamp)).action)

        self.assertEqual(actions, ["reject", "reject", "bind"])

    def test_lip_motion_is_mandatory_even_with_active_vad(self):
        engine = BinderEngine(confirmations=1)
        engine.ingest_audio(audio(1000))

        decision = engine.decide(video(1000, moving=False))

        self.assertEqual(decision.action, "reject")
        self.assertEqual(decision.reason, "hard_gate_failed")


if __name__ == "__main__":
    unittest.main()
