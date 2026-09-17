import unittest

from robot_attention_perception.async_fusion import AsyncPerceptionBuffer
from robot_attention_perception.types import AcousticTrack, VisionPerson


class AsyncPerceptionBufferTest(unittest.TestCase):
    def test_mixed_rate_audio_uses_recent_visual_context(self):
        buffer = AsyncPerceptionBuffer.with_algorithm("ros4hri_native")
        buffer.ingest_vision(
            [
                VisionPerson(
                    person_id="owner",
                    stamp_ms=1000,
                    voice_id="voice_1",
                    role="owner",
                    azimuth_deg=4.0,
                    face_visible=True,
                    gaze_score=0.85,
                    body_facing_score=0.8,
                    engagement_status="engaged",
                    proxemic_space="social",
                )
            ]
        )
        buffer.ingest_acoustic(
            [
                AcousticTrack(
                    track_id="voice_1",
                    stamp_ms=1120,
                    voice_activity=True,
                    speech_probability=0.9,
                    clarity=0.84,
                    azimuth_deg=6.0,
                )
            ]
        )

        state = buffer.update(1130)

        self.assertIsNotNone(state)
        self.assertEqual(state.person_id, "owner")
        self.assertTrue(state.listen)
        self.assertIn("async_vision_cache", state.reasons)

    def test_stale_audio_is_pruned_before_attention(self):
        buffer = AsyncPerceptionBuffer.with_algorithm("ros4hri_native")
        buffer.ingest_acoustic(
            [
                AcousticTrack(
                    track_id="old_voice",
                    stamp_ms=1000,
                    voice_activity=True,
                    speech_probability=0.95,
                    clarity=0.9,
                    azimuth_deg=0.0,
                )
            ]
        )

        state = buffer.update(1300)

        self.assertIsNotNone(state)
        self.assertEqual(state.target_id, "none")

    def test_rate_limit_returns_none_when_called_too_fast(self):
        buffer = AsyncPerceptionBuffer.with_algorithm("ros4hri_native")
        buffer.update(1000)

        self.assertIsNone(buffer.update(1005))


if __name__ == "__main__":
    unittest.main()
