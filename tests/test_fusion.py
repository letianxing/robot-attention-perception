import unittest

from robot_attention_perception.fusion import AttentionFusion
from robot_attention_perception.scenario import load_scenario
from robot_attention_perception.types import (
    AcousticTrack,
    PerceptionFrame,
    RobotState,
    TranscriptRecord,
    VisionPerson,
)


class AttentionFusionTest(unittest.TestCase):
    def test_default_algorithm_is_ros4hri_native(self):
        fusion = AttentionFusion()

        self.assertEqual(fusion.config.algorithm, "ros4hri_native")
        self.assertIn("ros4hri_native", AttentionFusion.available_algorithms())
        self.assertIn("weighted_audio_visual", AttentionFusion.available_algorithms())

    def test_prefers_gaze_matched_speaker_over_side_talker(self):
        frames = load_scenario("fixtures/cocktail_effect_three_people.json")
        state = AttentionFusion().update(frames[0])

        self.assertEqual(state.person_id, "owner")
        self.assertEqual(state.source_track_id, "a_owner")
        self.assertTrue(state.listen)
        self.assertTrue(state.addressed_to_robot)
        self.assertEqual(len(state.transcripts), 2)

    def test_robot_echo_does_not_win_during_user_barge_in(self):
        frames = load_scenario("fixtures/cocktail_effect_three_people.json")
        fusion = AttentionFusion()
        fusion.update(frames[0])
        state = fusion.update(frames[1])

        self.assertEqual(state.source_track_id, "a_owner")
        self.assertEqual(state.person_id, "owner")
        self.assertNotEqual(state.source_track_id, "robot_echo")

    def test_high_confidence_robot_echo_is_rejected(self):
        fusion = AttentionFusion()
        fusion.update(
            PerceptionFrame(
                stamp_ms=1000,
                acoustic_tracks=[
                    AcousticTrack(
                        track_id="owner_voice",
                        stamp_ms=1000,
                        voice_activity=True,
                        speech_probability=0.9,
                        clarity=0.85,
                        azimuth_deg=3.0,
                    )
                ],
                vision_people=[
                    VisionPerson(
                        person_id="owner",
                        stamp_ms=1000,
                        voice_id="owner_voice",
                        role="owner",
                        azimuth_deg=4.0,
                        face_visible=True,
                        gaze_score=0.8,
                        body_facing_score=0.8,
                        engagement_status="engaged",
                        proxemic_space="social",
                    )
                ],
            )
        )

        state = fusion.update(
            PerceptionFrame(
                stamp_ms=1200,
                acoustic_tracks=[
                    AcousticTrack(
                        track_id="robot_echo",
                        stamp_ms=1200,
                        voice_activity=True,
                        speech_probability=0.96,
                        clarity=0.9,
                        azimuth_deg=0.0,
                        self_echo_probability=0.95,
                    )
                ],
                vision_people=[
                    VisionPerson(
                        person_id="owner",
                        stamp_ms=1200,
                        role="owner",
                        azimuth_deg=4.0,
                        face_visible=True,
                        gaze_score=0.8,
                        body_facing_score=0.8,
                        engagement_status="engaged",
                        proxemic_space="social",
                    )
                ],
                robot=RobotState(stamp_ms=1200, speaking=True),
            )
        )

        self.assertEqual(state.target_id, "none")
        self.assertFalse(state.listen)

    def test_audio_only_array_can_select_sound_source(self):
        frame = PerceptionFrame(
            stamp_ms=500,
            acoustic_tracks=[
                AcousticTrack(
                    track_id="array_track_1",
                    stamp_ms=500,
                    voice_activity=True,
                    speech_probability=0.88,
                    clarity=0.8,
                    azimuth_deg=-30.0,
                    transcript=TranscriptRecord(
                        track_id="array_track_1",
                        text="小圆过来一下",
                        is_final=True,
                        language="zh-CN",
                        clarity=0.8,
                    ),
                )
            ],
        )

        state = AttentionFusion().update(frame)

        self.assertEqual(state.target_kind, "sound_source")
        self.assertEqual(state.azimuth_deg, -30.0)
        self.assertTrue(state.listen)
        self.assertTrue(state.addressed_to_robot)

    def test_no_voice_activity_keeps_transcript_log_but_no_attention(self):
        frame = PerceptionFrame(
            stamp_ms=100,
            acoustic_tracks=[
                AcousticTrack(
                    track_id="late_asr",
                    stamp_ms=100,
                    voice_activity=False,
                    speech_probability=0.0,
                    clarity=0.4,
                    transcript=TranscriptRecord(
                        track_id="late_asr",
                        text="之前的文字",
                        is_final=True,
                    ),
                )
            ],
            vision_people=[
                VisionPerson(
                    person_id="owner",
                    stamp_ms=100,
                    role="owner",
                    face_visible=True,
                    gaze_score=0.8,
                    body_facing_score=0.8,
                )
            ],
            robot=RobotState(stamp_ms=100, speaking=False),
        )

        state = AttentionFusion().update(frame)

        self.assertEqual(state.target_id, "none")
        self.assertEqual(len(state.transcripts), 1)

    def test_ros4hri_voice_id_association_overrides_bearing_ambiguity(self):
        frame = PerceptionFrame(
            stamp_ms=1000,
            acoustic_tracks=[
                AcousticTrack(
                    track_id="voice_a",
                    stamp_ms=1000,
                    voice_activity=True,
                    speech_probability=0.85,
                    clarity=0.8,
                    azimuth_deg=8.0,
                )
            ],
            vision_people=[
                VisionPerson(
                    person_id="left_person",
                    stamp_ms=1000,
                    voice_id="voice_other",
                    azimuth_deg=5.0,
                    face_visible=True,
                    gaze_score=0.2,
                    body_facing_score=0.2,
                ),
                VisionPerson(
                    person_id="matched_voice_person",
                    stamp_ms=1000,
                    voice_id="voice_a",
                    azimuth_deg=25.0,
                    face_visible=True,
                    gaze_score=0.7,
                    body_facing_score=0.75,
                    engagement_status="engaged",
                    proxemic_space="social",
                ),
            ],
        )

        state = AttentionFusion(algorithm="ros4hri_native").update(frame)

        self.assertEqual(state.person_id, "matched_voice_person")
        self.assertTrue(state.addressed_to_robot)
        self.assertIn("ros4hri_voice_match:matched_voice_person", state.reasons)


if __name__ == "__main__":
    unittest.main()
