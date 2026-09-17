import unittest

from robot_attention_perception.live_runtime import (
    acoustic_tracks_from_payload,
    gate_allows_transcript,
    vision_people_from_payload,
)


class LiveRuntimeMappingTest(unittest.TestCase):
    def test_maps_native_vision_dashboard_payload(self):
        people = vision_people_from_payload(
            {
                "state": {
                    "stamp": 10.0,
                    "people": [
                        {
                            "person_id": "p1",
                            "face_id": "f1",
                            "body_id": "b1",
                            "has_azimuth": True,
                            "azimuth_deg": 12.0,
                            "face_visible": True,
                            "gaze_score": 0.8,
                            "body_facing_score": 0.7,
                            "engagement_status": "engaged",
                        }
                    ],
                }
            },
            12000,
        )

        self.assertEqual(len(people), 1)
        self.assertEqual(people[0].stamp_ms, 10000)
        self.assertEqual(people[0].azimuth_deg, 12.0)
        self.assertTrue(people[0].face_visible)

    def test_maps_native_voice_dashboard_payload(self):
        tracks = acoustic_tracks_from_payload(
            {
                "last_track": {
                    "track_id": "voice_mono",
                    "stamp_ms": 1234,
                    "voice_activity": True,
                    "speech_probability": 0.9,
                    "clarity": 0.7,
                    "azimuth_deg": None,
                }
            }
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, "voice_mono")
        self.assertIsNone(tracks[0].azimuth_deg)

    def test_silence_maps_to_no_active_tracks(self):
        self.assertEqual(acoustic_tracks_from_payload({"last_track": None}), [])

    def test_final_transcript_can_use_recent_matching_attention_gate(self):
        remembered = {
            "listen": True,
            "addressed_to_robot": True,
            "source_track_id": "voice_1",
            "remembered_ms": 1000,
        }

        self.assertTrue(
            gate_allows_transcript({}, remembered, {"track_id": "voice_1"}, 3200)
        )
        self.assertFalse(
            gate_allows_transcript({}, remembered, {"track_id": "voice_2"}, 3200)
        )
        self.assertFalse(
            gate_allows_transcript({}, remembered, {"track_id": "voice_1"}, 3600)
        )


if __name__ == "__main__":
    unittest.main()
