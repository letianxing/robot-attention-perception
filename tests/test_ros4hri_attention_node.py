import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ros4hri_attention_node.py"
SPEC = importlib.util.spec_from_file_location("ros4hri_attention_node", SCRIPT)
NODE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(NODE)


class Ros4HriAttentionNodeParserTest(unittest.TestCase):
    def test_parse_voice_bridge_payload(self):
        tracks = NODE.parse_acoustic_payload(
            json.dumps({"engineering_track": {"track_id": "voice_1", "voice_activity": True, "speech_probability": 0.9, "clarity": 0.8}}),
            1000,
        )

        self.assertEqual(tracks[0].track_id, "voice_1")
        self.assertTrue(tracks[0].voice_activity)

    def test_parse_people_list_payload(self):
        people = NODE.parse_vision_payload(
            json.dumps([{"person_id": "person_1", "face_visible": True, "gaze_score": 0.7}]),
            1000,
        )

        self.assertEqual(people[0].person_id, "person_1")
        self.assertTrue(people[0].face_visible)

    def test_parse_robot_state_payload(self):
        state = NODE.parse_robot_state(json.dumps({"speaking": True, "playback_rms_db": -16.0}), 1000)

        self.assertTrue(state.speaking)
        self.assertEqual(state.playback_rms_db, -16.0)


if __name__ == "__main__":
    unittest.main()
