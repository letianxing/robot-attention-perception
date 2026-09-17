import time
import unittest
from unittest.mock import patch

from robot_attention_perception.dialogue import AcknowledgementBackend, SpeechToSpeechResponder


class DialogueTest(unittest.TestCase):
    def test_acknowledgement_backend_returns_short_reply(self):
        self.assertEqual(AcknowledgementBackend().reply("  你好   Reachy "), "我听到了。你好 Reachy")

    @patch("robot_attention_perception.dialogue.subprocess.run")
    def test_responder_deduplicates_and_emits_playback_state(self, run):
        playback = []
        responder = SpeechToSpeechResponder(on_playback=playback.append)
        try:
            self.assertTrue(responder.submit("1", "你好"))
            self.assertFalse(responder.submit("1", "你好"))
            deadline = time.monotonic() + 1.0
            while responder.last_turn is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(responder.last_turn)
            self.assertEqual(playback, [True, False])
            run.assert_called_once()
        finally:
            responder.close()


if __name__ == "__main__":
    unittest.main()
