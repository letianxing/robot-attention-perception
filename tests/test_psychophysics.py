import unittest

from robot_attention_perception.psychophysics import (
    cocktail_attention_benchmark,
    double_flash_curve,
    mcgurk_consistency_benchmark,
)


class PsychophysicsBenchmarkTest(unittest.TestCase):
    def test_double_flash_binding_declines_with_interval(self):
        curve = double_flash_curve([25, 50, 100, 200])
        strengths = [item["fission_strength"] for item in curve]
        self.assertEqual(strengths, sorted(strengths, reverse=True))

    def test_congruent_av_speech_scores_above_conflict(self):
        result = mcgurk_consistency_benchmark()
        self.assertGreater(result["consistency_gap"], 0.3)

    def test_cocktail_attention_locks_and_switches_quickly(self):
        result = cocktail_attention_benchmark()
        self.assertGreaterEqual(result["target_lock_rate"], 0.95)
        self.assertLessEqual(result["switch_delay_ms"], 100)


if __name__ == "__main__":
    unittest.main()
