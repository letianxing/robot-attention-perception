import unittest

from robot_attention_perception.causal_inference import BayesianCausalInference


class CausalInferenceTest(unittest.TestCase):
    def test_aligned_reliable_cues_are_integrated(self):
        result = BayesianCausalInference().infer(10.0, 12.0, 1000, 1030, 0.9, 0.9)

        self.assertTrue(result.should_associate)
        self.assertGreater(result.common_source_probability, 0.8)
        self.assertGreater(result.integrated_azimuth_deg, 10.0)

    def test_spatial_temporal_conflict_keeps_modalities_separate(self):
        result = BayesianCausalInference().infer(-70.0, 70.0, 1000, 1400, 0.9, 0.9)

        self.assertFalse(result.should_associate)
        self.assertLess(result.common_source_probability, 0.1)
        self.assertLess(result.model_averaged_azimuth_deg, -60.0)

    def test_single_channel_audio_can_associate_one_fresh_visual_person(self):
        result = BayesianCausalInference().infer(None, 5.0, 1000, 1020, 0.8, 0.9)

        self.assertTrue(result.should_associate)


if __name__ == "__main__":
    unittest.main()
