import ctypes
import unittest
from pathlib import Path

from robot_attention_perception.attention_plugin import (ABI_VERSION, AlgorithmRegistry, CandidateIn, CandidateOut,
                                                         FrameIn, FrameOut, KeyValue, SharedLibraryAlgorithm,
                                                         default_plugin_dir, reasons_from_mask)

PLUGIN = default_plugin_dir() / 'libattention_av_memory_language_v1.so'
SOURCES = {'audio_visual': True, 'memory_context': True, 'linguistic_context': True, 'internal_state': False}


def person(stamp, **extra):
    base = dict(candidate_id='visual:owner', modality='visual', person_id='owner', stamp_ms=stamp, ttl_ms=600,
                salience=.85, goal=.8, observability=.9, gaze=.9, body_facing=.9, identity_confidence=.8)
    base.update(extra)
    return base


@unittest.skipUnless(PLUGIN.exists(), f'{PLUGIN} 未编译，先运行 cmake --build build')
class SharedLibraryTests(unittest.TestCase):
    """The .so is the contract; a layout drift must fail loudly, not silently."""

    def test_struct_layout_is_verified_against_the_library(self):
        algorithm = SharedLibraryAlgorithm(PLUGIN)
        self.addCleanup(algorithm.close)
        for index, structure in enumerate((FrameIn, CandidateIn, FrameOut, CandidateOut, KeyValue)):
            self.assertEqual(ctypes.sizeof(structure), algorithm.library.attention_plugin_struct_size(index))
        self.assertEqual(algorithm.manifest['abi'], ABI_VERSION)
        self.assertFalse(algorithm.manifest['calibrated'])
        self.assertIn('audio_visual', algorithm.manifest['required_sources'])

    def test_a_repeated_stamp_is_rejected_rather_than_replayed(self):
        algorithm = SharedLibraryAlgorithm(PLUGIN)
        self.addCleanup(algorithm.close)
        self.assertIsNotNone(algorithm.update(1000, [person(1000)], SOURCES))
        self.assertIsNone(algorithm.update(1000, [person(1000)], SOURCES))

    def test_participant_gaze_reaches_invitation_and_memory_ablation_removes_it(self):
        algorithm = SharedLibraryAlgorithm(PLUGIN)
        self.addCleanup(algorithm.close)
        result = None
        for stamp in range(1000, 4000, 50):
            result = algorithm.update(stamp, [person(stamp, memory_participant=1., memory_recency=1.)], SOURCES)
        self.assertEqual(result['engagement']['state'], 'INVITED')
        self.assertIn('silent_invitation', result['engagement']['reasons'])

        ablated = SharedLibraryAlgorithm(PLUGIN)
        self.addCleanup(ablated.close)
        without_memory = dict(SOURCES, memory_context=False)
        for stamp in range(1000, 4000, 50):
            result = ablated.update(stamp, [person(stamp, memory_participant=1., memory_recency=1.)], without_memory)
        self.assertNotEqual(result['engagement']['state'], 'INVITED')
        self.assertNotIn('memory_participant', result['engagement']['reasons'])

    def test_memory_is_worth_no_more_than_the_recognition_behind_it(self):
        """Face/voiceprint accuracy reaches attention exactly here."""
        algorithm = SharedLibraryAlgorithm(PLUGIN)
        self.addCleanup(algorithm.close)
        result = None
        for stamp in range(1000, 4000, 50):
            result = algorithm.update(stamp, [person(stamp, identity_confidence=.05,
                                                     memory_participant=1., memory_recency=1.)], SOURCES)
        self.assertNotEqual(result['engagement']['state'], 'INVITED')
        self.assertNotIn('memory_participant', result['engagement']['reasons'])

    def test_cross_session_familiarity_arrives_through_the_abi_extension(self):
        algorithm = SharedLibraryAlgorithm(PLUGIN)
        self.addCleanup(algorithm.close)
        result = None
        for stamp in range(1000, 5000, 50):
            result = algorithm.update(stamp, [person(stamp, extra={'memory_familiarity': 1.})], SOURCES)
        self.assertEqual(result['engagement']['state'], 'INVITED')
        self.assertIn('familiar_across_sessions', result['engagement']['reasons'])
        # Familiarity is slower than taking part in the conversation happening now.
        stranger = SharedLibraryAlgorithm(PLUGIN)
        self.addCleanup(stranger.close)
        for stamp in range(1000, 5000, 50):
            result = stranger.update(stamp, [person(stamp)], SOURCES)
        self.assertNotEqual(result['engagement']['state'], 'INVITED')

    def test_naming_another_human_does_not_address_the_robot(self):
        algorithm = SharedLibraryAlgorithm(PLUGIN)
        self.addCleanup(algorithm.close)
        result = None
        for stamp in range(1000, 2000, 50):
            result = algorithm.update(stamp, [dict(candidate_id='audio:t1', modality='audio', person_id='guest',
                                                   stamp_ms=stamp, ttl_ms=350, salience=.8, voice_activity=1.,
                                                   memory_participant=1., memory_recency=1.,
                                                   lang_addresses_other=1., event_id='utt-3')], SOURCES)
        self.assertNotIn(result['engagement']['state'], {'ENGAGED', 'EXPECTED_ANSWER'})


class RegistryTests(unittest.TestCase):
    def test_builtin_is_always_available_and_declares_its_missing_layer(self):
        registry = AlgorithmRegistry(plugin_dir=Path('/nonexistent-attention-plugins'))
        self.addCleanup(registry.close)
        status = registry.status()
        self.assertEqual(status['active'], 'builtin_audio_visual_v1')
        self.assertIn('plugin_dir', status['load_errors'])
        result = registry.update(1000, [person(1000)], SOURCES)
        self.assertEqual(result['engagement']['unavailable_reason'], 'selected_algorithm_has_no_engagement_layer')

    def test_selection_is_reversible_and_an_unknown_id_is_refused(self):
        registry = AlgorithmRegistry()
        self.addCleanup(registry.close)
        self.assertFalse(registry.select('does_not_exist'))
        self.assertTrue(registry.select('builtin_audio_visual_v1'))
        self.assertEqual(registry.status()['active'], 'builtin_audio_visual_v1')
        if PLUGIN.exists():
            self.assertTrue(registry.select('av_memory_language_v1'))
            self.assertEqual(registry.status()['active'], 'av_memory_language_v1')

    def test_reason_bits_decode_to_named_evidence(self):
        self.assertEqual(reasons_from_mask(0), [])
        self.assertEqual(reasons_from_mask((1 << 2) | (1 << 4)), ['directed_call', 'expected_answer'])


if __name__ == '__main__':
    unittest.main()
