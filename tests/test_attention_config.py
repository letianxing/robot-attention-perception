"""Console control over which systems feed attention and which algorithm ranks them."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from robot_attention_perception.attention_plugin import AlgorithmRegistry
from robot_attention_perception.attention_sources import AttentionSources
from robot_attention_perception.live_runtime import LivePerceptionRuntime


class ConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.registry = AlgorithmRegistry()
        self.addCleanup(self.registry.close)
        self.host = SimpleNamespace(
            attention_sources=AttentionSources('s', enabled=['audio_visual']),
            interaction=SimpleNamespace(competition=SimpleNamespace(registry=self.registry)),
            attention_config_path=Path(directory.name) / 'attention-config.json',
            attention_choice_source='default')
        self.host.attention_config = lambda: LivePerceptionRuntime.attention_config(self.host)

    def config(self):
        return LivePerceptionRuntime.attention_config(self.host)

    def apply(self, payload):
        return LivePerceptionRuntime.set_attention_config(self.host, payload)

    def test_every_source_is_listed_with_its_state_and_meaning(self):
        sources = {item['id']: item for item in self.config()['sources']}
        self.assertEqual(set(sources), {'audio_visual', 'memory_context', 'linguistic_context',
                                        'cross_session_memory', 'internal_state'})
        self.assertTrue(sources['audio_visual']['enabled'])
        self.assertFalse(sources['memory_context']['enabled'])
        self.assertTrue(all(item['display_name'] and item['description'] for item in sources.values()))
        self.assertIn('不是关闭摄像头', self.config()['note'])

    def test_a_choice_is_applied_and_survives_a_restart(self):
        result = self.apply({'sources': ['audio_visual', 'memory_context', 'linguistic_context'],
                             'algorithm': 'builtin_audio_visual_v1'})
        self.assertEqual(result['algorithm']['active'], 'builtin_audio_visual_v1')
        self.assertEqual([item['id'] for item in result['sources'] if item['enabled']],
                         ['audio_visual', 'memory_context', 'linguistic_context'])

        saved = json.loads(self.host.attention_config_path.read_text('utf-8'))
        self.assertEqual(saved['algorithm'], 'builtin_audio_visual_v1')
        self.assertEqual(LivePerceptionRuntime._load_attention_config(self.host), saved)

    def test_a_checkbox_map_is_accepted_and_sources_can_all_be_unplugged(self):
        self.assertEqual([item['id'] for item in self.apply({'sources': {'audio_visual': True,
                                                                        'memory_context': False}})['sources']
                          if item['enabled']], ['audio_visual'])
        self.assertEqual([item['id'] for item in self.apply({'sources': []})['sources'] if item['enabled']], [])

    def test_an_unknown_algorithm_is_reported_and_the_running_one_is_kept(self):
        before = self.config()['algorithm']['active']
        result = self.apply({'algorithm': 'no_such_algorithm'})
        self.assertEqual(result['algorithm']['active'], before)
        self.assertTrue(any('no_such_algorithm' in message for message in result['messages']))

    def test_a_damaged_config_file_falls_back_instead_of_crashing(self):
        self.host.attention_config_path.write_text('{ not json', 'utf-8')
        self.assertEqual(LivePerceptionRuntime._load_attention_config(self.host), {})
        self.host.attention_config_path.write_text('{"sources": 5, "algorithm": 7}', 'utf-8')
        self.assertEqual(LivePerceptionRuntime._load_attention_config(self.host),
                         {'sources': None, 'algorithm': '7', 'addressee_weights': {}})

    def test_the_console_reports_where_the_choice_came_from(self):
        self.assertEqual(self.config()['algorithm']['chosen_by'], 'default')
        self.apply({'algorithm': 'builtin_audio_visual_v1'})
        self.assertEqual(self.config()['algorithm']['chosen_by'], 'saved_console_choice')

    def test_switching_gives_a_fresh_algorithm_not_its_earlier_state(self):
        """An instance that last ran minutes ago still holds that much
        habituation and engagement; switching in must not compare now against a
        situation that no longer exists."""
        registry = self.registry
        if not registry.select('av_memory_language_v1'):
            self.skipTest('plugins not built')
        person = {'candidate_id': 'visual:owner', 'modality': 'visual', 'person_id': 'owner', 'ttl_ms': 600,
                  'salience': .85, 'goal': .8, 'observability': .9, 'identity_confidence': .8,
                  'gaze': .9, 'body_facing': .9, 'memory_participant': 1., 'memory_recency': 1.}
        sources = {'audio_visual': True, 'memory_context': True, 'linguistic_context': True}
        for stamp in range(1000, 4000, 50):
            registry.update(stamp, [dict(person, stamp_ms=stamp)], sources)
        self.assertEqual(registry.update(4000, [dict(person, stamp_ms=4000)], sources)['engagement']['state'],
                         'INVITED')

        registry.select('builtin_audio_visual_v1')
        registry.select('av_memory_language_v1')
        fresh = registry.update(5000, [dict(person, stamp_ms=5000)], sources)
        self.assertNotEqual(fresh['engagement']['state'], 'INVITED')
        self.assertEqual(fresh['candidates']['visual:owner']['a'], 0.0)

    def test_each_algorithm_declares_its_sources_and_calibration_state(self):
        for algorithm in self.config()['algorithm']['algorithms']:
            self.assertTrue(algorithm['display_name'])
            self.assertIn('audio_visual', algorithm['required_sources'])
            self.assertFalse(algorithm['calibrated'])
            self.assertTrue(algorithm['limits'])


if __name__ == '__main__':
    unittest.main()


def person_snapshots(person="天行", end=12000, gaze=.9, count=20, others=(), lips=True, other_gaze=.1):
    """What the camera reported while the sentence was being said."""
    from robot_attention_perception.types import VisionPerson
    rows = []
    for stamp in range(end - count * 100, end, 100):
        people = [VisionPerson(person, stamp, face_visible=True, face_confidence=.9, gaze_score=gaze,
                               body_facing_score=.8, identity_confidence=.8, role='known',
                               lip_motion=lips)]
        people += [VisionPerson(name, stamp, face_visible=True, face_confidence=.9, gaze_score=other_gaze,
                                body_facing_score=.5, identity_confidence=.7, role='known') for name in others]
        rows.append((stamp, people))
    return rows


class AddresseeWeightTests(unittest.TestCase):
    """What counts as being spoken to is one weighted sum, and it is tunable.

    It used to be five hard-coded branches, each granting on its own and
    recording only the one that fired — so "why did it not answer" could not be
    answered from the trace, and no threshold could be adjusted without editing
    code.
    """

    def gates(self, person="天行", end=12000, phase="VISUAL_FOCUS"):
        return [(stamp, {"person_id": person, "interaction_phase": phase, "reasons": [],
                         "listen": True, "addressed_to_robot": False, "confidence": .9,
                         "target_kind": "person", "source_track_id": None})
                for stamp in range(end - 2400, end, 100)]

    def transcript(self, text="今天天气怎么样", end=12000):
        return {"track_id": "t1", "text": text, "started_ms": end - 800, "ended_ms": end,
                "is_final": True, "speaker": {"speaker_id": "unknown", "similarity": 0.0}}

    def test_every_term_that_fired_is_reported_with_its_contribution(self):
        from robot_attention_perception.interaction_fusion import finalize_gate
        gate = finalize_gate(self.transcript("小圆，今天天气怎么样"), self.gates(), 12000,
                             people_history=person_snapshots())

        detail = gate["addressee_detail"]
        self.assertIn("directed_call", detail["terms"])
        self.assertIn("gaze_while_speaking", detail["terms"])
        self.assertIn("lip_motion_while_looking", detail["terms"])
        self.assertGreaterEqual(detail["score"], detail["threshold"])

    def test_a_refusal_says_what_it_had_and_what_it_needed(self):
        from robot_attention_perception.interaction_fusion import finalize_gate
        gate = finalize_gate(self.transcript(), [], 12000)

        self.assertFalse(gate["addressed_to_robot"])
        self.assertEqual(gate["addressee_detail"]["terms"], {})
        self.assertEqual(gate["addressee_detail"]["decision"], "ignore")

    def test_raising_the_threshold_refuses_what_it_used_to_allow(self):
        from robot_attention_perception.interaction_fusion import finalize_gate
        people = person_snapshots()
        allowed = finalize_gate(self.transcript(), self.gates(), 12000, people_history=people)
        refused = finalize_gate(self.transcript(), self.gates(), 12000,
                                weights={"answer_threshold": 1.5, "threshold": 1.2}, people_history=people)

        self.assertTrue(allowed["addressed_to_robot"])
        self.assertFalse(refused["addressed_to_robot"])

    def test_weights_survive_a_restart(self):
        import json, tempfile, types
        from pathlib import Path
        from robot_attention_perception.live_runtime import LivePerceptionRuntime
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "attention-config.json"
            path.write_text(json.dumps({"sources": ["audio_visual"], "algorithm": "v1",
                                        "addressee_weights": {"attended_person": .2}}), "utf-8")
            host = types.SimpleNamespace(attention_config_path=path)

            saved = LivePerceptionRuntime._load_attention_config(host)

            self.assertEqual(saved["addressee_weights"], {"attended_person": .2})


class CrowdedRoomTests(unittest.TestCase):
    """A table of people, one of whom turns to the robot and speaks."""

    def gates(self, end=12000):
        rows = []
        for stamp in range(end - 2000, end, 100):
            # attention alternates between the two people present
            person = '天行' if (stamp // 100) % 2 else '二丫'
            rows.append((stamp, {'person_id': person, 'interaction_phase': 'VISUAL_FOCUS',
                                 'gaze_to_robot': person == '天行', 'reasons': [], 'listen': True,
                                 'addressed_to_robot': False, 'confidence': .8,
                                 'target_kind': 'person', 'source_track_id': None}))
        return rows

    def transcript_in_a_crowd(self):
        return {'track_id': 't1', 'text': '这个方案行吗', 'started_ms': 11200, 'ended_ms': 12000,
                'is_final': True, 'speaker': {'speaker_id': 'unknown', 'similarity': 0.0}}

    def test_the_one_looking_and_talking_is_answered_even_in_a_crowd(self):
        """How many people are present is not the question. One of them turned
        to the robot and spoke; the others are looking elsewhere."""
        from robot_attention_perception.interaction_fusion import finalize_gate

        gate = finalize_gate(self.transcript_in_a_crowd(), self.gates(), 12000,
                             people_history=person_snapshots(others=('二丫', '小林')))

        self.assertTrue(gate['addressed_to_robot'])
        self.assertEqual(gate['person_id'], '天行')
        self.assertIn('lip_motion_while_looking', gate['addressee_detail']['terms'])

    def test_a_glance_with_no_mouth_moving_is_only_remembered(self):
        """Looking over while somebody else talks is the ambiguous case; the
        sentence is written down against them rather than answered."""
        from robot_attention_perception.interaction_fusion import finalize_gate

        gate = finalize_gate(self.transcript_in_a_crowd(), self.gates(), 12000,
                             people_history=person_snapshots(others=('二丫',), lips=False))

        self.assertFalse(gate['addressed_to_robot'])
        self.assertTrue(gate['held_for_attention'])
        self.assertEqual(gate['addressee_detail']['decision'], 'hold')

    def test_two_people_both_looking_is_ambiguous_and_held(self):
        """Competition, not a coin toss: when two of them face the robot the
        dominance collapses and the score says so."""
        from robot_attention_perception.interaction_fusion import finalize_gate

        gate = finalize_gate(self.transcript_in_a_crowd(), self.gates(), 12000,
                             people_history=person_snapshots(others=('二丫',), other_gaze=.9))

        self.assertFalse(gate['addressed_to_robot'])

    def test_nobody_looking_at_the_robot_is_still_refused(self):
        from robot_attention_perception.interaction_fusion import finalize_gate
        rows = [(stamp, dict(gate, gaze_to_robot=False)) for stamp, gate in self.gates()]
        transcript = {'track_id': 't1', 'text': '这个方案行吗', 'started_ms': 11200, 'ended_ms': 12000,
                      'is_final': True, 'speaker': {'speaker_id': 'unknown', 'similarity': 0.0}}

        self.assertFalse(finalize_gate(transcript, rows, 12000)['addressed_to_robot'])
