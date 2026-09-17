"""Cross-session familiarity: a weak prior, never a claim about the current utterance."""
import unittest

from robot_attention_perception.attention_sources import AttentionSources
from robot_attention_perception.familiarity import FamiliarityIndex
from robot_attention_perception.interaction_fusion import InteractionFusion
from robot_attention_perception.types import VisionPerson


def hit(person, session, stamp):
    return {'record': {'entity': {'person_id': person}, 'scope': {'session_id': session}, 'observed_at_ms': stamp}}


def index(responses, session_id='today', **kwargs):
    calls = []

    def search(person_id):
        calls.append(person_id)
        result = responses.get(person_id)
        if isinstance(result, Exception):
            raise result
        return {'hits': result or []}

    built = FamiliarityIndex('http://memory', 'subject', 'robot', session_id, search=search, **kwargs)
    built.calls = calls
    return built


def drain(built):
    built.worker.shutdown(wait=True)
    from concurrent.futures import ThreadPoolExecutor
    built.worker = ThreadPoolExecutor(max_workers=1)


class IndexTests(unittest.TestCase):
    def test_distinct_prior_sessions_saturate_and_the_current_one_does_not_count(self):
        built = index({'owner': [hit('owner', 'mon', 1), hit('owner', 'tue', 2), hit('owner', 'mon', 3),
                                 hit('owner', 'today', 4)]})
        self.addCleanup(built.close)
        built.request(1000, ['owner'])
        drain(built)
        record = built.scores(2000)['owner']
        self.assertEqual(record['prior_sessions'], 2)
        self.assertEqual(record['prior_turns'], 3)
        self.assertAlmostEqual(record['familiarity'], 0.5)

        many = index({'owner': [hit('owner', f's{n}', n) for n in range(9)]})
        self.addCleanup(many.close)
        many.request(1000, ['owner'])
        drain(many)
        self.assertEqual(many.scores(2000)['owner']['familiarity'], 1.0)

    def test_records_belonging_to_somebody_else_are_ignored(self):
        built = index({'owner': [hit('guest', 'mon', 1), hit('owner', 'tue', 2)]})
        self.addCleanup(built.close)
        built.request(1000, ['owner'])
        drain(built)
        self.assertEqual(built.scores(2000)['owner']['prior_sessions'], 1)

    def test_an_unreachable_service_is_unknown_rather_than_stranger(self):
        built = index({'owner': RuntimeError('memory service down')})
        self.addCleanup(built.close)
        built.request(1000, ['owner'])
        drain(built)
        self.assertNotIn('owner', built.scores(2000))
        self.assertIn('memory service down', built.status(2000)['error'])

    def test_anonymous_ids_are_never_queried_and_answers_are_cached(self):
        built = index({'owner': [hit('owner', 'mon', 1)]})
        self.addCleanup(built.close)
        built.request(1000, ['unknown', 'robot', '', 'owner'])
        drain(built)
        self.assertEqual(built.calls, ['owner'])
        built.request(1500, ['owner'])
        drain(built)
        self.assertEqual(built.calls, ['owner'])
        # Past the refresh interval it asks again.
        built.request(1000 + built.refresh_ms + 1, ['owner'])
        drain(built)
        self.assertEqual(built.calls, ['owner', 'owner'])

    def test_a_cached_answer_expires(self):
        built = index({'owner': [hit('owner', 'mon', 1)]}, ttl_ms=5000)
        self.addCleanup(built.close)
        built.request(1000, ['owner'])
        drain(built)
        stamp = built.records['owner']['checked_ms']
        self.assertIn('owner', built.scores(stamp + 1000))
        self.assertEqual(built.scores(stamp + 9000), {})


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.person = VisionPerson('owner', 1000, face_visible=True, face_confidence=.9, gaze_score=.9,
                                   body_facing_score=.9, identity_confidence=.8)
        self.index = index({'owner': [hit('owner', f's{n}', n) for n in range(6)]})
        self.addCleanup(self.index.close)

    def test_familiarity_reaches_the_algorithm_only_when_the_source_is_plugged_in(self):
        sources = AttentionSources('today', enabled=['cross_session_memory'], familiarity=self.index)
        sources.context(1000, [self.person])
        drain(self.index)
        context = sources.context(1100, [self.person])
        self.assertEqual(context['person_memory']['owner']['memory_familiarity'], 1.0)
        self.assertTrue(context['sources']['cross_session_memory']['available'])

        sources.set_enabled(['audio_visual'])
        self.assertEqual(sources.context(1200, [self.person])['person_memory'], {})

    def test_without_an_index_the_source_says_so_instead_of_reporting_zero(self):
        sources = AttentionSources('today', enabled=['cross_session_memory'])
        status = sources.context(1000, [self.person])['sources']['cross_session_memory']
        self.assertTrue(status['enabled'])
        self.assertFalse(status['available'])
        self.assertEqual(status['reason'], 'no_memory_index_configured')


class FusionTests(unittest.TestCase):
    """A returning partner needs more gaze than a live participant, a stranger never qualifies."""

    def run_gaze(self, familiarity, frames=70):
        fusion = InteractionFusion()
        memory = {'owner': {'memory_familiarity': familiarity}} if familiarity else {}
        state = None
        for tick in range(frames):
            stamp = 1000 + tick * 50
            person = VisionPerson('owner', stamp, face_id='f1', face_visible=True, face_confidence=.9,
                                  gaze_score=.9, body_facing_score=.9, identity_confidence=.8)
            context = {'sources': {name: {'enabled': True} for name in
                                   ('audio_visual', 'memory_context', 'linguistic_context', 'cross_session_memory')},
                       'crossmodal_enabled': True, 'goals': [], 'importance': {}, 'motivation': {},
                       'person_memory': memory, 'linguistic': {}, 'linguistic_speaker': None, 'event_id': ''}
            state = fusion.update(stamp, {'algorithm': 'test'}, [person], [], {}, None, context=context)
        return state

    def test_a_familiar_person_can_invite_and_an_unfamiliar_one_cannot(self):
        familiar = self.run_gaze(1.0)
        self.assertEqual(familiar['interaction_phase'], 'INVITED')
        self.assertIn('familiar_across_sessions', familiar['engagement']['reasons'])
        self.assertFalse(familiar['addressed_to_robot'])

        self.assertNotEqual(self.run_gaze(0.0)['interaction_phase'], 'INVITED')

    def test_familiarity_is_slower_than_taking_part_in_the_current_conversation(self):
        self.assertNotEqual(self.run_gaze(1.0, frames=30)['interaction_phase'], 'INVITED')
        self.assertEqual(self.run_gaze(1.0, frames=60)['interaction_phase'], 'INVITED')


if __name__ == '__main__':
    unittest.main()
