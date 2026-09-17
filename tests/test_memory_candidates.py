"""Endogenous attention is the easiest thing here to turn into a nuisance."""
import unittest

from robot_attention_perception.global_workspace import CAPACITY, URGENT, GlobalWorkspace
from robot_attention_perception.memory_candidates import MemoryCandidates
from robot_attention_perception.types import VisionPerson


def person(stamp, person_id='owner'):
    return VisionPerson(person_id, stamp, face_visible=True, face_confidence=.9, identity_confidence=.8)


def brain(stamp, unfinished=None, session='s'):
    return {'session_id': session, 'stamp_ms': stamp,
            'attention_memory': {'recent_participants': [], 'expected_answer': None,
                                 'unfinished': unfinished if unfinished is not None else
                                 [{'id': 'turn-1', 'kind': 'unanswered_question', 'person_id': 'owner',
                                   'question': '周六去公园好不好？', 'stamp_ms': stamp - 60000,
                                   'summary': '周六去公园好不好？'}]}}


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.index = MemoryCandidates('s')
        self.stamp = 1_000_000

    def raise_one(self, **kwargs):
        stamp = kwargs.pop('stamp', self.stamp)
        people = kwargs.pop('people', [person(stamp)])
        snapshot = kwargs.pop('brain', brain(stamp))
        quiet_since = kwargs.pop('last_human_ms', stamp - 10000)
        return self.index.candidates(stamp, snapshot, people, quiet_since)

    def test_unfinished_business_with_the_person_present_and_the_room_quiet(self):
        raised = self.raise_one()
        self.assertEqual(len(raised), 1)
        self.assertEqual(raised[0]['candidate_id'], 'memory:turn-1')
        self.assertEqual(raised[0]['kind'], 'memory_event')
        self.assertEqual(raised[0]['person_id'], 'owner')
        self.assertEqual(raised[0]['urgency'], 0.0)
        self.assertIn('unfinished_business', raised[0]['reasons'])
        self.assertIn('不构成说话许可', raised[0]['limits'])

    def test_it_never_fires_twice_for_the_same_thing(self):
        self.assertEqual(len(self.raise_one()), 1)
        later = self.raise_one(stamp=self.stamp + 400000)
        self.assertEqual(later, [])
        self.assertEqual(self.index.status()['reason'], 'nothing_unfinished')

    def test_a_busy_room_an_empty_room_and_a_rate_limit_all_block_it(self):
        self.assertEqual(self.raise_one(people=[]), [])
        self.assertEqual(self.index.status()['reason'], 'nobody_present')

        self.assertEqual(self.raise_one(last_human_ms=self.stamp - 500), [])
        self.assertEqual(self.index.status()['reason'], 'room_is_busy')

        self.assertEqual(len(self.raise_one()), 1)
        fresh = MemoryCandidates('s')
        fresh.last_raised_ms = self.stamp
        fresh.candidates(self.stamp + 1000, brain(self.stamp + 1000), [person(self.stamp + 1000)], 0)
        self.assertEqual(fresh.status()['reason'], 'rate_limited')

    def test_it_is_raised_with_the_person_it_concerns(self):
        self.assertEqual(self.raise_one(people=[person(self.stamp, 'guest')]), [])

    def test_a_stale_or_foreign_snapshot_produces_nothing(self):
        self.assertEqual(self.raise_one(brain=brain(self.stamp - 9000)), [])
        self.assertEqual(self.index.status()['reason'], 'brain_snapshot_stale')
        self.assertEqual(self.raise_one(brain=brain(self.stamp, session='other')), [])
        self.assertEqual(self.index.status()['reason'], 'no_current_session_snapshot')

    def test_something_too_old_is_let_go(self):
        old = brain(self.stamp)
        old['attention_memory']['unfinished'][0]['stamp_ms'] = self.stamp - 3600000
        self.assertEqual(self.raise_one(brain=old), [])

    def test_it_can_be_switched_off_entirely(self):
        off = MemoryCandidates('s', enabled=False)
        self.assertEqual(off.candidates(self.stamp, brain(self.stamp), [person(self.stamp)], 0), [])
        self.assertEqual(off.status()['reason'], 'disabled')


class WorkspaceCompetitionTests(unittest.TestCase):
    """A memory event has no modality, so the workspace is where it competes."""

    def sensory(self, count, activation=0.4):
        return {'candidates': {f'visual:p{index}': {'modality': 'visual', 'person_id': f'p{index}',
                                                    'a': activation, 'valid': True, 'reasons': []}
                               for index in range(count)}}

    def test_a_memory_event_competes_against_live_perception(self):
        workspace = GlobalWorkspace('s')
        self.addCleanup(workspace.close)
        contents = workspace.publish(1000, {}, self.sensory(2), {}, {}, events=[
            {'candidate_id': 'memory:turn-1', 'kind': 'memory_event', 'person_id': 'owner',
             'salience': 0.55, 'summary': '周六去公园好不好？', 'reasons': ['unfinished_business']}])
        ranked = contents['coalition']
        self.assertEqual(ranked[0]['candidate_id'], 'memory:turn-1')
        self.assertEqual(ranked[0]['kind'], 'memory_event')
        self.assertIsNone(ranked[0]['modality'])
        self.assertAlmostEqual(sum(item['share'] for item in ranked), 1.0, places=4)

    def test_only_a_few_candidates_get_through_and_the_rest_say_so(self):
        workspace = GlobalWorkspace('s')
        self.addCleanup(workspace.close)
        contents = workspace.publish(1000, {}, self.sensory(CAPACITY + 3), {}, {})
        self.assertEqual(contents['capacity']['limit'], CAPACITY)
        self.assertEqual(contents['capacity']['admitted'], CAPACITY)
        self.assertEqual(contents['capacity']['suppressed'], 3)
        self.assertTrue(all(item['admission'] == 'suppressed_over_capacity'
                            for item in contents['coalition'] if not item['admitted']))

    def test_an_urgent_event_pre_empts_the_capacity_limit(self):
        workspace = GlobalWorkspace('s')
        self.addCleanup(workspace.close)
        contents = workspace.publish(1000, {}, self.sensory(CAPACITY + 2, activation=0.9), {}, {}, events=[
            {'candidate_id': 'reflex:bang', 'kind': 'reflex_event', 'salience': 0.2, 'urgency': URGENT,
             'reasons': ['reflex_audio']}])
        reflex = next(item for item in contents['coalition'] if item['candidate_id'] == 'reflex:bang')
        self.assertTrue(reflex['admitted'])
        self.assertLess(reflex['share'], 0.2)  # it got in on urgency, not on strength

    def test_an_empty_cycle_does_not_divide_by_zero(self):
        workspace = GlobalWorkspace('s')
        self.addCleanup(workspace.close)
        contents = workspace.publish(1000, {}, {'candidates': {}}, {}, {})
        self.assertEqual(contents['coalition'], [])
        self.assertEqual(contents['capacity']['admitted'], 0)


if __name__ == '__main__':
    unittest.main()
