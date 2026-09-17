"""The broadcast step: one contents, many consumers, never blocking perception."""
import json
import queue
import unittest

from robot_attention_perception.global_workspace import GlobalWorkspace, sse_event


def attention(target='person:owner', phase='INVITED'):
    return {'target_id': target, 'target_kind': 'person', 'person_id': 'owner', 'confidence': .78,
            'interaction_phase': phase, 'listen': True, 'addressed_to_robot': False,
            'reasons': ['silent_invitation'], 'transition_id': 12}


def distribution():
    return {'focus': {'visual': 'visual:owner', 'audio': None},
            'engagement': {'state': 'INVITED', 'person_id': 'owner', 'confidence': .78,
                           'reasons': ['gaze_engaged', 'memory_participant']},
            'candidates': {
                'visual:owner': {'modality': 'visual', 'person_id': 'owner', 'a': .91, 'engagement': .78,
                                 'valid': True, 'focus': True, 'reasons': ['gaze_engaged']},
                'visual:guest': {'modality': 'visual', 'person_id': 'guest', 'a': .22, 'engagement': .3,
                                 'valid': True, 'focus': False, 'reasons': []},
                'audio:stale': {'modality': 'audio', 'person_id': None, 'a': .0, 'valid': False}}}


class ContentsTests(unittest.TestCase):
    def setUp(self):
        self.workspace = GlobalWorkspace('s')
        self.addCleanup(self.workspace.close)

    def test_a_cycle_carries_the_winner_its_reasons_and_its_provenance(self):
        contents = self.workspace.publish(1000, attention(), distribution(),
                                          sources={'memory_context': {'enabled': True, 'available': True}},
                                          algorithm={'id': 'av_memory_language_v1', 'calibrated': False},
                                          provenance={'utterance_id': 'u1'})
        self.assertEqual(contents['cycle'], 1)
        self.assertEqual(contents['session_id'], 's')
        self.assertEqual(contents['target']['target_id'], 'person:owner')
        self.assertEqual(contents['engagement']['state'], 'INVITED')
        self.assertEqual(contents['provenance']['utterance_id'], 'u1')
        self.assertFalse(contents['algorithm']['calibrated'])
        self.assertIn('不是意识', contents['limits'])

    def test_the_coalition_is_ranked_and_excludes_invalid_candidates(self):
        contents = self.workspace.publish(1000, attention(), distribution(), {}, {})
        self.assertEqual([item['candidate_id'] for item in contents['coalition']],
                         ['visual:owner', 'visual:guest'])
        self.assertEqual(contents['coalition'][0]['person_id'], 'owner')
        self.assertTrue(contents['coalition'][0]['focus'])

    def test_before_any_cycle_it_says_so_instead_of_inventing_contents(self):
        snapshot = self.workspace.snapshot()
        self.assertEqual(snapshot['cycle'], 0)
        self.assertEqual(snapshot['reason'], 'no_cycle_broadcast_yet')
        self.assertNotIn('target', snapshot)

    def test_replay_returns_only_cycles_after_the_one_asked_for(self):
        for stamp in range(1000, 1500, 100):
            self.workspace.publish(stamp, attention(), distribution(), {}, {})
        self.assertEqual(len(self.workspace.replay(0)), 5)
        self.assertEqual([item['cycle'] for item in self.workspace.replay(3)], [4, 5])
        self.assertEqual(self.workspace.replay(5), [])

    def test_history_is_bounded(self):
        workspace = GlobalWorkspace('s', history=4)
        self.addCleanup(workspace.close)
        for stamp in range(1000, 2000, 100):
            workspace.publish(stamp, attention(), distribution(), {}, {})
        self.assertEqual(len(workspace.replay(0)), 4)
        self.assertEqual(workspace.status()['cycle'], 10)


class SubscriberTests(unittest.TestCase):
    def test_a_subscriber_gets_the_current_contents_then_every_new_cycle(self):
        workspace = GlobalWorkspace('s')
        self.addCleanup(workspace.close)
        workspace.publish(1000, attention(), distribution(), {}, {})
        key, channel = workspace.subscribe()
        self.assertEqual(channel.get_nowait()['cycle'], 1)
        workspace.publish(1100, attention(), distribution(), {}, {})
        self.assertEqual(channel.get_nowait()['cycle'], 2)
        workspace.unsubscribe(key)
        workspace.publish(1200, attention(), distribution(), {}, {})
        with self.assertRaises(queue.Empty):
            channel.get_nowait()

    def test_a_slow_consumer_drops_cycles_and_is_counted_not_waited_for(self):
        workspace = GlobalWorkspace('s', queue_size=2)
        self.addCleanup(workspace.close)
        workspace.subscribe()
        for stamp in range(1000, 1600, 100):
            workspace.publish(stamp, attention(), distribution(), {}, {})
        status = workspace.status()
        self.assertEqual(status['cycle'], 6)
        self.assertGreater(status['dropped_cycles'], 0)

    def test_the_event_framing_is_parseable_by_a_plain_sse_client(self):
        workspace = GlobalWorkspace('s')
        self.addCleanup(workspace.close)
        contents = workspace.publish(1000, attention(), distribution(), {}, {})
        frame = sse_event(contents).decode('utf-8')
        self.assertTrue(frame.startswith('event: workspace\ndata: '))
        self.assertTrue(frame.endswith('\n\n'))
        self.assertEqual(json.loads(frame.split('data: ', 1)[1])['cycle'], contents['cycle'])


if __name__ == '__main__':
    unittest.main()
