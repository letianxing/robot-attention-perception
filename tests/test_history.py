import unittest
from unittest.mock import patch
from robot_attention_perception.history import search_history, trace_history
from robot_attention_perception.live_runtime import LiveRuntimeConfig

class HistoryTests(unittest.TestCase):
    @patch('robot_attention_perception.history.post_json', return_value={'hits':[], 'total':0})
    def test_scope_pagination_and_strict_query(self, post):
        search_history(LiveRuntimeConfig(session_id='current'), {'session':['older'],'q':['公园'],'person':['person2'],'offset':['25'],'as_of_ms':['1000']})
        payload=post.call_args.args[1]
        self.assertEqual(payload['scope']['session_id'],'older')
        self.assertEqual(payload['offset'],25)
        self.assertTrue(payload['strict_text'])
        self.assertEqual(payload['entity_id'],'person2')
        self.assertEqual(payload['kinds'],['heard_utterance','dialogue_turn'])

    @patch('robot_attention_perception.history.post_json', return_value={'hits':[]})
    def test_trace_uses_turn_identity(self, post):
        trace_history(LiveRuntimeConfig(),{'turn':['voice:1:2']})
        self.assertEqual(post.call_args.args[1]['entity_id'],'voice:1:2')
        self.assertEqual(post.call_args.args[1]['kind'],'brain_trace')
