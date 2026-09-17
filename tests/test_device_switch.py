import unittest
from types import SimpleNamespace
from unittest.mock import patch
from robot_attention_perception.live_runtime import LivePerceptionRuntime


class DeviceSwitchTests(unittest.TestCase):
    def test_camera_and_microphone_switch_are_independent(self):
        runtime=LivePerceptionRuntime.__new__(LivePerceptionRuntime)
        runtime.config=SimpleNamespace(vision_enabled=True,vision_url='http://vision',voice_url='http://voice')
        with patch('robot_attention_perception.live_runtime.post_json',return_value={'success':True}) as post:
            runtime.start_inputs('1','opencv','mac_builtin','2',target='camera')
            self.assertEqual(post.call_count,2)
            self.assertTrue(all(call.args[0].startswith('http://vision') for call in post.call_args_list))
            post.reset_mock()
            runtime.start_inputs('1','opencv','mac_builtin','2',target='microphone')
            post.assert_called_once()
            self.assertTrue(post.call_args.args[0].startswith('http://voice'))
