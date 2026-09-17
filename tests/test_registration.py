import unittest
from robot_attention_perception.registration import validate

class RegistrationTests(unittest.TestCase):
    def test_free_speech_and_quality_checks(self):
        payload={'name':'小天','role':'owner','utterance_id':'u'}
        t={'utterance_id':'u','started_ms':6000,'ended_ms':9000,'text':'我喜欢去公园散步和听音乐'}
        state={'vision':{'stamp':9.9,'people':[{'face_confidence':.9}]},'voice':{'last_transcript':t}}
        self.assertEqual(validate(state,payload,10000),'小天')
        t['source']='robot_echo'
        with self.assertRaises(ValueError):validate(state,payload,10000)
        t['source']='microphone';state['vision']['people'].append({'face_confidence':.9})
        with self.assertRaises(ValueError):validate(state,payload,10000)

    def test_same_name_and_exact_utterance_sent_to_both_services(self):
        from unittest.mock import Mock, patch
        from types import SimpleNamespace
        from robot_attention_perception.registration import register
        runtime=Mock(config=SimpleNamespace(vision_url='http://vision',voice_url='http://voice'))
        runtime.state.return_value={'voice':{'last_transcript':{'speaker':{}}}}
        with patch('robot_attention_perception.registration.validate',return_value='小天'), patch('robot_attention_perception.registration.post_json',return_value={'success':True}) as post:
            result=register(runtime,{'name':'小天','role':'owner','utterance_id':'u1'})
        self.assertTrue(result['success'])
        self.assertEqual(post.call_args_list[0].args[1]['user_id'],'小天')
        self.assertTrue(post.call_args_list[0].args[1]['require_single'])
        self.assertEqual(post.call_args_list[1].args[1]['speaker_id'],'小天')
        self.assertEqual(post.call_args_list[1].args[1]['utterance_id'],'u1')

    def test_recognized_owner_is_not_enrolled_again(self):
        from unittest.mock import Mock, patch
        from robot_attention_perception.registration import register
        runtime=Mock()
        runtime.state.return_value={'voice':{'last_transcript':{'speaker':{'speaker_id':'小天','speaker_role':'owner','similarity':.9}}}}
        with patch('robot_attention_perception.registration.validate',return_value='小天'), patch('robot_attention_perception.registration.post_json') as post:
            result=register(runtime,{'name':'小天','role':'owner','utterance_id':'u1'})
        post.assert_not_called()
        self.assertTrue(result['already_registered'])
        self.assertIn('不要来挑逗',result['message'])

    def test_failed_modality_validation_prevents_either_profile_commit(self):
        from unittest.mock import Mock,patch
        from robot_attention_perception.registration import register
        runtime=Mock();runtime.config.vision_url='http://vision';runtime.config.voice_url='http://voice';runtime.state.return_value={'voice':{'last_transcript':{'speaker':{}}}}
        with patch('robot_attention_perception.registration.validate',return_value='小天'),patch('robot_attention_perception.registration.post_json',side_effect=[{'success':True},{'success':False,'message':'声纹不一致'}]) as post:
            result=register(runtime,{'name':'小天','role':'owner','utterance_id':'u'})
        self.assertFalse(result['success'])
        self.assertTrue(all(call.args[1]['dry_run'] for call in post.call_args_list))
