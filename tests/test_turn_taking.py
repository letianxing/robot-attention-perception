import unittest
from robot_attention_perception.turn_taking import interruption_plan

class SocialTests(unittest.TestCase):
    def test_backchannel_is_not_a_floor_request(self):
        for word in ['嗯','对对','好的']:
            self.assertEqual(interruption_plan(word,True,{},1000)['mode'],'continue')
        self.assertEqual(interruption_plan('对吗？',True,{},1000)['mode'],'yield')
    def test_name_stop_and_normal_turn_have_different_delivery(self):
        self.assertEqual(interruption_plan('小圆',True,{},1000)['fade_ms'],20)
        self.assertEqual(interruption_plan('小圆，停止',True,{},1000)['fade_ms'],10)
        self.assertEqual(interruption_plan('我想补充一下',True,{},1000)['mode'],'yield')
    def test_vap_never_grants_permission_it_only_adjusts_confirmation(self):
        plan=interruption_plan('我想补充',True,{'audio_scene':{'turn_prediction':{'valid':True,'stamp_ms':1000,'p_user_now':.9}}},1000)
        self.assertEqual(plan['confirmation_ms'],350);self.assertNotIn('listen',plan)
        # 没有模型预测时要等更久：几十毫秒的重叠多半是应声或口误，不是要接话
        self.assertEqual(interruption_plan('我想补充',True,{},1000)['confirmation_ms'],500)
