import unittest
from robot_attention_perception.session_attention import SessionAttention
from robot_attention_perception.types import VisionPerson

class SessionAttentionTests(unittest.TestCase):
    def test_active_dialogue_support_then_cancel_expire_or_conflict(self):
        bridge=SessionAttention('test');person=VisionPerson('owner',1000,face_visible=True)
        state={'session_id':'test','stamp_ms':1000,'current_turn':{'turn_id':'t','person_id':'owner','status':'speaking','attention':{}},'s2s':{'speaking':True}}
        bridge.receive(state)
        goal=bridge.context(1050,[person])['goals'][0]
        self.assertEqual(goal['candidate_id'],'visual:owner');self.assertNotIn('listen',goal)
        self.assertEqual(bridge.context(2300,[person])['goals'],[])
        bridge.receive(dict(state,current_turn=dict(state['current_turn'],status='interrupted')))
        self.assertEqual(bridge.context(1050,[person])['goals'],[])
        bridge.receive(dict(state,current_turn=dict(state['current_turn'],attention={'identity_conflict':True})))
        self.assertEqual(bridge.context(1050,[person])['goals'],[])
    def test_other_session_cannot_create_goal(self):
        bridge=SessionAttention('test')
        bridge.receive({'session_id':'foreign','stamp_ms':1000,'s2s':{'speaking':True}})
        self.assertEqual(bridge.context(1000,[])['goals'],[])
