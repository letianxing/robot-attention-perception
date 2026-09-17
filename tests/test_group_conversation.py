import unittest
from robot_attention_perception.group_conversation import GroupConversation

class GroupTests(unittest.TestCase):
    def turn(self,p,text,t,addressed=False):
        return {'person_id':p,'stamp_ms':t,'source':'microphone','transcript':{'text':text,'started_ms':t-200,'ended_ms':t},'attention':{'listen':addressed,'addressed_to_robot':addressed,'confidence':.8}}
    def test_many_people_and_explicit_invitation(self):
        group=GroupConversation()
        for i in range(5):group.annotate(self.turn('person'+str(i),'我有一个想法',1000+i*500))
        invitation=group.annotate(self.turn('person0','小圆，一起讨论吧',4000,True))
        self.assertEqual(len(invitation['group_context']['participants']),6)
        q=group.annotate(self.turn('person4','大家觉得这个方案怎么样呢',4500))
        self.assertEqual(q['addressee'],'group');self.assertTrue(q['attention']['listen'])
        aside=group.annotate(self.turn('person3','person1，你来解释一下',5000,True))
        self.assertEqual(aside['addressee'],'person:person1');self.assertFalse(aside['attention']['listen'])
    def test_membership_does_not_mean_every_sentence_needs_robot_reply(self):
        group=GroupConversation();group.annotate(self.turn('a','小圆，一起聊',1000,True))
        aside=group.annotate(self.turn('b','我也这么想',2000))
        self.assertFalse(aside['attention']['listen'])
        group.annotate(self.turn('a','小圆，你先别说',3000,True))
        self.assertFalse(group.annotate(self.turn('b','大家怎么看',4000))['attention']['listen'])
    def test_group_call_without_joining_and_expired_membership_do_not_interrupt(self):
        group=GroupConversation()
        self.assertFalse(group.annotate(self.turn('a','大家怎么看',1000))['attention']['listen'])
        group.annotate(self.turn('a','小圆，一起聊',2000,True))
        self.assertFalse(group.annotate(self.turn('a','大家怎么看',123000))['attention']['listen'])
