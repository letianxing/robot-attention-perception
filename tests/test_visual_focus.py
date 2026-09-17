import unittest
from dataclasses import replace
from robot_attention_perception.visual_focus import OwnerVisualFocus
from robot_attention_perception.types import VisionPerson,AcousticTrack
from robot_attention_perception.person_manager import ProbabilisticPersonManager

class VisualFocusTests(unittest.TestCase):
    def setUp(self):
        self.policy=OwnerVisualFocus()
        self.owner=VisionPerson(person_id='owner',stamp_ms=1000,role='owner',face_visible=True,face_confidence=.95,identity_confidence=.78,gaze_score=.89,body_facing_score=.9)
        self.none={'target_id':'none','listen':False,'addressed_to_robot':False,'confidence':0}
    def test_owner_looking_without_speech_gains_visual_focus(self):
        for stamp in range(1000,1801,100):
            state=self.policy.update(stamp,self.none,[replace(self.owner,stamp_ms=stamp)],[])
        self.assertEqual(state['target_id'],'person:owner')
        self.assertTrue(state['listen']);self.assertFalse(state['addressed_to_robot'])
        self.assertIsNone(state['source_track_id'])
    def test_look_away_or_other_target_prevents_silent_takeover(self):
        for stamp in range(1000,2001,100):
            state=self.policy.update(stamp,self.none,[replace(self.owner,stamp_ms=stamp,gaze_score=.2)],[])
        self.assertEqual(state['target_id'],'none')
        busy=dict(self.none,target_id='person:other',listen=True,addressed_to_robot=True)
        self.assertEqual(self.policy.update(2000,busy,[replace(self.owner,stamp_ms=2000)],[])['target_id'],'person:other')
    def test_person_manager_preserves_confirmed_owner_role(self):
        manager=ProbabilisticPersonManager()
        self.assertEqual(manager.update([self.owner],[],1000)[0].role,'owner')

    def test_stranger_gaze_can_gain_focus_despite_background_vad(self):
        for stamp in range(1000,1801,100):
            person=replace(self.owner,person_id='stranger1',role='anonymous',identity_confidence=0,stamp_ms=stamp)
            track=AcousticTrack('background',stamp,True,.8,.6)
            state=self.policy.update(stamp,self.none,[person],[track])
        self.assertEqual(state['person_id'],'stranger1')
        self.assertFalse(state['addressed_to_robot'])

    def test_loud_sound_orients_then_habituates_without_inventing_direction(self):
        for stamp in range(1000,2801,100):
            acoustic={'stamp_ms':stamp,'raw_rms_dbfs':-12,'direction_valid':False,'direction_deg':60}
            state=self.policy.update(stamp,self.none,[replace(self.owner,stamp_ms=stamp)],[],acoustic)
            if stamp==1000:
                self.assertEqual(state['target_id'],'sound:salient')
                self.assertIsNone(state['azimuth_deg']);self.assertFalse(state['listen'])
        self.assertEqual(state['target_id'],'person:owner')

    def test_equal_gaze_is_ambiguous_and_reflex_has_priority(self):
        for stamp in range(1000,1801,100):
            a=replace(self.owner,person_id='a',role='known',stamp_ms=stamp)
            b=replace(a,person_id='b')
            state=self.policy.update(stamp,self.none,[a,b],[])
        self.assertEqual(state['target_id'],'none')
        reflex={'active':True,'stamp_ms':1800,'source':'audio','target_id':'sound:bang','spatial':{'azimuth_deg':30}}
        state=self.policy.update(1800,self.none,[a],[],reflex=reflex)
        self.assertEqual(state['target_id'],'sound:bang');self.assertFalse(state['listen'])

    def test_salience_proposal_never_grants_conversation_permission(self):
        for stamp in range(1000,1801,100):
            p=replace(self.owner,stamp_ms=stamp,lip_motion=True,azimuth_deg=10)
            t=AcousticTrack('voice1',stamp,True,.9,.7,azimuth_deg=12)
            state=self.policy.update(stamp,self.none,[p],[t])
        self.assertFalse(state['addressed_to_robot'])
        self.assertIsNone(state['source_track_id'])
        self.assertEqual(state['algorithm'],'visual_engagement')
        # A conflicting bearing cannot upgrade a visual target to dialogue.
        state=self.policy.update(1900,self.none,[replace(p,stamp_ms=1900)],[replace(t,stamp_ms=1900,azimuth_deg=120)])
        self.assertFalse(state['addressed_to_robot'])
        self.assertEqual(state['person_id'],'owner')

    def test_threshold_jitter_holds_target_but_sustained_look_away_releases(self):
        for stamp in range(1000,2000,100):
            state=self.policy.update(stamp,self.none,[replace(self.owner,stamp_ms=stamp,gaze_score=.73 if stamp%200 else .77)],[])
        self.assertEqual(state['person_id'],'owner')
        for stamp in range(2000,3000,100):
            state=self.policy.update(stamp,self.none,[replace(self.owner,stamp_ms=stamp,gaze_score=.56,face_confidence=.78)],[])
            self.assertEqual(state['person_id'],'owner')
        state=self.policy.update(3100,self.none,[],[])
        self.assertEqual(state['person_id'],'owner');self.assertFalse(state['listen'])
        for stamp in range(3200,3600,100):state=self.policy.update(stamp,self.none,[replace(self.owner,stamp_ms=stamp,gaze_score=.1)],[])
        self.assertEqual(state['target_id'],'none')
