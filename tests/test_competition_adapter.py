import unittest
from dataclasses import replace
from robot_attention_perception.competition_adapter import CompetitionAdapter
from robot_attention_perception.types import VisionPerson,AcousticTrack

class AdapterTests(unittest.TestCase):
    def test_separate_visual_goal_and_audio_speaker_and_cancel(self):
        a=CompetitionAdapter()
        p=VisionPerson('owner',0,face_visible=True,face_confidence=.95,gaze_score=.5,body_facing_score=.7,lip_motion=True,azimuth_deg=0)
        for t in range(0,2001,100):
            track=AcousticTrack('v',t,True,.9,.8,azimuth_deg=0)
            obj={'candidate_id':'visual:book','stamp_ms':t,'salience':.3,'confidence':.95}
            context={'goals':[{'candidate_id':'visual:book','stamp_ms':t,'strength':1,'confirmed':True}]}
            result=a.update(t,[replace(p,stamp_ms=t)],[track],{},[obj],context)
        self.assertEqual(result['focus']['visual'],'visual:book')
        self.assertEqual(result['focus']['audio'],'audio:v')
        self.assertTrue(all(not r['executable'] for r in result['allocation_requests']))
        result=a.update(5000,[],[],{},[],{})
        self.assertIsNone(result['focus']['visual']);self.assertIsNone(result['focus']['audio'])
    def test_identity_conflict_does_not_create_crossmodal_gain(self):
        a=CompetitionAdapter()
        p=VisionPerson('faceA',1000,face_visible=True,face_confidence=.95,gaze_score=.9,body_facing_score=.9,lip_motion=True,azimuth_deg=0)
        t=AcousticTrack('v',1000,True,.9,.9,azimuth_deg=0,speaker_label='voiceB',speaker_similarity=.9)
        a.update(1000,[p],[t],{})
        # A conflicting identity does not become an audio conversational goal.
        self.assertEqual(a.result['candidates']['audio:v']['goal'],0)
