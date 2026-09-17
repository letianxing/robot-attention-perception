import unittest
from robot_attention_perception.reflex_attention import ReflexAttention, spatial_source


class ReflexTests(unittest.TestCase):
    def test_audio_orients_without_permission_and_expires(self):
        reflex=ReflexAttention()
        event={'id':'bang','kind':'startle','stamp_ms':10000,'direction_deg':45,'direction_valid':True}
        focus=reflex.update(10000,{'audio_scene':{'events':[event]}},{})
        self.assertEqual(focus['spatial']['azimuth_deg'],45)
        self.assertIsNone(focus['spatial']['position_xyz_m'])
        self.assertFalse(focus['listen'])
        self.assertFalse(focus['addressed_to_robot'])
        self.assertFalse(reflex.update(12000,{'audio_scene':{'events':[event]}},{})['active'])

    def test_3d_requires_valid_finite_coordinates_and_frame(self):
        valid={'position_xyz_m':[1,2,3],'frame_id':'base_link','stamp_ms':12}
        self.assertTrue(spatial_source('audio',valid)['position_valid'])
        for bad in ({'position_xyz_m':[1,2,3]},dict(valid,position_valid=False),dict(valid,position_xyz_m=[1,float('nan'),3])):
            self.assertFalse(spatial_source('vision',bad)['position_valid'])
        self.assertEqual(spatial_source('audio',{'direction_deg':0,'direction_valid':False})['mode'],'unknown')

    def test_visual_rapid_approach_not_first_detection(self):
        reflex=ReflexAttention()
        def vision(stamp,area): return {'stamp':stamp/1000,'people':[{'person_id':'p1','bbox_area_ratio':area,'face_confidence':.82,'azimuth_deg':12}]}
        # 0.06 -> 0.36 of the frame in 250 ms is about 15 -> 36 degrees, i.e. 86
        # deg/s, inside the range that drives flight rather than freezing.
        self.assertIsNone(reflex.update(10000,{},vision(10000,.06)))
        focus=reflex.update(10250,{},vision(10250,.36))
        self.assertEqual(focus['source'],'vision')
        self.assertEqual(focus['target_id'],'person:p1')
        self.assertGreaterEqual(focus['evidence']['expansion_deg_per_s'],57)
        self.assertFalse(focus['spatial']['position_valid'])

    def test_someone_walking_closer_is_not_a_threat(self):
        reflex=ReflexAttention()
        def vision(stamp,area): return {'stamp':stamp/1000,'people':[{'person_id':'p1','bbox_area_ratio':area,'face_confidence':.95}]}
        reflex.update(10000,{},vision(10000,.06))
        # doubling the angle over a second is a person walking up, not a lunge
        for index,area in enumerate([.09,.13,.18,.24],start=1):
            self.assertIsNone(reflex.update(10000+index*500,{},vision(10000+index*500,area)))

    def test_small_bbox_jitter_does_not_startle(self):
        reflex=ReflexAttention()
        for i,area in enumerate([.15,.16,.14,.17,.15]):
            self.assertIsNone(reflex.update(10000+i*200,{}, {'stamp':(10000+i*200)/1000,'people':[{'person_id':'p1','bbox_area_ratio':area,'face_confidence':.99}]}))

    def test_large_object_appearance_requires_warmup_and_confirmation(self):
        reflex=ReflexAttention();reflex.update(10000,{}, {'stamp':10})
        def frame(stamp):return {'stamp':stamp/1000,'objects':[{'label':'book','confidence':.95,'bbox_area_ratio':.4,'center':[.5,.5],'stamp_ms':stamp}]}
        self.assertIsNone(reflex.update(12500,{},frame(12500)))
        focus=reflex.update(12650,{},frame(12650))
        self.assertEqual(focus['reason'],'sudden_large_object')
        self.assertFalse(focus['listen'])


class OrientingIsNotStartleTest(unittest.TestCase):
    """Orienting must not clear the floor the way a startle does.

    Giving the robot a general orienting response (so speech off camera could
    reach attention at all) made it fire every couple of seconds on ordinary
    talking. Everything downstream treated a reflex as a startle: the engagement
    layer suspends name calls and answer continuations while reflex_active is
    set, and the workspace admitted the reflex at urgency 1.0 past its capacity
    limit. The robot stopped answering its own name.
    """

    def fusion_with(self, kind):
        from robot_attention_perception.interaction_fusion import InteractionFusion
        from robot_attention_perception.types import AcousticTrack, VisionPerson
        fusion = InteractionFusion()
        person = VisionPerson('owner', 0, face_id='f1', face_visible=True, face_confidence=.9,
                              gaze_score=.9, body_facing_score=.9, identity_confidence=.8, role='known',
                              lip_motion=True)
        state = None
        for index in range(12):
            stamp = 10000 + index * 50
            track = AcousticTrack('t1', stamp, voice_activity=True, speech_probability=.9, clarity=.8,
                                  speaker_label='owner', speaker_similarity=.9)
            reflex = {'id': 'sound', 'active': True, 'kind': kind, 'stamp_ms': stamp, 'source': 'audio',
                      'target_id': 'sound:1'}
            from dataclasses import replace
            state = fusion.update(stamp, {'algorithm': 'test'}, [replace(person, stamp_ms=stamp)], [track],
                                  {}, reflex)
        return state

    def test_a_startle_still_clears_the_floor(self):
        self.assertEqual(self.fusion_with('startle')['interaction_phase'], 'ORIENTING')

    def test_orienting_does_not_take_the_floor_from_somebody_speaking(self):
        # Orienting now fires on ordinary speech by design; if it also cleared
        # the floor, the person being listened to lost the robot mid-sentence.
        self.assertNotEqual(self.fusion_with('orient')['interaction_phase'], 'ORIENTING')
