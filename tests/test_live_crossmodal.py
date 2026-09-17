import unittest
from robot_attention_perception.live_crossmodal import LiveCrossmodal

class CrossmodalTests(unittest.TestCase):
    def test_real_event_timing_binds_two_beeps_one_flash_without_changing_raw_counts(self):
        model=LiveCrossmodal()
        tones=[{'id':'b1','kind':'tone_pulse','stamp_ms':1000},{'id':'b2','kind':'tone_pulse','stamp_ms':1100}]
        flash={'id':'f1','kind':'flash_pulse','stamp_ms':1020}
        result=model.update(1500,{'audio_scene':{'events':tones}},{'flash_events':[flash]},[])
        self.assertEqual(len(result),1);self.assertTrue(result[0]['double_flash_candidate'])
        self.assertEqual(result[0]['observed_flash_count'],1)
        self.assertEqual(len(model.update(1600,{'audio_scene':{'events':tones}},{'flash_events':[flash]},[])),1)

    def test_far_apart_events_do_not_bind(self):
        result=LiveCrossmodal().update(2000,{'audio_scene':{'events':[{'id':'b','kind':'tone_pulse','stamp_ms':1700}]}},{'flash_events':[{'id':'f','kind':'flash_pulse','stamp_ms':1000}]},[])
        self.assertFalse(result[0]['double_flash_candidate'])

    def test_bilabial_conflict_uses_observed_mouth_geometry(self):
        model=LiveCrossmodal()
        for stamp in [1000,1050,1100]:model.update(stamp,{}, {'stamp':stamp/1000,'people':[{'person_id':'p','lip_motion_valid':True,'mouth_open_ratio':.2}]},[])
        result=model.update(1400,{}, {},[{'utterance_id':'u','person_id':'p','transcript':{'text':'巴','started_ms':1000,'ended_ms':1200}}])
        self.assertTrue(result[0]['conflict']);self.assertEqual(result[0]['audio_text'],'巴')
