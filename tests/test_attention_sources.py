import unittest
from robot_attention_perception.attention_sources import AttentionSources
from robot_attention_perception.types import VisionPerson
from robot_attention_perception.competition_adapter import CompetitionAdapter

class SourcesTests(unittest.TestCase):
    def setUp(self):self.person=VisionPerson('a',1000,face_visible=True,face_confidence=.9,gaze_score=.9,body_facing_score=.9)
    def test_default_has_no_invented_internal_state(self):
        p=AttentionSources('s',enabled=['audio_visual','memory_context'])
        c=p.context(1000,[self.person]);self.assertNotIn('arousal_gain',c)
        self.assertEqual(c['motivation'],{});self.assertFalse(c['sources']['internal_state']['available'])
        result=CompetitionAdapter().update(1000,[self.person],[],{},context=c)
        self.assertFalse(result['internal_state_used'])
    def test_context_memory_is_scoped_bounded_and_needs_current_person(self):
        p=AttentionSources('s',enabled=['memory_context'])
        state={'session_id':'s','stamp_ms':1000,'attention_memory':{'recent_participants':[{'person_id':'a','last_heard_ms':950},{'person_id':'invisible','last_heard_ms':950}],
                'expected_answer':{'person_id':'a','expires_ms':3000}},'current_turn':{},'s2s':{}}
        p.receive_brain(state);c=p.context(1050,[self.person])
        self.assertGreater(c['importance']['visual:a'],0);self.assertNotIn('visual:invisible',c['importance'])
        self.assertEqual(c['goals'][0]['purpose'],'expected_answer');self.assertNotIn('listen',c['goals'][0])
        self.assertEqual(p.context(3000,[self.person])['goals'],[])
        other=AttentionSources('x',enabled=['memory_context']);other.receive_brain(state)
        self.assertFalse(other.context(1050,[self.person])['sources']['memory_context']['available'])
    def test_optional_internal_provider_expires_and_cannot_create_permission(self):
        p=AttentionSources('s',enabled=['internal_state'])
        self.assertNotIn('arousal_gain',p.context(1000,[]) )
        self.assertFalse(p.receive_internal({'schema_version':1,'session_id':'s','stamp_ms':1000,'arousal_gain':float('nan')}))
        self.assertTrue(p.receive_internal({'schema_version':1,'session_id':'s','stamp_ms':1000,'arousal_gain':1.3,'candidate_support':{'visual:a':.5}}))
        c=p.context(1100,[self.person]);self.assertEqual(c['arousal_gain'],1.3);self.assertNotIn('listen',c)
        self.assertNotIn('arousal_gain',p.context(3000,[self.person]))
    def test_disable_memory_is_real_ablation(self):
        p=AttentionSources('s',enabled=['audio_visual'])
        p.receive_brain({'session_id':'s','stamp_ms':1000,'attention_memory':{'recent_participants':[{'person_id':'a','last_heard_ms':1000}]}})
        self.assertEqual(p.context(1100,[self.person])['importance'],{})

    def test_memory_reports_participants_expectations_and_the_live_dialogue_target(self):
        p=AttentionSources('s',enabled=['memory_context'])
        p.receive_brain({'session_id':'s','stamp_ms':1000,'s2s':{},
            'attention_memory':{'recent_participants':[{'person_id':'a','last_heard_ms':900}],
                                'expected_answer':{'person_id':'a','expires_ms':9000,'question':'要我说说吗？'}},
            'current_turn':{'person_id':'a','status':'speaking'}})
        memory=p.context(1100,[self.person])['person_memory']['a']
        self.assertEqual(memory['memory_expected_answer'],1.)
        self.assertEqual(memory['memory_dialogue_target'],1.)
        self.assertEqual(memory['memory_participant'],1.)
        self.assertLess(memory['memory_recency'],1.)

    def test_language_is_scored_once_per_utterance_and_carries_the_open_question(self):
        p=AttentionSources('s',enabled=['memory_context','linguistic_context'])
        p.receive_brain({'session_id':'s','stamp_ms':1000,'s2s':{},'current_turn':{},
            'attention_memory':{'recent_participants':[],'expected_answer':{'person_id':'a','expires_ms':9000,'question':'周六去公园吗？'}}})
        voice={'last_transcript':{'is_final':True,'utterance_id':'u1','emitted_ms':1000,'text':'周六去公园吧',
                                  'speaker':{'speaker_id':'a','similarity':.9},'track_id':'t1'}}
        result=p.context(1100,[self.person],(),voice)
        self.assertEqual(result['event_id'],'u1:final')
        self.assertEqual(result['linguistic_speaker'],'a')
        self.assertGreater(result['linguistic']['lang_answer_continuation'],0)
        self.assertTrue(result['sources']['linguistic_context']['available'])

    def test_robot_echo_and_stale_text_are_not_linguistic_evidence(self):
        p=AttentionSources('s',enabled=['linguistic_context'])
        echo={'last_transcript':{'is_final':True,'utterance_id':'u2','emitted_ms':1000,'text':'小圆你好','source':'robot_echo'}}
        self.assertEqual(p.context(1100,[],(),echo)['linguistic'],{})
        stale={'last_transcript':{'is_final':True,'utterance_id':'u3','emitted_ms':1000,'text':'小圆你好'}}
        self.assertEqual(p.context(9000,[],(),stale)['linguistic'],{})

    def test_brain_goals_bias_the_person_they_concern_and_expire(self):
        p=AttentionSources('s',enabled=['memory_context'])
        p.receive_brain({'session_id':'s','stamp_ms':1000,'s2s':{},'current_turn':{},
            'attention_memory':{'recent_participants':[],'expected_answer':None,
                'goals':[{'kind':'awaiting_answer','person_id':'a','strength':.65,'expires_ms':5000},
                         {'kind':'greet_owner','person_id':'ghost','strength':.4,'expires_ms':500}]}})
        c=p.context(1100,[self.person])
        self.assertAlmostEqual(c['importance']['visual:a'],.65)
        self.assertNotIn('visual:ghost',c['importance'])
        self.assertEqual([g['kind'] for g in c['brain_goals']],['awaiting_answer'])

    def test_a_goal_cannot_grant_listening_or_invent_a_person(self):
        p=AttentionSources('s',enabled=['memory_context'])
        p.receive_brain({'session_id':'s','stamp_ms':1000,'s2s':{},'current_turn':{},
            'attention_memory':{'recent_participants':[],'expected_answer':None,
                'goals':[{'kind':'holding_floor','person_id':'a','strength':2.5,'expires_ms':5000}]}})
        c=p.context(1100,[self.person])
        self.assertEqual(c['importance']['visual:a'],1.)
        self.assertNotIn('listen',c)
        self.assertEqual(c['person_memory'],{})

    def test_sources_can_be_swapped_while_running(self):
        p=AttentionSources('s',enabled=['audio_visual'])
        self.assertEqual(p.enabled_names(),['audio_visual'])
        self.assertEqual(p.set_enabled(['audio_visual','memory_context']),['audio_visual','memory_context'])
        self.assertTrue(p.context(1000,[self.person])['sources']['memory_context']['enabled'])
        with self.assertRaises(ValueError):p.set_enabled(['audio_visual','telepathy'])
        self.assertEqual(p.enabled_names(),['audio_visual','memory_context'])


class PerSpeakerCandidateTests(unittest.TestCase):
    """Three people talking is three auditory candidates, not one loud track.

    The ear used to hand up a single dominant track per frame, so attention had
    nothing to choose between and a cocktail-party focus could not switch to
    anybody: there was only ever one audio candidate in the competition.
    """

    def speaker(self, speaker_id, age_ms, person=None, azimuth=None, confidence=.5):
        return {'speaker_id': speaker_id, 'person_id': person, 'age_ms': age_ms,
                'last_heard_ms': 1000 - age_ms, 'azimuth_deg': azimuth,
                'identity_confidence': confidence, 'speech_ms': 1200, 'windows': 4}

    def run_adapter(self, speakers, cycles=12):
        """Activation is an integrator, so one cycle shows nothing."""
        from robot_attention_perception.competition_adapter import CompetitionAdapter
        from robot_attention_perception.types import AcousticTrack
        adapter = CompetitionAdapter()
        result = {}
        for index in range(cycles):
            stamp = 1000 + index * 50
            track = AcousticTrack('t1', stamp, voice_activity=True, speech_probability=.9, clarity=.8)
            rows = [dict(row, last_heard_ms=stamp - int(row['age_ms'])) for row in speakers]
            voice = {'speakers': {'speakers': rows, 'active': [row['speaker_id'] for row in rows]},
                     'last_streaming_transcript': {}, 'audio_scene': {}}
            result = adapter.update(stamp, [], [track], voice, context={'sources': {}})
        return {key: row for key, row in (result.get('candidates') or {}).items()
                if row.get('modality') == 'audio'}

    def test_each_speaker_becomes_its_own_candidate(self):
        rows = self.run_adapter([self.speaker('天行', 100, person='天行', confidence=.7),
                                 self.speaker('spk_02', 1200, azimuth=40),
                                 self.speaker('spk_03', 2800)])

        self.assertEqual(set(rows), {'audio:天行', 'audio:spk_02', 'audio:spk_03'})

    def test_the_one_talking_now_outweighs_the_ones_who_just_stopped(self):
        rows = self.run_adapter([self.speaker('天行', 100, person='天行', confidence=.7),
                                 self.speaker('spk_02', 2500)])

        self.assertGreater(rows['audio:天行']['a'], rows['audio:spk_02']['a'])

    def test_with_no_speaker_model_it_still_reports_the_microphone_track(self):
        rows = self.run_adapter([])

        self.assertEqual(set(rows), {'audio:t1'})
