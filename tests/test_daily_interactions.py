"""Replay sensor event sequences through fusion -> turn gate -> Brain -> TTS."""
import sys,time,unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'biomimetic-brain-test'))
from brain.runtime import BrainRuntime
from robot_attention_perception.interaction_fusion import InteractionFusion
from robot_attention_perception.conversation import ConversationObserver
from robot_attention_perception.types import VisionPerson,AcousticTrack


class DailyInteractionTests(unittest.TestCase):
    def setUp(self):
        self.fusion=InteractionFusion();self.observer=ConversationObserver();self.brain=BrainRuntime(memory_path=':memory:')
        self.brain.memory.search=Mock(return_value=[]);self.brain.llm=Mock()
        self.brain.llm.respond.return_value='按你们刚才讨论的安排，下午三点比较合适。'
        self.brain.tts=Mock(enabled=True);self.brain.tts.speak.return_value=True
        self.addCleanup(self.brain.stop)
        self.owner=VisionPerson('owner',100000,role='owner',face_id='face_owner',face_visible=True,face_confidence=.95,
                                identity_confidence=.9,gaze_score=.9,body_facing_score=.9,azimuth_deg=0)
        self.other=replace(self.owner,person_id='other',role='known',face_id='face_other',azimuth_deg=25)
        self.none={'target_id':'none','listen':False,'addressed_to_robot':False,'confidence':0}
        self.state={}
    def tick(self,stamp,people,tracks=(),voice=None,proposal=None,reflex=None):
        people=[replace(p,stamp_ms=stamp) for p in people];tracks=[replace(t,stamp_ms=stamp) for t in tracks]
        voice=dict(voice or {});voice.setdefault('last_vad',{'active':bool(tracks),'stamp_ms':stamp})
        gate=self.fusion.update(stamp,proposal or self.none,people,tracks,voice,reflex)
        self.observer.update(stamp,voice,tracks,people,bool((voice.get('playback') or {}).get('speaking')),gate)
        self.state={'stamp_ms':stamp,'attention':gate,'voice':voice,'utterances':list(self.observer.turns),
                    'interruption_candidate':self.observer.interruption,'reflex_attention':reflex,
                    'person_hypotheses':[{'resolved_person_id':p.person_id,'role':p.role,'active':True,'identity_confidence':p.identity_confidence} for p in people]}
        with patch('brain.runtime.now_ms',return_value=stamp):self.brain.step(self.state)
        return gate
    def wait_reply(self):
        until=time.monotonic()+2
        while self.brain.jobs and time.monotonic()<until:time.sleep(.01)
        self.assertFalse(self.brain.jobs)
    def utterance(self,key,text,person,start,end,**extra):
        return dict(utterance_id=key,track_id=key,text=text,is_final=True,started_ms=start,ended_ms=end,emitted_ms=end+200,
                    speaker={'speaker_id':person,'speaker_role':'owner' if person=='owner' else 'known','similarity':.9},**extra)
    def talk(self,key,text,person,start,people,bearing):
        track=AcousticTrack(key,start,True,.95,.8,azimuth_deg=bearing)
        for stamp in range(start,start+601,100):self.tick(stamp,people,[track])
        transcript=self.utterance(key,text,person,start,start+600)
        self.tick(start+800,[replace(p,lip_motion=False) for p in people],voice={'transcripts':[transcript]})
        self.wait_reply()
        return self.observer.turns[-1]

    def test_silent_gaze_opens_once_through_attention_not_a_second_rule(self):
        """Somebody familiar looking at the robot and saying nothing eventually
        gets spoken to. It used to be Brain's own owner rule that did this, in
        parallel with attention's invitation, and both could fire."""
        for stamp in range(100000,105800,100):self.tick(stamp,[self.owner])
        self.brain.tts.speak.assert_not_called()
        self.assertEqual(self.state['attention']['interaction_phase'],'VISUAL_FOCUS')
        for stamp in range(105800,112000,100):
            self.tick(stamp,[self.owner])
        self.wait_reply()
        self.assertLessEqual(self.brain.tts.speak.call_count,1)
        if self.brain.tts.speak.call_count:
            kwargs=self.brain.llm.respond.call_args.kwargs
            speaker=(kwargs.get('perception') or {}).get('current_speaker') or {}
            self.assertIsNone(kwargs.get('proactive'))
            self.assertTrue(speaker.get('attention_trigger'))


    def test_bystander_discussion_then_owner_asks_isnt_it(self):
        side=[replace(self.owner,gaze_score=.1,lip_motion=True),replace(self.other,gaze_score=.1)]
        a=self.talk('a','我们下午三点去公园','owner',100000,side,0)
        b=self.talk('b','那时候太阳没有那么晒','other',101000,[replace(side[0],lip_motion=False),replace(side[1],lip_motion=True)],25)
        self.assertFalse(a['attention']['listen']);self.assertFalse(b['attention']['listen'])
        self.brain.tts.speak.assert_not_called()
        q=self.talk('q','你说是吧','owner',102000,[replace(self.owner,lip_motion=True),side[1]],0)
        self.assertTrue(q['attention']['listen']);self.assertEqual(q['person_id'],'owner')
        self.brain.tts.speak.assert_called_once()
        request=self.brain.llm.respond.call_args
        self.assertEqual(request.args[0],'你说是吧');self.assertIsNone(request.kwargs['proactive'])
        recent=request.kwargs['perception']['current_speaker']['recent_conversation']
        self.assertEqual([x['person_id'] for x in recent],['owner','other'])
        for stamp in range(103000,109000,100):self.tick(stamp,[self.owner,self.other])
        self.brain.tts.speak.assert_called_once()

    def test_upstream_permission_without_current_evidence_is_rejected(self):
        stale={'target_id':'person:owner','person_id':'owner','listen':True,'addressed_to_robot':True,'confidence':.99,'source_track_id':'old'}
        for stamp in range(100000,101000,100):gate=self.tick(stamp,[self.owner],proposal=stale)
        self.assertEqual(gate['interaction_phase'],'VISUAL_FOCUS');self.assertFalse(gate['addressed_to_robot'])
        self.brain.tts.speak.assert_not_called()

    def test_robot_echo_is_never_a_new_directed_call(self):
        track=AcousticTrack('echo',100000,True,.99,.8,self_echo_probability=.95)
        for stamp in range(100000,100600,100):self.tick(stamp,[replace(self.owner,lip_motion=True)],[track])
        t=self.utterance('echo','小圆请回答我','owner',100000,100500,source='robot_echo',self_echo_probability=.95)
        self.tick(100700,[self.owner],voice={'transcripts':[t]})
        self.brain.tts.speak.assert_not_called()
        self.assertEqual(len(self.brain.state['heard']),0)

    def test_voice_identity_wins_without_merging_conflicting_face(self):
        turn=self.talk('conflict','刚才约了几点','other',100000,[replace(self.owner,lip_motion=True)],0)
        self.assertTrue(turn['identity_conflict']);self.assertEqual(turn['person_id'],'other')
        self.assertEqual(turn['visual_person_id'],'owner')

    def test_look_during_reply_does_not_steal_but_addressed_speech_can(self):
        for stamp in range(100000,100500,100):self.tick(stamp,[replace(self.owner,lip_motion=True)],[AcousticTrack('a',stamp,True,.9,.8,azimuth_deg=0)])
        people=[replace(self.owner,gaze_score=.1),replace(self.other,gaze_score=.99)]
        for stamp in range(100500,101500,100):gate=self.tick(stamp,people,voice={'playback':{'speaking':True}})
        self.assertEqual(gate['person_id'],'owner');self.assertEqual(gate['interaction_phase'],'RESPONDING')
        for stamp in range(101500,102100,100):gate=self.tick(stamp,[people[0],replace(people[1],lip_motion=True)],[AcousticTrack('b',stamp,True,.95,.8,azimuth_deg=25)],voice={'playback':{'speaking':True},'last_streaming_transcript':{'text':'我插一句','emitted_ms':stamp,'is_final':False,'source':'microphone'}})
        self.assertEqual(gate['person_id'],'other');self.assertEqual(gate['interaction_phase'],'LISTENING')
        self.assertIsNotNone(self.observer.interruption)

    def test_loud_event_releases_then_visual_focus_returns(self):
        for stamp in range(100000,101000,100):self.tick(stamp,[self.owner])
        reflex={'id':'bang','active':True,'kind':'orient','stamp_ms':101000,'source':'audio','target_id':'sound:bang'}
        gate=self.tick(101000,[self.owner],reflex=reflex)
        self.assertEqual(gate['interaction_phase'],'ORIENTING');self.assertFalse(gate['listen'])
        for stamp in range(102800,103700,100):gate=self.tick(stamp,[self.owner])
        self.assertEqual(gate['interaction_phase'],'VISUAL_FOCUS')
        self.brain.tts.speak.assert_not_called()

    def test_normal_volume_name_call_without_visible_people_focuses_source(self):
        t=AcousticTrack('caller',100000,True,.85,.6,azimuth_deg=-45)
        voice={'last_streaming_transcript':{'track_id':'caller','text':'小圆','emitted_ms':100000,'source':'microphone'},
               'audio_scene':{'acoustic':{'stamp_ms':100000,'raw_rms_dbfs':-45,'direction_valid':True,'direction_deg':-45}}}
        gate=self.tick(100000,[],[t],voice)
        self.assertEqual(gate['target_id'],'sound:caller');self.assertTrue(gate['addressed_to_robot'])
        self.assertEqual(gate['azimuth_deg'],-45)
        final=self.utterance('caller','小圆','owner',100000,100500,source='microphone',direction={'direction_valid':True,'direction_deg':-45})
        self.tick(100700,[],voice={'last_transcript':final,'transcripts':[final]});self.wait_reply()
        self.brain.tts.speak.assert_called_once();self.assertEqual(self.brain.tts.speak.call_args.args[0],'我在，你说。')
        self.assertEqual(self.state['attention']['interaction_phase'],'AUDIO_FOCUS')
        self.assertFalse(self.state['attention']['addressed_to_robot'])
        self.assertFalse(any(e['kind']=='startle_attention' for e in self.brain.state['timeline']))

    def test_third_party_name_mention_is_not_a_call(self):
        t=AcousticTrack('other',100000,True,.85,.6,azimuth_deg=40)
        for stamp in range(100000,100700,100):
            gate=self.tick(stamp,[],[t],{'last_streaming_transcript':{'track_id':'other','text':'昨天小圆回答过这个问题','emitted_ms':stamp,'source':'microphone'}})
        self.assertFalse(gate['addressed_to_robot'])
        self.assertNotEqual(gate['interaction_phase'],'AUDIO_FOCUS')
        self.brain.tts.speak.assert_not_called()

    def test_offscreen_named_question_uses_speaker_memory_and_unknown_direction(self):
        self.brain.memory.search.return_value=[{'trace_id':'past','text':'主人喜欢公园','entity':{'person_id':'owner'}}]
        final=self.utterance('call','小圆，我之前说喜欢哪里','owner',100000,100600,source='microphone',direction={'direction_valid':False,'direction_deg':60})
        self.tick(100800,[],voice={'last_transcript':final,'transcripts':[final]});self.wait_reply()
        self.assertIsNone(self.state['attention']['azimuth_deg'])
        self.assertEqual(self.brain.llm.respond.call_args.args[1][0]['text'],'主人喜欢公园')
        self.assertEqual(self.observer.turns[-1]['person_id'],'owner')
        self.assertEqual(self.observer.turns[-1]['visual_person_id'],'unknown')

    def test_known_voice_can_finish_question_when_lip_measurement_is_missing(self):
        for stamp in range(100000,101000,100):self.tick(stamp,[self.owner])
        turn=self.talk('known','刚才我们约了几点','owner',101000,[self.owner],0)
        self.assertTrue(turn['attention']['listen'])
        self.assertIn('confirmed_voice_at_engaged_target',turn['attention']['reasons'])
        self.brain.tts.speak.assert_called_once()

    def test_later_eye_contact_does_not_replay_an_old_bystander_sentence(self):
        side=replace(self.owner,gaze_score=.1,lip_motion=True)
        t=AcousticTrack('ambient',100000,True,.9,.8,azimuth_deg=0)
        for stamp in range(100000,100600,100):self.tick(stamp,[side],[t])
        for stamp in range(100600,101600,100):self.tick(stamp,[self.owner])
        final=self.utterance('ambient','下午去公园吧','owner',100000,100500)
        final['emitted_ms']=101600
        self.tick(101600,[self.owner],voice={'transcripts':[final]})
        self.brain.tts.speak.assert_not_called()
        self.assertFalse(self.observer.turns[-1]['attention']['listen'])

    def test_completed_partial_cannot_retrigger_name_call_on_new_audio(self):
        t=AcousticTrack('same',100000,True,.9,.8)
        gate=self.tick(100000,[],[t],{'last_streaming_transcript':{'track_id':'same','text':'小圆','is_final':True,'emitted_ms':99900}})
        self.assertFalse(gate['addressed_to_robot'])
        self.assertIsNone(self.observer.interruption)
