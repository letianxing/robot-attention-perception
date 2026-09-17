import unittest
from dataclasses import replace
from robot_attention_perception.conversation import ConversationObserver
from robot_attention_perception.types import AcousticTrack, VisionPerson


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.observer = ConversationObserver()
        self.track = AcousticTrack("voice1", 1000, True, .98, .95, azimuth_deg=0)
        self.person = VisionPerson(person_id="person1", stamp_ms=1000, azimuth_deg=0,
                                   face_visible=True, face_confidence=.99, gaze_score=.9,
                                   body_facing_score=.9, lip_motion=True)

    def test_directed_call_not_third_party_mention(self):
        for text, expected in [("小圆，等一下", True), ("他说机器人很有趣", False), ("下午去公园", False)]:
            gate = self.observer.gate({"text": text, "track_id": "voice1"}, self.track, [], 1000, True)
            self.assertEqual(gate["addressed_to_robot"], expected)

    def test_visual_attention_and_echo_rejection(self):
        gate = self.observer.gate({"text": "等一下"}, self.track, [self.person], 1000, True)
        self.assertTrue(gate["listen"])
        gate = self.observer.gate({"text": "小圆，等一下"}, replace(self.track, self_echo_probability=.95), [self.person], 1000, True)
        self.assertFalse(gate["listen"])

    def test_delayed_final_keeps_speaker_and_original_attention(self):
        self.observer.update(1000, {}, [self.track], [self.person], False)
        transcript = {"utterance_id": "u", "track_id": "voice1", "started_ms": 500, "ended_ms": 1000,
                      "emitted_ms": 2000, "text": "下午去公园", "is_final": True,
                      "speaker": {"speaker_id": "person1", "similarity": .9}}
        self.observer.update(2000, {"transcripts": [transcript]}, [], [], False)
        self.assertEqual(self.observer.turns[0]["person_id"], "person1")
        self.assertTrue(self.observer.turns[0]["attention"]["addressed_to_robot"])
        self.observer.update(2050, {"transcripts": [transcript]}, [], [], False)
        self.assertEqual(len(self.observer.turns), 1)

    def test_a_moving_mouth_alone_never_interrupts_the_robot(self):
        """要么不说，说了就说完：只有真的说出词来才抢得走话轮。
        机器人说话时人看着它、跟着动嘴是常态，那不是打断。"""
        for stamp in (1000,1100,1200,1300,1400,1500):
            self.observer.update(stamp,{'last_vad':{'active':True},'active_utterance':{'started_ms':900}},
                                 [replace(self.track,stamp_ms=stamp)],[replace(self.person,stamp_ms=stamp)],True)
        self.assertIsNone(self.observer.interruption)

    def test_words_spoken_over_the_robot_do_interrupt(self):
        for stamp in (1000,1100,1200,1300,1400,1500):
            partial={'text':'等一下我想说','emitted_ms':stamp,'is_final':False,'source':'microphone'}
            self.observer.update(stamp,{'last_vad':{'active':True},'active_utterance':{'started_ms':900},
                                        'last_streaming_transcript':partial},
                                 [replace(self.track,stamp_ms=stamp)],[replace(self.person,stamp_ms=stamp)],True)
        self.assertIsNotNone(self.observer.interruption)
    def test_our_own_playback_is_never_an_interruption(self):
        for stamp in (1000,1100,1200,1300,1400,1500):
            partial={'text':'等一下我想说','emitted_ms':stamp,'is_final':False,'source':'microphone'}
            self.observer.update(stamp,{'last_vad':{'active':True},'last_streaming_transcript':partial},
                                 [replace(self.track,stamp_ms=stamp,self_echo_probability=.95)],
                                 [replace(self.person,stamp_ms=stamp)],True)
        self.assertIsNone(self.observer.interruption)

    def test_echo_final_is_retained_but_never_addressed(self):
        transcript={'utterance_id':'echo','track_id':'voice1','ended_ms':1000,'emitted_ms':1000,'is_final':True,'text':'小圆，你好','source':'robot_echo'}
        self.observer.update(1000,{'transcripts':[transcript]},[self.track],[self.person],True)
        self.assertFalse(self.observer.turns[0]['attention']['listen'])
        self.assertEqual(self.observer.turns[0]['person_id'],'robot')

    def test_final_uses_same_live_gate_despite_local_fusion_disagreement(self):
        gate={'listen':True,'addressed_to_robot':True,'confidence':.8,'person_id':'person1','source_track_id':'voice1','reasons':['native_engaged']}
        self.observer.update(1000,{},[self.track],[self.person],False,gate)
        t={'utterance_id':'native','track_id':'voice1','text':'你觉得今天天气怎么样','started_ms':500,'ended_ms':1000,'emitted_ms':1500,'is_final':True}
        self.observer.update(1500,{'transcripts':[t]},[],[],False)
        final=self.observer.turns[0]['attention']
        self.assertTrue(final['listen']);self.assertIn('contemporaneous_live_gate',final['reasons'])

    def test_attention_turning_later_does_not_replay_bystander_question(self):
        t={'utterance_id':'old','track_id':'voice1','text':'下午几点走','ended_ms':1000,'emitted_ms':2000,'is_final':True}
        gate={'listen':True,'addressed_to_robot':True,'confidence':.9,'person_id':'person1','source_track_id':'voice1'}
        self.observer.update(1000,{},[self.track],[],False)
        self.observer.update(2000,{'transcripts':[t]},[self.track],[self.person],False,gate)
        self.assertFalse(self.observer.turns[0]['attention']['listen'])

    def test_fresh_voice_cluster_loses_to_a_face_the_camera_is_sure_about(self):
        """An auto-created cluster is a handle for a voice, not a person.

        In the field the owner asked about the weather, the voiceprint scored
        0.083 against his own enrolment (rank 18 of 21), a cluster was minted
        for the sentence with similarity 1.0 by fiat, and the turn went into
        memory as stranger_c3b41448 even though the face was identified. The
        face is independent evidence; the minted cluster is not evidence at all.
        """
        person = replace(self.person, person_id="天行", identity_confidence=.82)
        gate = {'listen': True, 'addressed_to_robot': True, 'confidence': .9, 'person_id': '天行',
                'source_track_id': 'voice1'}
        self.observer.update(1000, {}, [self.track], [person], False, gate)
        t = {'utterance_id': 'cluster', 'track_id': 'voice1', 'text': '今天天气怎么样', 'started_ms': 500,
             'ended_ms': 1000, 'emitted_ms': 1500, 'is_final': True,
             'speaker': {'speaker_id': 'stranger_c3b41448ec74', 'speaker_role': 'stranger', 'similarity': 1.0}}
        self.observer.update(1500, {'transcripts': [t]}, [self.track], [person], False)
        turn = self.observer.turns[0]
        self.assertEqual(turn['person_id'], '天行')
        self.assertEqual(turn['identity_source'], 'face_recognition')
        self.assertEqual(turn['identity_resolution'], 'face_over_voice_cluster')
        self.assertEqual(turn['voice_cluster_id'], 'stranger_c3b41448ec74')
        self.assertFalse(turn['identity_conflict'])

    def test_voice_cluster_still_names_the_speaker_when_nobody_is_on_camera(self):
        """Off camera the cluster is all there is, and separate voices must stay separate."""
        t = {'utterance_id': 'crowd', 'track_id': 'voice1', 'text': '下午几点走', 'started_ms': 500,
             'ended_ms': 1000, 'emitted_ms': 1500, 'is_final': True,
             'speaker': {'speaker_id': 'stranger_9ea9b338402b', 'speaker_role': 'stranger', 'similarity': 1.0}}
        self.observer.update(1000, {}, [self.track], [], False)
        self.observer.update(1500, {'transcripts': [t]}, [self.track], [], False)
        turn = self.observer.turns[0]
        self.assertEqual(turn['person_id'], 'stranger_9ea9b338402b')
        self.assertEqual(turn['identity_source'], 'voice_cluster')

    def test_conflicting_face_does_not_replace_reliable_voice_identity(self):
        gate={'listen':True,'addressed_to_robot':True,'confidence':.9,'person_id':'face_person','source_track_id':'voice1'}
        self.observer.update(1000,{},[self.track],[self.person],False,gate)
        t={'utterance_id':'conflict','track_id':'voice1','text':'几点出发','started_ms':500,'ended_ms':1000,'emitted_ms':1500,'is_final':True,'speaker':{'speaker_id':'voice_person','similarity':.8}}
        self.observer.update(1500,{'transcripts':[t]},[],[],False)
        turn=self.observer.turns[0]
        self.assertEqual(turn['person_id'],'voice_person')
        self.assertEqual(turn['visual_person_id'],'face_person')
        self.assertTrue(turn['identity_conflict'])


class TurningRoundTest(unittest.TestCase):
    """人群在镜头外说话，其中一个人回头看机器人。

    The case the field kept hitting. Nobody is on camera, so the voices are
    clusters; then one of them turns round, and the sentence they said a moment
    ago is filed under an identity nothing recognises. Binding it to the face
    that appears where that voice was coming from is what makes it answerable.
    """

    def setUp(self):
        self.observer = ConversationObserver()
        self.track = AcousticTrack("voice1", 1000, True, .98, .95, azimuth_deg=28)

    def crowd_member(self, azimuth, person_id="guest_2", confidence=.78):
        return VisionPerson(person_id=person_id, stamp_ms=4000, azimuth_deg=azimuth,
                            face_visible=True, face_confidence=.9, gaze_score=.88,
                            body_facing_score=.8, identity_confidence=confidence, lip_motion=False)

    def payload(self, azimuth=30, heard=3000):
        return {"speakers": {"speakers": [{"speaker_id": "stranger_dd", "person_id": None,
                                           "azimuth_deg": azimuth, "last_heard_ms": heard,
                                           "display_name": "陌生人1"}]}}

    def test_a_face_where_the_voice_was_is_the_same_person(self):
        self.observer.update(4000, self.payload(), [self.track], [self.crowd_member(30)], False)
        self.assertEqual(sorted(self.observer.voice_bindings["guest_2"]), ["stranger_dd"])

    def test_two_people_in_the_same_direction_bind_to_neither(self):
        people = [self.crowd_member(28, "guest_2"), self.crowd_member(34, "guest_3")]
        self.observer.update(4000, self.payload(), [self.track], people, False)
        self.assertEqual(self.observer.voice_bindings, {})

    def test_a_voice_from_the_other_side_of_the_room_is_somebody_else(self):
        self.observer.update(4000, self.payload(azimuth=-60), [self.track], [self.crowd_member(30)], False)
        self.assertEqual(self.observer.voice_bindings, {})

    def test_an_old_voice_is_not_bound_to_whoever_walks_in_later(self):
        self.observer.update(90000, self.payload(heard=3000), [self.track], [self.crowd_member(30)], False)
        self.assertEqual(self.observer.voice_bindings, {})

    def test_bindings_expire(self):
        self.observer.update(4000, self.payload(), [self.track], [self.crowd_member(30)], False)
        self.observer._bind_voice("guest_9", "stranger_zz", 4000 + 700000)
        self.assertNotIn("guest_2", self.observer.voice_bindings)
