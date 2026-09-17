"""The cases that used to need scripted handling downstream, decided in one place."""
import unittest

from robot_attention_perception.interaction_fusion import InteractionFusion, finalize_gate
from robot_attention_perception.types import AcousticTrack, VisionPerson

ALL_SOURCES = ('audio_visual', 'memory_context', 'linguistic_context')


def context(person_memory=None, language=None, speaker=None, event='', sources=ALL_SOURCES):
    return {'sources': {name: {'enabled': name in sources} for name in
                        ('audio_visual', 'memory_context', 'linguistic_context', 'internal_state')},
            'crossmodal_enabled': 'audio_visual' in sources, 'goals': [], 'importance': {}, 'motivation': {},
            'person_memory': person_memory or {}, 'linguistic': language or {},
            'linguistic_speaker': speaker, 'event_id': event, 'known_people': []}


def looker(stamp, person='owner'):
    return VisionPerson(person, stamp, face_id='f1', face_visible=True, face_confidence=.9, gaze_score=.9,
                        body_facing_score=.9, identity_confidence=.8, role='known')


def voice_track(stamp, person='owner', track='t1'):
    return AcousticTrack(track, stamp, voice_activity=True, speech_probability=.9, clarity=.8,
                         speaker_label=person, speaker_similarity=.9)


class EngagementTests(unittest.TestCase):
    def setUp(self):
        self.fusion = InteractionFusion()

    def run_frames(self, start, count, people=(), tracks=(), **kwargs):
        state = None
        for index in range(count):
            stamp = start + index * 50
            state = self.fusion.update(stamp, {'algorithm': 'test'},
                                       [person for person in people] if not callable(people) else people(stamp),
                                       [track for track in tracks] if not callable(tracks) else tracks(stamp),
                                       {}, None, context=kwargs.get('context') or context())
        return state

    def test_silent_gaze_from_a_participant_opens_a_turn_without_addressing_it(self):
        memory = {'owner': {'memory_participant': 1., 'memory_recency': 1.}}
        state = self.run_frames(1000, 50, people=lambda stamp: [looker(stamp)],
                                context=context(person_memory=memory))
        self.assertEqual(state['interaction_phase'], 'INVITED')
        self.assertEqual(state['person_id'], 'owner')
        # Nothing was said, so this must not be treated as speech to the robot.
        self.assertFalse(state['addressed_to_robot'])
        self.assertIn('silent_invitation', state['reasons'])
        self.assertEqual(state['engagement']['state'], 'INVITED')

    def test_the_same_gaze_without_the_memory_source_stays_visual_focus(self):
        memory = {'owner': {'memory_participant': 1., 'memory_recency': 1.}}
        state = self.run_frames(1000, 50, people=lambda stamp: [looker(stamp)],
                                context=context(person_memory=memory, sources=('audio_visual', 'linguistic_context')))
        self.assertNotEqual(state['interaction_phase'], 'INVITED')
        self.assertFalse(state['addressed_to_robot'])

    def test_a_stranger_staring_is_never_an_invitation(self):
        state = self.run_frames(1000, 60, people=lambda stamp: [looker(stamp, 'guest')], context=context())
        self.assertNotEqual(state['interaction_phase'], 'INVITED')

    def test_an_answer_with_the_head_turned_away_is_addressed_to_the_robot(self):
        payload = context(person_memory={'owner': {'memory_expected_answer': 1.}},
                          language={'lang_answer_continuation': 1.}, speaker='owner', event='utt-9:final')
        state = self.run_frames(1000, 6, tracks=lambda stamp: [voice_track(stamp)], context=payload)
        self.assertEqual(state['interaction_phase'], 'LISTENING')
        self.assertTrue(state['addressed_to_robot'])
        self.assertEqual(state['person_id'], 'owner')
        self.assertIn('expected_answer_to_robot_question', state['reasons'])
        self.assertGreaterEqual(state['confidence'], .62)

        # The same live evidence must carry the final sentence through the gate.
        history = [(state['stamp_ms'], state)]
        final = finalize_gate({'track_id': 't1', 'text': '是的', 'started_ms': state['stamp_ms'] - 200,
                               'speaker': {'speaker_id': 'owner', 'similarity': .9}},
                              history, state['stamp_ms'])
        self.assertTrue(final['addressed_to_robot'])
        self.assertIn('final_from_live_evidence', final['reasons'])

    def test_an_utterance_aimed_at_another_human_is_not_taken_up(self):
        payload = context(person_memory={'owner': {'memory_participant': 1., 'memory_recency': 1.}},
                          language={'lang_addresses_other': 1.}, speaker='owner', event='utt-4:final')
        state = self.run_frames(1000, 20, tracks=lambda stamp: [voice_track(stamp)], context=payload)
        self.assertFalse(state['addressed_to_robot'])
        self.assertNotEqual(state['interaction_phase'], 'LISTENING')

    def test_an_expected_answer_from_a_different_voice_is_refused(self):
        payload = context(person_memory={'owner': {'memory_expected_answer': 1.}},
                          language={'lang_answer_continuation': 1.}, speaker='owner', event='utt-9:final')
        state = self.run_frames(1000, 6, tracks=lambda stamp: [voice_track(stamp, 'guest', 't2')], context=payload)
        self.assertFalse(state['addressed_to_robot'])

    def test_turning_to_us_mid_conversation_is_the_normal_case_for_an_invitation(self):
        """Somebody turning to the robot while the others keep talking is what
        this is for, so other voices must not block it. Only the inviter being
        the one speaking does."""
        memory = {'owner': {'memory_participant': 1., 'memory_recency': 1.}}
        others = self.run_frames(1000, 50, people=lambda stamp: [looker(stamp)],
                                 tracks=lambda stamp: [voice_track(stamp, 'guest', 't9')],
                                 context=context(person_memory=memory))
        self.assertEqual(others['interaction_phase'], 'INVITED')
        self.assertEqual(others['person_id'], 'owner')
        self.assertFalse(others['addressed_to_robot'])

        themselves = self.run_frames(1000, 50, people=lambda stamp: [looker(stamp)],
                                     tracks=lambda stamp: [voice_track(stamp, 'owner', 't1')],
                                     context=context(person_memory=memory))
        self.assertNotEqual(themselves['interaction_phase'], 'INVITED')

    def test_an_open_exchange_survives_losing_sight_of_them(self):
        """A follow-up from somebody who was addressing us moments ago, now off
        camera. Requiring them to be visible is what left this unanswered."""
        payload = context(person_memory={'owner': {'memory_participant': 1., 'memory_recency': 1.,
                                                   'memory_open_exchange': .82}})
        state = self.run_frames(1000, 40, tracks=lambda stamp: [voice_track(stamp)], context=payload)
        self.assertEqual(state['interaction_phase'], 'LISTENING')
        self.assertTrue(state['addressed_to_robot'])
        self.assertEqual(state['person_id'], 'owner')
        self.assertIn('engaged_speaker', state['reasons'])

    def test_the_follow_up_window_fades_and_never_covers_a_bystander(self):
        faded = context(person_memory={'owner': {'memory_participant': 1., 'memory_recency': 1.,
                                                 'memory_open_exchange': .13}})
        self.assertFalse(self.run_frames(1000, 40, tracks=lambda stamp: [voice_track(stamp)],
                                         context=faded)['addressed_to_robot'])
        # Somebody who has been talking in the room but never to us.
        bystander = context(person_memory={'owner': {'memory_participant': 1., 'memory_recency': 1.}})
        self.assertFalse(self.run_frames(1000, 40, tracks=lambda stamp: [voice_track(stamp)],
                                         context=bystander)['addressed_to_robot'])

    def test_an_unreliable_voice_cannot_continue_somebody_else_exchange(self):
        payload = context(person_memory={'owner': {'memory_participant': 1., 'memory_recency': 1.,
                                                   'memory_open_exchange': .82}})
        state = self.run_frames(1000, 40, context=payload,
                                tracks=lambda stamp: AcousticTrack('t1', stamp, voice_activity=True,
                                                                   speech_probability=.9, clarity=.8,
                                                                   speaker_label='owner',
                                                                   speaker_similarity=.5) and
                                [AcousticTrack('t1', stamp, voice_activity=True, speech_probability=.9,
                                               clarity=.8, speaker_label='owner', speaker_similarity=.5)])
        self.assertFalse(state['addressed_to_robot'])


if __name__ == '__main__':
    unittest.main()


class CrossModalBindingTests(unittest.TestCase):
    """Binding a moving mouth to a voice must not depend on an uncalibrated array."""

    def run_frames(self, count, people, tracks):
        fusion = InteractionFusion()
        state = None
        for index in range(count):
            stamp = 1000 + index * 50
            state = fusion.update(stamp, {'algorithm': 'test'}, people(stamp), tracks(stamp), {}, None,
                                  context=context())
        return state

    def speaker(self, stamp, azimuth=None, person='guest'):
        return VisionPerson(person, stamp, face_id='f1', face_visible=True, face_confidence=.9, gaze_score=.9,
                            body_facing_score=.9, lip_motion=True, identity_confidence=.3, azimuth_deg=azimuth)

    def unknown_voice(self, stamp, azimuth=None):
        return AcousticTrack('t1', stamp, voice_activity=True, speech_probability=.9, clarity=.8,
                             speaker_label='unknown', speaker_similarity=.0, azimuth_deg=azimuth)

    def test_one_mouth_and_one_voice_bind_whatever_the_array_reports(self):
        """The regression: a microphone array started reporting angles, the
        single-person fallback only applied when it did not, and an unenrolled
        person speaking straight at the camera stopped being answered."""
        for label, vision_az, audio_az in (('array only', None, 30.),
                                           ('bearings disagree by 70 degrees', -40., 30.),
                                           ('no bearings at all', None, None),
                                           ('bearings agree', 28., 30.)):
            state = self.run_frames(30, lambda stamp, a=vision_az: [self.speaker(stamp, a)],
                                    lambda stamp, a=audio_az: [self.unknown_voice(stamp, a)])
            self.assertTrue(state['addressed_to_robot'], label)
            self.assertEqual(state['person_id'], 'guest', label)

    def test_direction_still_separates_a_crowd(self):
        def people(stamp):
            return [self.speaker(stamp, -40.),
                    VisionPerson('other', stamp, face_id='f2', face_visible=True, face_confidence=.9,
                                 gaze_score=.9, body_facing_score=.9, lip_motion=True,
                                 identity_confidence=.3, azimuth_deg=85.)]
        state = self.run_frames(30, people, lambda stamp: [self.unknown_voice(stamp, 30.)])
        self.assertFalse(state['addressed_to_robot'])

    def test_a_bearing_disagreement_is_reported_rather_than_hidden(self):
        fusion = InteractionFusion()
        for index in range(10):
            stamp = 1000 + index * 50
            fusion.update(stamp, {'algorithm': 'test'}, [self.speaker(stamp, -40.)],
                          [self.unknown_voice(stamp, 30.)], {}, None, context=context())
        notes = fusion.competition.result.get('binding_notes') or []
        self.assertTrue(notes)
        self.assertGreater(notes[0]['bearing_gap_deg'], 30)
        self.assertEqual(notes[0]['person_id'], 'guest')


class MeasuredThresholdTests(unittest.TestCase):
    """Thresholds tuned on one camera silently disabled whole paths on another."""

    def test_a_face_the_detector_accepted_counts_as_present(self):
        """The detector reports at 0.80. Field scores sat at 0.81 to 0.83, and
        requiring 0.85 downstream made a clearly detected face count as absent,
        which removed the invitation, lip motion and cross-modal binding at once."""
        fusion = InteractionFusion()
        payload = context(person_memory={'owner': {'memory_participant': 1., 'memory_recency': 1.}})
        state = None
        for index in range(60):
            stamp = 1000 + index * 50
            person = VisionPerson('owner', stamp, face_id='f1', face_visible=True, face_confidence=.82,
                                  gaze_score=.9, body_facing_score=.9, identity_confidence=.55)
            state = fusion.update(stamp, {'algorithm': 'test'}, [person], [], {}, None, context=payload)
        self.assertEqual(state['interaction_phase'], 'INVITED')
        self.assertEqual(state['person_id'], 'owner')

    def test_a_middling_identity_still_supports_the_invitation(self):
        """Field identity confidence for an enrolled person measured 0.50 to
        0.59, so the paths that matter must not need more than that."""
        fusion = InteractionFusion()
        payload = context(person_memory={'owner': {'memory_participant': 1., 'memory_recency': 1.}})
        state = None
        for index in range(60):
            stamp = 1000 + index * 50
            person = VisionPerson('owner', stamp, face_id='f1', face_visible=True, face_confidence=.82,
                                  gaze_score=.9, body_facing_score=.9, identity_confidence=.52)
            other = AcousticTrack('t9', stamp, voice_activity=True, speech_probability=.9, clarity=.8,
                                  speaker_label='guest', speaker_similarity=.55)
            state = fusion.update(stamp, {'algorithm': 'test'}, [person], [other], {}, None, context=payload)
        self.assertEqual(state['interaction_phase'], 'INVITED')

    def test_somebody_we_cannot_identify_cannot_borrow_a_history(self):
        """The other half of the same rule: memory hangs on an identity, so a
        face we cannot place must not inherit somebody else's participation."""
        fusion = InteractionFusion()
        payload = context(person_memory={'owner': {'memory_participant': 1., 'memory_recency': 1.}})
        state = None
        for index in range(60):
            stamp = 1000 + index * 50
            person = VisionPerson('owner', stamp, face_id='f1', face_visible=True, face_confidence=.82,
                                  gaze_score=.9, body_facing_score=.9, identity_confidence=.05)
            state = fusion.update(stamp, {'algorithm': 'test'}, [person], [], {}, None, context=payload)
        self.assertNotEqual(state['interaction_phase'], 'INVITED')


class AnswerContinuityTest(unittest.TestCase):
    """The robot asked somebody a question and a voice answered.

    Field trace: the owner leaned back out of frame, said "我想出去玩", was
    recorded as stranger_9a227aac8372 because the voiceprint scored 0.0 off-axis,
    the answer was dropped as unaddressed, and the robot asked its question
    again. Nothing in the audio identified him — but the conversation did.
    """

    def history(self, person="天行", asked_ms=10000, until_ms=12000):
        gates = [(asked_ms, {"person_id": person, "interaction_phase": "RESPONDING",
                             "reasons": ["active_reply_target"], "listen": True,
                             "addressed_to_robot": False, "confidence": .8,
                             "target_kind": "person", "source_track_id": None})]
        gates += [(stamp, {"person_id": None, "interaction_phase": "IDLE", "reasons": [],
                           "listen": False, "addressed_to_robot": False, "confidence": 0.,
                           "target_kind": "none", "source_track_id": None})
                  for stamp in range(asked_ms + 100, until_ms, 100)]
        return gates

    def transcript(self, speaker_id="stranger_9a227aac8372", similarity=0.0, end=12000):
        return {"track_id": "voice_269", "text": "我想出去玩", "started_ms": end - 900, "ended_ms": end,
                "is_final": True, "speaker": {"speaker_id": speaker_id, "similarity": similarity}}

    def test_an_answer_with_no_usable_voiceprint_is_still_our_answer(self):
        gate = finalize_gate(self.transcript(), self.history(), 12000)

        self.assertTrue(gate["addressed_to_robot"])
        self.assertEqual(gate["person_id"], "天行")
        self.assertIn("answer_to_our_question_unverified_voice", gate["reasons"])

    def test_it_expires_rather_than_claiming_anything_said_later(self):
        gate = finalize_gate(self.transcript(end=60000), self.history(until_ms=60000), 60000)

        self.assertFalse(gate["addressed_to_robot"])

    def test_a_voiceprint_that_names_somebody_else_wins(self):
        gate = finalize_gate(self.transcript(speaker_id="二丫", similarity=.7), self.history(), 12000)

        self.assertNotIn("answer_to_our_question_unverified_voice", gate["reasons"])

    def test_somebody_else_holding_the_floor_cancels_it(self):
        gates = self.history()
        gates.append((11500, {"person_id": "二丫", "interaction_phase": "LISTENING", "reasons": [],
                              "listen": True, "addressed_to_robot": True, "confidence": .8,
                              "target_kind": "person", "source_track_id": "voice_9"}))
        gate = finalize_gate(self.transcript(), sorted(gates, key=lambda item: item[0]), 12000)

        self.assertFalse(gate["addressed_to_robot"])


class NameCallWindowTest(unittest.TestCase):
    """Being called by name puts the robot in listening mode, camera or not.

    Field trace: "小圆，明天是。" was granted, and "星期几？" one second later was
    refused for want of contemporaneous visual evidence — the rest of the same
    thought, dropped. Later, off camera, "小圆，天气怎么样" met the same wall.
    """

    def called(self, at=10000, until=11500):
        gates = [(at, {"person_id": None, "interaction_phase": "AUDIO_FOCUS", "focus_origin": "direct_call",
                       "reasons": ["name_call_attention", "waiting_for_called_speaker"], "listen": True,
                       "addressed_to_robot": False, "confidence": .8, "target_kind": "sound_source",
                       "source_track_id": "voice_260"})]
        gates += [(stamp, {"person_id": None, "interaction_phase": "IDLE", "reasons": [], "listen": False,
                           "addressed_to_robot": False, "confidence": 0., "target_kind": "none",
                           "source_track_id": None}) for stamp in range(at + 100, until, 100)]
        return gates

    def follow_up(self, end=11500, speaker_id="unknown"):
        return {"track_id": "voice_264", "text": "星期几", "started_ms": end - 700, "ended_ms": end,
                "is_final": True, "speaker": {"speaker_id": speaker_id, "similarity": 0.0}}

    def test_the_sentence_after_the_name_is_still_addressed_to_us(self):
        gate = finalize_gate(self.follow_up(), self.called(), 11500)

        self.assertTrue(gate["addressed_to_robot"])
        self.assertIn("within_name_call_window", gate["reasons"])

    def test_the_window_closes(self):
        gate = finalize_gate(self.follow_up(end=40000), self.called(until=40000), 40000)

        self.assertFalse(gate["addressed_to_robot"])

    def test_the_field_sequence_name_then_reply_then_the_actual_question(self):
        """12:10:29 "小圆。" granted; 12:10:31 robot answers "我在，你说。";
        12:10:33 "今天天气怎么样?" refused. The question is the whole point of
        calling the name, and by then the voiceprint had even named him."""
        gates = self.called(at=10000, until=11000)
        gates += [(stamp, {"person_id": None, "interaction_phase": "RESPONDING",
                           "reasons": ["active_reply_target"], "listen": True, "addressed_to_robot": False,
                           "confidence": .8, "target_kind": "person", "source_track_id": None})
                  for stamp in range(11000, 12000, 100)]
        question = {"track_id": "voice_9", "text": "今天天气怎么样", "started_ms": 13000, "ended_ms": 14000,
                    "is_final": True, "speaker": {"speaker_id": "天行", "similarity": .52}}

        gate = finalize_gate(question, gates, 14000)

        self.assertTrue(gate["addressed_to_robot"])
        self.assertEqual(gate["person_id"], "天行")


class NoWakeWordNeededTest(unittest.TestCase):
    """Talking to a robot that is looking at you does not need its name.

    Field trace 12:30:47: 天行, recognised, in frame, asks "今天天气怎么样？" and is
    refused as unaddressed. For a while the only thing that worked was saying the
    name first, which is not how anybody talks.
    """

    def attended(self, person="天行", end=12000):
        return [(stamp, {"person_id": person, "interaction_phase": "VISUAL_FOCUS",
                         "reasons": ["gaze_engaged"], "listen": True, "addressed_to_robot": False,
                         "confidence": .9, "target_kind": "person", "source_track_id": None})
                for stamp in range(end - 1400, end, 100)]

    def people(self, person="天行", end=12000, others=()):
        from robot_attention_perception.types import VisionPerson
        rows = []
        for stamp in range(end - 2000, end, 100):
            seen = [VisionPerson(person, stamp, face_visible=True, face_confidence=.9, gaze_score=.9,
                                 body_facing_score=.8, identity_confidence=.8, role='known',
                                 lip_motion=True)]
            seen += [VisionPerson(name, stamp, face_visible=True, face_confidence=.9, gaze_score=.1,
                                  body_facing_score=.5, identity_confidence=.7, role='known') for name in others]
            rows.append((stamp, seen))
        return rows

    def test_the_person_being_looked_at_does_not_have_to_say_the_name(self):
        transcript = {"track_id": "t9", "text": "今天天气怎么样", "started_ms": 11200, "ended_ms": 12000,
                      "is_final": True, "speaker": {"speaker_id": "unknown", "similarity": 0.0}}

        gate = finalize_gate(transcript, self.attended(), 12000, people_history=self.people())

        self.assertTrue(gate["addressed_to_robot"])
        self.assertEqual(gate["person_id"], "天行")
        self.assertIn("looked_at_the_robot_while_speaking", gate["reasons"])
        self.assertIn("looker_is_the_speaker", gate["reasons"])

    def test_two_people_both_facing_it_is_held_not_guessed(self):
        gates = self.attended()
        gates += [(11950, {"person_id": "二丫", "interaction_phase": "VISUAL_FOCUS", "reasons": [],
                           "listen": False, "addressed_to_robot": False, "confidence": .8,
                           "target_kind": "person", "source_track_id": None})]
        transcript = {"track_id": "t9", "text": "今天天气怎么样", "started_ms": 11200, "ended_ms": 12000,
                      "is_final": True, "speaker": {"speaker_id": "unknown", "similarity": 0.0}}

        from robot_attention_perception.types import VisionPerson
        both = [(stamp, [VisionPerson(name, stamp, face_visible=True, face_confidence=.9, gaze_score=.9,
                                      body_facing_score=.8, identity_confidence=.8, role='known',
                                      lip_motion=True) for name in ('天行', '二丫')])
                for stamp in range(10000, 12000, 100)]

        gate = finalize_gate(transcript, sorted(gates, key=lambda item: item[0]), 12000, people_history=both)

        self.assertFalse(gate["addressed_to_robot"])


class MidExchangeTest(unittest.TestCase):
    """Field trace 14:22 — the clearest refusal of the lot.

        14:22:49  天行 "眼睛痒。"              granted
        14:22:54  robot "眼睛痒可能是过敏哦。"
        14:22:59  天行 "怎么办呀？"            refused

    Five seconds into a conversation the robot itself started, the follow-up was
    refused because every cue on offer needed a face, and at that moment there
    was not a good one. Whether somebody happened to be facing the camera when
    they said "怎么办呀" has nothing to do with whom they meant.
    """

    def follow_up(self, end=14000):
        return {"track_id": "t9", "text": "怎么办呀", "started_ms": end - 700, "ended_ms": end,
                "is_final": True, "speaker": {"speaker_id": "unknown", "similarity": 0.0}}

    def test_a_follow_up_mid_exchange_is_addressed_to_us(self):
        gate = finalize_gate(self.follow_up(), [], 14000,
                             exchange={"person_id": "天行", "stamp_ms": 9000})

        self.assertTrue(gate["addressed_to_robot"])
        self.assertEqual(gate["person_id"], "天行")
        self.assertIn("mid_exchange_with_this_person", gate["reasons"])

    def test_the_exchange_goes_stale(self):
        gate = finalize_gate(self.follow_up(end=60000), [], 60000,
                             exchange={"person_id": "天行", "stamp_ms": 9000})

        self.assertFalse(gate["addressed_to_robot"])

    def test_somebody_else_speaking_ends_it(self):
        history = [(12000, {"person_id": "二丫", "interaction_phase": "LISTENING", "listen": True,
                            "addressed_to_robot": True, "confidence": .8, "reasons": [],
                            "target_kind": "person", "source_track_id": "t2"})]

        gate = finalize_gate(self.follow_up(), history, 14000,
                             exchange={"person_id": "天行", "stamp_ms": 9000})

        self.assertNotIn("mid_exchange_with_this_person", gate.get("reasons") or [])
