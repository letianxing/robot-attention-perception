import unittest
from robot_attention_perception.affect_candidates import AffectCandidates
from robot_attention_perception.types import VisionPerson

def person(stamp, label='sad', valid=True, valence=-.6, pid='owner', conf=.8):
    return VisionPerson(pid, stamp, face_visible=True, face_confidence=.9, identity_confidence=conf,
                        emotion_valid=valid, emotion_label=label, emotion_valence=valence)

class AffectTests(unittest.TestCase):
    def feed(self, index, start, count, step=800, **kw):
        """Default spacing puts  samples across more than the minimum span."""
        for i in range(count):
            index.observe(start + i*step, [person(start + i*step, **kw)])
        return start + count*step

    def test_sustained_low_expression_during_silence_raises_one_candidate(self):
        a = AffectCandidates()
        end = self.feed(a, 1_000_000, 20)
        out = a.candidates(end, [person(end)], end - 30000)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['kind'], 'affect_event')
        self.assertEqual(out[0]['person_id'], 'owner')
        self.assertGreaterEqual(out[0]['detail']['samples'], 12)
        self.assertGreater(out[0]['detail']['negative_share'], .6)
        self.assertIn('不是情绪或健康诊断', out[0]['limits'])
        self.assertEqual(a.candidates(end+1000, [person(end+1000)], end-30000), [])

    def test_a_frown_at_video_rate_does_not_qualify_as_persistent(self):
        """Vision runs at 20 fps, so a sample count alone would let six tenths of
        a second through. The samples have to span real time."""
        a = AffectCandidates()
        for i in range(40):                      # 40 frames at 50 ms = 2 seconds
            a.observe(1_000_000 + i*50, [person(1_000_000 + i*50)])
        end = 1_000_000 + 40*50
        self.assertEqual(a.candidates(end, [person(end)], end-30000), [])
        for i in range(40, 240):                 # keep going past eight seconds
            a.observe(1_000_000 + i*50, [person(1_000_000 + i*50)])
        end = 1_000_000 + 240*50
        self.assertEqual(len(a.candidates(end, [person(end)], end-30000)), 1)

    def test_a_few_frames_or_a_passing_frown_is_not_a_mood(self):
        a = AffectCandidates()
        end = self.feed(a, 1_000_000, 6)
        self.assertEqual(a.candidates(end, [person(end)], end-30000), [])
        b = AffectCandidates()
        end = b and self.feed(b, 1_000_000, 20, label='happy', valence=.5)
        self.assertEqual(b.candidates(end, [person(end, label='happy')], end-30000), [])
        self.assertEqual(b.status()['reason'], 'no_sustained_low_expression')

    def test_invalid_classifier_frames_are_not_counted(self):
        a = AffectCandidates()
        end = self.feed(a, 1_000_000, 20, valid=False)
        self.assertEqual(a.candidates(end, [person(end)], end-30000), [])

    def test_it_stays_quiet_while_people_are_talking(self):
        a = AffectCandidates()
        end = self.feed(a, 1_000_000, 20)
        self.assertEqual(a.candidates(end, [person(end)], end-2000), [])
        self.assertEqual(a.status()['reason'], 'not_silent_long_enough')

    def test_an_unidentified_face_does_not_qualify(self):
        a = AffectCandidates()
        end = self.feed(a, 1_000_000, 20, conf=.2)
        self.assertEqual(a.candidates(end, [person(end, conf=.2)], end-30000), [])
        self.assertEqual(a.status()['reason'], 'nobody_identified_present')

    def test_it_can_be_switched_off(self):
        a = AffectCandidates(enabled=False)
        end = self.feed(a, 1_000_000, 20)
        self.assertEqual(a.candidates(end, [person(end)], end-30000), [])
        self.assertEqual(a.status()['reason'], 'disabled')

if __name__ == '__main__':
    unittest.main()
