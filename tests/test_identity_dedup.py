import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from robot_attention_perception.live_runtime import LivePerceptionRuntime

class IdentityDedupTests(unittest.TestCase):
    def test_profile_does_not_repeat_per_utterance_and_has_stable_trace_on_restart(self):
        def runtime():
            r=LivePerceptionRuntime.__new__(LivePerceptionRuntime)
            r.config=SimpleNamespace(subject_id='s',robot_id='r',session_id='test')
            r.memory=Mock();return r
        profile={'speaker_id':'stranger_a','speaker_role':'stranger','finalized_identity':True,'identity_embedding':[.1]*192,'embedding':[.2]*192,'stamp_ms':1}
        a=runtime();a._record_speaker_identity(1,{'last_speaker':profile})
        a._record_speaker_identity(2,{'last_speaker':dict(profile,stamp_ms=2,similarity=.8)})
        a.memory.submit.assert_called_once()
        b=runtime();b._record_speaker_identity(3,{'last_speaker':dict(profile,stamp_ms=3)})
        self.assertEqual(a.memory.submit.call_args.args[0]['trace_id'],b.memory.submit.call_args.args[0]['trace_id'])
