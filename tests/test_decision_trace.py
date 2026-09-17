import json,tempfile,unittest
from pathlib import Path
from robot_attention_perception.decision_trace import DecisionTrace

class TraceTests(unittest.TestCase):
    def test_rotates_and_flushes_numeric_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.jsonl';trace=DecisionTrace(path,max_bytes=100,backups=2)
            for i in range(15):trace.submit({'stamp_ms':i,'phase':'VISUAL_FOCUS'})
            trace.close()
            self.assertFalse(trace.error)
            files=list(Path(tmp).iterdir());self.assertLessEqual(len(files),3)
            rows=[json.loads(line) for p in files for line in p.read_text().splitlines()]
            self.assertTrue(any(r['stamp_ms']==14 for r in rows))
