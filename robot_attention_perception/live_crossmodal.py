"""Measured AV pulse binding and limited articulatory conflict evidence."""
from collections import deque
import re

class LiveCrossmodal:
    def __init__(self):
        self.tones=deque(maxlen=100);self.flashes=deque(maxlen=100);self.seen=deque(maxlen=1000)
        self.mouth=deque(maxlen=1000);self.results=deque(maxlen=50)
    def update(self,stamp,voice,vision,turns):
        raw=vision.get('state') or vision
        frame_ms=int(raw.get('stamp',0)*1000)
        for p in raw.get('people',[]):
            if p.get('lip_motion_valid'):
                self.mouth.append((frame_ms,p.get('person_id'),float(p.get('mouth_open_ratio') or 0)))
        for event in (voice.get('audio_scene') or {}).get('events',[]):
            key=event.get('id')
            if event.get('kind')=='tone_pulse' and key not in self.seen:
                self.seen.append(key);self.tones.append(event)
        for event in raw.get('flash_events',[]):
            if event['id'] not in self.seen:self.seen.append(event['id']);self.flashes.append(event)
        for flash in self.flashes:
            key='bound:'+flash['id']
            if key in self.seen or stamp-flash['stamp_ms']<400:continue
            self.seen.append(key)
            tones=[t for t in self.tones if abs(t['stamp_ms']-flash['stamp_ms'])<=150]
            flashes=[f for f in self.flashes if abs(f['stamp_ms']-flash['stamp_ms'])<=150]
            self.results.appendleft({'id':key,'kind':'av_flash_binding','stamp_ms':stamp,'observed_flash_count':len(flashes),'observed_tone_count':len(tones),
                'double_flash_candidate':len(flashes)==1 and len(tones)==2,'flash':flash,'tones':tones,'policy':'preserve_observed_counts'})
        for turn in turns:
            key='speech:'+turn['utterance_id']
            if key in self.seen:continue
            self.seen.append(key)
            t=turn['transcript'];text=re.sub(r'[\s，,。.!！?？]','',t.get('text','')).lower()
            if text not in {'ba','pa','ma','巴','爸','八','吧','怕','啪','妈','马'}:continue
            if turn.get('source')=='robot_echo':continue
            person=turn.get('visual_person_id') or turn.get('person_id')
            samples=[ratio for ts,p,ratio in self.mouth if t.get('started_ms',0)<=ts<=t.get('ended_ms',0) and p==person]
            valid=len(samples)>=3
            self.results.appendleft({'id':key,'kind':'av_speech_consistency','stamp_ms':stamp,'utterance_id':turn['utterance_id'],'person_id':person,
                'audio_text':text,'expected_articulation':'bilabial_closure','visual_min_open_ratio':min(samples) if samples else None,
                'valid':valid,'conflict':bool(valid and min(samples)>.08),'sample_count':len(samples),
                'capability':'bilabial_closure_check_only','policy':'retain_asr_do_not_invent_fused_phoneme'})
        return list(self.results)
