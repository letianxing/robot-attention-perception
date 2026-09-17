"""Social speech cues select how to yield, never who may interrupt."""
import re
from .interaction_intent import directed_call


def interruption_plan(text, speaking, voice, stamp, track_id=None):
    text=str(text or '').strip()
    normalized=re.sub(r'[\s，,。.!！?？]','',text)
    scene=voice.get('audio_scene') or {};bio=(scene.get('acoustic') or {}).get('bio') or {};vap=scene.get('turn_prediction') or {}
    fresh=0<=stamp-int(bio.get('stamp_ms') or 0)<500 and (track_id is None or bio.get("source_track_id")==track_id)
    rising=bool(fresh and bio.get('voicing_confidence',0)>=.8 and (bio.get('f0_slope_semitones_per_second') or 0)>4)
    if speaking and normalized in {'嗯','嗯嗯','对','对对','对对对','好','好的','是的'} and not any(c in text for c in '?？') and not rising:
        return {'speech_act':'backchannel','mode':'continue','confirmation_ms':0,'fade_ms':0,'reason':'acknowledgement_not_floor_request'}
    command=re.sub(r'^(?:你好)?(?:小圆|小园|小元|小袁|机器人|reachy)[，,。！!\s]*','',text,flags=re.I)
    if re.match(r'^(?:请)?(?:停一下|先停|停止|别说了|闭嘴|打住|等一下)',command):
        return {'speech_act':'stop_request','mode':'interrupt','confirmation_ms':80,'fade_ms':10,'reason':'explicit_stop_request'}
    if directed_call(text):
        return {'speech_act':'directed_call','mode':'interrupt','confirmation_ms':120,'fade_ms':20,'reason':'name_call'}
    if re.match(r'^(?:小心|救命|当心)',command):
        return {'speech_act':'urgent_request','mode':'interrupt','confirmation_ms':80,'fade_ms':10,'reason':'explicit_urgent_words'}
    model_valid=bool(vap.get('valid') and 0<=stamp-int(vap.get('stamp_ms') or 0)<800)
    projected=model_valid and float(vap.get('p_user_now') or 0)>.7
    # How long the other voice has to keep going before we treat it as taking the
    # floor. Ordinary turn transitions run around 200 ms and short overlaps are
    # usually backchannels or false starts rather than claims (Stivers et al.
    # 2009 on transition timing), so a couple of frames of speech is not a
    # reason to stop talking. Explicit stops and name calls above keep their
    # short windows — those are unambiguous.
    return {'speech_act':'turn_request','mode':'yield','confirmation_ms':350 if projected else 500,'fade_ms':40,
            'max_boundary_wait_ms':350,'reason':'projected_user_turn' if projected else 'addressed_speech_confirmed',
            'vap_assisted':bool(projected),'prosody_valid':bool(fresh)}
