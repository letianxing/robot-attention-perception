"""Guided face/voice enrollment using a specific fresh utterance."""
import time
from .live_runtime import get_json, post_json


def validate(state, payload, stamp):
    name = str(payload.get('name') or '').strip()
    if not name or len(name)>64 or any(ord(c)<32 or c in '/\\' for c in name):
        raise ValueError('请填写称呼（1–64字）')
    if payload.get('role') not in {'owner','known'}:
        raise ValueError('请选择主人或熟人')
    voice=state.get('voice') or {}
    vision=state.get('vision') or {}
    vision=vision.get('state') or vision
    people=[p for p in vision.get('people',[]) if p.get('face_visible') or float(p.get('face_confidence') or 0)>=.75]
    if len(people)!=1:
        raise ValueError('请只让注册者面对镜头，等待清晰的人脸')
    frame=vision.get('stamp_ms') or float(vision.get('stamp') or 0)*1000
    if not 0<=stamp-frame<2000:
        raise ValueError('摄像头画面已过期，请检查采集')
    t=voice.get('last_transcript') or {}
    if not payload.get('utterance_id') or t.get('utterance_id')!=payload['utterance_id']:
        raise ValueError('声音样本已变化，请再说一段话')
    if not 0<=stamp-t.get('ended_ms',0)<15000 or t.get('ended_ms',0)-t.get('started_ms',0)<2000 or len(t.get('text','').strip())<6:
        raise ValueError('请自然说一句稍长的话，至少2秒')
    if t.get('source')=='robot_echo' or float(t.get('self_echo_probability') or 0)>=.65 or float(t.get('overlap_probability') or 0)>=.5:
        raise ValueError('检测到回声或重叠说话，请注册者单独再说一次')
    return name


def register(runtime,payload):
    state=runtime.state()
    try:
        name=validate(state,payload,int(time.time()*1000))
    except ValueError as exc:
        return {"success":False,"message":str(exc)}
    role=payload['role']
    speaker=((state.get('voice') or {}).get('last_transcript') or {}).get('speaker') or {}
    if speaker.get('speaker_role') in {'owner','known'} and speaker.get('speaker_id') not in {None,'unknown','robot'} and float(speaker.get('similarity') or 0)>=.6:
        message='你已经是主人啦，不要来挑逗我了。' if speaker.get('speaker_role')=='owner' else '我已经认识你啦，不用再注册了。'
        return {'success':True,'already_registered':True,'person_id':speaker['speaker_id'],'message':message}
    transcript=(state.get('voice') or {}).get('last_transcript') or {}
    face_payload={'user_id':name,'user_role':role,'require_single':True,'started_ms':transcript.get('started_ms',0),'ended_ms':transcript.get('ended_ms',0)}
    voice_payload={'speaker_id':name,'speaker_role':role,'utterance_id':payload['utterance_id']}
    # Validate both modalities before either owning service commits its profile.
    for url, request in [(runtime.config.vision_url+'/api/enroll',face_payload),(runtime.config.voice_url+'/api/enroll-speaker',voice_payload)]:
        checked=post_json(url,dict(request,dry_run=True),timeout=5.)
        if not checked.get('success'):return checked
    face=post_json(runtime.config.vision_url+'/api/enroll',face_payload,timeout=5.)
    if not face.get('success'):
        return {'success':False,'message':face.get('message','人脸注册失败')}
    try:
        voice=post_json(runtime.config.voice_url+'/api/enroll-speaker',voice_payload,timeout=5.)
    except Exception as exc:
        voice={'success':False,'message':str(exc)}
    if not voice.get('success'):
        return {'success':False,'message':'人脸已保存，声纹尚未完成：'+voice.get('message','请重新说话后重试')}
    if face.get('embedding'):
        runtime._submit_memory_event('vision-detection','/vision/identity_profiles','face_identity_profile',{'person_id':name,'role':role,'embedding_model':'opencv-sface'},int(time.time()*1000),embedding=face['embedding'],identity_scope=True,entity={'person_id':name})
    runtime._submit_memory_event('robot-attention-perception','/attention/person_registration','person_registration',{'person_id':name,'role':role,'modalities':['face','voice'],'utterance_id':payload['utterance_id']},int(time.time()*1000))
    return {'success':True,'person_id':name,'message':f'{name}，人脸和声纹注册完成'}
