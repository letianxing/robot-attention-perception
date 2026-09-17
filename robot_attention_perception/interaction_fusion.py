"""Authoritative AV interaction state: proposals never bypass current evidence."""
import os
from dataclasses import asdict
from collections import deque
from .visual_focus import VisualAttentionFocus
from .interaction_intent import directed_call
from .competition_adapter import CompetitionAdapter

# Matches the YuNet detector's own confidence threshold; override per camera.
FACE_VISIBLE_CONFIDENCE=float(os.environ.get('ATTENTION_FACE_CONFIDENCE','0.80'))


class InteractionFusion:
    def __init__(self):
        self.visual=VisualAttentionFocus()
        self.competition=CompetitionAdapter()
        self.last_lips={}
        self.speech_key=None;self.speech_since=None;self.last_speech_evidence=None
        self.dialogue_person=None;self.dialogue_ms=0
        self.call_focus=None
        self.seen_final_calls=deque(maxlen=256)
        self.transitions=deque(maxlen=80);self.sequence=0;self.last_signature=None

    def _finish(self,stamp,state,phase,proposal,tracks):
        state=dict(state,stamp_ms=stamp,interaction_phase=phase,policy='unified_av_interaction_v1',
                   upstream_algorithm=proposal.get('algorithm'),upstream_target=proposal.get('target_id'),
                   observed_audio_track_ids=[t.track_id for t in tracks])
        signature=(phase,state.get('target_id'),bool(state.get('addressed_to_robot')))
        if signature!=self.last_signature:
            self.sequence+=1
            self.transitions.appendleft({'id':self.sequence,'stamp_ms':stamp,'from':self.last_signature,'to':signature,'reasons':state.get('reasons',[])})
            self.last_signature=signature
        state['transition_id']=self.sequence
        state["attention_distribution"]=self.competition.result
        state["engagement"]=(self.competition.result or {}).get("engagement") or {}
        return state

    def update(self,stamp,proposal,people,tracks,voice=None,reflex=None,objects=(),context=None):
        proposal=dict(proposal) if isinstance(proposal,dict) else asdict(proposal)
        voice=voice or {}
        distribution=self.competition.update(stamp,people,tracks,voice,objects,context)
        self.visual.distribution=distribution
        empty={'stamp_ms':stamp,'target_id':'none','target_kind':'none','person_id':None,'confidence':0.,
               'listen':False,'addressed_to_robot':False,'source_track_id':None,'azimuth_deg':None,'reasons':['no_current_evidence']}
        # The detector is configured to report at 0.80; demanding more of it
        # downstream, with no separate justification, is what made a clearly
        # detected face count as absent. Measured in the field: 0.81 to 0.83.
        visible=[p for p in people if p.face_visible and p.face_confidence>=FACE_VISIBLE_CONFIDENCE
                 and 0<=stamp-p.stamp_ms<400]
        fresh=[t for t in tracks if t.voice_activity and 0<=stamp-t.stamp_ms<350]
        clean=[t for t in fresh if t.speech_probability>=.6 and t.clarity>=.35 and t.self_echo_probability<.4
               and t.overlap_probability<.35 and not t.target_speech_rejected]
        for p in visible:
            if p.lip_motion:self.last_lips[p.person_id]=stamp
        self.last_lips={k:t for k,t in self.last_lips.items() if stamp-t<=250}
        # A startle drops everything and looks. Orienting does not: turning an ear
        # towards a novel sound is the cheap response, and once orienting fired on
        # ordinary speech (which is the point of it) letting it clear the floor
        # every two seconds meant the person being talked to kept losing the
        # robot's attention mid-sentence.
        if (reflex and reflex.get('active') and 0<=stamp-int(reflex.get('stamp_ms') or 0)<1800
                and (reflex.get('kind')=='startle' or not clean)):
            self.speech_key=None;self.speech_since=None
            out=self.visual.update(stamp,empty,people,[],reflex=reflex)
            return self._finish(stamp,out,'ORIENTING',proposal,fresh)
        # Explicit calls are read from fresh microphone ASR, never a stale upstream boolean.
        final=voice.get('last_transcript') or {}
        final_key=final.get('utterance_id') or str(final.get('emitted_ms'))
        if (final.get('is_final') and final_key not in self.seen_final_calls and final.get('source')!='robot_echo'
                and 0<=stamp-int(final.get('emitted_ms') or 0)<1200 and float(final.get('self_echo_probability') or 0)<.65
                and directed_call(str(final.get('text') or ''))):
            self.seen_final_calls.append(final_key)
            speaker=final.get('speaker') or {};direction=final.get('direction') or {}
            self.call_focus={'track_id':final.get('track_id'),'since':stamp,
                             'person_id':speaker.get('speaker_id') if float(speaker.get('similarity') or 0)>=.48 else None,
                             'azimuth_deg':direction.get('direction_deg') if direction.get('direction_valid') else None}
        partial=voice.get('last_streaming_transcript') or {}
        direct=bool(not partial.get('is_final') and partial.get('source')!='robot_echo' and 0<=stamp-int(partial.get('emitted_ms') or 0)<800
                    and directed_call(str(partial.get('text') or '')))
        active=voice.get('active_utterance') or {}
        if active.get('started_ms') and partial.get('started_ms') and active['started_ms']!=partial['started_ms']:direct=False
        called=[t for t in clean if t.track_id==partial.get('track_id')]
        matches=[]
        alone=len(visible)==1 and len(clean)==1
        for t in clean:
            for p in visible:
                if p.gaze_score<.65 or p.body_facing_score<.6 or p.person_id not in self.last_lips:continue
                if abs(t.stamp_ms-p.stamp_ms)>200:continue
                known=t.azimuth_deg is not None and p.azimuth_deg is not None
                # Direction separates a crowd; with one mouth moving and one
                # voice it is not needed, and an uncalibrated array disagreeing
                # must not be read as a second speaker nobody can see.
                if known and abs((t.azimuth_deg-p.azimuth_deg+180)%360-180)>=30 and not alone:continue
                matches.append((t,p))
        chosen=matches[0] if len(matches)==1 else None
        if direct and len(called)==1:
            t=called[0]
            p=chosen[1] if chosen and chosen[0].track_id==t.track_id else None
            self.dialogue_person=p.person_id if p else None;self.dialogue_ms=stamp
            if p:self.visual.adopt(p,stamp)
            self.call_focus={"track_id":t.track_id,"since":stamp,"person_id":p.person_id if p else (t.speaker_label if t.speaker_label!="unknown" else None),"azimuth_deg":t.azimuth_deg}
            out=dict(empty,target_id='person:'+p.person_id if p else 'sound:'+t.track_id,target_kind='person' if p else 'sound_source',
                     person_id=p.person_id if p else None,confidence=.8,listen=True,addressed_to_robot=True,source_track_id=t.track_id,
                     face_id=p.face_id if p else None,azimuth_deg=p.azimuth_deg if p else t.azimuth_deg,reasons=['fresh_directed_call'])
            return self._finish(stamp,out,'LISTENING',proposal,fresh)
        key=(chosen[0].track_id,chosen[1].person_id) if chosen else None
        if key is None:
            self.speech_key=None;self.speech_since=None
        else:
            if key!=self.speech_key or self.last_speech_evidence is None or stamp-self.last_speech_evidence>250:
                self.speech_key=key;self.speech_since=stamp
            self.last_speech_evidence=stamp
            if stamp-self.speech_since>=120:
                t,p=chosen
                self.dialogue_person=p.person_id;self.dialogue_ms=stamp
                self.visual.adopt(p,stamp)
                out=dict(empty,target_id='person:'+p.person_id,target_kind='person',person_id=p.person_id,face_id=p.face_id,
                         confidence=min(p.face_confidence,p.gaze_score,t.speech_probability),listen=True,addressed_to_robot=True,
                         source_track_id=t.track_id,azimuth_deg=p.azimuth_deg,reasons=['gaze_to_robot','lip_audio_synchrony','confirmed_speech_target'])
                return self._finish(stamp,out,'LISTENING',proposal,fresh)
        # Keep attention (not permission) while a response that began inside the
        # call window is still being spoken; final identity evidence decides reply.
        active=voice.get('active_utterance') or {}
        if self.call_focus and len(clean)==1 and active.get('started_ms'):
            started=int(active['started_ms']);focus=self.call_focus;t=clean[0]
            same_direction=(focus.get('azimuth_deg') is None or t.azimuth_deg is None or abs((t.azimuth_deg-focus['azimuth_deg']+180)%360-180)<=15)
            if started<=focus['since']+6000 and 0<=stamp-started<30000 and same_direction:
                focus['since']=stamp
        # A name call opens a short auditory engagement window even off camera.
        if self.call_focus and stamp-self.call_focus['since']<6000:
            focus=self.call_focus
            known=focus.get('person_id')
            followup=[t for t in clean if known and t.speaker_label==known and t.speaker_similarity>=.6]
            if len(followup)==1:
                t=followup[0];focus['since']=stamp
                out=dict(empty,target_id='sound:'+t.track_id,target_kind='sound_source',person_id=known,
                         confidence=min(.9,t.speaker_similarity),listen=True,addressed_to_robot=True,source_track_id=t.track_id,
                         azimuth_deg=t.azimuth_deg,reasons=['recognized_voice_after_direct_call'],focus_origin='direct_call')
                return self._finish(stamp,out,'LISTENING',proposal,fresh)
            out=dict(empty,target_id='sound:'+str(focus['track_id']),target_kind='sound_source',person_id=known,confidence=.8,
                     listen=True,azimuth_deg=focus.get('azimuth_deg'),focus_source_track_id=focus['track_id'],
                     focus_origin='direct_call',reasons=['name_call_attention','waiting_for_called_speaker'])
            return self._finish(stamp,out,'AUDIO_FOCUS',proposal,fresh)
        # Engagement accumulated by the selected algorithm from gaze, working
        # memory and what was said. It is the same decision as the paths above,
        # only reached by evidence that arrives over several seconds, so an
        # answer given with the head turned away and a silent invitation into an
        # ongoing discussion no longer need their own handling downstream.
        engagement=(distribution or {}).get('engagement') or {}
        phase_state=str(engagement.get('state') or '')
        engaged=engagement.get('person_id')
        engaged_confidence=min(.9,float(engagement.get('confidence') or 0))
        engagement_reasons=list(engagement.get('reasons') or [])
        if phase_state=='EXPECTED_ANSWER' and engaged:
            answering=[t for t in clean if t.speaker_label==engaged and t.speaker_similarity>=.6]
            if len(answering)==1:
                t=answering[0]
                p=next((q for q in visible if q.person_id==engaged),None)
                self.dialogue_person=engaged;self.dialogue_ms=stamp
                if p:self.visual.adopt(p,stamp)
                out=dict(empty,target_id='person:'+engaged if p else 'sound:'+t.track_id,
                         target_kind='person' if p else 'sound_source',person_id=engaged,
                         confidence=min(engaged_confidence,max(.65,float(t.speaker_similarity))),listen=True,
                         addressed_to_robot=True,source_track_id=t.track_id,face_id=p.face_id if p else None,
                         azimuth_deg=p.azimuth_deg if p else t.azimuth_deg,
                         reasons=['expected_answer_to_robot_question']+engagement_reasons)
                return self._finish(stamp,out,'LISTENING',proposal,fresh)
        if phase_state=='ENGAGED' and engaged:
            # An exchange already under way does not need the camera. Somebody
            # who was addressing us moments ago and speaks again is still
            # addressing us, and requiring them to be visible is what made a
            # follow-up from the dark go unanswered. Identity still has to be
            # reliable, because this is the only evidence tying the voice to the
            # conversation it is continuing.
            speaking=[t for t in clean if t.speaker_label==engaged and t.speaker_similarity>=.6]
            if len(speaking)==1:
                t=speaking[0]
                p=next((q for q in visible if q.person_id==engaged),None)
                self.dialogue_person=engaged;self.dialogue_ms=stamp
                if p:self.visual.adopt(p,stamp)
                out=dict(empty,target_id='person:'+engaged if p else 'sound:'+t.track_id,
                         target_kind='person' if p else 'sound_source',person_id=engaged,
                         confidence=min(engaged_confidence,max(.65,float(t.speaker_similarity))),listen=True,
                         addressed_to_robot=True,source_track_id=t.track_id,face_id=p.face_id if p else None,
                         azimuth_deg=p.azimuth_deg if p else t.azimuth_deg,
                         reasons=['engaged_speaker']+engagement_reasons)
                return self._finish(stamp,out,'LISTENING',proposal,fresh)
        if phase_state=='INVITED' and engaged:
            # Other people talking is the normal case for this: somebody turns
            # to us in the middle of a conversation they were part of. What
            # matters is that *this* person is not the one speaking, which the
            # engagement layer already required before reporting INVITED.
            p=next((q for q in visible if q.person_id==engaged),None)
            speaking=[t for t in clean if t.speaker_label==engaged and t.speaker_similarity>=.48]
            if p and not speaking:
                self.visual.adopt(p,stamp)
                # Nothing was said, so this opens a turn; it does not address one.
                out=dict(empty,target_id='person:'+engaged,target_kind='person',person_id=engaged,face_id=p.face_id,
                         confidence=engaged_confidence,listen=True,addressed_to_robot=False,
                         azimuth_deg=p.azimuth_deg,gaze_to_robot=p.gaze_score>=.58,
                         reasons=['silent_invitation']+engagement_reasons)
                return self._finish(stamp,out,'INVITED',proposal,fresh)
        # A new confirmed speaker above can barge in; mere visual interest cannot steal TTS.
        playing=bool((voice.get('playback') or {}).get('speaking') or voice.get('robot_speaking'))
        current=next((p for p in visible if p.person_id==self.dialogue_person),None)
        if playing and current and stamp-self.dialogue_ms<30000:
            out=dict(empty,target_id='person:'+current.person_id,target_kind='person',person_id=current.person_id,
                     confidence=min(current.face_confidence,.85),listen=True,gaze_to_robot=current.gaze_score>=.65,reasons=['active_reply_target'])
            return self._finish(stamp,out,'RESPONDING',proposal,fresh)
        # Visual policy receives no unverified speech permissions from upstream.
        # It only chooses where to look; conversation permission is owned above.
        out=self.visual.update(stamp,empty,people,[],(voice.get('audio_scene') or {}).get('acoustic'))
        out['addressed_to_robot']=False;out['source_track_id']=None
        if out.get('target_kind')=='sound_source':phase='ORIENTING'
        elif out.get('target_kind')=='person':phase='VISUAL_FOCUS'
        elif fresh:phase='AMBIENT_SPEECH'
        else:phase='ACQUIRING' if self.visual.person else 'IDLE'
        return self._finish(stamp,out,phase,proposal,fresh)


# Every way a sentence can turn out to be addressed to the robot, as one
# weighted sum instead of five separate branches, so that the answer to "why
# did it not reply" is a row of numbers rather than a hunt through the code.
# Tunable live from the 8092 console; weights persist in .run/attention-config.json.
# Which cues say a sentence was meant for the robot, and roughly how much each
# is worth. The ordering follows the addressee-detection literature: head pose
# and gaze towards the addressee, plus lexical address, carry most of the signal
# in multi-party settings (Katzenmaier, Stiefelhagen & Schultz 2004, Identifying
# the addressee in human-human-robot interactions based on head pose and speech;
# Vertegaal et al. 2001, Eye gaze patterns in conversations, showing gaze
# predicts who is being spoken to far better than chance; Op den Akker & Traum
# 2009 on addressee prediction in meetings). Conversational continuity — an
# answer to a question we just asked — comes from adjacency-pair structure
# (Sacks, Schegloff & Jefferson 1974). None of these weights is calibrated on
# this hardware; they are a starting point you can move from the console.
ADDRESSEE_DEFAULTS = {
    'directed_call': 1.0,          # said the robot's name at the start of the sentence
    'live_gate': 0.9,              # the live policy already judged it addressed while it was said
    'reliable_voiceprint': 0.75,   # voiceprint above the reliability bar, on somebody we are attending to
    'name_call_window': 0.8,       # the name was called a moment ago (no camera needed)
    'answer_to_question': 0.8,     # we asked them something and nobody else has spoken since
    'open_exchange': 0.8,          # we are mid-conversation with them; no camera needed for that
    'lip_motion_while_looking': 0.3,  # the one looking at us is also the one whose mouth moved
    'gaze_while_speaking': 0.6,    # somebody was looking at us while this was said; scaled by how much
    # Two bars, not one. Above the first the robot answers straight away; between
    # the two it writes the sentence down and says nothing, because a single weak
    # cue — somebody glancing over while talking to a person next to them — is a
    # guess. Those held sentences are not lost: when attention does settle on that
    # person, the reply is made from them. Below the lower bar it is ambient talk.
    'answer_threshold': 0.8,       # answer now
    'threshold': 0.45,             # at or above this: remember it, do not speak
    'name_call_window_ms': 12000,  # how long being called by name keeps the floor open
    'answer_window_ms': 25000,     # how long an unanswered question of ours stays open
    'attended_window_ms': 2500,    # how far back the camera snapshots are read
}
# The names these terms have always had downstream; the trace, the memory events
# and the console all key on them, so the scoring keeps speaking that language.
ADDRESSEE_REASONS = {
    'directed_call': 'explicit_final_directed_call',
    'live_gate': 'final_from_live_evidence',
    'reliable_voiceprint': 'confirmed_voice_at_engaged_target',
    'name_call_window': 'within_name_call_window',
    'answer_to_question': 'answer_to_our_question_unverified_voice',
    'open_exchange': 'mid_exchange_with_this_person',
    'lip_motion_while_looking': 'looker_is_the_speaker',
    'gaze_while_speaking': 'looked_at_the_robot_while_speaking',
}
ADDRESSEE_LABELS = {
    'directed_call': '点名（本句开头就叫了名字）',
    'live_gate': '说话当下现场证据已判定对机器人说',
    'reliable_voiceprint': '可靠声纹 + 持续注视焦点',
    'name_call_window': '刚才叫过名字（12 秒窗口，不需要摄像头）',
    'answer_to_question': '这是在回答机器人刚问的问题（25 秒窗口）',
    'open_exchange': '正在和这个人一来一往（不依赖摄像头）',
    'lip_motion_while_looking': '看着机器人的那个人嘴也在动（就是他在说）',
    'gaze_while_speaking': '说这句话的时候正看着机器人（多人在场也算）',
    'answer_threshold': '立刻回复的阈值',
    'threshold': '记住但先不回的阈值',
    'name_call_window_ms': '点名后的听话窗口（毫秒）',
    'answer_window_ms': '机器人提问后等回答的窗口（毫秒）',
    'attended_window_ms': '读多久的摄像头快照来判断谁在看（毫秒）',
}


def addressee_weights(config=None):
    weights = dict(ADDRESSEE_DEFAULTS)
    for key, value in (config or {}).items():
        if key in weights:
            try:
                weights[key] = float(value)
            except (TypeError, ValueError):
                pass
    return weights


def finalize_gate(transcript,history,end,evidence=None,weights=None,exchange=None,people_history=()):
    """Resolve the final sentence against the same policy's contemporaneous evidence.

    One score, several pieces of evidence, all of them reported. The branches this
    replaced each granted on their own and recorded only the one that fired, so a
    sentence that nearly qualified three different ways looked identical to one
    with no evidence at all.
    """
    weights=addressee_weights(weights)
    track=transcript.get('track_id')
    denied={'stamp_ms':end,'target_id':'none','target_kind':'none','person_id':None,'confidence':0.,'listen':False,
            'addressed_to_robot':False,'source_track_id':track,'policy':'unified_av_interaction_v1','reasons':['no_contemporaneous_dialogue_permission']}
    clean=(transcript.get('source')!='robot_echo' and float(transcript.get('self_echo_probability') or 0)<.65
           and not (evidence and evidence.target_speech_rejected))
    if not clean:return dict(denied,reasons=['echo_or_rejected_speech'])
    start=int(transcript.get('started_ms') or end-800)
    window=[(ts,g) for ts,g in history if max(start,end-800)<=ts<=end+100]
    relevant=[(ts,g) for ts,g in window if g.get('source_track_id')==track or track in g.get('observed_audio_track_ids',[])]
    speaker=transcript.get('speaker') or {}
    scores={};person=None;source_gate=None

    if directed_call(str(transcript.get('text') or '')):
        scores['directed_call']=weights['directed_call']

    positives=[(ts,g) for ts,g in relevant if g.get('listen') and g.get('addressed_to_robot') and float(g.get('confidence') or 0)>=.62]
    if positives:
        ts,gate=positives[-1]
        later=relevant[-1][1]
        same_target=later.get('person_id')==gate.get('person_id') and later.get('interaction_phase') in {'VISUAL_FOCUS','RESPONDING','LISTENING'}
        if ts==relevant[-1][0] or (end-ts<=500 and same_target):
            scores['live_gate']=weights['live_gate'];person=person or gate.get('person_id');source_gate=gate

    voiced=speaker.get('speaker_id')
    focused=[(ts,g) for ts,g in relevant if g.get('person_id')==voiced and (g.get('interaction_phase')=='VISUAL_FOCUS' or (g.get('interaction_phase')=='RESPONDING' and g.get('gaze_to_robot')) or (g.get('interaction_phase')=='AUDIO_FOCUS' and g.get('focus_origin')=='direct_call'))]
    if (voiced not in {None,'unknown','robot'} and float(speaker.get('similarity') or 0)>=.6 and len(focused)>=2
            and focused[-1][0]-focused[0][0]>=100 and focused[-1][0]==relevant[-1][0]):
        scores['reliable_voiceprint']=weights['reliable_voiceprint'];person=person or voiced;source_gate=source_gate or focused[-1][1]

    # An exchange in progress is not a visual fact. The robot answered this
    # person five seconds ago; whether they happened to be facing the camera
    # when they asked the follow-up has nothing to do with whom they meant.
    # In the field this was the clearest refusal of all: "眼睛痒" answered, then
    # "怎么办呀" refused, because every other cue needed a face.
    if exchange and exchange.get('person_id'):
        age=end-int(exchange.get('stamp_ms') or 0)
        if 0<=age<=int(weights['answer_window_ms']):
            others=[gate.get('person_id') for ts,gate in history
                    if int(exchange['stamp_ms'])<ts<=end and gate.get('listen')
                    and gate.get('person_id') not in {None,'unknown','robot',exchange['person_id']}]
            if not others:
                scores['open_exchange']=weights['open_exchange']
                person=person or exchange['person_id']

    answered=_answer_to_our_question(history,end,speaker,int(weights['answer_window_ms']))
    if answered:
        scores['answer_to_question']=weights['answer_to_question'];person=person or answered
    name_window=int(weights['name_call_window_ms'])
    if (_spoke_since(history,end,name_window,'explicit_final_directed_call')
            or _spoke_since(history,end,name_window,'name_call_attention')
            or _called_since(history,end,name_window)):
        scores['name_call_window']=weights['name_call_window']
        if voiced and voiced not in {'unknown','robot'} and not str(voiced).startswith('stranger_'):
            person=person or voiced
    # Read from what the camera actually reported, not from the gate. The live
    # gate carries gaze_to_robot only in some branches and its person_id is null
    # whenever visual attention has not locked on, so terms built on those fields
    # scored zero in the field while vision was reporting a face at gaze 0.95.
    # How many people are present is not the question. Anybody looking at the
    # robot while something was said is a candidate, and the candidates compete:
    # whoever looked for most of the sentence, and whose mouth was moving while
    # they did, is the one who said it.
    looking,dominance=_looked_at_us_while_speaking(people_history,start,end,
                                                   int(weights['attended_window_ms']))
    if looking:
        scores['gaze_while_speaking']=round(weights['gaze_while_speaking']*dominance,3)
        if _lips_moved(people_history,start,end,looking):
            scores['lip_motion_while_looking']=weights['lip_motion_while_looking']
        # A name can be called from anywhere in the room, including out of frame,
        # so a sentence that called the name must not borrow the identity of
        # whoever happens to be on camera.
        if 'directed_call' not in scores:
            person=person or looking

    total=round(sum(scores.values()),3)
    decision=('answer' if total>=weights['answer_threshold']
              else 'hold' if total>=weights['threshold'] else 'ignore')
    detail={'score':total,'threshold':weights['threshold'],
            'answer_threshold':weights['answer_threshold'],'decision':decision,
            'terms':{k:round(v,3) for k,v in scores.items()}}
    # Who the camera says this is, as opposed to who the sentence is attributed
    # to. A name called from out of frame is attributed to somebody by memory or
    # by the open exchange; that is not the camera claiming to have seen them.
    seen_person=person if ({'live_gate','reliable_voiceprint'} & set(scores)) else None
    if decision!='answer':
        # Held, not refused. The sentence is recorded with who probably said it
        # and why we were unsure, so that if attention lands on them a moment
        # later there is something to answer.
        return dict(denied,person_id=person if decision=='hold' else None,
                    held_for_attention=decision=='hold',
                    reasons=['held_pending_attention'] if decision=='hold' else denied['reasons'],
                    addressee_detail=detail)
    if source_gate is not None and 'live_gate' in scores:
        return dict(source_gate,listen=True,addressed_to_robot=True,source_track_id=track,
                    confidence=max(float(source_gate.get('confidence') or 0),min(1.,total)),
                    reasons=list(source_gate.get('reasons',[]))+[ADDRESSEE_REASONS[k] for k in sorted(scores)],
                    visual_person_id=seen_person or 'unknown',addressee_detail=detail)
    return dict(denied,target_id=('person:'+person) if person else 'sound:'+str(track),
                target_kind='person' if person else 'sound_source',person_id=person,
                confidence=min(1.,total),listen=True,addressed_to_robot=True,
                reasons=[ADDRESSEE_REASONS[k] for k in sorted(scores)],
                visual_person_id=seen_person or 'unknown',addressee_detail=detail)


ANSWER_WINDOW_MS=25000
NAME_CALL_WINDOW_MS=12000


def _called_since(history,end,window_ms):
    for ts,gate in reversed(list(history)):
        if end-ts>window_ms:
            return False
        if gate.get('focus_origin')=='direct_call':
            return True
    return False


def _spoke_since(history,end,window_ms,reason):
    for ts,gate in reversed(list(history)):
        if end-ts>window_ms:
            return False
        if reason in (gate.get('reasons') or ()):
            return True
    return False


ATTENDED_WINDOW_MS=2500
ATTENDED_SHARE=.35


GAZE_AT_ROBOT=.6
ATTENDED_GAZE=.5     # the one person present also has to be looking this way


def _people_in_window(people_history,start,end,pad_ms=600):
    """Camera snapshots taken while the sentence was being said."""
    return [(ts,people) for ts,people in people_history if start-pad_ms<=ts<=end+200 and people]


def _looked_at_us_while_speaking(people_history,start,end,window_ms=ATTENDED_WINDOW_MS):
    """Who was looking at the robot while this was said, and how clearly.

    Returns the winner of a small competition rather than insisting the room hold
    one person: a table of people where one turns to the robot is the case this
    exists for. Dominance is how much of the sentence they spent looking, scaled
    down when somebody else was looking about as much — two people both facing
    the robot is genuinely ambiguous and the score should say so.
    """
    snapshots=_people_in_window(people_history,max(start,end-window_ms),end)
    if not snapshots:
        return None,0.
    looking={}
    for ts,people in snapshots:
        for person in people:
            key=str(getattr(person,'person_id','') or '')
            if not key or key in {'unknown','robot'} or not getattr(person,'face_visible',False):
                continue
            if float(getattr(person,'gaze_score',0) or 0)>=GAZE_AT_ROBOT:
                looking[key]=looking.get(key,0)+1
    if not looking:
        return None,0.
    ranked=sorted(looking.items(),key=lambda item:-item[1])
    share=ranked[0][1]/float(len(snapshots))
    runner_up=ranked[1][1]/float(len(snapshots)) if len(ranked)>1 else 0.
    dominance=max(0.,min(1.,share*(1.-runner_up)))
    return (ranked[0][0],round(dominance,3)) if dominance>0 else (None,0.)


def _lips_moved(people_history,start,end,person_id):
    """Was that person's mouth moving while they looked? Then they said it."""
    for ts,people in _people_in_window(people_history,start,end):
        for person in people:
            if str(getattr(person,'person_id','') or '')==person_id and getattr(person,'lip_motion',False):
                return True
    return False


def _answer_to_our_question(history,end,speaker,window_ms=ANSWER_WINDOW_MS):
    """We just spoke to somebody and a voice answers. Whose voice is it?

    Not a guess about the voiceprint — the voiceprint failed, which is exactly
    the case this covers. It is a claim about the conversation: the robot asked
    one person something a moment ago, nobody else has spoken since, and a reply
    arrived. In the field this was the whole failure — the owner leaned back out
    of frame, answered, was recorded as a new stranger, the answer was dropped,
    and the robot asked the same question again.

    Refused whenever it could be somebody else: a second person speaking, a
    voiceprint that names a different person, or too long a silence. The reason
    string travels with the gate so the trace never shows this as recognition.
    """
    named=str((speaker or {}).get('speaker_id') or '')
    identified=bool(named) and not named.startswith('stranger_') and named not in {'unknown','robot'}
    asked=None
    for ts,gate in reversed(list(history)):
        if end-ts>window_ms:
            break
        person=gate.get('person_id')
        if not person or person in {'unknown','robot'}:
            continue
        if gate.get('interaction_phase')=='RESPONDING' or 'active_reply_target' in (gate.get('reasons') or ()):
            asked=(ts,person)
            break
    if asked is None:
        return None
    # The voiceprint naming somebody else means this is not our answer. The
    # voiceprint naming the person we asked is agreement, not a conflict — an
    # earlier version refused both alike, so a recognised owner answering the
    # robot's own question was still dropped.
    if identified and named!=asked[1]:
        return None
    since=[gate.get('person_id') for ts,gate in history
           if asked[0]<ts<=end and gate.get('listen') and gate.get('person_id') not in {None,'unknown','robot'}]
    if any(person!=asked[1] for person in since):
        return None   # somebody else has held the floor since; this is not our answer
    return asked[1]
