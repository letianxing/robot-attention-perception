"""Pluggable attention evidence: availability is not a fabricated neutral state.

Each system feeding attention — audio-visual perception, this session's working
memory, the linguistic context of the current utterance, cross-session
familiarity from the memory service, and later an internal-state service — is an
independent source that the console can plug in or unplug. Providers may
suggest bounded evidence; none can grant dialogue or action permission, create
an unseen person or rewrite an identity. Perception and safety reflexes stay
operational whatever is unplugged.
"""
import math,os,threading
from .linguistic_context import LinguisticContext
from .session_attention import SessionAttention

SOURCE_NAMES=('audio_visual','memory_context','linguistic_context','cross_session_memory','internal_state')
DEFAULT_SOURCES=('audio_visual','memory_context','linguistic_context','cross_session_memory')


def parse_sources(value):
    names={name.strip() for name in str(value or '').split(',') if name.strip()}
    unknown=names-set(SOURCE_NAMES)
    if unknown:raise ValueError('Unknown attention sources: '+','.join(sorted(unknown)))
    return names


class AttentionSources:
    def __init__(self,session,enabled=None,familiarity=None):
        self.session=session
        self.familiarity=familiarity
        configured=enabled if enabled is not None else os.environ.get('ATTENTION_SOURCES',','.join(DEFAULT_SOURCES))
        self.enabled=parse_sources(configured) if isinstance(configured,str) else parse_sources(','.join(configured))
        self.dialogue=SessionAttention(session);self.language=LinguisticContext()
        self.lock=threading.Lock();self.brain=None;self.internal=None

    def set_enabled(self,names):
        """Swap sources while running; returns the accepted set."""
        selected=parse_sources(names) if isinstance(names,str) else parse_sources(','.join(names))
        with self.lock:self.enabled=selected
        return sorted(selected)

    def enabled_names(self):
        with self.lock:return sorted(self.enabled)

    def receive_brain(self,payload):
        if payload.get('session_id')!=self.session:return
        self.dialogue.receive(payload)
        with self.lock:self.brain=dict(payload)

    def receive_internal(self,payload):
        if payload.get('session_id')!=self.session or payload.get('schema_version')!=1:return False
        try:
            arousal=float(payload['arousal_gain']);stamp=int(payload['stamp_ms'])
            if not math.isfinite(arousal) or not .5<=arousal<=2:return False
            support=payload.get('candidate_support') or {}
            if len(support)>64:return False
            if any(not isinstance(k,str) or not math.isfinite(float(v)) or not 0<=float(v)<=1 for k,v in support.items()):return False
        except (TypeError,ValueError,KeyError):return False
        with self.lock:self.internal={'stamp_ms':stamp,'arousal_gain':arousal,'candidate_support':dict(support)}
        return True

    def _current_utterance(self,stamp,voice):
        """Pick the text to score and the id that makes a phasic effect fire once.

        A partial is used only to catch the robot's name early; the complete
        sentence is scored again when the final transcript arrives.
        """
        final=voice.get('last_transcript') or {}
        if (final.get('is_final') and final.get('source')!='robot_echo'
                and 0<=stamp-int(final.get('emitted_ms') or 0)<1500
                and float(final.get('self_echo_probability') or 0)<.65):
            key=str(final.get('utterance_id') or final.get('emitted_ms') or '')
            return str(final.get('text') or ''),key+':final',final
        partial=voice.get('last_streaming_transcript') or {}
        if (not partial.get('is_final') and partial.get('source')!='robot_echo'
                and 0<=stamp-int(partial.get('emitted_ms') or 0)<800):
            from .interaction_intent import directed_call
            if directed_call(str(partial.get('text') or '')):
                key=str(partial.get('utterance_id') or partial.get('started_ms') or '')
                return str(partial.get('text') or ''),key+':partial',partial
        return '','',{}

    def context(self,stamp,people,tracks=(),voice=None,turns=()):
        voice=voice or {}
        with self.lock:
            brain=self.brain;internal=self.internal;enabled=set(self.enabled)
        status={name:{'enabled':name in enabled,'available':False,
                      'reason':'disabled' if name not in enabled else 'missing'} for name in SOURCE_NAMES}
        output={'goals':[],'importance':{},'motivation':{},'person_memory':{},'linguistic':{},
                'linguistic_speaker':None,'event_id':'','known_people':[],
                'crossmodal_enabled':'audio_visual' in enabled,'sources':status}

        if 'audio_visual' in enabled:
            available=(any(p.face_visible and 0<=stamp-p.stamp_ms<600 for p in people)
                       or any(t.voice_activity and 0<=stamp-t.stamp_ms<350 for t in tracks))
            status['audio_visual'].update(available=available,
                                          reason='live_evidence' if available else 'no_fresh_observation')

        visible={p.person_id for p in people if p.face_visible and 0<=stamp-p.stamp_ms<600}
        known=set(visible)
        pending={}
        if 'memory_context' in enabled:
            fresh=brain is not None and 0<=stamp-int(brain.get('stamp_ms') or 0)<1200
            status['memory_context'].update(available=fresh,
                                            reason='current_session_snapshot' if fresh else 'missing_or_stale')
            if fresh:
                output['goals'].extend(self.dialogue.context(stamp,people).get('goals',[]))
                summary=brain.get('attention_memory') or {}
                for item in summary.get('recent_participants',[])[:16]:
                    person=item.get('person_id');age=stamp-int(item.get('last_heard_ms') or 0)
                    if not person or not 0<=age<90000:continue
                    known.add(person)
                    recency=math.exp(-age/30000)
                    output['person_memory'].setdefault(person,{}).update(memory_participant=1.,memory_recency=recency)
                    # Addressing the robot is not the same as speaking. An exchange
                    # already under way survives losing sight of them, and fades
                    # over roughly fifteen seconds rather than latching.
                    addressed=item.get('last_addressed_ms')
                    if addressed:
                        open_age=stamp-int(addressed)
                        if 0<=open_age<60000:
                            output['person_memory'][person]['memory_open_exchange']=math.exp(-open_age/15000)
                    if person in visible:output['importance']['visual:'+person]=.2*recency
                pending=summary.get('expected_answer') or {}
                if pending.get('person_id') and stamp<int(pending.get('expires_ms') or 0):
                    person=pending['person_id'];known.add(person)
                    output['person_memory'].setdefault(person,{})['memory_expected_answer']=1.
                    if person in visible:
                        output['goals'].append({'candidate_id':'visual:'+person,'confirmed':True,
                                                'stamp_ms':brain['stamp_ms'],'strength':.65,'purpose':'expected_answer'})
                # An exchange in progress is a memory fact; it must not require
                # the partner to keep looking at the camera.
                turn=brain.get('current_turn') or {}
                if (turn.get('person_id') and turn.get('status') in {'thinking','synthesizing','speaking'}
                        and not turn.get('reflex')):
                    person=turn['person_id'];known.add(person)
                    output['person_memory'].setdefault(person,{})['memory_dialogue_target']=1.
                # What Brain currently intends, as top-down bias. A goal weights
                # the competition for the person it concerns; it grants nothing.
                for goal in (summary.get('goals') or [])[:8]:
                    person=goal.get('person_id')
                    if not person or stamp>=int(goal.get('expires_ms') or 0):continue
                    known.add(person)
                    strength=max(0.,min(1.,float(goal.get('strength') or 0.)))
                    key='visual:'+person
                    output['importance'][key]=max(output['importance'].get(key,0.),strength)
                    output.setdefault('brain_goals',[]).append(
                        {'kind':goal.get('kind'),'person_id':person,'strength':strength})

        if 'linguistic_context' in enabled:
            self.language.observe(turns)
            text,event,record=self._current_utterance(stamp,voice)
            status['linguistic_context'].update(available=bool(text),
                                                reason='current_utterance' if text else 'no_fresh_transcript')
            if text:
                speaker=record.get('speaker') or {}
                speaker_id=speaker.get('speaker_id') if float(speaker.get('similarity') or 0)>=.48 else None
                if speaker_id in {'unknown','robot'}:speaker_id=None
                features=self.language.features(stamp,text,speaker_id or '',sorted(known),
                                                str(pending.get('question') or ''))
                output['linguistic']=features
                output['linguistic_speaker']=speaker_id
                output['event_id']=event
                output['linguistic_track_id']=record.get('track_id')
        output['known_people']=sorted(known)

        if 'cross_session_memory' in enabled:
            index=self.familiarity
            if index is None:
                status['cross_session_memory'].update(reason='no_memory_index_configured')
            else:
                # Only people we can actually see or hear are looked up, and the
                # lookup itself runs on a background worker.
                index.request(stamp,[p.person_id for p in people]+[t.speaker_label for t in tracks])
                scores=index.scores(stamp)
                status['cross_session_memory'].update(available=bool(scores),
                    reason='memory_service_records' if scores else 'no_cached_records_yet')
                for person,record in scores.items():
                    if record['familiarity']<=0:continue
                    known.add(person)
                    output['person_memory'].setdefault(person,{})['memory_familiarity']=record['familiarity']
                output['familiarity_detail']={p:{k:r[k] for k in ('familiarity','prior_sessions','prior_turns','last_seen_ms')}
                                              for p,r in scores.items()}
                output['familiarity_status']=index.status(stamp)

        if 'internal_state' in enabled:
            fresh=internal is not None and 0<=stamp-internal['stamp_ms']<1500
            status['internal_state'].update(available=fresh,
                                            reason='external_state' if fresh else 'missing_or_stale')
            if fresh:
                output['arousal_gain']=internal['arousal_gain']
                output['motivation']=internal['candidate_support']
        return output
