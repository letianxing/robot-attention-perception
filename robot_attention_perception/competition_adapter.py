"""Bind measured observations to whichever attention algorithm is plugged in.

This adapter owns no policy. It normalises what each source reported into the
plugin ABI's evidence fields, hands them to the selected algorithm and passes
the result back. Which algorithm runs, and which sources feed it, are console
settings; nothing here silently substitutes a missing source.
"""
import math

from .attention_plugin import (EXTRA_AV_SYNC_ONSET, EXTRA_AZIMUTH_DEG, EXTRA_GESTURE_INVITE,
                               EXTRA_GESTURE_REJECT, EXTRA_MEMORY_FAMILIARITY, EXTRA_OPEN_EXCHANGE,
                               EXTRA_RETURNED,
                               AlgorithmRegistry)
from .interaction_intent import directed_call

MEMORY_FIELDS=('memory_participant','memory_recency','memory_expected_answer','memory_dialogue_target')
# Vision publishes '<role>_wave' / '<role>_invite' / '<role>_reject'.
INVITING_GESTURES=('wave','invite','hello','hi','come','call')
REJECTING_GESTURES=('reject','stop','no')
RETURN_ABSENCE_MS=3000
LANGUAGE_FIELDS=('lang_directed_call','lang_question','lang_answer_continuation',
                 'lang_topic_continuation','lang_addresses_other','lang_backchannel')


def unit(value):
    try:
        value=float(value)
    except (TypeError,ValueError):
        return 0.
    return min(1.,max(0.,value)) if math.isfinite(value) else 0.


class CompetitionAdapter:
    def __init__(self,registry=None):
        self.registry=registry or AlgorithmRegistry()
        self.result={};self.baselines={};self.bound=set();self.binding_notes=[]
        self.last_gesture={};self.last_seen={}

    def close(self):
        self.registry.close()

    def update(self,stamp,people,tracks,voice,objects=(),context=None):
        context=context or {};goals=context.get('goals') or []
        sources={name:bool(entry.get('enabled')) for name,entry in (context.get('sources') or {}).items()}
        supported={g.get('candidate_id'):unit(g.get('strength',1)) for g in goals
                   if g.get('confirmed') and 0<=stamp-int(g.get('stamp_ms',0))<2000}
        person_memory=context.get('person_memory') or {}
        language=context.get('linguistic') or {}
        language_speaker=context.get('linguistic_speaker')
        language_track=context.get('linguistic_track_id')
        event_id=str(context.get('event_id') or '')
        importance=context.get('importance') or {}
        motivation=context.get('motivation') or {}

        previous=self.result.get('candidates') or {}
        dt=0. if not self.result.get('stamp_ms') else max(0.,(stamp-int(self.result['stamp_ms']))/1000)

        def surprise(key,value):
            baseline=self.baselines.get(key,value)
            change=abs(value-baseline)
            self.baselines[key]=baseline+(1-math.exp(-dt/5))*(value-baseline)
            return min(1.,change/.5)

        partial=voice.get('last_streaming_transcript') or {}
        call=(directed_call(str(partial.get('text') or '')) and partial.get('source')!='robot_echo'
              and 0<=stamp-int(partial.get('emitted_ms') or 0)<800)
        fresh_people=[p for p in people if p.face_visible and 0<=stamp-p.stamp_ms<600]
        fresh_tracks=[t for t in tracks if t.voice_activity and 0<=stamp-t.stamp_ms<350
                      and t.self_echo_probability<.4 and not t.target_speech_rejected]

        links={};voice_of_track={};onsets=set();bound_now=set();self.binding_notes=[]
        # One visible mouth moving and one voice is the binding, whatever the
        # array says. Direction disambiguates a crowd; it is not a precondition,
        # and on an uncalibrated array a disagreement is more likely a bearing
        # error than a second hidden speaker. Requiring agreement here is what
        # silenced the robot as soon as a microphone array started reporting
        # angles at all.
        alone=len(fresh_people)==1 and len(fresh_tracks)==1
        for t in fresh_tracks:
            matched=[]
            for p in fresh_people:
                if not p.lip_motion or abs(t.stamp_ms-p.stamp_ms)>200:continue
                known=t.azimuth_deg is not None and p.azimuth_deg is not None
                gap=abs((t.azimuth_deg-p.azimuth_deg+180)%360-180) if known else None
                if known and gap>=30 and not alone:continue
                if known and gap>=30:
                    self.binding_notes.append({'track_id':t.track_id,'person_id':p.person_id,
                                               'bearing_gap_deg':round(gap,1),
                                               'note':'bound_as_only_pair_despite_bearing_disagreement'})
                matched.append(p)
            if len(matched)==1:
                p=matched[0]
                if t.speaker_label not in {'','unknown',p.person_id} and t.speaker_similarity>=.48:continue
                links['audio:'+t.track_id]=('visual:'+p.person_id,.8)
                links['visual:'+p.person_id]=('audio:'+t.track_id,.8)
                voice_of_track[t.track_id]=p.person_id
                # Somebody facing us who starts speaking is an onset, not a level.
                # The host fires it once per episode; the algorithm decides what
                # it is worth. Same evidence bar as the fast path it replaces.
                if p.gaze_score>=.65 and p.body_facing_score>=.6:
                    pair=(t.track_id,p.person_id)
                    if pair not in self.bound:onsets.add(p.person_id)
                    bound_now.add(pair)
        if not context.get('crossmodal_enabled',True):links={};onsets.clear()
        self.bound=bound_now

        def memory_of(person):
            return {name:unit(person_memory.get(person,{}).get(name,0.)) for name in MEMORY_FIELDS}

        def gesture_of(p):
            """A gesture is an event: the same wave must not be scored every frame."""
            label=str(p.gesture or '').lower()
            score=unit(p.gesture_score)
            if not label or score<=0:
                self.last_gesture.pop(p.person_id,None)
                return {}
            if self.last_gesture.get(p.person_id)==label:return {}
            self.last_gesture[p.person_id]=label
            if any(word in label for word in REJECTING_GESTURES):
                return {EXTRA_GESTURE_REJECT:score}
            if any(word in label for word in INVITING_GESTURES):
                # Beckoning counts as an approach to us only to the degree they
                # are actually turned towards us.
                return {EXTRA_GESTURE_INVITE:score*unit(p.body_facing_score)}
            return {}

        def returned_of(person,stamp_ms):
            """Coming back after being away is a change, so it lifts habituation."""
            previous=self.last_seen.get(person)
            self.last_seen[person]=stamp_ms
            if len(self.last_seen)>32:
                self.last_seen={k:v for k,v in self.last_seen.items() if stamp-v<300000}
            return previous is not None and stamp_ms-previous>=RETURN_ABSENCE_MS

        def extras_of(person,azimuth_deg=None,gesture=None):
            """Evidence added after the ABI froze rides in the key/value array.

            Bearing is only sent when it was actually measured: an algorithm
            with a spatial inhibitory surround must not suppress a neighbour on
            a guessed direction.
            """
            extra={}
            familiarity=unit(person_memory.get(person,{}).get('memory_familiarity',0.))
            if familiarity>0:extra[EXTRA_MEMORY_FAMILIARITY]=familiarity
            open_exchange=unit(person_memory.get(person,{}).get('memory_open_exchange',0.))
            if open_exchange>0:extra[EXTRA_OPEN_EXCHANGE]=open_exchange
            if person in onsets:extra[EXTRA_AV_SYNC_ONSET]=1.
            extra.update(gesture or {})
            if azimuth_deg is not None and math.isfinite(float(azimuth_deg)):
                extra[EXTRA_AZIMUTH_DEG]=float(azimuth_deg)
            return {'extra':extra} if extra else {}

        def language_of(person,track_id):
            """Language belongs to the utterance, so only its speaker carries it."""
            if not language:return {}
            if language_speaker:
                if person!=language_speaker:return {}
            elif not (track_id and language_track and track_id==language_track):
                return {}
            return {name:unit(language.get(name,0.)) for name in LANGUAGE_FIELDS}

        candidates=[]
        for p in fresh_people:
            key='visual:'+p.person_id
            # Gaze expresses a current social purpose, so habituation must not
            # erase a still-engaged partner. Identity only contributes a small prior.
            goal=max(supported.get(key,0),unit((p.gaze_score-.45)/.5)*unit(p.body_facing_score))
            salience=unit(.6*p.gaze_score+.4*p.body_facing_score)*unit(p.face_confidence)
            link=links.get(key,('',0.))
            candidates.append(dict(
                candidate_id=key,modality='visual',person_id=p.person_id,stamp_ms=p.stamp_ms,ttl_ms=600,
                salience=salience,goal=goal,surprise=surprise(key,salience),
                importance=max(.5 if p.role=='owner' and p.identity_confidence>=.6 else 0.,unit(importance.get(key,0))),
                uncertainty=1-unit(p.identity_confidence),observability=unit(p.face_confidence),
                gaze=unit(p.gaze_score),body_facing=unit(p.body_facing_score),
                lip_sync=1. if p.lip_motion else 0.,voice_activity=0.,
                identity_confidence=unit(p.identity_confidence),
                motivation=unit(motivation.get(key,0)),link_id=link[0],link_confidence=link[1],
                change_id=f"return:{p.person_id}:{p.stamp_ms}" if returned_of(p.person_id,p.stamp_ms) else '',
                event_id=event_id,**memory_of(p.person_id),
                **extras_of(p.person_id,p.azimuth_deg,gesture_of(p)),
                **language_of(p.person_id,None)))

        bio=((voice.get('audio_scene') or {}).get('acoustic') or {}).get('bio') or {}
        novelty=bool(bio.get('novelty_onset') and 0<=stamp-int(bio.get('stamp_ms') or 0)<200)

        # One candidate per person in the conversation, not one per microphone
        # track. The ear used to hand up a single dominant track, so with three
        # people talking attention had one thing to choose from and could not
        # switch between speakers at all. When the ear reports speakers, they are
        # the auditory candidates and the live track's acoustics are attached to
        # whoever is speaking right now; with no speaker model we fall back to
        # tracks, which is the old behaviour.
        speakers=[row for row in ((voice.get('speakers') or {}).get('speakers') or [])
                  if row.get('speaker_id') and int(row.get('age_ms') or 10**9)<=3000]
        if speakers and fresh_tracks:
            live=min(speakers,key=lambda row:int(row.get('age_ms') or 10**9))
            track=fresh_tracks[0]
            for row in speakers:
                key='audio:'+str(row['speaker_id'])
                speaking=int(row.get('age_ms') or 10**9)<=500
                person=str(row.get('person_id') or '')
                if not person and speaking:person=voice_of_track.get(track.track_id,'')
                link=links.get('audio:'+track.track_id,('',0.)) if row is live else ('',0.)
                if link[0]:
                    links[link[0]]=(key,link[1])
                recency=max(0.,1.-int(row.get('age_ms') or 0)/3000.)
                goal=max(supported.get(key,0),1. if call and speaking else .8 if link[0] else 0.)
                candidates.append(dict(
                    candidate_id=key,modality='audio',person_id=person,
                    stamp_ms=int(row.get('last_heard_ms') or track.stamp_ms),ttl_ms=3000,
                    salience=(unit(track.speech_probability)*unit(track.clarity)) if speaking else .35*recency,
                    goal=goal,
                    surprise=unit(bio.get('salience_score',0)) if novelty and speaking else 0.,
                    change_id=f"sound:{bio.get('stamp_ms')}" if novelty and speaking else '',
                    observability=unit(track.clarity) if speaking else recency,
                    uncertainty=1-unit(row.get('identity_confidence',0)),
                    voice_activity=1. if speaking else 0.,
                    lip_sync=1. if (row is live and voice_of_track.get(track.track_id)) else 0.,
                    identity_confidence=unit(row.get('identity_confidence',0)),
                    link_id=link[0],link_confidence=link[1],
                    event_id=event_id,**memory_of(person),
                    **extras_of(person,row.get('azimuth_deg')),
                    **language_of(person,track.track_id if speaking else None)))
            fresh_tracks=[]

        for t in fresh_tracks:
            key='audio:'+t.track_id
            link=links.get(key,('',0.))
            person=t.speaker_label if t.speaker_label not in {'','unknown','robot'} and t.speaker_similarity>=.48 else ''
            if not person:person=voice_of_track.get(t.track_id,'')
            goal=max(supported.get(key,0),1. if call and partial.get('track_id')==t.track_id else .8 if link[0] else 0.)
            candidates.append(dict(
                candidate_id=key,modality='audio',person_id=person,stamp_ms=t.stamp_ms,ttl_ms=350,
                salience=unit(t.speech_probability)*unit(t.clarity),goal=goal,
                surprise=unit(bio.get('salience_score',0)) if novelty else 0.,
                change_id=f"sound:{bio.get('stamp_ms')}" if novelty else '',
                observability=unit(t.clarity),uncertainty=1-unit(t.speaker_similarity),
                voice_activity=1.,lip_sync=1. if voice_of_track.get(t.track_id) else 0.,
                identity_confidence=unit(t.speaker_similarity),
                link_id=link[0],link_confidence=link[1],
                event_id=event_id,**memory_of(person),**extras_of(person,t.azimuth_deg),
                **language_of(person,t.track_id)))

        for obj in objects:
            key=obj.get('candidate_id')
            if not key:continue # anonymous boxes cannot support persistent habituation
            candidates.append(dict(
                candidate_id=key,modality='visual',person_id='',stamp_ms=int(obj.get('stamp_ms') or 0),ttl_ms=600,
                salience=unit(obj.get('salience',0)),goal=supported.get(key,0),
                uncertainty=unit(obj.get('uncertainty',0)),observability=unit(obj.get('confidence',0)),
                motivation=unit(motivation.get(key,0)),change_id=str(obj.get('change_id') or '')))

        active=self.registry.active
        engine=getattr(active,'engine',None)
        if engine is not None:
            for event in context.get('observation_results') or []:
                if event.get('candidate_id') in {c['candidate_id'] for c in candidates} and event.get('id'):
                    engine.observe_result(event['id'],event['candidate_id'],event.get('problem','identity'),
                                          event.get('condition','normal'),event.get('completed',False),
                                          event.get('new_information',False))

        keys={c['candidate_id'] for c in candidates}
        self.baselines={k:v for k,v in self.baselines.items() if k in previous or k in keys}

        result=self.registry.update(stamp,candidates,sources,arousal_gain=context.get('arousal_gain',1.),
                                    robot_speaking=bool((voice.get('playback') or {}).get('speaking') or voice.get('robot_speaking')),
                                    reflex_active=bool(context.get('reflex_active')))
        if result is None:
            return self.result
        # The algorithm ranks candidates; only the adapter knows which person a
        # candidate resolved to, so it stamps that back for the workspace.
        owners={c['candidate_id']:c.get('person_id') or None for c in candidates}
        for key,row in (result.get('candidates') or {}).items():
            row['person_id']=owners.get(key)
        result['sources']=context.get('sources',{})
        result['familiarity']=context.get('familiarity_detail') or {}
        result['binding_notes']=list(self.binding_notes)
        result['familiarity_status']=context.get('familiarity_status') or {}
        result['internal_state_used']='arousal_gain' in context
        result['feedback_supported']=engine is not None
        result['algorithm']={'id':getattr(active,'id',''),'kind':getattr(active,'kind',''),
                             'load_errors':self.registry.errors}
        # Shares are requested extra capacity, not CPU percentages or action permission.
        result['allocation_requests']=[{'candidate_id':k,'share':v['a'],
                                        'information_priority':v['a']*v.get('uncertainty',0.)*v.get('learning',0.),
                                        'executable':False,'reason':'extra_perception_scheduler_not_connected'}
                                       for k,v in result['candidates'].items() if v.get('valid')]
        self.result=result
        return result
