"""Visual salience with separate acquisition, hold and release thresholds."""
from dataclasses import asdict


class VisualAttentionFocus:
    def __init__(self,dwell_ms=800):
        self.distribution=None
        self.dwell_ms=dwell_ms
        self.person=None;self.since=None;self.last_ms=None
        self.locked=None;self.last_good_ms=None;self.away_since=None
        self.loud_since=None;self.last_loud_ms=None

    def reset_gaze(self):
        self.person=None;self.since=None;self.last_ms=None
        self.locked=None;self.last_good_ms=None;self.away_since=None

    def adopt(self,person,stamp):
        """Dialogue -> quiet gaze carries the same target, without reacquisition."""
        self.person=person.person_id;self.since=stamp-self.dwell_ms;self.last_ms=stamp
        self.locked=person;self.last_good_ms=stamp;self.away_since=None

    def _focus(self,state,stamp,person,observed):
        return dict(state,stamp_ms=stamp,target_id='person:'+person.person_id,target_kind='person',
                    person_id=person.person_id,confidence=.7*person.gaze_score+.3*person.face_confidence,
                    listen=observed,addressed_to_robot=False,source_track_id=None,azimuth_deg=person.azimuth_deg,
                    focus_status='tracking' if observed else 'held_briefly',focus_observed=observed,
                    focus_age_ms=max(0,stamp-(self.last_good_ms or stamp)),algorithm='visual_engagement',
                    reasons=['sustained_gaze','visual_focus_waiting_for_speech'] if observed else ['brief_visual_dropout_hold','no_dialogue_permission'])

    def update(self,stamp,selected,people,tracks,acoustic=None,reflex=None):
        state=dict(selected) if isinstance(selected,dict) else asdict(selected)
        if reflex and reflex.get('active') and 0<=stamp-int(reflex.get('stamp_ms') or 0)<1800:
            self.reset_gaze()
            return dict(state,stamp_ms=stamp,target_id=reflex.get('target_id','environment'),target_kind='sound_source' if reflex.get('source')=='audio' else 'visual_event',
                        person_id=None,confidence=.85,listen=False,addressed_to_robot=False,source_track_id=None,
                        azimuth_deg=(reflex.get('spatial') or {}).get('azimuth_deg'),algorithm='reflex_orienting',reasons=['reflex_orienting','no_dialogue_permission'])
        state.update(listen=False,addressed_to_robot=False,source_track_id=None)
        acoustic=acoustic or {}
        fresh=0<=stamp-int(acoustic.get('stamp_ms') or 0)<400
        level=acoustic.get('raw_rms_dbfs',acoustic.get('rms_dbfs',-120))
        loud=fresh and level is not None and (level>=-18 or ((acoustic.get("bio") or {}).get("novelty_onset") and (acoustic.get("bio") or {}).get("salience_score",0)>=.8)) and float(acoustic.get('echo_probability') or 0)<.4
        if loud:
            if self.loud_since is None:self.loud_since=stamp
            self.last_loud_ms=stamp
        elif not fresh or level is None or level<-25:self.loud_since=None
        if loud and stamp-self.loud_since<900:
            self.reset_gaze()
            return dict(state,stamp_ms=stamp,target_id='sound:salient',target_kind='sound_source',person_id=None,confidence=.75,
                        listen=False,addressed_to_robot=False,source_track_id=None,azimuth_deg=acoustic.get('direction_deg') if acoustic.get('direction_valid') else None,
                        algorithm='acoustic_salience',reasons=['loud_sound_onset','no_dialogue_permission'])
        distribution=self.distribution or {}
        selected_key=(distribution.get('focus') or {}).get('visual')
        selected_person=selected_key.removeprefix('visual:') if selected_key and selected_key.startswith('visual:') else None
        fresh_people=[p for p in people if p.face_visible and 0<=stamp-p.stamp_ms<600]
        if self.locked is not None:
            current=next((p for p in fresh_people if p.person_id==self.locked.person_id),None)
            good=bool(current and current.face_confidence>=.65 and current.gaze_score>=.5 and current.body_facing_score>=.45)
            challenger=next((p for p in fresh_people if p.person_id==selected_person and p.gaze_score>=.72 and p.face_confidence>=.8),None)
            if good and challenger and challenger.person_id!=current.person_id:
                self.adopt(challenger,stamp)
                return self._focus(state,stamp,challenger,True)
            if good:
                self.locked=current;self.last_good_ms=stamp;self.last_ms=stamp;self.away_since=None
                return self._focus(state,stamp,current,True)
            away=bool(current and (current.gaze_score<.25 or current.body_facing_score<.25))
            if away and self.away_since is None:self.away_since=stamp
            if not away:self.away_since=None
            release=(self.away_since is not None and stamp-self.away_since>=300) or stamp-(self.last_good_ms or 0)>650
            if not release:return self._focus(state,stamp,self.locked,False)
            self.reset_gaze()
        candidates=[p for p in fresh_people if p.face_confidence>=.8 and p.gaze_score>=.72 and p.body_facing_score>=.55]
        def score(p):return .7*p.gaze_score+.3*p.body_facing_score+(.04 if p.role=='owner' and p.identity_confidence>=.6 else 0)+(.04 if p.person_id==self.person else 0)
        candidates.sort(key=score,reverse=True)
        if not candidates or (len(candidates)>1 and score(candidates[0])-score(candidates[1])<.035):
            # Brief fluctuations during acquisition need not erase all prior dwell.
            near=next((p for p in fresh_people if p.person_id==self.person and p.gaze_score>=.5 and p.face_confidence>=.65),None)
            if not near or self.last_ms is None or stamp-self.last_ms>300:self.reset_gaze()
            return state
        person=candidates[0]
        if self.person!=person.person_id or self.last_ms is None or stamp-self.last_ms>400:
            self.person=person.person_id;self.since=stamp
        self.last_ms=stamp
        if stamp-self.since<self.dwell_ms:return state
        if self.distribution is not None and selected_person!=person.person_id:return state
        self.adopt(person,stamp)
        return self._focus(state,stamp,person,True)


OwnerVisualFocus=VisualAttentionFocus
