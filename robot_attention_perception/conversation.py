"""Bind each ASR turn to contemporaneous attention evidence, not a later frame."""
from collections import deque
from dataclasses import asdict, replace
import re
from .fusion import AttentionFusion, FusionConfig
from .types import AcousticTrack, PerceptionFrame, RobotState, TranscriptRecord


from .interaction_intent import directed_call
from .interaction_fusion import finalize_gate
from .turn_taking import interruption_plan
from .group_conversation import GroupConversation


class ConversationObserver:
    def __init__(self):
        self.history = deque(maxlen=1200)
        self.live_gates = deque(maxlen=1200)
        self.turns = deque(maxlen=256)
        self.seen = set()
        self.unified_policy = False
        self.group=GroupConversation()
        self.interruption = None
        self.visual_speech_since = {}
        # Set by the runtime from the saved console choice; the defaults live in
        # interaction_fusion so the algorithm owns its own parameters.
        self.addressee_weights = None
        # The scoring of the most recent finished sentence, for the console: the
        # point of tunable weights is being able to see what they just did.
        self.last_addressee = None
        # Who we are in the middle of talking with, kept without reference to the
        # camera: granted speech from them, and our own playback, both extend it.
        self.exchange = None
        # Which voices belong to which person, learned from the utterances where
        # the camera named somebody whose voiceprint had only been filed as a
        # cluster. This is what lets a sentence said off camera be answered
        # when its speaker turns round: the held sentence is under the cluster,
        # attention arrives on the face, and without the binding they are two
        # different people.
        self.voice_bindings: dict[str, dict[str, int]] = {}
        self.episode_start = 0
        self.last_utterance_end = 0

    BINDING_MEMORY_MS = 600000
    BINDING_TOLERANCE_DEG = 25      # array direction is not that precise
    BINDING_WINDOW_MS = 30000       # a voice heard longer ago than this is not this face

    def _bind_by_direction(self, stamp, payload, people):
        """The crowd talked off camera; now one of them is on camera.

        Nothing named those voices while nobody could be seen — they are
        clusters. When a face appears where one of those voices was coming
        from, and there is exactly one face in that direction, the voice and
        the face are the same person. This is the only evidence available in
        the case the field kept hitting: people talking, one of them turning
        round, and the sentence they said a moment ago sitting under an
        identity nobody recognises.

        Exactly one candidate on purpose. Two people standing in the same
        direction is not a binding, it is a coin toss.
        """
        speakers = ((payload.get("speakers") or {}).get("speakers")) or []
        seen = [person for person in people
                if person.face_visible and float(person.identity_confidence or 0) >= .5
                and person.azimuth_deg is not None
                and str(person.person_id or "") not in {"", "unknown", "robot"}]
        if not seen:
            return
        for track in speakers:
            cluster = str(track.get("speaker_id") or "")
            azimuth = track.get("azimuth_deg")
            if not cluster or track.get("person_id") or azimuth is None:
                continue
            if stamp - int(track.get("last_heard_ms") or 0) > self.BINDING_WINDOW_MS:
                continue
            near = [person for person in seen
                    if abs(float(person.azimuth_deg) - float(azimuth)) <= self.BINDING_TOLERANCE_DEG]
            if len(near) == 1:
                self._bind_voice(str(near[0].person_id), cluster, stamp)

    def _bind_voice(self, person_id, cluster_id, stamp):
        if not person_id or not cluster_id or person_id == cluster_id:
            return
        bound = self.voice_bindings.setdefault(str(person_id), {})
        bound[str(cluster_id)] = int(stamp)
        for person, voices in list(self.voice_bindings.items()):
            for voice, when in list(voices.items()):
                if stamp - when > self.BINDING_MEMORY_MS:
                    voices.pop(voice, None)
            if not voices:
                self.voice_bindings.pop(person, None)

    def gate(self, transcript, track, people, stamp, speaking):
        track_id = str(transcript.get("track_id") or (track.track_id if track else ""))
        text = str(transcript.get("text") or "")
        record = TranscriptRecord(track_id=track_id, text=text,
                                  is_final=bool(transcript.get("is_final")), clarity=float(transcript.get("clarity") or .7))
        if track is None:
            track = AcousticTrack(track_id=track_id, stamp_ms=stamp, voice_activity=True,
                                  speech_probability=1., clarity=.7,
                                  self_echo_probability=float(transcript.get("self_echo_probability") or 0))
        track = replace(track, transcript=record, stamp_ms=stamp)
        # Wake mentions inside third-party speech are not directed calls.
        fusion = AttentionFusion(FusionConfig(use_transcript_wake_words=False))
        state = asdict(fusion.update(PerceptionFrame(stamp_ms=stamp, acoustic_tracks=[track],
                                                   vision_people=people, robot=RobotState(stamp_ms=stamp, speaking=speaking))))
        person = next((p for p in people if p.person_id == state.get("person_id")), None)
        visual_address = bool(person and person.gaze_score >= .58 and
                              (person.lip_motion or person.body_facing_score >= .68))
        call = directed_call(text)
        clean = track.self_echo_probability < .65 and not track.target_speech_rejected
        addressed = clean and (call or visual_address)
        confidence = max(float(state["confidence"]), .75 if call and clean else 0.)
        state.update(addressed_to_robot=addressed, listen=addressed and confidence >= .62,
                     confidence=confidence, source_track_id=track_id,
                     reasons=list(state.get("reasons", [])) + (["directed_call"] if call else []) +
                     (["gaze_and_facing"] if visual_address else []))
        return state

    def update(self, stamp, payload, tracks, people, speaking, live_attention=None):
        track = tracks[0] if tracks else None
        # Store actual live evidence without stale final-transcript overrides.
        self.history.append((stamp, track, list(people), speaking))
        if live_attention is not None:
            self.live_gates.append((stamp, dict(live_attention)))
            self.unified_policy = self.unified_policy or live_attention.get("policy")=="unified_av_interaction_v1"
        self._bind_by_direction(stamp, payload, people)
        partial = payload.get("last_streaming_transcript") or {}
        vad = payload.get("last_vad") or {}
        fresh = 0 <= stamp - int(partial.get("emitted_ms") or 0) < 800
        usable_partial = fresh and not partial.get("is_final") and partial.get("source") != "robot_echo"
        if speaking and self.exchange:
            self.exchange = dict(self.exchange, stamp_ms=stamp)
        gate = self.gate(partial if usable_partial else {}, track, people, stamp, speaking) if track else {}
        if (live_attention is not None and track and live_attention.get("source_track_id")==track.track_id
                and track.self_echo_probability<.65 and not track.target_speech_rejected and not directed_call(str(partial.get("text") or ""))):
            gate = dict(live_attention)
        if self.unified_policy:
            gate=dict(live_attention or {})
        matched = next((person for person in people if person.person_id == gate.get("person_id")), None)
        # A mouth moving while the robot talks is not yet an interruption: people
        # look at the robot and react while it speaks. Only a mouth that keeps
        # moving, with the microphone not hearing our own playback, is somebody
        # actually starting to talk over us.
        quiet_microphone = track is None or track.self_echo_probability < .3
        for person in people:
            key = str(person.person_id or '')
            if key and person.lip_motion and person.gaze_score >= .65 and quiet_microphone:
                self.visual_speech_since.setdefault(key, stamp)
            else:
                self.visual_speech_since.pop(key, None)
        started = self.visual_speech_since.get(str(matched.person_id or '')) if matched else None
        visual_speech = bool(started is not None and stamp - started >= 400)
        self.interruption = None
        social=interruption_plan(partial.get("text") if usable_partial else "",speaking,payload,stamp,track.track_id if track else None)
        # While the robot is talking, the most likely source of "speech" is the
        # robot. AEC leaves a residual and the recogniser will happily transcribe
        # it into something that does not match the playback text, at which point
        # the robot interrupts itself — which is what it kept doing in the field.
        own_voice = track is not None and track.self_echo_probability >= .3
        # Words, not movement. A mouth moving while the robot talks is somebody
        # reacting to it; only somebody actually saying something takes the floor.
        # The rule the robot is held to: either do not start, or finish — being
        # cut off mid-sentence is worse than answering half a second later.
        said_something = bool(usable_partial and str(partial.get("text") or "").strip())
        if (social["mode"]!="continue" and track and vad.get("active") and not own_voice
                and gate.get("listen") and gate.get("addressed_to_robot")
                and said_something and (visual_speech or not speaking)):
            segment = payload.get("active_utterance") or {}
            self.interruption = {"key": f"{track.track_id}:{segment.get('started_ms') or partial.get('started_ms')}",
                                 "stamp_ms": stamp, "transcript": partial if usable_partial else {}, "attention": gate,
                                 "source": "asr_attention" if usable_partial else "visual_audio_attention", "plan":social}
        incoming = payload.get("transcripts") or [payload.get("last_transcript") or {}]
        for transcript in incoming:
            if not transcript.get("is_final") or not str(transcript.get("text") or "").strip():
                continue
            key = transcript.get("utterance_id") or f"{transcript.get('track_id')}:{transcript.get('emitted_ms')}"
            if key in self.seen:
                continue
            self.seen.add(key)
            if len(self.seen) > 4096:
                self.seen = {item["utterance_id"] for item in self.turns} | {key}
            end = int(transcript.get("ended_ms") or transcript.get("emitted_ms") or stamp)
            candidates = [h for h in self.history if max(int(transcript.get("started_ms") or end-800),end-800)<=h[0]<=end+100 and h[1] and h[1].track_id == transcript.get("track_id")]
            sample = min(candidates, key=lambda h: abs(h[0] - end)) if candidates else None
            if sample:
                _, evidence, visible, playing = sample
                evidence = replace(evidence, self_echo_probability=float(transcript.get("self_echo_probability") or 0))
            else:
                evidence, visible, playing = None, [], False
            attention = self.gate(transcript, evidence, visible, end, playing)
            if not sample and not directed_call(str(transcript.get("text"))):
                attention.update(listen=False, addressed_to_robot=False, reasons=["missing_contemporaneous_evidence"])
            recorded = [(ts,g) for ts,g in self.live_gates if end-800 <= ts <= end+100
                        and g.get("source_track_id") == transcript.get("track_id")]
            selected = min(recorded, key=lambda item:abs(item[0]-end)) if recorded else None
            if selected and not directed_call(str(transcript.get("text"))):
                selected_ms, selected_gate = selected
                clean = float(transcript.get("self_echo_probability") or 0)<.65 and not (evidence and evidence.target_speech_rejected)
                attention = dict(selected_gate)
                allowed = bool(clean and attention.get("listen") and attention.get("addressed_to_robot") and float(attention.get("confidence") or 0)>=.62)
                attention.update(listen=allowed,addressed_to_robot=allowed,
                                 reasons=list(attention.get("reasons",[]))+["contemporaneous_live_gate"],
                                 gate_observed_ms=selected_ms)
            if self.unified_policy:
                attention=finalize_gate(transcript,self.live_gates,end,evidence,self.addressee_weights,self.exchange,
                                        [(entry[0],entry[2]) for entry in self.history])
                if attention.get('addressee_detail'):
                    self.last_addressee=dict(attention['addressee_detail'],
                                             text=str(transcript.get('text') or '')[:40],
                                             person_id=attention.get('person_id'),
                                             granted=bool(attention.get('addressed_to_robot')),
                                             held=bool(attention.get('held_for_attention')),
                                             stamp_ms=end)
            if (transcript.get("separation") or {}).get("mode")=="mixed_fallback":
                attention.update(listen=False,addressed_to_robot=False,reasons=["unresolved_overlapping_speech"])
            speaker = transcript.get("speaker") or {}
            speaker_id = str(speaker.get("speaker_id") or "unknown")
            # The gate now says separately who the camera saw; attribution by
            # memory or by an open exchange is not a visual claim.
            visual_id = str(attention.get("visual_person_id") or "") or (
                (attention.get("person_id") or "unknown") if attention.get("target_kind", "person")=="person" else "unknown")
            # Three kinds of claim about who spoke, and they are not equal.
            # An enrolled voiceprint match is a real identity. A face the camera
            # is confident about is a real identity. An auto-created voice
            # cluster is only a handle for "this voice, whoever it is" — and
            # remember_stranger hands those back with similarity 1.0, which used
            # to make a cluster minted one second ago outrank the owner's own
            # face and get his sentence filed under stranger_c3b41448.
            role = str(speaker.get("speaker_role") or "")
            similarity = float(speaker.get("similarity") or 0)
            enrolled_voice = role != "stranger" and speaker_id not in {"unknown", "robot"} and similarity >= .48
            voice_cluster = role == "stranger" and speaker_id not in {"unknown", "robot"}
            visual_confidence = next((p.identity_confidence for p in visible if p.person_id == visual_id), 0.0)
            named_by_face = visual_id not in {"", "unknown", "robot"} and float(visual_confidence or 0) >= .5
            reliable_voice = enrolled_voice
            if enrolled_voice:
                person_id = speaker_id
            elif named_by_face:
                person_id = visual_id
            elif voice_cluster:
                person_id = speaker_id
            else:
                person_id = visual_id
            if named_by_face and voice_cluster:
                self._bind_voice(visual_id, speaker_id, end)
            identity_conflict = bool(enrolled_voice and visual_id not in {"unknown", "", speaker_id})
            identity_resolution = ("voice_priority_conflict" if identity_conflict
                                   else "audio_visual_agreement" if enrolled_voice and visual_id == speaker_id
                                   else "voice_only" if enrolled_voice
                                   else "face_over_voice_cluster" if named_by_face and voice_cluster
                                   else "face_only" if named_by_face
                                   else "voice_cluster_only" if voice_cluster
                                   else "visual_or_unresolved")
            if attention.get('addressed_to_robot') and person_id not in {None,'unknown','robot'}:
                self.exchange={'person_id':person_id,'stamp_ms':end}
            if 'answer_to_our_question_unverified_voice' in (attention.get('reasons') or ()):
                # Attributed by the shape of the conversation, not by recognising
                # the voice. Downstream must be able to tell those apart.
                identity_resolution = "answer_continuity_unverified"
            source = transcript.get("source", "microphone")
            if source == "robot_echo":
                attention.update(listen=False, addressed_to_robot=False, reasons=["robot_playback_echo"])
                person_id = "robot"
            attention["visual_person_id"] = visual_id
            attention["identity_conflict"] = identity_conflict
            attention["person_id"] = person_id
            if not self.episode_start or end - self.last_utterance_end > 30000:
                self.episode_start = int(transcript.get("started_ms") or end)
            self.last_utterance_end = max(self.last_utterance_end, end)
            recent_reply=(payload.get("playback") or {})
            after_reply=bool(recent_reply.get("turn_id") and 0<=int(transcript.get("started_ms") or 0)-int(recent_reply.get("ended_ms") or 0)<1500)
            act=interruption_plan(transcript.get("text"),playing or after_reply,payload,stamp,transcript.get("track_id"))
            self.turns.append({"speech_act":act["speech_act"],"turn_taking":act,"source": source, "evidence": {
                                   "audio": asdict(evidence) if evidence else None,
                                   "vision": [{key: asdict(person).get(key) for key in ("person_id", "face_id", "identity_confidence", "stamp_ms", "role", "azimuth_deg", "face_confidence", "gaze_score", "body_facing_score", "lip_motion", "emotion_label", "emotion_valid")} for person in visible],
                                   "policy": "unified_av_interaction_v1" if self.unified_policy else "conversation_gate", "observed_at_ms": sample[0] if sample else None, "transcript_end_ms":end,
                               }, "identity_conflict": identity_conflict, "identity_resolution":identity_resolution, "visual_person_id":visual_id, "temporal_group_id": str(self.episode_start), "utterance_id": key, "transcript": transcript, "attention": attention,
                               "person_id": person_id, "identity_confidence": float(speaker.get("similarity") or 0),
                               "voice_identity_confidence": float(speaker.get("similarity") or 0),
                               "visual_identity_confidence": next((p.identity_confidence for p in visible if p.person_id==visual_id),0.0),
                               "identity_source": ("voice_enrollment" if enrolled_voice else "face_recognition" if named_by_face
                                                   else "voice_cluster" if voice_cluster
                                                   else "av_association" if person_id not in {"unknown", ""} else "unresolved"),
                               "voice_cluster_id": speaker_id if voice_cluster else None,
                               "addressee": "robot" if attention.get("addressed_to_robot") else "non_robot_or_unknown",
                               # Probably meant for us, not certainly. Written down
                               # rather than answered; if attention settles on this
                               # person shortly, Brain answers from it.
                               "held_for_attention": bool(attention.get("held_for_attention")),
                               "addressee_detail": attention.get("addressee_detail"),
                               "visible_participants": [p.person_id for p in visible],
                               "overlap_probability": transcript.get("overlap_probability", 0),
                               "stamp_ms": end})
            self.group.annotate(self.turns[-1])
