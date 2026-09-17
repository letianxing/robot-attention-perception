from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import queue
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import Any
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler, urlopen

import numpy as np

from .cocktail_view import cocktail_rows
from .conversation import ConversationObserver
from .reflex_attention import ReflexAttention
from .live_crossmodal import LiveCrossmodal
from .interaction_fusion import addressee_weights, ADDRESSEE_LABELS, InteractionFusion
from .decision_trace import DecisionTrace
from .attention_sources import SOURCE_NAMES, AttentionSources
from .familiarity import FamiliarityIndex
from .global_workspace import GlobalWorkspace
from .affect_candidates import AffectCandidates
from .presence_candidates import PresenceCandidates
from .memory_candidates import MemoryCandidates
from .async_fusion import AsyncPerceptionBuffer
from .av_binding import AudioFeatureFrame, BinderEngine, VideoFeatureFrame
from .dialogue import AcknowledgementBackend, OpenAiCompatibleBackend, SpeechToSpeechResponder
from .person_manager import IdentityEvidence, ProbabilisticPersonManager
from .rosbridge_transport import RosbridgeTransport
from .types import AcousticTrack, AttentionState, RobotState, TranscriptRecord, VisionPerson


ATTENTION_SOURCE_LABELS = {
    "audio_visual": ("视听感知", "摄像头与麦克风的当前观测：注视、朝向、唇动、声源与声纹。关闭后跨模态关联证据不再进入注意力，硬件与安全反射不受影响。"),
    "memory_context": ("记忆上下文", "本会话工作记忆：近期参与者、正在进行的对话对象、机器人还在等谁回答。来自 Brain 快照，过期 1200ms 即撤销。"),
    "linguistic_context": ("当前语境文本", "进行中这句话的通用语言特征：称呼形式、问句/应答句式、与机器人参与过的对话的词面延续。不是语义理解，也不调用大模型。"),
    "cross_session_memory": ("跨会话熟悉度", "从记忆服务读取这个人在以往会话里真实交谈过多少次，作为「如何理解一次沉默注视」的弱先验。后台线程查询，快循环只读缓存；查不到与没记录分开标注。熟悉本身不能把谁认定为在对机器人说话。"),
    "internal_state": ("内部状态", "外部内部状态服务的唤醒调制与候选偏好。当前没有生产者，缺失时标记 missing，不生成心理数值。"),
}


def now_ms() -> int:
    return int(time.time() * 1000)


def get_json(url: str, timeout: float = 0.25) -> dict[str, Any]:
    opener = _local_opener(url)
    with opener.open(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 0.5) -> dict[str, Any]:
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    opener = _local_opener(url)
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _local_opener(url: str):
    host = (urlparse(url).hostname or "").lower()
    if host in {"127.0.0.1", "localhost", "::1"}:
        return build_opener(ProxyHandler({}))
    return build_opener()


def vision_people_from_payload(payload: dict[str, Any], received_ms: int) -> list[VisionPerson]:
    state = payload.get("state") if isinstance(payload.get("state"), dict) else payload
    stamp_value = state.get("stamp_ms", state.get("stamp", received_ms))
    stamp_ms = int(float(stamp_value) * 1000) if float(stamp_value) < 10_000_000_000 else int(stamp_value)
    people = []
    for item in state.get("people") or []:
        people.append(
            VisionPerson(
                person_id=str(item.get("person_id") or "unknown"),
                stamp_ms=stamp_ms,
                face_id=str(item.get("face_id") or ""),
                body_id=str(item.get("body_id") or ""),
                voice_id=str(item.get("voice_id") or ""),
                role=str(item.get("role") or "unknown"),
                azimuth_deg=_optional_number(item, "azimuth_deg", "has_azimuth"),
                elevation_deg=_optional_number(item, "elevation_deg", "has_elevation"),
                distance_m=_optional_number(item, "distance_m", "has_distance"),
                distance_confidence=float(item.get("distance_confidence", 0.0) or 0.0),
                depth_source=str(item.get("depth_source") or "none"),
                proxemic_space=str(item.get("proxemic_space") or "unknown"),
                engagement_status=str(item.get("engagement_status") or "unknown"),
                face_visible=bool(item.get("face_visible", False)),
                face_confidence=float(item.get("face_confidence", 0.0) or 0.0),
                mouth_open_ratio=float(item.get("mouth_open_ratio", 0.0) or 0.0),
                lip_motion=bool(item.get("lip_motion", False)),
                mouth_roi_features=tuple(float(value) for value in item.get("mouth_roi_features", ())),
                gaze_score=float(item.get("gaze_score", 0.0) or 0.0),
                body_facing_score=float(item.get("body_facing_score", 0.0) or 0.0),
                bbox_area_ratio=float(item.get("bbox_area_ratio", 0.0) or 0.0),
                gesture=str(item.get("gesture") or ""),
                gesture_score=float(item.get("gesture_score", 0.0) or 0.0),
                identity_confidence=float(item.get("identity_confidence", 0.0) or 0.0),
                emotion_valence=float(item.get("emotion_valence", 0.0) or 0.0),
                emotion_arousal=float(item.get("emotion_arousal", 0.0) or 0.0),
                emotion_valid=bool(item.get("emotion_valid", False)),
                emotion_label=str(item.get("emotion_label") or "unknown"),
            )
        )
    return people


def acoustic_tracks_from_payload(payload: dict[str, Any]) -> list[AcousticTrack]:
    item = payload.get("last_track")
    last_speaker = payload.get("last_speaker") if isinstance(payload.get("last_speaker"), dict) else {}
    transcript_payload = payload.get("last_transcript")
    if isinstance(item, dict) and isinstance(item.get("transcript"), dict):
        transcript_payload = item.get("transcript")
    transcript = None
    if isinstance(transcript_payload, dict) and transcript_payload.get("text"):
        transcript = TranscriptRecord(
            track_id=str(transcript_payload.get("track_id") or (item or {}).get("track_id") or ""),
            text=str(transcript_payload["text"]),
            is_final=bool(transcript_payload.get("is_final", False)),
            language=str(transcript_payload.get("language") or "unknown"),
            clarity=float(transcript_payload.get("clarity", 0.0) or 0.0),
            started_ms=transcript_payload.get("started_ms"),
            ended_ms=transcript_payload.get("ended_ms"),
            emitted_ms=transcript_payload.get("emitted_ms"),
        )
    if not isinstance(item, dict):
        # Keep the final transcript long enough for the attention poller to
        # observe it even when ASR finalization and the next fusion cycle are
        # separated by a few hundred milliseconds.
        if transcript is None or now_ms() - int(transcript.emitted_ms or 0) > 5000:
            return []
        item = {
            "track_id": transcript.track_id,
            "stamp_ms": transcript.emitted_ms,
            # The live frontend may clear last_track immediately after an
            # utterance while last_transcript is still fresh. Preserve a
            # short-lived acoustic candidate so wake-word attention can gate
            # the final transcript instead of dropping it as silence.
            "voice_activity": True,
            "speech_probability": 1.0,
            "clarity": max(float(transcript.clarity), 0.7),
        }
    elif transcript is not None:
        # ASR finalization runs on a worker and can arrive after the frontend
        # has already replaced last_track with a silent/old track. Prefer the
        # fresh transcript as the active acoustic candidate so attention does
        # not depend on callback ordering between the two fields.
        item = dict(item)
        item["track_id"] = transcript.track_id or str(item.get("track_id") or "")
        item["stamp_ms"] = transcript.emitted_ms or transcript.ended_ms or item.get("stamp_ms")
        item["voice_activity"] = True
        item["speech_probability"] = max(float(item.get("speech_probability", 0.0) or 0.0), 1.0)
        item["clarity"] = max(float(item.get("clarity", 0.0) or 0.0), float(transcript.clarity), 0.7)
        item["transcript"] = {
            "track_id": transcript.track_id,
            "text": transcript.text,
            "is_final": transcript.is_final,
            "language": transcript.language,
            "clarity": transcript.clarity,
            "started_ms": transcript.started_ms,
            "ended_ms": transcript.ended_ms,
            "emitted_ms": transcript.emitted_ms,
        }
    if last_speaker and (
        not item.get("speaker_label")
        or item.get("speaker_label") in {"unknown", ""}
        or str(item.get("track_id") or "") == str(last_speaker.get("track_id") or "")
    ):
        item = dict(item)
        item["speaker_label"] = str(last_speaker.get("speaker_id") or "unknown")
        item["speaker_similarity"] = float(last_speaker.get("similarity", 0.0) or 0.0)
    tse = payload.get("last_tse") if isinstance(payload.get("last_tse"), dict) else {}
    if tse:
        item = dict(item)
        item["target_speaker_probability"] = float(tse.get("target_probability", 1.0) or 1.0)
        item["tse_enabled"] = bool(tse.get("enabled", False))
        item["tse_healthy"] = bool(tse.get("healthy", True))
        item["tse_latency_ms"] = float(tse.get("latency_ms", 0.0) or 0.0)
        item["target_speech_rejected"] = item["target_speaker_probability"] < 0.35
    track = AcousticTrack(
        track_id=str(item.get("track_id") or "voice_mono"),
        stamp_ms=int(item.get("stamp_ms") or now_ms()),
        voice_activity=bool(item.get("voice_activity", False)),
        speech_probability=float(item.get("speech_probability", 0.0) or 0.0),
        clarity=float(item.get("clarity", 0.0) or 0.0),
        azimuth_deg=_optional_number(item, "azimuth_deg"),
        elevation_deg=_optional_number(item, "elevation_deg"),
        distance_m=_optional_number(item, "distance_m"),
        overlap_probability=float(item.get("overlap_probability", 0.0) or 0.0),
        self_echo_probability=float(item.get("self_echo_probability", 0.0) or 0.0),
        speaker_label=str(item.get("speaker_label") or "unknown"),
        speaker_similarity=float(item.get("speaker_similarity", 0.0) or 0.0),
        target_speaker_probability=float(item.get("target_speaker_probability", 1.0) or 1.0),
        tse_enabled=bool(item.get("tse_enabled", False)),
        tse_healthy=bool(item.get("tse_healthy", True)),
        tse_latency_ms=float(item.get("tse_latency_ms", 0.0) or 0.0),
        target_speech_rejected=bool(item.get("target_speech_rejected", False)),
        transcript=transcript,
    )
    if track.track_id.lower() in {"voice_demo", "demo_voice"}:
        return []
    return [
        track
    ]


def _optional_number(item: dict[str, Any], key: str, present_key: str = "") -> float | None:
    if present_key and item.get(present_key) is False:
        return None
    value = item.get(key)
    return None if value is None else float(value)


@dataclass(frozen=True)
class LiveRuntimeConfig:
    vision_url: str = "http://127.0.0.1:8080"
    voice_url: str = "http://127.0.0.1:8090"
    memory_url: str = "http://127.0.0.1:8788"
    poll_hz: float = 20.0
    algorithm: str = "ros4hri_native"
    subject_id: str = "local-user"
    robot_id: str = "reachy-mini"
    session_id: str = "mac-first-test"
    rosbridge_url: str = "ws://127.0.0.1:9090"
    s2s_enabled: bool = False
    llm_url: str = ""
    llm_model: str = "local-model"
    llm_api_key: str = ""
    vision_enabled: bool = True
    use_transcript_wake_words: bool = True
    brain_url: str = "http://127.0.0.1:8094"


class MemoryWriter:
    def __init__(self, base_url: str):
        self.url = base_url.rstrip("/") + "/v1/events"
        self.items: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=128)
        self.stop_event = threading.Event()
        self.last_error = ""
        self.thread = threading.Thread(target=self._run, name="attention-memory", daemon=True)
        self.thread.start()

    def submit(self, event: dict[str, Any]) -> None:
        try:
            self.items.put_nowait(event)
        except queue.Full:
            self.last_error = "memory_queue_full"

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                event = self.items.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                post_json(self.url, event)
                self.last_error = ""
            except Exception as exc:
                self.last_error = str(exc)


class LivePerceptionRuntime:
    def __init__(self, config: LiveRuntimeConfig):
        self.config = config
        self.buffer = AsyncPerceptionBuffer.with_algorithm(
            config.algorithm,
            use_transcript_wake_words=config.use_transcript_wake_words,
        )
        self.person_manager = ProbabilisticPersonManager()
        self.conversation = ConversationObserver()
        self.reflex_attention = ReflexAttention()
        self.interaction=InteractionFusion()
        self.decision_trace=DecisionTrace(os.environ.get("ATTENTION_TRACE_PATH",str(Path(__file__).resolve().parents[1]/".run/attention-decisions.jsonl")))
        self.crossmodal = LiveCrossmodal()
        self.crossmodal_state=[]
        self._published_utterances = deque(maxlen=512)
        self.av_binder = BinderEngine()
        self.av_decisions: list[dict[str, Any]] = []
        self._last_av_audio_ms = 0
        self._last_av_video_ms = 0
        self.memory = MemoryWriter(config.memory_url)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.events: deque[dict[str, Any]] = deque(maxlen=30)
        self.latest: dict[str, Any] = self._initial_state()
        self.ros_attention: dict[str, Any] | None = None
        self.ros_attention_ms = 0
        self.voice_pool_worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix="voice-pool")
        self.voice_pool_future=None
        self.voice_pool_seen=""
        self.attention_config_path=Path(os.environ.get("ATTENTION_CONFIG_PATH",str(Path(__file__).resolve().parents[1]/".run/attention-config.json")))
        saved=self._load_attention_config()
        self.familiarity=FamiliarityIndex(config.memory_url,config.subject_id,config.robot_id,config.session_id)
        self.attention_sources=AttentionSources(config.session_id,enabled=saved.get("sources"),familiarity=self.familiarity)
        self.workspace=GlobalWorkspace(config.session_id)
        self.memory_candidates=MemoryCandidates(config.session_id,
            enabled=os.environ.get("ATTENTION_MEMORY_EVENTS","1") not in {"0","false","off"})
        self._last_human_ms=0
        self._robot_spoke_ms=0
        self.affect_candidates=AffectCandidates(
            enabled=os.environ.get("ATTENTION_AFFECT_EVENTS","1") not in {"0","false","off"})
        # Quiet company is its own reason to speak, and it does not require the
        # person to look unhappy first.
        self.presence_candidates=PresenceCandidates(
            enabled=os.environ.get("ATTENTION_PRESENCE_EVENTS","1") not in {"0","false","off"})
        # An explicit environment variable is a decision made for this run, so it
        # wins over a choice the console saved earlier; otherwise setting it
        # would appear to do nothing.
        self.addressee_weights = addressee_weights(saved.get("addressee_weights"))
        self.conversation.addressee_weights = self.addressee_weights
        self.attention_choice_source = "default"
        if os.environ.get("ATTENTION_ALGORITHM"):
            self.attention_choice_source = "environment"
        elif saved.get("algorithm") and self.interaction.competition.registry.select(saved["algorithm"]):
            self.attention_choice_source = "saved_console_choice"
        self.rosbridge = RosbridgeTransport(config.rosbridge_url, self._on_ros_attention,self.attention_sources.receive_brain,self.attention_sources.receive_internal)
        backend = (
            OpenAiCompatibleBackend(config.llm_url, config.llm_model, config.llm_api_key)
            if config.llm_url
            else AcknowledgementBackend()
        )
        self.s2s = SpeechToSpeechResponder(backend, self._on_playback)
        self.robot_speaking = False
        self._last_memory_signature = ""
        self._last_memory_ms = 0
        self._last_source_memory_ms = 0
        self._last_speaker_memory_signature = ""
        self._last_dialogue_ended_ms = 0
        self._recorded_person_events: set[str] = set()
        self._last_addressed_gate: dict[str, Any] | None = None
        self._last_signal_active: bool | None = None
        self._last_signal_event_ms = 0
        self.thread = threading.Thread(target=self._run, name="live-perception", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.rosbridge.close()
        self.s2s.close()
        self.memory.close()
        self.decision_trace.close()
        self.interaction.competition.close()
        self.familiarity.close()
        self.workspace.close()
        self.voice_pool_worker.shutdown(wait=False,cancel_futures=True)

    def state(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.latest)

    def _load_attention_config(self) -> dict[str, Any]:
        """Console choices outlive a restart; a damaged file falls back to env."""
        try:
            saved = json.loads(self.attention_config_path.read_text("utf-8"))
        except (OSError, ValueError):
            return {}
        sources = saved.get("sources")
        weights = saved.get("addressee_weights")
        return {
            "sources": sources if isinstance(sources, list) and all(isinstance(item, str) for item in sources) else None,
            "algorithm": str(saved.get("algorithm") or ""),
            "addressee_weights": weights if isinstance(weights, dict) else {},
        }

    def attention_config(self) -> dict[str, Any]:
        """Which systems feed attention, which algorithm ranks them, and what
        counts as being spoken to."""
        registry = self.interaction.competition.registry
        return {
            "sources": [
                {
                    "id": name,
                    "enabled": name in self.attention_sources.enabled_names(),
                    "display_name": ATTENTION_SOURCE_LABELS[name][0],
                    "description": ATTENTION_SOURCE_LABELS[name][1],
                }
                for name in SOURCE_NAMES
            ],
            "algorithm": dict(registry.status(), chosen_by=self.attention_choice_source),
            # What counts as "this sentence was addressed to me": one weighted
            # sum, every term visible and editable while it runs.
            "addressee_last": getattr(getattr(self, "conversation", None), "last_addressee", None),
            "addressee": [{"id": key, "weight": round(float(value), 3),
                           "label": ADDRESSEE_LABELS.get(key, key)}
                          for key, value in getattr(self, "addressee_weights", addressee_weights()).items()],
            "config_path": str(self.attention_config_path),
            "note": "关闭某来源是消融该通道证据，不是关闭摄像头、麦克风或安全反射；算法与来源均未做现场标定。",
        }

    def set_attention_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        registry = self.interaction.competition.registry
        messages = []
        if "sources" in payload:
            requested = payload.get("sources")
            if isinstance(requested, dict):
                requested = [name for name, value in requested.items() if value]
            self.attention_sources.set_enabled(list(requested or []))
        if payload.get("algorithm"):
            if not registry.select(str(payload["algorithm"])):
                messages.append(f"算法 {payload['algorithm']} 未加载，保持原算法")
        if isinstance(payload.get("addressee_weights"), dict):
            # What counts as "they are talking to me", tunable while it runs.
            # Anything the field says is wrong should be adjustable without a
            # code change and without guessing which branch fired.
            self.addressee_weights = addressee_weights(
                {**getattr(self, "addressee_weights", addressee_weights()), **payload["addressee_weights"]})
            self.conversation.addressee_weights = self.addressee_weights
        chosen = {"sources": self.attention_sources.enabled_names(), "algorithm": registry.status()["active"],
                  "addressee_weights": getattr(self, "addressee_weights", addressee_weights())}
        self.attention_choice_source = "saved_console_choice"
        try:
            self.attention_config_path.parent.mkdir(parents=True, exist_ok=True)
            self.attention_config_path.write_text(json.dumps(chosen, ensure_ascii=False), "utf-8")
        except OSError as exc:
            messages.append(f"未能保存选择：{exc}")
        result = self.attention_config()
        result["messages"] = messages
        return result

    def devices(self) -> dict[str, Any]:
        return {
            "cameras": get_json(self.config.vision_url + "/api/cameras").get("cameras", []) if self.config.vision_enabled else [],
            "microphones": get_json(self.config.voice_url + "/api/devices"),
            "profiles": get_json(self.config.voice_url + "/api/profiles"),
        }

    def start_inputs(self, camera_index: str, camera_source_type: str, profile: str, device: str, target: str = "all") -> dict[str, Any]:
        if target not in {"all", "camera", "microphone"}:
            raise ValueError("unsupported capture target")
        vision = {"success": True, "message": "vision unchanged or disabled"}
        if self.config.vision_enabled and target in {"all", "camera"}:
            vision = post_json(
                self.config.vision_url + "/api/connect",
                {"source_type": camera_source_type, "camera_index": camera_index},
            )
            post_json(self.config.vision_url + "/api/start")
        query = urlencode({"profile": profile, "device": device, "seconds": "0"})
        voice = post_json(self.config.voice_url + "/api/start?" + query, timeout=12.0) if target in {"all", "microphone"} else {"success": True, "message": "microphone unchanged"}
        return {"vision": vision, "voice": voice}

    def stop_inputs(self) -> dict[str, Any]:
        result = {"voice": post_json(self.config.voice_url + "/api/stop")}
        if not self.config.vision_enabled:
            result["vision"] = {"success": True, "message": "vision disabled"}
            return result
        try:
            result["vision"] = post_json(self.config.vision_url + "/api/stop")
        except Exception as exc:
            result["vision"] = {"success": False, "message": str(exc)}
        return result

    def ingest_identity_evidence(self, payload: dict[str, Any]) -> dict[str, Any]:
        evidence = IdentityEvidence(
            modality=str(payload.get("modality") or "face"),
            observation_id=str(payload.get("observation_id") or ""),
            candidates={str(key): float(value) for key, value in (payload.get("candidates") or {}).items()},
            stamp_ms=int(payload.get("stamp_ms") or now_ms()),
        )
        accepted = self.person_manager.ingest_identity_evidence(evidence)
        if accepted:
            self._submit_memory_event(
                "robot-attention-perception",
                "/attention/identity_evidence",
                "identity_evidence",
                asdict(evidence),
                evidence.stamp_ms,
            )
        return {"accepted": accepted, "persons": self.person_manager.snapshots(now_ms())}

    def register_person(self, temporary_id: str, persistent_id: str) -> dict[str, Any]:
        vision_enrollment = {}
        try:
            vision_enrollment = post_json(
                self.config.vision_url + "/api/enroll",
                {"user_id": persistent_id, "user_role": "known"},
                timeout=2.0,
            )
        except Exception as exc:
            vision_enrollment = {"success": False, "message": str(exc)}
        accepted = self.person_manager.register(temporary_id, persistent_id, now_ms())
        if accepted:
            self._submit_memory_event(
                "robot-attention-perception",
                "/attention/person_registration",
                "person_registration",
                {"temporary_id": temporary_id, "persistent_id": persistent_id},
                now_ms(),
            )
        if vision_enrollment.get("success") and vision_enrollment.get("embedding"):
            self._submit_memory_event(
                "vision-detection",
                "/vision/identity_profiles",
                "face_identity_profile",
                {
                    "person_id": persistent_id,
                    "role": vision_enrollment.get("user_role", "known"),
                    "embedding_model": vision_enrollment.get("embedding_model", "opencv-sface"),
                },
                now_ms(),
                embedding=vision_enrollment["embedding"],
                identity_scope=True,
                entity={"person_id": persistent_id},
            )
        return {
            "accepted": accepted,
            "vision_enrollment": vision_enrollment,
            "persons": self.person_manager.snapshots(now_ms()),
        }

    def _run(self) -> None:
        period = 1.0 / max(self.config.poll_hz, 1.0)
        while not self.stop_event.is_set():
            started = time.perf_counter()
            current_ms = now_ms()
            if self.config.vision_enabled:
                vision_payload, vision_error = self._read_source(self.config.vision_url + "/api/state")
            else:
                vision_payload, vision_error = {}, ""
            voice_payload, voice_error = self._read_source(self.config.voice_url + "/api/state?compact=1", timeout=1.0)
            raw_people = vision_people_from_payload(vision_payload, current_ms) if vision_payload else []
            tracks = acoustic_tracks_from_payload(voice_payload) if voice_payload else []
            live_tracks = acoustic_tracks_from_payload({"last_track": voice_payload.get("last_track")})
            self._record_audio_signal_event(current_ms, voice_payload)
            people = self.person_manager.update(raw_people, live_tracks, current_ms)
            people = self.person_manager.active_people(current_ms)
            if people:
                self.buffer.ingest_vision(people)
            if tracks:
                self.buffer.ingest_acoustic(tracks)
            if not self.config.s2s_enabled:
                self.robot_speaking = bool(voice_payload.get("robot_speaking", False))
            self.buffer.ingest_robot(RobotState(stamp_ms=current_ms, speaking=self.robot_speaking))
            local_attention = self.buffer.update(current_ms)
            local_attention = self._final_transcript_fallback(
                local_attention,
                voice_payload,
                current_ms,
            )
            self.reflex_attention.update(current_ms, voice_payload, vision_payload)
            self._publish_ros_inputs(current_ms, people, tracks)
            attention = self._select_attention(current_ms, local_attention)
            object_candidates=[dict(obj,candidate_id="visual:"+key,salience=min(1.,float(obj.get("bbox_area_ratio",0))*2),uncertainty=1-float(obj.get("confidence",0))) for key,obj in self.reflex_attention.object_tracks.items()]
            reflex = self.reflex_attention.focus
            context = self.attention_sources.context(current_ms,people,live_tracks,voice_payload,list(self.conversation.turns))
            # This flag suspends the language terms in the engagement layer — a
            # name call does not count while the robot is flinching. Only a
            # startle is that; orienting happens constantly on ordinary speech,
            # and letting it set this flag silently disabled name calls and
            # answer continuations for most of a conversation.
            context["reflex_active"] = bool(reflex and reflex.get("active") and reflex.get("kind")=="startle"
                                            and 0<=current_ms-int(reflex.get("stamp_ms") or 0)<1800)
            attention = self.interaction.update(current_ms,attention,people,live_tracks,voice_payload,reflex,objects=object_candidates,context=context)
            self._broadcast_workspace(current_ms,attention,context,voice_payload,people)
            self._update_av_binding(raw_people,live_tracks,voice_payload,attention)
            self.conversation.update(current_ms, voice_payload, live_tracks, people, self.robot_speaking, asdict(attention) if not isinstance(attention, dict) else attention)
            self.crossmodal_state=self.crossmodal.update(current_ms,voice_payload,vision_payload,list(self.conversation.turns))
            self.decision_trace.submit({"stamp_ms":current_ms,"session_id":self.config.session_id,"attention":attention,
                "people":[{k:asdict(p).get(k) for k in ("person_id","role","stamp_ms","face_visible","face_confidence","identity_confidence","gaze_score","body_facing_score","lip_motion")} for p in people],
                "audio":[{k:asdict(t).get(k) for k in ("track_id","stamp_ms","voice_activity","speech_probability","clarity","self_echo_probability","azimuth_deg")} for t in live_tracks],
                "vad":voice_payload.get("last_vad"),"playback":voice_payload.get("playback")})
            self._remember_attention_gate(attention, current_ms)
            self._maybe_respond(voice_payload, attention)
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            self._publish_state(current_ms, vision_payload, voice_payload, people, tracks, attention, elapsed_ms, vision_error, voice_error)
            self._record_sources(current_ms, people, tracks)
            self._record_speaker_identity(current_ms, voice_payload)
            self._adapt_voice_pool(current_ms,voice_payload,raw_people)
            self._record_dialogue()
            self._record_person_events()
            self.stop_event.wait(max(0.0, period - (time.perf_counter() - started)))

    def _last_heard_ms(self, stamp_ms) -> int:
        """Newest transcribed human utterance. Raw voice activity is not usable:
        it goes active on the robot's own playback."""
        return max((int(turn.get("stamp_ms") or 0) for turn in self.conversation.turns
                    if turn.get("person_id") not in {"robot", None, "unknown"}), default=0)

    def _broadcast_workspace(self, stamp_ms, attention, context, voice_payload, people) -> None:
        """The broadcast step of the cycle: publish what won, never re-decide it."""
        state = attention if isinstance(attention, dict) else asdict(attention) if attention is not None else {}
        distribution = state.get("attention_distribution") or {}
        transcript = voice_payload.get("last_transcript") or {}
        registry = self.interaction.competition.registry
        active = registry.active

        vad = voice_payload.get("last_vad") or {}
        if vad.get("active") and 0 <= stamp_ms - int(vad.get("stamp_ms") or 0) < 1000:
            self._last_human_ms = stamp_ms
        # Memory may put forward something nobody is doing right now. It competes
        # in the workspace rather than in the plugin, because it has no modality.
        events = list(self.memory_candidates.candidates(
            stamp_ms, self.attention_sources.brain, people, self._last_human_ms))
        # A different question from engagement: not "are they talking to me" but
        # "does somebody here seem to want me to open". Expression belongs here.
        self.affect_candidates.observe(stamp_ms, people)
        heard_ms = self._last_heard_ms(stamp_ms)
        events += self.affect_candidates.candidates(stamp_ms, people, heard_ms)
        if (voice_payload.get("playback") or {}).get("speaking"):
            self._robot_spoke_ms = stamp_ms
        self.presence_candidates.observe(stamp_ms, people)
        # Only when nothing better is on the table: quiet company is the weakest
        # reason there is to speak, and it must never crowd out a real event.
        if not events:
            events += self.presence_candidates.candidates(stamp_ms, people, heard_ms, self._robot_spoke_ms)
        reflex = self.reflex_attention.focus or {}
        if reflex.get("active") and 0 <= stamp_ms - int(reflex.get("stamp_ms") or 0) < 1800:
            # Urgency is what lets an item bypass the workspace capacity limit.
            # A startle earns that. Orienting is a nudge and competes normally,
            # or every sentence anybody speaks pre-empts whatever was being
            # attended to.
            startling = reflex.get("kind") == "startle"
            events = list(events) + [{"candidate_id": "reflex:" + str(reflex.get("id") or reflex.get("source") or ""),
                                      "kind": "reflex_event", "person_id": reflex.get("target_id"),
                                      "salience": 1.0 if startling else .35,
                                      "urgency": 1.0 if startling else .25,
                                      "summary": str(reflex.get("kind") or "orient"),
                                      "reasons": ["reflex_" + str(reflex.get("source") or "unknown")]}]

        self.workspace.publish(
            stamp_ms, state, distribution,
            sources=(context or {}).get("sources") or {},
            algorithm={"id": getattr(active, "id", ""), "kind": getattr(active, "kind", ""),
                       "calibrated": bool((getattr(active, "manifest", {}) or {}).get("calibrated")),
                       "load_errors": registry.errors},
            provenance={"utterance_id": transcript.get("utterance_id"),
                        "transition_id": state.get("transition_id"),
                        "trace_path": self.decision_trace.status().get("path"),
                        "familiarity": (context or {}).get("familiarity_status") or {},
                        "memory_events": self.memory_candidates.status(),
                        "affect_events": self.affect_candidates.status(),
                        "presence_events": self.presence_candidates.status()},
            events=events)
        self.rosbridge.publish_json("/attention/workspace", self.workspace.snapshot())

    def _final_transcript_fallback(
        self,
        attention: AttentionState | None,
        voice_payload: dict[str, Any],
        current_ms: int,
    ) -> AttentionState | None:
        """Guarantee a fresh final ASR can address the robot in no-wake mode.

        The voice dashboard updates ``last_track`` and ``last_transcript`` from
        different callback paths. If a valid transcript is visible but the
        fusion candidate is empty, synthesize only the short-lived attention
        state; this is still subject to the downstream VLA freshness/track
        checks and does not alter the strong-wake default.
        """
        if self.config.use_transcript_wake_words:
            return attention
        transcript_payload = voice_payload.get("last_transcript")
        if not isinstance(transcript_payload, dict) or not transcript_payload.get("is_final"):
            return attention
        emitted_ms = int(
            transcript_payload.get("emitted_ms")
            or transcript_payload.get("ended_ms")
            or 0
        )
        if emitted_ms <= 0 or current_ms - emitted_ms > 5000 or emitted_ms - current_ms > 1000:
            return attention
        if attention is not None and attention.listen and attention.addressed_to_robot:
            return attention
        track_id = str(transcript_payload.get("track_id") or "")
        if not track_id:
            return attention
        transcript = TranscriptRecord(
            track_id=track_id,
            text=str(transcript_payload.get("text") or ""),
            is_final=True,
            language=str(transcript_payload.get("language") or "unknown"),
            clarity=float(transcript_payload.get("clarity", 0.0) or 0.0),
            started_ms=transcript_payload.get("started_ms"),
            ended_ms=transcript_payload.get("ended_ms"),
            emitted_ms=emitted_ms,
        )
        if not transcript.text.strip():
            return attention
        return AttentionState(
            stamp_ms=current_ms,
            target_id=f"sound:{track_id}",
            target_kind="sound_source",
            confidence=0.59,
            listen=True,
            addressed_to_robot=True,
            source_track_id=track_id,
            transcripts=(transcript,),
            reasons=("final_transcript_fallback", "no_wake_word"),
        )

    @staticmethod
    def _read_source(url: str, timeout: float = 0.25) -> tuple[dict[str, Any], str]:
        try:
            return get_json(url, timeout=timeout), ""
        except Exception as exc:
            return {}, str(exc)

    def _publish_state(
        self,
        current_ms: int,
        vision_payload: dict[str, Any],
        voice_payload: dict[str, Any],
        people: list[VisionPerson],
        tracks: list[AcousticTrack],
        attention,
        fusion_latency_ms: float,
        vision_error: str,
        voice_error: str,
    ) -> None:
        previous_attention = self.latest.get("attention")
        attention_data = (
            attention if isinstance(attention, dict) else asdict(attention) if attention is not None else previous_attention
        )
        event = None
        if attention_data and attention_data != previous_attention:
            event = {
                "stamp_ms": current_ms,
                "target_id": attention_data.get("target_id", "none"),
                "listen": attention_data.get("listen", False),
                "confidence": attention_data.get("confidence", 0.0),
            }
            self.events.appendleft(event)
            self._record_attention(attention_data, current_ms)
        state = {
            "stamp_ms": current_ms,
            "session_id": self.config.session_id,
            "algorithm": self.config.algorithm,
            "vision": vision_payload,
            "voice": voice_payload,
            "people": [asdict(person) for person in people],
            "tracks": [asdict(track) for track in tracks],
            "attention": attention_data,
            "diagnostics": {
                "vision_ok": not vision_error,
                "voice_ok": not voice_error,
                "memory_ok": not self.memory.last_error,
                "vision_error": vision_error,
                "voice_error": voice_error,
                "memory_error": self.memory.last_error,
                "fusion_cycle_ms": fusion_latency_ms,
                "decision_trace":self.decision_trace.status(),
                "vision_age_ms": _age_ms(people, current_ms),
                "audio_age_ms": _age_ms(tracks, current_ms),
                "rosbridge_connected": self.rosbridge.connected,
                "rosbridge_error": self.rosbridge.last_error,
                "attention_transport": (
                    "ros2" if self.ros_attention and current_ms - self.ros_attention_ms <= 750 else "local_fallback"
                ),
            },
            "events": list(self.events),
            "dialogue": self.s2s.last_turn,
            "reflex_attention": self.reflex_attention.focus, "crossmodal_events":self.crossmodal_state,"interaction_transitions":list(self.interaction.transitions),
            "utterances": list(self.conversation.turns),
            "interruption_candidate": self.conversation.interruption,
            "person_hypotheses": self.person_manager.snapshots(current_ms),
            "active_behaviors": list(self.person_manager.active_behaviors),
            "person_events": list(self.person_manager.events),
            "av_binding": list(self.av_decisions),
            "workspace_events": self._workspace_events(),
            # 谁说了什么: the voiceprint, the person, and the sentence, joined.
            "cocktail": cocktail_rows(current_ms, voice_payload, list(self.conversation.turns), attention_data),
            # Which voices are this person's, so a sentence said off camera can
            # still be answered when its speaker turns round.
            "voice_bindings": {person: sorted(voices) for person, voices in self.conversation.voice_bindings.items()},
        }
        for turn in self.conversation.turns:
            if turn["utterance_id"] not in self._published_utterances:
                if self.rosbridge.publish_json("/attention/utterance", {"session_id": self.config.session_id, "utterance": turn}):
                    self._published_utterances.append(turn["utterance_id"])
        self.rosbridge.publish_json("/attention/brain_input", {
            "session_id": self.config.session_id, "stamp_ms": current_ms,
            "voice": {"last_streaming_transcript":voice_payload.get("last_streaming_transcript"), "last_vad": voice_payload.get("last_vad"), "playback": voice_payload.get("playback"), "audio_scene": voice_payload.get("audio_scene")},
            "workspace_events": self._workspace_events(),
            "person_hypotheses": state["person_hypotheses"], "attention": attention_data, "reflex_attention": self.reflex_attention.focus, "crossmodal_events":self.crossmodal_state,"interaction_transitions":list(self.interaction.transitions),
            "voice_bindings": state["voice_bindings"],
            "utterances": list(self.conversation.turns)[-8:], "interruption_candidate": self.conversation.interruption,
        })
        with self.lock:
            self.latest = state

    def _workspace_events(self) -> list[dict[str, Any]]:
        """Non-sensory candidates that won a place this cycle, for Brain to read.

        Only admitted ones: something suppressed by the capacity limit did not
        get in, and must not be acted on as if it had.
        """
        return [item for item in (self.workspace.snapshot().get("coalition") or [])
                if item.get("kind") not in {"sensory", None} and item.get("admitted")]

    def _record_attention(self, state: dict[str, Any], stamp_ms: int) -> None:
        signature = json.dumps(
            [state.get("target_id"), state.get("listen"), state.get("addressed_to_robot")],
            separators=(",", ":"),
        )
        if signature == self._last_memory_signature and stamp_ms - self._last_memory_ms < 1000:
            return
        self._last_memory_signature = signature
        self._last_memory_ms = stamp_ms
        self.memory.submit(
            {
                "scope": {
                    "subject_id": self.config.subject_id,
                    "robot_id": self.config.robot_id,
                    "session_id": self.config.session_id,
                },
                "source_system": "robot-attention-perception",
                "source_topic": "/attention/state",
                "kind": "attention_state",
                "entity": {
                    "person_id": state.get("person_id"),
                    "voice_id": state.get("source_track_id"),
                },
                "structured": state,
                "observed_at_ms": stamp_ms,
                "trace_id": f"{self.config.session_id}:{stamp_ms}",
            }
        )

    def _record_audio_signal_event(self, stamp_ms: int, voice_payload: dict[str, Any]) -> None:
        vad = voice_payload.get("last_vad") if isinstance(voice_payload.get("last_vad"), dict) else {}
        if not vad:
            return
        active = bool(vad.get("active", False))
        rms = float(vad.get("rms_dbfs", -120.0) or -120.0)
        snr = float(vad.get("snr_db", 0.0) or 0.0)
        changed = self._last_signal_active is None or active != self._last_signal_active
        # Idle is a state, not an event.  Record it once on the falling edge;
        # otherwise a quiet microphone would fill the event list with a
        # heartbeat every second.  Active audio may still emit a one-second
        # health sample for latency/debugging.
        periodic = active and stamp_ms - self._last_signal_event_ms >= 1000 and rms > -70.0
        if not changed and not periodic:
            return
        self._last_signal_active = active
        self._last_signal_event_ms = stamp_ms
        self.events.appendleft(
            {
                "stamp_ms": stamp_ms,
                "kind": "vad_signal",
                "target_id": "audio",
                "listen": active,
                "confidence": float(vad.get("probability", 0.0) or 0.0),
                "rms_dbfs": round(rms, 1),
                "snr_db": round(snr, 1),
                "message": "VAD active" if active else "microphone signal / VAD idle",
            }
        )
    def _record_sources(
        self,
        stamp_ms: int,
        people: list[VisionPerson],
        tracks: list[AcousticTrack],
    ) -> None:
        if stamp_ms - self._last_source_memory_ms < 1000:
            return
        self._last_source_memory_ms = stamp_ms
        for source_system, source_topic, kind, structured in (
            ("vision-detection", "/perception/vision/people", "person_observation", [asdict(item) for item in people]),
            ("voice-detection", "/perception/voice/tracks", "acoustic_track", [asdict(item) for item in tracks]),
        ):
            self.memory.submit(
                {
                    "scope": {
                        "subject_id": self.config.subject_id,
                        "robot_id": self.config.robot_id,
                        "session_id": self.config.session_id,
                    },
                    "source_system": source_system,
                    "source_topic": source_topic,
                    "kind": kind,
                    "entity": {},
                    "structured": structured,
                    "observed_at_ms": stamp_ms,
                    "trace_id": f"{self.config.session_id}:{stamp_ms}",
                }
            )

    def _adapt_voice_pool(self,stamp,voice,people):
        t=voice.get("last_transcript") or {};key=t.get("utterance_id")
        if not key or key==self.voice_pool_seen or (self.voice_pool_future and not self.voice_pool_future.done()):return
        self.voice_pool_seen=key
        speaker=t.get("speaker") or {}
        voiced=str(speaker.get("speaker_id") or "")
        # Whose voice this is gets decided by the face here, not by the voiceprint
        # we are trying to improve. Requiring the voiceprint to already match at
        # .7 meant the pool could only ever learn what it already knew.
        ended=int(t.get("ended_ms") or 0)
        visible=[p for p in people if p.face_visible and p.identity_confidence>=.8 and abs(p.stamp_ms-ended)<700
                 and str(p.person_id or "") not in {"","unknown","stranger"}
                 and not str(p.person_id).startswith(("vision_","anonymous_"))]
        if len(visible)!=1:return
        identity=str(visible[0].person_id)
        if voiced and voiced!=identity and not voiced.startswith("stranger_") and voiced!="unknown":
            return   # the voiceprint names somebody else: a conflict to report, not a sample to learn
        payload={"utterance_id":key,"person_id":identity,"face_person_id":identity,
                 "face_confidence":visible[0].identity_confidence,
                 # The cluster this voice was auto-registered as, so it can be
                 # folded into the person instead of staying a rival identity.
                 "cluster_id":voiced if voiced.startswith("stranger_") else ""}
        self.voice_pool_future=self.voice_pool_worker.submit(post_json,self.config.voice_url+"/api/voice-pool",payload,timeout=1.)

    def _record_speaker_identity(self, stamp_ms: int, voice_payload: dict[str, Any]) -> None:
        profile = voice_payload.get("last_speaker") if isinstance(voice_payload.get("last_speaker"), dict) else None
        embedding = profile.get("identity_embedding") if profile else None
        if not profile or not profile.get("finalized_identity") or not isinstance(embedding, list) or len(embedding) < 32:
            return
        speaker_id = str(profile.get("speaker_id") or "unknown")
        if speaker_id == "unknown":
            return
        signature = json.dumps([speaker_id, profile.get("speaker_role"), embedding, profile.get("aliases",[])],sort_keys=True)
        if not hasattr(self,"_identity_memory_signatures"):
            self._identity_memory_signatures=set()
        if signature in self._identity_memory_signatures:
            return
        self._identity_memory_signatures.add(signature)
        self._last_speaker_memory_signature = signature
        self._submit_memory_event(
            "voice-detection",
            "/voice/identity_profiles",
            "voice_identity_profile",
            {
                "speaker_id": speaker_id,
                "role": str(profile.get("speaker_role") or "known"),
                "aliases": profile.get("aliases", []),
                "embedding_model": "3dspeaker-eres2netv2-zh-cn",
            },
            int(profile.get("stamp_ms") or stamp_ms),
            embedding=embedding,
            identity_scope=True,
            entity={"voice_id": speaker_id, "person_id": speaker_id},
        )

    def _record_dialogue(self) -> None:
        turn = self.s2s.last_turn
        if not turn or int(turn.get("ended_ms", 0)) <= self._last_dialogue_ended_ms:
            return
        self._last_dialogue_ended_ms = int(turn["ended_ms"])
        self.memory.submit(
            {
                "scope": {
                    "subject_id": self.config.subject_id,
                    "robot_id": self.config.robot_id,
                    "session_id": self.config.session_id,
                },
                "source_system": "robot-attention-perception",
                "source_topic": "/speech/target_stream",
                "kind": "dialogue_turn",
                "entity": {},
                "text": str(turn.get("user_text") or ""),
                "structured": turn,
                "observed_at_ms": int(turn["ended_ms"]),
                "trace_id": f"{self.config.session_id}:dialogue:{turn['ended_ms']}",
            }
        )

    def _submit_memory_event(
        self,
        source_system: str,
        source_topic: str,
        kind: str,
        structured: Any,
        stamp_ms: int,
        embedding: list[float] | None = None,
        identity_scope: bool = False,
        entity: dict[str, Any] | None = None,
    ) -> None:
        self.memory.submit(
            {
                "scope": {
                    "subject_id": self.config.subject_id,
                    "robot_id": self.config.robot_id,
                    "session_id": "identity-registry" if identity_scope else self.config.session_id,
                },
                "source_system": source_system,
                "source_topic": source_topic,
                "kind": kind,
                "entity": entity or {},
                "embedding": embedding,
                "structured": structured,
                "observed_at_ms": stamp_ms,
                "trace_id": ("identity:"+kind+":"+__import__("hashlib").sha256(json.dumps([entity, structured, embedding],sort_keys=True).encode()).hexdigest()) if identity_scope else f"{self.config.session_id}:{kind}:{stamp_ms}",
            }
        )

    def _record_person_events(self) -> None:
        for event in self.person_manager.events:
            signature = json.dumps(event, ensure_ascii=False, sort_keys=True)
            if signature in self._recorded_person_events:
                continue
            self._recorded_person_events.add(signature)
            self._submit_memory_event(
                "robot-attention-perception",
                "/attention/person_events",
                str(event.get("kind") or "person_event"),
                event,
                int(event.get("stamp_ms") or now_ms()),
            )
        if len(self._recorded_person_events) > 1000:
            self._recorded_person_events.clear()

    def _initial_state(self) -> dict[str, Any]:
        return {
            "stamp_ms": now_ms(),
            "session_id": self.config.session_id,
            "algorithm": self.config.algorithm,
            "people": [],
            "tracks": [],
            "attention": None,
            "diagnostics": {"vision_ok": False, "voice_ok": False, "memory_ok": False},
            "events": [],
        }

    def _publish_ros_inputs(
        self,
        stamp_ms: int,
        people: list[VisionPerson],
        tracks: list[AcousticTrack],
    ) -> None:
        self.rosbridge.publish_json(
            "/vision/people_json",
            {"stamp_ms": stamp_ms, "people": [asdict(person) for person in people]},
        )
        self.rosbridge.publish_json(
            "/voice/acoustic_tracks",
            {"stamp_ms": stamp_ms, "tracks": [asdict(track) for track in tracks]},
        )
        self.rosbridge.publish_json(
            "/robot/playback_state",
            {"stamp_ms": stamp_ms, "speaking": self.robot_speaking, "moving": False, "playback_rms_db": 0.0},
        )
        self.rosbridge.publish_json(
            "/attention/person_hypotheses",
            {"stamp_ms": stamp_ms, "persons": self.person_manager.snapshots(stamp_ms)},
        )
        self.rosbridge.publish_json(
            "/attention/active_behaviors",
            {"stamp_ms": stamp_ms, "behaviors": list(self.person_manager.active_behaviors)},
        )
        header = {"stamp": _ros_time(stamp_ms), "frame_id": "base_link"}
        self.rosbridge.publish(
            "/perception/vision/people",
            {"header": header, "people": [_typed_vision(person) for person in people]},
        )
        self.rosbridge.publish(
            "/perception/voice/tracks",
            {"header": header, "tracks": [_typed_acoustic(track) for track in tracks]},
        )

    def _on_ros_attention(self, state: dict[str, Any]) -> None:
        self.ros_attention = state
        self.ros_attention_ms = now_ms()

    def _select_attention(self, current_ms: int, local_attention):
        if self.ros_attention is not None and current_ms - self.ros_attention_ms <= 750:
            return dict(self.ros_attention)
        return local_attention

    def _maybe_respond(self, voice_payload: dict[str, Any], attention) -> None:
        if not self.config.s2s_enabled or attention is None:
            return
        state = attention if isinstance(attention, dict) else asdict(attention)
        transcript = voice_payload.get("last_transcript")
        if not isinstance(transcript, dict) or not transcript.get("is_final", False):
            return
        if not gate_allows_transcript(state, self._last_addressed_gate, transcript, now_ms()):
            return
        key = str(transcript.get("emitted_ms") or transcript.get("text") or "")
        self.s2s.submit(key, str(transcript.get("text") or ""))

    def _remember_attention_gate(self, attention, stamp_ms: int) -> None:
        if attention is None:
            return
        state = attention if isinstance(attention, dict) else asdict(attention)
        if state.get("listen") and state.get("addressed_to_robot"):
            self._last_addressed_gate = {**state, "remembered_ms": stamp_ms}

    def _on_playback(self, speaking: bool) -> None:
        self.robot_speaking = speaking
        query = urlencode({"speaking": "1" if speaking else "0"})
        try:
            post_json(self.config.voice_url + "/api/robot-speaking?" + query)
        except Exception:
            pass

    def _update_av_binding(
        self,
        people: list[VisionPerson],
        tracks: list[AcousticTrack],
        voice_payload: dict[str, Any],
        attention: dict[str, Any] | None = None,
    ) -> None:
        if tracks:
            track = tracks[0]
            if track.stamp_ms > self._last_av_audio_ms:
                features = ((voice_payload.get("last_track") or {}).get("features") or {}).get("mfcc") or []
                mfcc = [float(value) for value in features[:13]]
                mfcc.extend([0.0] * (13 - len(mfcc)))
                self.av_binder.ingest_audio(
                    AudioFeatureFrame(
                        stamp_ms=track.stamp_ms,
                        mfcc=np.asarray([mfcc], dtype=np.float32),
                        vad_active=track.voice_activity,
                        speaker_cluster_id=track.track_id,
                    )
                )
                self._last_av_audio_ms = track.stamp_ms
        if not people:
            return
        frame_stamp_ms = max(person.stamp_ms for person in people)
        if frame_stamp_ms <= self._last_av_video_ms:
            return
        decisions = []
        for person in people:
            if not person.face_id:
                continue
            mouth_features = list(person.mouth_roi_features[:32])
            mouth_features.extend([0.0] * (32 - len(mouth_features)))
            decision = self.av_binder.decide(
                VideoFeatureFrame(
                    stamp_ms=person.stamp_ms,
                    face_id=person.face_id,
                    face_visible=person.face_visible,
                    lip_motion=person.lip_motion,
                    mouth_open_ratio=person.mouth_open_ratio,
                    mouth_roi_features=np.asarray([mouth_features], dtype=np.float32),
                )
            )
            decisions.append(asdict(decision))
            speaker=voice_payload.get("last_speaker") or {}
            conflict=(speaker.get("track_id")==decision.speaker_cluster_id and abs(person.stamp_ms-int(speaker.get("stamp_ms") or 0))<1000
                      and float(speaker.get("similarity") or 0)>=.48 and speaker.get("speaker_id") not in {None,"unknown",person.person_id}
                      and person.role in {"owner","known"})
            if conflict:
                decisions[-1].update(action="reject",reason="identity_conflict_voice_priority_no_face_binding")
            permitted=bool(attention and attention.get("addressed_to_robot") and attention.get("source_track_id")==decision.speaker_cluster_id and attention.get("face_id")==decision.face_id)
            if decision.action == "bind" and not permitted:
                decisions[-1].update(action="reject",reason="no_shared_interaction_evidence")
            if decision.action == "bind" and not conflict and permitted:
                self.person_manager.apply_av_binding(
                    decision.face_id,
                    decision.speaker_cluster_id,
                    decision.confidence,
                    person.stamp_ms,
                )
        self.av_decisions = decisions
        self._last_av_video_ms = frame_stamp_ms


def _age_ms(items: list[Any], current_ms: int) -> int | None:
    if not items:
        return None
    return max(0, current_ms - max(item.stamp_ms for item in items))


def _ros_time(stamp_ms: int) -> dict[str, int]:
    return {"sec": int(stamp_ms // 1000), "nanosec": int(stamp_ms % 1000) * 1_000_000}


def _typed_vision(person: VisionPerson) -> dict[str, Any]:
    data = asdict(person)
    data["observed_at"] = _ros_time(person.stamp_ms)
    data["has_azimuth"] = person.azimuth_deg is not None
    data["has_elevation"] = person.elevation_deg is not None
    data["has_distance"] = person.distance_m is not None
    data["azimuth_deg"] = float(person.azimuth_deg or 0.0)
    data["elevation_deg"] = float(person.elevation_deg or 0.0)
    data["distance_m"] = float(person.distance_m or 0.0)
    data.pop("stamp_ms")
    return data


def _typed_acoustic(track: AcousticTrack) -> dict[str, Any]:
    data = asdict(track)
    transcript = data.pop("transcript") or {}
    data["observed_at"] = _ros_time(track.stamp_ms)
    data["has_azimuth"] = track.azimuth_deg is not None
    data["has_elevation"] = track.elevation_deg is not None
    data["has_distance"] = track.distance_m is not None
    data["azimuth_deg"] = float(track.azimuth_deg or 0.0)
    data["elevation_deg"] = float(track.elevation_deg or 0.0)
    data["distance_m"] = float(track.distance_m or 0.0)
    data["transcript_text"] = str(transcript.get("text") or "")
    data["transcript_is_final"] = bool(transcript.get("is_final", False))
    data["language"] = str(transcript.get("language") or "")
    data["transcript_confidence"] = float(transcript.get("confidence", 0.0) or 0.0)
    data.pop("stamp_ms")
    return data


def gate_allows_transcript(
    current_attention: dict[str, Any],
    remembered_gate: dict[str, Any] | None,
    transcript: dict[str, Any],
    current_ms: int,
    ttl_ms: int = 2500,
) -> bool:
    gate = (
        current_attention
        if current_attention.get("listen") and current_attention.get("addressed_to_robot")
        else remembered_gate
    )
    if gate is None or current_ms - int(gate.get("remembered_ms", current_ms)) > ttl_ms:
        return False
    transcript_track = str(transcript.get("track_id") or "")
    gate_track = str(gate.get("source_track_id") or "")
    return not (transcript_track and gate_track and transcript_track != gate_track)
