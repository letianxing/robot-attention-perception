"""Linguistic evidence about who the utterance in progress is for.

Only generic utterance features are produced here — address form, question and
answer shape, lexical continuation of the exchange the robot took part in. What
they mean is decided by the fused attention algorithm, so no scenario ever gets
its own branch. Keyword lists are engineering baselines for Mandarin, not a
complete speech-act classifier, and none of this is semantic understanding.

Evidence type follows the addressee-detection literature that combines what was
said with how the speaker was oriented:
  Katzenmaier, Stiefelhagen & Schultz 2004, ICMI, doi:10.1145/1027933.1027959
  Mallidi et al. 2018, Interspeech, doi:10.21437/Interspeech.2018-1531
"""
import re
from collections import deque

from .interaction_intent import directed_call

QUESTION = re.compile(r"[?？]|(?:吗|么|呢)[。！!\s]*$")
ANSWER_OPENING = re.compile(
    r"^(?:是|对|嗯|好|行|可以|没有|没|不|应该|大概|也许|可能|当然|差不多|我?觉得|我想|我是|我在)")
BACKCHANNEL = {"嗯", "嗯嗯", "对", "对对", "对对对", "好", "好的", "是的", "哦", "噢", "嗯哼"}
ROBOT_NAMES = ("小圆", "小园", "小元", "小袁", "机器人", "reachy")  # 小园/小元 是识别器常把「小圆」听成的同音写法
PUNCTUATION = re.compile(r"[\s，,。.!！?？：:；;、\"'“”‘’（）()]")
TOKEN = re.compile(r"[一-鿿]|[a-z0-9]+")


def _normalize(text):
    return PUNCTUATION.sub("", str(text or "")).lower()


def _bigrams(text):
    tokens = TOKEN.findall(_normalize(text))
    if len(tokens) < 2:
        return set(tokens)
    return {tokens[index] + tokens[index + 1] for index in range(len(tokens) - 1)}


def _overlap(text, reference):
    """Containment of the utterance in the reference, not a similarity score."""
    left, right = _bigrams(text), _bigrams(reference)
    if not left or not right:
        return 0.0
    return len(left & right) / float(min(len(left), len(right)))


def addressed_human(text, known_people, speaker_id=""):
    """Vocative aimed at a named human: the strongest evidence against the robot.

    Only names the system actually knows are detectable. Unregistered speakers
    have no name here, so this stays 0 and the attention log says so rather than
    pretending the utterance was checked.
    """
    compact = _normalize(text)
    if not compact:
        return 0.0, ""
    for person in known_people:
        name = _normalize(person)
        # A one-character id matches far too much ordinary speech to be a vocative.
        if len(name) < 2 or person == speaker_id or any(robot in name for robot in ROBOT_NAMES):
            continue
        if compact.startswith(name):
            return 1.0, person
        if re.search(re.escape(name) + r"(?:你|您|你们)", compact):
            return 0.8, person
    return 0.0, ""


class LinguisticContext:
    """Keeps the short text window the robot itself took part in."""

    def __init__(self, window_ms=120000, limit=24):
        self.window_ms = window_ms
        self.turns = deque(maxlen=limit)
        self.seen = set()

    def observe(self, turns):
        """Record turns the robot was part of: its own replies and speech to it.

        Called every perception tick, so only the newest turns are inspected and
        the seen-set is bounded rather than grown.
        """
        for turn in list(turns)[-32:]:
            key = turn.get("utterance_id")
            if not key or key in self.seen:
                continue
            attention = turn.get("attention") or {}
            engaged = bool(attention.get("addressed_to_robot")) or turn.get("source") == "robot_echo"
            if not engaged:
                continue
            if len(self.seen) > 512:
                self.seen = {item_key for item_key in self.seen if item_key in {t["key"] for t in self.turns}}
            self.seen.add(key)
            self.turns.append({"key": key, "stamp_ms": int(turn.get("stamp_ms") or 0),
                               "text": str((turn.get("transcript") or {}).get("text") or "")})

    def robot_context(self, stamp):
        return " ".join(turn["text"] for turn in self.turns if 0 <= stamp - turn["stamp_ms"] < self.window_ms)

    def features(self, stamp, text, speaker_id="", known_people=(), open_question=""):
        """Per-utterance features in [0,1]; an empty utterance yields all zeros."""
        text = str(text or "").strip()
        if not text:
            return {}
        compact = _normalize(text)
        other, named = addressed_human(text, known_people, speaker_id)
        call = 1.0 if directed_call(text) else 0.0
        question = 1.0 if QUESTION.search(text) else 0.0
        backchannel = 1.0 if compact in BACKCHANNEL else 0.0
        answer = 0.0
        if not other and not call:
            answer = max(0.7 if ANSWER_OPENING.match(compact) else 0.0,
                         _overlap(text, open_question) if open_question else 0.0)
        topic = 0.0 if other or call else _overlap(text, self.robot_context(stamp))
        return {"lang_directed_call": call,
                "lang_question": question,
                "lang_answer_continuation": round(min(1.0, answer), 4),
                "lang_topic_continuation": round(min(1.0, topic), 4),
                "lang_addresses_other": other,
                "lang_backchannel": backchannel,
                "addressed_human_id": named,
                "mentions_robot_without_calling": float(bool(not call and any(name in compact for name in ROBOT_NAMES)))}
