"""谁说了什么：one row per voice in a noisy room.

The console could already show the transcript and, separately, the list of
voices the tracker was following. In a room with several people talking that is
not enough to check the thing that matters — whether the system is keeping the
sentences apart and attaching each one to the right person. So this builds the
join: the voiceprint, who it is taken to be, and what that person said.

Deliberately a view and nothing else. No decisions are made here: identity comes
from the conversation record (enrolled voiceprint, then face, then the
auto-registered cluster), the names come from the voice service's registry, and
attention says which row it is currently listening to. If a row looks wrong on
screen, the fault is upstream and this is where it shows.
"""
from __future__ import annotations

from typing import Any

WINDOW_MS = 120000     # how far back a row keeps its sentences
MAX_ROWS = 8
MAX_UTTERANCES = 6


def _row(key: str, name: str) -> dict[str, Any]:
    return {"key": key, "display_name": name, "person_id": None, "registered": False,
            "identity_source": "", "identity_confidence": 0.0, "digest": [],
            "azimuth_deg": None, "speech_ms": 0, "last_heard_ms": 0, "attended": False,
            "reference_available": False, "utterances": []}


def cocktail_rows(stamp_ms: int, voice: dict[str, Any], utterances: list[dict[str, Any]],
                  attention: dict[str, Any] | None = None, window_ms: int = WINDOW_MS) -> list[dict[str, Any]]:
    """Voices heard recently, each with its name and its sentences."""
    registry = voice.get("voice_registry") or {}
    speakers = ((voice.get("speakers") or {}).get("speakers")) or []
    rows: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}

    for track in speakers:
        if not isinstance(track, dict):
            continue
        heard = int(track.get("last_heard_ms") or 0)
        if heard and stamp_ms - heard > window_ms:
            continue
        person = str(track.get("person_id") or "")
        key = person or str(track.get("speaker_id") or "")
        if not key:
            continue
        entry = registry.get(person) or registry.get(str(track.get("speaker_id") or "")) or {}
        row = rows.setdefault(key, _row(key, str(entry.get("display_name") or track.get("display_name") or key)))
        row.update(person_id=person or None,
                   registered=bool(entry) and str(entry.get("role") or "") != "",
                   digest=list(track.get("embedding_digest") or []),
                   azimuth_deg=track.get("azimuth_deg"),
                   speech_ms=int(track.get("speech_ms") or 0),
                   last_heard_ms=heard,
                   identity_confidence=float(track.get("identity_confidence") or 0),
                   reference_available=bool(track.get("reference_available")))
        for alias in (entry.get("aliases") or []):
            aliases[str(alias)] = key
        if str(track.get("speaker_id") or "") != key:
            aliases[str(track.get("speaker_id"))] = key

    for turn in utterances:
        if not isinstance(turn, dict):
            continue
        end = int(turn.get("stamp_ms") or 0)
        if end and stamp_ms - end > window_ms:
            continue
        text = str((turn.get("transcript") or {}).get("text") or "").strip()
        if not text or turn.get("source") == "robot_echo":
            continue
        person = str(turn.get("person_id") or "")
        cluster = str(turn.get("voice_cluster_id") or "")
        key = (aliases.get(person) or aliases.get(cluster)
               or (person if person in rows else "")
               or (cluster if cluster in rows else "")
               or person or cluster or "unknown")
        entry = registry.get(person) or registry.get(cluster) or {}
        row = rows.setdefault(key, _row(key, str(entry.get("display_name") or key)))
        if not row.get("person_id"):
            row["person_id"] = person or None
        if entry:
            row["display_name"] = str(entry.get("display_name") or row["display_name"])
            row["registered"] = True
        row["identity_source"] = str(turn.get("identity_source") or row["identity_source"])
        row["identity_confidence"] = max(row["identity_confidence"], float(turn.get("identity_confidence") or 0))
        row["last_heard_ms"] = max(row["last_heard_ms"], end)
        row["utterances"].append({
            "text": text, "stamp_ms": end,
            "addressed": turn.get("addressee") == "robot",
            "held": bool(turn.get("held_for_attention")),
            "score": round(float(((turn.get("addressee_detail") or {}).get("score") or 0)), 2),
        })

    target = str(((attention or {}).get("target_id") or "")).split(":")[-1]
    listening = bool((attention or {}).get("listen"))
    for key, row in rows.items():
        row["utterances"] = sorted(row["utterances"], key=lambda item: -item["stamp_ms"])[:MAX_UTTERANCES]
        row["attended"] = bool(listening and target and target in {key, str(row.get("person_id") or "")})
        row["age_ms"] = max(0, stamp_ms - int(row["last_heard_ms"] or stamp_ms))
    ordered = sorted(rows.values(), key=lambda item: -int(item["last_heard_ms"] or 0))
    return [row for row in ordered if row["utterances"] or row["digest"]][:MAX_ROWS]
