from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import queue
import subprocess
import threading
import time
from typing import Any, Callable
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DialogueTurn:
    started_ms: int
    ended_ms: int
    user_text: str
    assistant_text: str
    backend: str
    error: str = ""


class DialogueBackend:
    name = "base"

    def reply(self, text: str) -> str:
        raise NotImplementedError


class AcknowledgementBackend(DialogueBackend):
    name = "local_acknowledgement"

    def reply(self, text: str) -> str:
        compact = " ".join(text.split())
        return f"我听到了。{compact}" if compact else "我在听。"


class OpenAiCompatibleBackend(DialogueBackend):
    name = "openai_compatible"

    def __init__(self, url: str, model: str, api_key: str = ""):
        self.url = url
        self.model = model
        self.api_key = api_key

    def reply(self, text: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是机器人语音助手。回复简洁、自然。"},
                    {"role": "user", "content": text},
                ],
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        request = Request(self.url, data=body, headers=headers, method="POST")
        with urlopen(request, timeout=20.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"]).strip()


class SpeechToSpeechResponder:
    def __init__(
        self,
        backend: DialogueBackend | None = None,
        on_playback: Callable[[bool], None] | None = None,
    ):
        self.backend = backend or AcknowledgementBackend()
        self.on_playback = on_playback or (lambda _speaking: None)
        self.items: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.last_key = ""
        self.speaking = False
        self.last_turn: dict[str, Any] | None = None
        self.thread = threading.Thread(target=self._run, name="speech-to-speech", daemon=True)
        self.thread.start()

    def submit(self, key: str, text: str) -> bool:
        if not text.strip() or key == self.last_key:
            return False
        self.last_key = key
        try:
            self.items.put_nowait((key, text))
            return True
        except queue.Full:
            return False

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                _key, text = self.items.get(timeout=0.2)
            except queue.Empty:
                continue
            started_ms = int(time.time() * 1000)
            assistant_text = ""
            error = ""
            try:
                assistant_text = self.backend.reply(text)
                self.speaking = True
                self.on_playback(True)
                subprocess.run(["say", assistant_text], check=True)
            except Exception as exc:
                error = str(exc)
            finally:
                self.speaking = False
                self.on_playback(False)
            turn = DialogueTurn(
                started_ms=started_ms,
                ended_ms=int(time.time() * 1000),
                user_text=text,
                assistant_text=assistant_text,
                backend=self.backend.name,
                error=error,
            )
            self.last_turn = asdict(turn)
