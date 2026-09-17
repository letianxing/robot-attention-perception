from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .live_runtime import LivePerceptionRuntime, LiveRuntimeConfig, get_json, post_json


class LiveDashboardHandler(BaseHTTPRequestHandler):
    runtime: LivePerceptionRuntime
    web_dir: Path

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send_file("live_index.html", "text/html; charset=utf-8")
        elif path == "/live_app.js":
            self._send_file("live_app.js", "text/javascript; charset=utf-8")
        elif path == "/live_styles.css":
            self._send_file("live_styles.css", "text/css; charset=utf-8")
        elif path == "/api/state":
            self._send_json(self.runtime.state())
        elif path == "/api/gates":
            self._send_json({"utterances": [{"utterance_id": turn["utterance_id"], "attention": turn["attention"]} for turn in self.runtime.state().get("utterances", [])]})
        elif path == "/api/memory-trace":
            from .history import trace_history
            self._call(lambda: trace_history(self.runtime.config, parse_qs(urlparse(self.path).query)))
        elif path == "/api/memory-search":
            from .history import search_history
            self._call(lambda: search_history(self.runtime.config, parse_qs(urlparse(self.path).query)))
        elif path == "/api/brain-state":
            self._call(lambda: get_json(self.runtime.config.brain_url + "/api/state?view=dashboard", timeout=.7))
        elif path == "/api/devices":
            self._call(self.runtime.devices)
        elif path == "/api/attention-config":
            self._call(self.runtime.attention_config)
        elif path == "/api/workspace":
            query = parse_qs(urlparse(self.path).query)
            since = query.get("since", [""])[0]
            if since:
                self._send_json({"cycles": self.runtime.workspace.replay(since),
                                 "status": self.runtime.workspace.status()})
            else:
                self._send_json(dict(self.runtime.workspace.snapshot(),
                                     status=self.runtime.workspace.status()))
        elif path == "/api/workspace/stream":
            self._stream_workspace()
        elif path == "/api/config":
            self._send_json(
                {
                    "vision_stream_url": self.runtime.config.vision_url + "/stream.mjpg",
                    "vision_url": self.runtime.config.vision_url,
                    "voice_url": self.runtime.config.voice_url,
                    "brain_url": self.runtime.config.brain_url,
                    "rosbridge_url": self.runtime.config.rosbridge_url,
                    "session_id": self.runtime.config.session_id,
                }
            )
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("content-length", "0") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/brain-tts":
                from urllib.parse import urlencode
                query = urlencode({"enabled": "true" if payload.get("enabled") else "false", "voice": str(payload.get("voice") or "Tingting"), "rate": max(120, min(300, int(payload.get("rate") or 210)))})
                self._send_json(post_json(self.runtime.config.brain_url + "/api/tts?" + query, timeout=.7))
            elif path == "/api/guided-registration":
                from .registration import register
                self._send_json(register(self.runtime, payload))
            elif path == "/api/start":
                self._send_json(
                    self.runtime.start_inputs(
                        str(payload.get("camera_index", "0")),
                        str(payload.get("camera_source_type") or "opencv"),
                        str(payload.get("profile") or "mac_builtin"),
                        str(payload.get("device") or "default"),
                        str(payload.get("target") or "all"),
                    )
                )
            elif path == "/api/stop":
                self._send_json(self.runtime.stop_inputs())
            elif path == "/api/attention-config":
                self._send_json(self.runtime.set_attention_config(payload))
            elif path == "/api/identity-evidence":
                self._send_json(self.runtime.ingest_identity_evidence(payload))
            elif path == "/api/register-person":
                self._send_json(
                    self.runtime.register_person(
                        str(payload.get("temporary_id") or ""),
                        str(payload.get("persistent_id") or ""),
                    )
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_GATEWAY)

    def _stream_workspace(self) -> None:
        """Push every broadcast cycle to a subscriber; never block perception."""
        import queue as queue_module
        from .global_workspace import sse_event
        key, channel = self.runtime.workspace.subscribe()
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                try:
                    contents = channel.get(timeout=15.0)
                except queue_module.Empty:
                    self.wfile.write(b": heartbeat\n\n")  # keeps proxies from closing an idle stream
                    self.wfile.flush()
                    continue
                self.wfile.write(sse_event(contents))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return
        finally:
            self.runtime.workspace.unsubscribe(key)

    def log_message(self, _format: str, *_args) -> None:
        return

    def _call(self, function) -> None:
        try:
            self._send_json(function())
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_GATEWAY)

    def _send_file(self, name: str, content_type: str) -> None:
        path = self.web_dir / name
        body = path.read_bytes()
        self._send(body, content_type, HTTPStatus.OK)

    def _send_json(self, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def _send(self, body: bytes, content_type: str, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(host: str, port: int, config: LiveRuntimeConfig) -> None:
    runtime = LivePerceptionRuntime(config)
    handler = type(
        "ConfiguredLiveDashboardHandler",
        (LiveDashboardHandler,),
        {"runtime": runtime, "web_dir": Path(__file__).resolve().parents[1] / "web"},
    )
    server = ThreadingHTTPServer((host, port), handler)
    try:
        print(f"live perception console: http://{host}:{port}", flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--vision-url", default="http://127.0.0.1:8080")
    parser.add_argument("--voice-url", default="http://127.0.0.1:8090")
    parser.add_argument("--brain-url", default="http://127.0.0.1:8094")
    parser.add_argument("--memory-url", default="http://127.0.0.1:8788")
    parser.add_argument("--rosbridge-url", default="ws://127.0.0.1:9090")
    parser.add_argument("--algorithm", default="ros4hri_native")
    parser.add_argument("--poll-hz", type=float, default=20.0)
    parser.add_argument("--subject-id", default="local-user")
    parser.add_argument("--robot-id", default="reachy-mini")
    parser.add_argument("--session-id", default="mac-first-test")
    parser.add_argument("--s2s", action="store_true")
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Do not open a user-facing camera; use acoustic attention/wake words only.",
    )
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="Do not require a transcript wake word for acoustic-only attention.",
    )
    parser.add_argument("--llm-url", default="")
    parser.add_argument("--llm-model", default="local-model")
    args = parser.parse_args()
    serve(
        args.host,
        args.port,
        LiveRuntimeConfig(
            vision_url=args.vision_url,
            voice_url=args.voice_url,
            brain_url=args.brain_url,
            memory_url=args.memory_url,
            rosbridge_url=args.rosbridge_url,
            poll_hz=args.poll_hz,
            algorithm=args.algorithm,
            subject_id=args.subject_id,
            robot_id=args.robot_id,
            session_id=args.session_id,
            s2s_enabled=args.s2s,
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            llm_api_key=os.environ.get("S2S_API_KEY", ""),
            vision_enabled=not args.no_vision,
            use_transcript_wake_words=not args.no_wake_word,
        ),
    )


if __name__ == "__main__":
    main()
