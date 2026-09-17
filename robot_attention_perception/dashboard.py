from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .fusion import AttentionFusion
from .scenario import load_scenario


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>robot-attention-perception</title>
  <style>
    body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111719; color: #eef5f2; }
    header { padding: 16px 22px; border-bottom: 1px solid #293536; display: flex; justify-content: space-between; gap: 14px; align-items: center; }
    h1, h2, p { margin: 0; }
    h1 { font-size: 20px; }
    main { padding: 18px; display: grid; grid-template-columns: minmax(320px, 0.8fr) minmax(420px, 1.2fr); gap: 14px; }
    section { border: 1px solid #2b3a3b; background: #151d1f; border-radius: 8px; padding: 14px; }
    .controls { display: grid; grid-template-columns: 1fr 1fr auto; gap: 10px; align-items: end; }
    label { display: grid; gap: 5px; color: #a8bbb7; font-size: 13px; }
    select, input, button { font: inherit; min-height: 36px; border-radius: 6px; border: 1px solid #3a4a4b; background: #0f1416; color: #eef5f2; padding: 0 10px; }
    button { background: #2e6650; border: 0; cursor: pointer; }
    .metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; background: #2b3a3b; margin-top: 12px; }
    .metric { min-height: 78px; background: #101719; padding: 12px; display: grid; align-content: space-between; }
    .metric span { color: #99aba7; font-size: 12px; }
    .metric strong { font-size: 20px; overflow-wrap: anywhere; }
    pre { background: #080b0c; color: #cfe9df; border-radius: 6px; padding: 12px; min-height: 420px; overflow: auto; }
    .reasons { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .reason { border: 1px solid #3a4a4b; border-radius: 6px; padding: 6px 8px; color: #dbe8e4; background: #0f1416; }
    @media (max-width: 860px) { main, .controls { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>robot-attention-perception</h1>
    <p id="status">loading</p>
  </header>
  <main>
    <section>
      <div class="controls">
        <label>algorithm<select id="algorithm"></select></label>
        <label>frame<input id="frame" value="0" inputmode="numeric"></label>
        <button id="step">Step</button>
      </div>
      <div class="metrics">
        <div class="metric"><span>target</span><strong id="target">none</strong></div>
        <div class="metric"><span>source track</span><strong id="source">none</strong></div>
        <div class="metric"><span>confidence</span><strong id="confidence">0.00</strong></div>
        <div class="metric"><span>listen</span><strong id="listen">false</strong></div>
        <div class="metric"><span>addressed</span><strong id="addressed">false</strong></div>
        <div class="metric"><span>azimuth</span><strong id="azimuth">--</strong></div>
      </div>
      <div id="reasons" class="reasons"></div>
    </section>
    <section>
      <h2>state</h2>
      <pre id="json"></pre>
    </section>
  </main>
  <script>
    const $ = id => document.getElementById(id);
    async function loadAlgorithms() {
      const data = await (await fetch('/api/algorithms')).json();
      $('algorithm').innerHTML = data.algorithms.map(item => `<option value="${item.name}">${item.name}</option>`).join('');
    }
    async function update() {
      const params = new URLSearchParams({algorithm: $('algorithm').value, frame: $('frame').value});
      const data = await (await fetch(`/api/state?${params.toString()}`)).json();
      const state = data.state || {};
      $('status').textContent = `frame ${data.frame_index + 1}/${data.frame_count}`;
      $('target').textContent = state.target_id || 'none';
      $('source').textContent = state.source_track_id || 'none';
      $('confidence').textContent = Number(state.confidence || 0).toFixed(2);
      $('listen').textContent = String(Boolean(state.listen));
      $('addressed').textContent = String(Boolean(state.addressed_to_robot));
      $('azimuth').textContent = state.azimuth_deg === null || state.azimuth_deg === undefined ? '--' : `${Number(state.azimuth_deg).toFixed(1)} deg`;
      $('reasons').innerHTML = (state.reasons || []).map(reason => `<span class="reason">${reason}</span>`).join('');
      $('json').textContent = JSON.stringify(data, null, 2);
    }
    $('step').onclick = () => { $('frame').value = String(Number($('frame').value || 0) + 1); update(); };
    $('algorithm').onchange = update;
    $('frame').onchange = update;
    loadAlgorithms().then(update);
  </script>
</body>
</html>
"""


class AttentionDashboard:
    def __init__(self, scenario_path: str | Path):
        self.lock = threading.Lock()
        self.scenario_path = Path(scenario_path)
        self.frames = load_scenario(self.scenario_path)

    def algorithms(self) -> list[dict[str, str]]:
        return [
            {
                "name": name,
                "description": AttentionFusion.create_algorithm(name).description,
            }
            for name in AttentionFusion.available_algorithms()
        ]

    def state(self, algorithm: str, frame_index: int) -> dict[str, object]:
        with self.lock:
            frames = self.frames
        if not frames:
            return {"frame_index": 0, "frame_count": 0, "state": None}
        bounded_index = max(0, min(frame_index, len(frames) - 1))
        fusion = AttentionFusion(algorithm=algorithm)
        state = None
        for frame in frames[: bounded_index + 1]:
            state = fusion.update(frame)
        return {
            "scenario": str(self.scenario_path),
            "algorithm": algorithm,
            "frame_index": bounded_index,
            "frame_count": len(frames),
            "state": asdict(state) if state is not None else None,
        }


def make_handler(runtime: AttentionDashboard):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
            elif parsed.path == "/api/algorithms":
                self._send_json({"algorithms": runtime.algorithms()})
            elif parsed.path == "/api/state":
                self._send_json(
                    runtime.state(
                        algorithm=query.get("algorithm", ["ros4hri_native"])[0],
                        frame_index=int(query.get("frame", ["0"])[0] or 0),
                    )
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, fmt, *args):  # noqa: A003
            return

        def _send_json(self, data):
            self._send_text(json.dumps(data, ensure_ascii=False), "application/json; charset=utf-8")

        def _send_text(self, text: str, content_type: str):
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(host: str, port: int, scenario: str | Path) -> None:
    runtime = AttentionDashboard(scenario)
    server = ThreadingHTTPServer((host, port), make_handler(runtime))
    try:
        print(f"attention dashboard: http://{host}:{port}", flush=True)
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--scenario", default="fixtures/cocktail_effect_three_people.json")
    args = parser.parse_args()
    serve(args.host, args.port, args.scenario)


if __name__ == "__main__":
    main()
