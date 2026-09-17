#!/usr/bin/env python3
"""Render the checked-in architecture Markdown as a PDF."""

from __future__ import annotations

import html
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/ARCHITECTURE_REPORT.md"
HTML = ROOT / "docs/ARCHITECTURE_REPORT.html"
PDF = Path("/Applications/权重/robot_perception_architecture_and_test_guide.pdf")
SCENARIO_FILE = ROOT.parent / "voice-detection/data/asr_acceptance_scenarios.jsonl"


def scenario_table() -> str:
    rows = [json.loads(line) for line in SCENARIO_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = ["| # | 场景 ID / 名称 | 阶段 | 测试原句 | 所需能力 |", "| ---: | --- | --- | --- | --- |"]
    for number, item in enumerate(rows, 1):
        capabilities = ", ".join(str(value) for value in item.get("required_capabilities", []))
        utterance = str(item.get("utterance_expected") or "无固定原句").replace("|", "\\|")
        name = f"`{item.get('scenario_id', '')}` / {item.get('scenario_name', '')}"
        result.append(f"| {number} | {name} | `{item.get('test_stage', '')}` | {utterance} | {capabilities} |")
    return "\n".join(result)


def markdown_to_html() -> None:
    markdown = REPORT.read_text(encoding="utf-8").replace("<!-- SCENARIO_TABLE -->", scenario_table())
    temporary_report = ROOT / "docs/.ARCHITECTURE_REPORT.generated.md"
    temporary_report.write_text(markdown, encoding="utf-8")
    subprocess.run(
        [
            "pandoc",
            str(temporary_report),
            "--standalone",
            "--from=markdown",
            "--to=html5",
            "--css=architecture_report.css",
            "--metadata",
            "title=仿生视听感知系统架构与手工测试指南",
            "--output",
            str(HTML),
        ],
        cwd=ROOT,
        check=True,
    )
    temporary_report.unlink(missing_ok=True)


def render_pdf() -> None:
    PDF.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "npx",
            "--yes",
            "playwright",
            "pdf",
            "--browser=chromium",
            "--paper-format",
            "A4",
            "--wait-for-timeout",
            "500",
            f"file://{HTML}",
            str(PDF),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    markdown_to_html()
    render_pdf()
    print(f"generated {PDF}")


if __name__ == "__main__":
    main()
