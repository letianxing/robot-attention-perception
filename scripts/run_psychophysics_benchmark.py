#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_attention_perception.psychophysics import full_benchmark


if __name__ == "__main__":
    print(json.dumps(full_benchmark(), ensure_ascii=False, indent=2))
