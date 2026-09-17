from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .fusion import AttentionFusion
from .scenario import load_scenario
from .dashboard import serve as serve_dashboard


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("algorithms")
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--scenario", required=True)
    simulate.add_argument("--algorithm", default="ros4hri_native")
    dashboard = subparsers.add_parser("dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8091)
    dashboard.add_argument("--scenario", default="fixtures/cocktail_effect_three_people.json")
    args = parser.parse_args()

    if args.command == "algorithms":
        for name in AttentionFusion.available_algorithms():
            algorithm = AttentionFusion.create_algorithm(name)
            print(f"{name}\t{algorithm.description}")
    elif args.command == "simulate":
        fusion = AttentionFusion(algorithm=args.algorithm)
        for frame in load_scenario(args.scenario):
            state = fusion.update(frame)
            print(json.dumps(asdict(state), ensure_ascii=False, separators=(",", ":")))
    elif args.command == "dashboard":
        serve_dashboard(args.host, args.port, args.scenario)


if __name__ == "__main__":
    main()
