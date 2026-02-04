from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from forechecking_pressure_topology.pipeline import ForecheckPipelineConfig, load_config, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Forechecking Pressure Topology pipeline.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "default_config.json")
    parser.add_argument("--max-games", type=int, default=None, help="Optional override for quick runs.")
    parser.add_argument("--frame-stride", type=int, default=None, help="Optional frame-stride override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config if args.config and args.config.exists() else None)
    if args.max_games is not None:
        cfg.max_games = args.max_games
    if args.frame_stride is not None:
        cfg.frame_stride = args.frame_stride

    summary = run_pipeline(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

