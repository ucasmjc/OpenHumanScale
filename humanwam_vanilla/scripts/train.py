#!/usr/bin/env python3
"""Configure and launch RoboTwin fine-tuning; --check-config does no GPU work."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from openhumanscale.config import load_config, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/robotwin.yaml')
    parser.add_argument('--paths', type=Path, help='Local path bindings, shaped like configs/paths.example.yaml')
    parser.add_argument('--check-config', action='store_true', help='Validate config and input metadata without loading models')
    parser.add_argument('overrides', nargs='*', help='Explicit OmegaConf key=value overrides')
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths, args.overrides)
    report = validate_config(cfg, check_files=True)
    print(json.dumps(report, indent=2))
    if args.check_config:
        return
    from humanwam.runtime import run_training
    run_training(cfg)


if __name__ == '__main__':
    main()
