#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaussianwam.config import load_config
from gaussianwam.precompute import precompute


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute GaussianWAM Stage 1 teacher cache.")
    parser.add_argument("--config", default="configs/gaussianwam/stage1_robotwin.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    counts = precompute(cfg, limit=args.limit, dry_run=args.dry_run, overwrite=True if args.overwrite else None)
    print(counts)


if __name__ == "__main__":
    main()
