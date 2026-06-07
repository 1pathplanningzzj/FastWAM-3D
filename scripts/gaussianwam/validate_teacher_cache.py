#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaussianwam.cache import cache_root
from gaussianwam.config import load_config
from gaussianwam.validate import tensor_summary, validate_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GaussianWAM teacher cache files.")
    parser.add_argument("--config", default="configs/gaussianwam/stage1_robotwin.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    root = cache_root(cfg.output_dir, int(cfg.cache.version), str(cfg.cache.namespace), str(cfg.split))
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    checked = 0
    errors = []
    latest_by_sample = {}
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = row.get("idx", row.get("cache_key"))
            latest_by_sample[key] = row
    for row in latest_by_sample.values():
        if row.get("status") != "ok":
            continue
        path = Path(row["path"])
        if not path.exists():
            errors.append(f"missing cache file: {path}")
            continue
        payload = torch.load(path, map_location="cpu")
        errs = validate_payload(payload, cfg)
        if errs:
            errors.extend([f"{path}: {e}" for e in errs])
        if checked == 0:
            print("First cache summaries:")
            for key, value in payload.get("targets", {}).items():
                if torch.is_tensor(value):
                    print(key, tensor_summary(value))
        checked += 1
        if args.limit is not None and checked >= args.limit:
            break
    print({"checked": checked, "errors": len(errors), "root": str(root)})
    for err in errors[:20]:
        print("ERROR", err)
    if checked == 0:
        errors.append("no ok cache entries were checked")
        print("ERROR no ok cache entries were checked")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
