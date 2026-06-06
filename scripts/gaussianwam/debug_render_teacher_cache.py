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
from gaussianwam.debug import save_debug_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Export debug PNGs from one GaussianWAM teacher cache file.")
    parser.add_argument("--config", default="configs/gaussianwam/stage1_robotwin.yaml")
    parser.add_argument("--cache-file", default=None)
    parser.add_argument("--output-dir", default="./debug/gaussianwam_cache")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    cache_file = args.cache_file
    if cache_file is None:
        root = cache_root(cfg.output_dir, int(cfg.cache.version), str(cfg.cache.namespace), str(cfg.split))
        manifest = root / "manifest.jsonl"
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("status") == "ok":
                    cache_file = row["path"]
                    break
    if cache_file is None:
        raise FileNotFoundError("No ok cache file found")
    payload = torch.load(cache_file, map_location="cpu")
    targets = payload["targets"]
    render = {
        "feature_map": targets["T_gaussian_feature"].float(),
        "dep": targets["T_depth"].float(),
        "alpha": targets["T_alpha"].float(),
    }
    if "T_dense_feature_target" in targets:
        render["feature_target"] = targets["T_dense_feature_target"].float()
    if "T_teacher_valid_mask" in targets:
        render["teacher_valid_mask"] = targets["T_teacher_valid_mask"].float()
    if "T_render_valid_mask" in targets:
        render["render_valid_mask"] = targets["T_render_valid_mask"].float()
    save_debug_maps(args.output_dir, render)
    print({"cache_file": cache_file, "output_dir": args.output_dir})


if __name__ == "__main__":
    main()
