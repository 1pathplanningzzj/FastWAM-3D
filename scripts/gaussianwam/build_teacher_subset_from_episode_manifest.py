#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaussianwam.config import load_config
from gaussianwam.data import video_sample_indices


def _episode_anchors(
    start: int,
    end: int,
    num_frames: int,
    action_video_freq_ratio: int,
    frame_policy: str,
    frame_stride: int,
) -> list[int]:
    max_anchor = max(start, end - int(num_frames))
    if frame_policy == "video_indices":
        offsets = video_sample_indices(num_frames, action_video_freq_ratio)
        anchors = [start + off for off in offsets if start + off <= max_anchor]
        return anchors or [start]
    if frame_policy == "video_stride":
        return list(range(start, max_anchor + 1, max(int(action_video_freq_ratio), 1)))
    if frame_policy == "all":
        return list(range(start, max_anchor + 1))
    return list(range(start, max_anchor + 1, max(int(frame_stride), 1)))


def _load_episode_spans(path: str | Path) -> dict[int, tuple[int, int]]:
    spans: dict[int, tuple[int, int]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ep_idx = int(row["episode_index"])
            stats = row["stats"]["index"]
            start = int(stats["min"][0])
            end = int(stats["max"][0]) + 1
            spans[ep_idx] = (start, end)
    if not spans:
        raise ValueError(f"No episode spans found in {path}")
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand an episode-level subset manifest into a frame/window-level teacher subset manifest."
    )
    parser.add_argument("--config", default="configs/gaussianwam/stage1_robotwin_demo_subset.yaml")
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--episodes-stats", default="/data/zijianzhang/gaussianwam_data/data/robotwin2.0/robotwin2.0/meta/episodes_stats.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--frame-policy",
        choices=["video_indices", "video_stride", "all", "stride"],
        default="all",
    )
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    episode_spans = _load_episode_spans(args.episodes_stats)
    num_frames = int(cfg.source.num_frames)
    action_video_freq_ratio = int(cfg.source.action_video_freq_ratio)

    episode_rows = []
    with Path(args.episode_manifest).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episode_rows.append(json.loads(line))

    rows: list[dict] = []
    for subset_episode_i, episode_row in enumerate(episode_rows):
        ep_idx = int(episode_row["episode_index"])
        if ep_idx not in episode_spans:
            raise KeyError(f"episode_index={ep_idx} missing from {args.episodes_stats}")
        start, end = episode_spans[ep_idx]
        task = ",".join(str(x) for x in episode_row.get("matched_keywords", []))
        anchors = _episode_anchors(
            start,
            end,
            num_frames,
            action_video_freq_ratio,
            args.frame_policy,
            int(args.frame_stride),
        )
        for local_i, idx in enumerate(anchors):
            row = {
                "idx": int(idx),
                "episode_index": int(ep_idx),
                "subset_episode_index": int(subset_episode_i),
                "episode_start": int(start),
                "episode_end": int(end),
                "episode_frame_index": int(idx - start),
                "subset_frame_index": int(local_i),
                "task": task,
            }
            for key in ("matched_keywords", "num_prompts", "length", "dataset_dir"):
                if key in episode_row:
                    row[key] = episode_row[key]
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        {
            "output": str(output),
            "episodes": len(episode_rows),
            "rows": len(rows),
            "frame_policy": args.frame_policy,
            "num_frames": num_frames,
            "action_video_freq_ratio": action_video_freq_ratio,
        }
    )


if __name__ == "__main__":
    main()
