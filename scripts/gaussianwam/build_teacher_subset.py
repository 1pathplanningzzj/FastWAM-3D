#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaussianwam.config import load_config
from gaussianwam.data import build_raw_dataset, video_sample_indices


def _episode_anchors(start: int, end: int, num_frames: int, action_video_freq_ratio: int, frame_policy: str, frame_stride: int) -> list[int]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a demo/episode-balanced GaussianWAM teacher subset manifest.")
    parser.add_argument("--config", default="configs/gaussianwam/stage1_robotwin_learn_feature_z.yaml")
    parser.add_argument("--output", default="/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_subsets/demo_subset.jsonl")
    parser.add_argument("--mode", choices=["episodes", "instruction_tasks"], default="episodes")
    parser.add_argument("--num-demos", type=int, default=100)
    parser.add_argument("--start-demo", type=int, default=0)
    parser.add_argument("--shuffle-demos", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-task-read", action="store_true")
    parser.add_argument("--demos-per-task", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--frame-policy", choices=["video_indices", "video_stride", "all", "stride"], default="video_stride")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    ds = build_raw_dataset(cfg.source)
    starts = ds.episode_data_index["from"].tolist()
    ends = ds.episode_data_index["to"].tolist()
    rows = []

    if args.mode == "episodes":
        max_episode_count = len(starts) if args.max_episodes is None else min(int(args.max_episodes), len(starts))
        candidates = list(range(max_episode_count))
        if args.shuffle_demos:
            rng = random.Random(int(args.seed))
            rng.shuffle(candidates)
        start_demo = min(max(int(args.start_demo), 0), len(candidates))
        stop_demo = min(start_demo + int(args.num_demos), len(candidates))
        selected = candidates[start_demo:stop_demo]
        for subset_demo_i, ep_idx in enumerate(selected):
            start, end = int(starts[ep_idx]), int(ends[ep_idx])
            anchors = _episode_anchors(start, end, int(cfg.source.num_frames), int(cfg.source.action_video_freq_ratio), args.frame_policy, int(args.frame_stride))
            if args.skip_task_read:
                task = f"episode_{ep_idx}"
            else:
                sample = ds[start]
                task = str(sample.get("task", ""))
            for local_i, idx in enumerate(anchors):
                rows.append(
                    {
                        "idx": int(idx),
                        "episode_index": int(ep_idx),
                        "subset_demo_index": int(subset_demo_i),
                        "episode_start": int(start),
                        "episode_end": int(end),
                        "episode_frame_index": int(idx - start),
                        "subset_frame_index": int(local_i),
                        "task": task,
                    }
                )
        episodes_count = len(selected)
        tasks_count = len({r["task"] for r in rows})
    else:
        by_task: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        max_episodes = len(starts) if args.max_episodes is None else min(args.max_episodes, len(starts))
        for ep_idx in range(max_episodes):
            start, end = int(starts[ep_idx]), int(ends[ep_idx])
            if end <= start:
                continue
            sample = ds[start]
            task = str(sample.get("task", ""))
            by_task[task].append((ep_idx, start, end))
        tasks = sorted(by_task.keys())
        if args.max_tasks is not None:
            tasks = tasks[: int(args.max_tasks)]
        for task in tasks:
            for ep_idx, start, end in by_task[task][: int(args.demos_per_task)]:
                anchors = _episode_anchors(start, end, int(cfg.source.num_frames), int(cfg.source.action_video_freq_ratio), args.frame_policy, int(args.frame_stride))
                for local_i, idx in enumerate(anchors):
                    rows.append(
                        {
                            "idx": int(idx),
                            "episode_index": int(ep_idx),
                            "episode_start": int(start),
                            "episode_end": int(end),
                            "episode_frame_index": int(idx - start),
                            "subset_frame_index": int(local_i),
                            "task": task,
                        }
                    )
        episodes_count = sum(min(len(by_task[t]), int(args.demos_per_task)) for t in tasks)
        tasks_count = len(tasks)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print({"output": str(output), "mode": args.mode, "tasks": tasks_count, "episodes": episodes_count, "rows": len(rows)})


if __name__ == "__main__":
    main()
