#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a heuristic RobotWin clean-only subset by slicing a fixed-size clean window "
            "from each task block."
        )
    )
    parser.add_argument(
        "--episodes",
        default="/data/zijianzhang/gaussianwam_data/data/robotwin2.0/robotwin2.0/meta/episodes.jsonl",
        help="Path to the packed RobotWin episodes.jsonl file.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="/data/zijianzhang/gaussianwam_data/data/robotwin2.0/robotwin2.0",
        help="Dataset root to store in each manifest row.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path for per-block summaries.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=550,
        help="Episodes per task block in the packed dataset.",
    )
    parser.add_argument(
        "--clean-size",
        type=int,
        default=50,
        help="Number of episodes to keep from each task block.",
    )
    parser.add_argument(
        "--position",
        choices=("first", "last"),
        default="first",
        help="Whether to keep the first or last clean_size episodes from each block.",
    )
    return parser.parse_args()


def load_episode_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            expected_episode_index = len(rows)
            episode_index = int(row["episode_index"])
            if episode_index != expected_episode_index:
                raise ValueError(
                    f"Expected contiguous episode_index={expected_episode_index} at line {line_no}, "
                    f"got {episode_index}."
                )
            rows.append(row)
    return rows


def build_subset(
    rows: list[dict],
    dataset_dir: str,
    group_size: int,
    clean_size: int,
    position: str,
) -> tuple[list[dict], list[dict]]:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if clean_size <= 0:
        raise ValueError(f"clean_size must be positive, got {clean_size}.")
    if clean_size > group_size:
        raise ValueError(f"clean_size ({clean_size}) cannot exceed group_size ({group_size}).")
    if len(rows) % group_size != 0:
        raise ValueError(
            f"Episode count {len(rows)} is not divisible by group_size {group_size}; "
            "the heuristic block assumption does not hold."
        )

    subset_rows: list[dict] = []
    report_rows: list[dict] = []
    num_blocks = len(rows) // group_size

    for block_id in range(num_blocks):
        block_start = block_id * group_size
        block_end = block_start + group_size - 1
        if position == "first":
            selected_start = block_start
        else:
            selected_start = block_end - clean_size + 1
        selected_end = selected_start + clean_size - 1

        block_prompt = rows[block_start]["tasks"][0]
        selected_prompt = rows[selected_start]["tasks"][0]

        report_rows.append(
            {
                "block_id": block_id,
                "block_episode_start": block_start,
                "block_episode_end": block_end,
                "selected_episode_start": selected_start,
                "selected_episode_end": selected_end,
                "group_size": group_size,
                "clean_size": clean_size,
                "position": position,
                "block_prompt_preview": block_prompt,
                "selected_prompt_preview": selected_prompt,
            }
        )

        for episode_index in range(selected_start, selected_end + 1):
            subset_rows.append(
                {
                    "dataset_dir": dataset_dir,
                    "episode_index": episode_index,
                    "heuristic_position": position,
                    "heuristic_group_size": group_size,
                    "heuristic_clean_size": clean_size,
                    "heuristic_block_id": block_id,
                }
            )

    return subset_rows, report_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    episode_path = Path(args.episodes)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report is not None else output_path.with_suffix(".report.json")

    rows = load_episode_rows(episode_path)
    subset_rows, report_rows = build_subset(
        rows=rows,
        dataset_dir=args.dataset_dir,
        group_size=args.group_size,
        clean_size=args.clean_size,
        position=args.position,
    )

    write_jsonl(output_path, subset_rows)
    write_json(
        report_path,
        {
            "episodes_path": str(episode_path),
            "dataset_dir": args.dataset_dir,
            "position": args.position,
            "group_size": args.group_size,
            "clean_size": args.clean_size,
            "total_episodes": len(rows),
            "num_blocks": len(report_rows),
            "selected_episodes": len(subset_rows),
            "blocks": report_rows,
        },
    )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "report": str(report_path),
                "position": args.position,
                "group_size": args.group_size,
                "clean_size": args.clean_size,
                "total_episodes": len(rows),
                "num_blocks": len(report_rows),
                "selected_episodes": len(subset_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
