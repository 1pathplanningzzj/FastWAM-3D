#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SWITCH_INCLUDE = re.compile(
    r"(press|tap|activate|click|engage|operate|interact|push|trigger|toggle).{0,50}\bswitch\b|"
    r"\bswitch\b.{0,50}(press|tap|activate|click|engage|operate|interact|push|trigger|toggle)|"
    r"control switch|rectangular switch|electrical switch"
)
SWITCH_EXCLUDE = re.compile(
    r"other arm|other hand|opposite arm|opposite side|switch hands|pass it|hand it|transfer it|move it to the opposite|microphone"
)
MICROWAVE_INCLUDE = re.compile(r"\bmicrowave\b")
MUG_INCLUDE = re.compile(r"\bmug\b")


def classify_episode(tasks: list[str]) -> str | None:
    text = " ".join(tasks).lower()
    hits: list[str] = []
    if SWITCH_INCLUDE.search(text) and not SWITCH_EXCLUDE.search(text):
        hits.append("switch")
    if MICROWAVE_INCLUDE.search(text):
        hits.append("microwave")
    if MUG_INCLUDE.search(text):
        hits.append("mug")
    if len(hits) != 1:
        return None
    return hits[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean focus3 RoboTwin episode subset using true prompt semantics.")
    parser.add_argument("--episodes", default="data/robotwin2.0/robotwin2.0/meta/episodes.jsonl")
    parser.add_argument("--output", default="data/robotwin2.0/subsets/stage2_focus3_switch_microwave_mug_clean.jsonl")
    args = parser.parse_args()

    rows: list[dict] = []
    counts = {"switch": 0, "microwave": 0, "mug": 0}
    frames = {"switch": 0, "microwave": 0, "mug": 0}

    with Path(args.episodes).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            matched = classify_episode(row["tasks"])
            if matched is None:
                continue
            out_row = {
                "dataset_dir": "./data/robotwin2.0/robotwin2.0",
                "episode_index": int(row["episode_index"]),
                "matched_keywords": [matched],
                "num_prompts": len(row["tasks"]),
                "length": int(row["length"]),
            }
            rows.append(out_row)
            counts[matched] += 1
            frames[matched] += int(row["length"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        {
            "output": str(output),
            "episodes": len(rows),
            "counts": counts,
            "frame_counts": frames,
        }
    )


if __name__ == "__main__":
    main()
