#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path('/data/zijianzhang/FastWAM/data/robotwin2.0/subsets/clean_15tasks_table_first50.jsonl')
OUT_DIR = Path('/data/zijianzhang/FastWAM/data/robotwin2.0/subsets/clean_15tasks_table_per_task')
TASKS = [
    'adjust_bottle',
    'beat_block_hammer',
    'blocks_ranking_rgb',
    'blocks_ranking_size',
    'click_alarmclock',
    'click_bell',
    'grab_roller',
    'move_playingcard_away',
    'pick_diverse_bottles',
    'pick_dual_bottles',
    'place_a2b_left',
    'place_a2b_right',
    'place_container_plate',
    'place_fan',
    'put_bottles_dustbin',
]


def main() -> None:
    rows_by_task: dict[str, list[dict]] = {task: [] for task in TASKS}
    with SOURCE.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get('canonical_task') or row.get('task_name')
            if task in rows_by_task:
                rows_by_task[task].append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for task in TASKS:
        rows = rows_by_task[task]
        if len(rows) != 50:
            raise RuntimeError(f'{task}: expected 50 rows, got {len(rows)}')
        path = OUT_DIR / f'{task}_first50.jsonl'
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        report[task] = {
            'path': str(path),
            'episodes': [int(row['episode_index']) for row in rows],
            'count': len(rows),
        }
        print(f'{task}\t{path}\t{len(rows)}')

    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
