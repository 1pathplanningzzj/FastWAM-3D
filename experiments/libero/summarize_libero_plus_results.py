import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_plus_benchmark import PLUS_CATEGORY_TO_SHORT, PLUS_SHORT_ORDER


def summarize_libero_plus_results(output_dir: str) -> dict:
    root = Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(f"Output directory not found: {root}")

    category_stats = defaultdict(lambda: {"tasks": 0, "successes": 0, "episodes": 0})
    suite_stats = defaultdict(lambda: {"tasks": 0, "successes": 0, "episodes": 0})
    total_successes = 0
    total_episodes = 0
    total_tasks = 0

    for result_file in root.rglob("*_results.json"):
        result = json.loads(result_file.read_text(encoding="utf-8"))
        category_long = result.get("plus_category")
        short = result.get("plus_category_short") or PLUS_CATEGORY_TO_SHORT.get(category_long)
        if short is None:
            continue

        successes = int(result["successes"])
        episodes = int(result["total_episodes"])
        suite = str(result.get("task_suite", result_file.parent.name))

        category_stats[short]["tasks"] += 1
        category_stats[short]["successes"] += successes
        category_stats[short]["episodes"] += episodes

        suite_stats[suite]["tasks"] += 1
        suite_stats[suite]["successes"] += successes
        suite_stats[suite]["episodes"] += episodes

        total_successes += successes
        total_episodes += episodes
        total_tasks += 1

    category_success_rates = {}
    for short in PLUS_SHORT_ORDER:
        stats = category_stats.get(short, {"tasks": 0, "successes": 0, "episodes": 0})
        episodes = stats["episodes"]
        category_success_rates[short] = (
            stats["successes"] / episodes * 100.0 if episodes > 0 else None
        )

    avg_success_rate = (
        sum(rate for rate in category_success_rates.values() if rate is not None)
        / sum(1 for rate in category_success_rates.values() if rate is not None)
        if any(rate is not None for rate in category_success_rates.values())
        else None
    )

    output = {
        "output_dir": str(root),
        "total_tasks_finished": total_tasks,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "overall_micro_success_rate": (
            total_successes / total_episodes * 100.0 if total_episodes > 0 else None
        ),
        "category_success_rate_percent": category_success_rates,
        "category_raw": {k: dict(v) for k, v in category_stats.items()},
        "suite_raw": {k: dict(v) for k, v in suite_stats.items()},
        "avg_category_success_rate_percent": avg_success_rate,
    }

    summary_path = root / "libero_plus_summary.json"
    summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    headers = PLUS_SHORT_ORDER + ["Avg"]
    values = [category_success_rates[h] for h in PLUS_SHORT_ORDER] + [avg_success_rate]

    print(" ".join(headers))
    print(
        " ".join(
            "N/A" if value is None else f"{value:.2f}"
            for value in values
        )
    )
    print(f"\nFinished tasks: {total_tasks}")
    print(
        "Overall micro success rate: "
        + ("N/A" if output["overall_micro_success_rate"] is None else f"{output['overall_micro_success_rate']:.2f}%")
    )
    print(f"Saved: {summary_path}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    summarize_libero_plus_results(args.output_dir)


if __name__ == "__main__":
    main()
