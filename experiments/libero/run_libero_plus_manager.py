import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_plus_benchmark import (
    DEFAULT_LIBERO_PLUS_CONFIG_DIR,
    DEFAULT_LIBERO_PLUS_ROOT,
    PLUS_SHORT_ORDER,
    LiberoPlusBenchmark,
    configure_libero_plus_runtime,
    normalize_plus_category,
    short_plus_category,
)


def _resolve_optional_path(raw_value, env_name: str, default_value: Path) -> Path:
    if raw_value is None:
        raw = os.environ.get(env_name, str(default_value))
    else:
        text = str(raw_value).strip()
        raw = os.environ.get(env_name, str(default_value)) if text.lower() in {"", "none", "null"} else text
    return Path(os.path.expanduser(os.path.expandvars(str(raw))))


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.task_plan_file",
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    hydra_overrides = list(HydraConfig.get().overrides.task)
    return [ov for ov in hydra_overrides if not _is_blocked_override(ov)]


def _resolve_worker_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError(
            "Hydra task choice is empty. Please pass task=... (e.g., task=libero_gaussianwam_stage2_current_2cam224_1e-4)."
        )
    return str(task_choice)


def _parse_visible_gpus(num_gpus: int) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        gpu_list = [item.strip() for item in visible.split(",") if item.strip()]
        if len(gpu_list) < num_gpus:
            raise ValueError(
                f"Requested MULTIRUN.num_gpus={num_gpus}, but CUDA_VISIBLE_DEVICES only has {len(gpu_list)} entries: {gpu_list}"
            )
        return gpu_list[:num_gpus]
    return [str(i) for i in range(num_gpus)]


def _grouped_task_specs(
    task_suite_names: list[str],
    *,
    plus_root: Path,
    plus_config_dir: Path,
    plus_category: str | None,
) -> list[tuple[str, int, str]]:
    configure_libero_plus_runtime(plus_root, plus_config_dir)
    grouped: dict[str, list[tuple[str, int, str]]] = defaultdict(list)

    for suite_name in task_suite_names:
        task_suite = LiberoPlusBenchmark(
            suite_name,
            plus_root=plus_root,
            category=plus_category,
        )
        for task_id, task in enumerate(task_suite.tasks):
            grouped[short_plus_category(task.plus_category)].append((suite_name, task_id, task.name))

    ordered = []
    for short in PLUS_SHORT_ORDER:
        bucket = grouped.get(short, [])
        ordered.extend(bucket)

    for short, bucket in grouped.items():
        if short not in PLUS_SHORT_ORDER:
            ordered.extend(bucket)
    return ordered


def _write_worker_plans(
    output_dir: Path,
    gpu_ids: list[str],
    task_specs: list[tuple[str, int, str]],
) -> list[Path]:
    plan_dir = output_dir / "worker_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    assignments: list[list[tuple[str, int, str]]] = [[] for _ in gpu_ids]

    for idx, spec in enumerate(task_specs):
        assignments[idx % len(gpu_ids)].append(spec)

    plan_files: list[Path] = []
    for worker_idx, (gpu_id, worker_specs) in enumerate(zip(gpu_ids, assignments)):
        plan_file = plan_dir / f"worker{worker_idx}_gpu{gpu_id}.txt"
        with plan_file.open("w", encoding="utf-8") as f:
            for suite_name, task_id, _ in worker_specs:
                f.write(f"{suite_name},{task_id}\n")
        plan_files.append(plan_file)
    return plan_files


def _start_tmux_workers(
    *,
    task_choice: str,
    ckpt: str,
    output_dir: Path,
    gpu_ids: list[str],
    plan_files: list[Path],
    extra_overrides: list[str],
    plus_root: Path,
    plus_config_dir: Path,
) -> None:
    session_name = "libero_plus_worker"
    python_bin = os.environ.get("PYTHON_BIN", "/data/miniconda3/envs/fastwam-libero/bin/python")
    mplconfigdir = os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib-fastwam-libero")
    mujoco_gl = os.environ.get("MUJOCO_GL", "egl")
    pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM", "egl")
    extra_args = " ".join(shlex.quote(arg) for arg in extra_overrides)

    subprocess.run(["tmux", "kill-session", "-t", session_name], check=False)
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name], check=True)

    for worker_idx, (gpu_id, plan_file) in enumerate(zip(gpu_ids, plan_files)):
        if worker_idx > 0:
            subprocess.run(["tmux", "new-window", "-t", f"{session_name}:{worker_idx}"], check=True)
        target = f"{session_name}:{worker_idx}.0"
        cmd = (
            f"source ~/.bashrc && cd {shlex.quote(str(project_root))} && "
            f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_id)} "
            f"MPLCONFIGDIR={shlex.quote(mplconfigdir)} "
            f"MUJOCO_GL={shlex.quote(mujoco_gl)} "
            f"PYOPENGL_PLATFORM={shlex.quote(pyopengl_platform)} "
            f"LIBERO_PLUS_ROOT={shlex.quote(str(plus_root))} "
            f"FASTWAM_LIBERO_PLUS_CONFIG_DIR={shlex.quote(str(plus_config_dir))} "
            f"{shlex.quote(python_bin)} experiments/libero/eval_libero_plus.py "
            f"task={shlex.quote(task_choice)} "
            f"ckpt={shlex.quote(ckpt)} "
            f"gpu_id={shlex.quote(gpu_id)} "
            f"EVALUATION.output_dir={shlex.quote(str(output_dir))} "
            f"EVALUATION.task_plan_file={shlex.quote(str(plan_file))} "
            f"EVALUATION.num_trials=1 "
            f"{extra_args}"
        ).strip()
        subprocess.run(["tmux", "send-keys", "-t", target, cmd, "C-m"], check=True)

    print(f"Started tmux session: {session_name}")
    print(f"Attach with: tmux attach -t {session_name}")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")

    task_choice = _resolve_worker_task_choice()
    manager = cfg.MULTIRUN

    plus_root = _resolve_optional_path(
        cfg.EVALUATION.get("plus_root", None),
        "LIBERO_PLUS_ROOT",
        DEFAULT_LIBERO_PLUS_ROOT,
    ).resolve()
    plus_config_dir = _resolve_optional_path(
        cfg.EVALUATION.get("plus_config_dir", None),
        "FASTWAM_LIBERO_PLUS_CONFIG_DIR",
        DEFAULT_LIBERO_PLUS_CONFIG_DIR,
    ).resolve()
    plus_category = normalize_plus_category(
        cfg.EVALUATION.get("plus_category", os.environ.get("LIBERO_PLUS_CATEGORY"))
    )

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    gpu_ids = _parse_visible_gpus(int(manager.num_gpus))
    task_specs = _grouped_task_specs(
        list(manager.task_suite_names),
        plus_root=plus_root,
        plus_config_dir=plus_config_dir,
        plus_category=plus_category,
    )
    plan_files = _write_worker_plans(output_dir, gpu_ids, task_specs)

    summary = output_dir / "worker_plan_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"total_tasks={len(task_specs)}\n")
        for gpu_id, plan_file in zip(gpu_ids, plan_files):
            count = len([line for line in plan_file.read_text(encoding='utf-8').splitlines() if line.strip()])
            f.write(f"gpu={gpu_id} plan={plan_file.name} tasks={count}\n")

    if bool(manager.get("create_only", False)):
        print(f"Worker plans created under: {output_dir / 'worker_plans'}")
        return

    _start_tmux_workers(
        task_choice=task_choice,
        ckpt=str(cfg.ckpt),
        output_dir=output_dir,
        gpu_ids=gpu_ids,
        plan_files=plan_files,
        extra_overrides=collect_worker_overrides(),
        plus_root=plus_root,
        plus_config_dir=plus_config_dir,
    )


if __name__ == "__main__":
    main()
