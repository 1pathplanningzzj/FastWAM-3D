import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_plus_benchmark import (
    DEFAULT_LIBERO_PLUS_CONFIG_DIR,
    DEFAULT_LIBERO_PLUS_ROOT,
    LiberoPlusBenchmark,
    configure_libero_plus_runtime,
    normalize_plus_category,
    short_plus_category,
)
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _resolve_optional_path(
    cfg_value: Any,
    env_name: str,
    default_value: Path,
) -> Path:
    if cfg_value is None:
        raw = os.environ.get(env_name, str(default_value))
    else:
        text = str(cfg_value).strip()
        raw = os.environ.get(env_name, str(default_value)) if text.lower() in {"", "none", "null"} else text
    return Path(os.path.expanduser(os.path.expandvars(str(raw))))


def _isolate_plus_config_dir(base_dir: Path) -> Path:
    """Avoid cross-process races on LIBERO_CONFIG_PATH.

    Each worker writes its own config.yaml under a pid-specific subdirectory.
    """
    return base_dir / f"pid_{os.getpid()}"


def _repeat_initial_states(initial_states, num_trials: int):
    if num_trials <= 0:
        return initial_states

    if isinstance(initial_states, list):
        if len(initial_states) == 0:
            raise ValueError("Initial states list is empty.")
        while len(initial_states) < num_trials:
            initial_states.extend(initial_states[: (num_trials - len(initial_states))])
        return initial_states[:num_trials]

    if isinstance(initial_states, tuple):
        return _repeat_initial_states(list(initial_states), num_trials)

    if torch.is_tensor(initial_states):
        if initial_states.ndim == 1:
            initial_states = initial_states.unsqueeze(0)
        if initial_states.shape[0] >= num_trials:
            return initial_states[:num_trials]
        repeats = (num_trials + int(initial_states.shape[0]) - 1) // int(initial_states.shape[0])
        return initial_states.repeat((repeats,) + (1,) * (initial_states.ndim - 1))[:num_trials]

    if isinstance(initial_states, np.ndarray):
        if initial_states.ndim == 1:
            initial_states = initial_states[None, ...]
        if initial_states.shape[0] >= num_trials:
            return initial_states[:num_trials]
        repeats = (num_trials + int(initial_states.shape[0]) - 1) // int(initial_states.shape[0])
        tiled = np.concatenate([initial_states] * repeats, axis=0)
        return tiled[:num_trials]

    raise TypeError(f"Unsupported initial_states type: {type(initial_states)}")


def _prepare_plus_runtime(cfg: DictConfig) -> tuple[Path, Path, Path, str | None]:
    plus_root = _resolve_optional_path(
        cfg.EVALUATION.get("plus_root", None),
        "LIBERO_PLUS_ROOT",
        DEFAULT_LIBERO_PLUS_ROOT,
    )
    plus_config_dir = _resolve_optional_path(
        cfg.EVALUATION.get("plus_config_dir", None),
        "FASTWAM_LIBERO_PLUS_CONFIG_DIR",
        DEFAULT_LIBERO_PLUS_CONFIG_DIR,
    )
    plus_config_dir = _isolate_plus_config_dir(plus_config_dir)
    resolved_plus_root, config_file = configure_libero_plus_runtime(plus_root, plus_config_dir)
    logging.info("Using LIBERO-Plus root: %s", resolved_plus_root)
    logging.info("Using LIBERO config file: %s", config_file)
    category = normalize_plus_category(
        cfg.EVALUATION.get("plus_category", os.environ.get("LIBERO_PLUS_CATEGORY"))
    )
    return resolved_plus_root, plus_config_dir, config_file, category


def _load_eval_components(cfg: DictConfig):
    from hydra.utils import instantiate

    from experiments.libero.eval_libero_single import (
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _resolve_dataset_stats_path,
        _resolve_eval_device,
    )
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)
    return model, processor, model_device, dataset_stats_path


def _build_plus_task_suite(
    cfg: DictConfig,
    *,
    suite_name: str,
    plus_root: Path,
    category: str | None,
) -> LiberoPlusBenchmark:
    task_classification_path = cfg.EVALUATION.get("plus_task_classification_path", None)
    task_suite = LiberoPlusBenchmark(
        suite_name,
        plus_root=plus_root,
        category=category,
        task_classification_path=task_classification_path,
    )
    logging.info(
        "LIBERO-Plus suite=%s category=%s tasks=%d",
        suite_name,
        category or "<all>",
        task_suite.n_tasks,
    )
    return task_suite


def _get_rollout_dimensions(cfg: DictConfig) -> tuple[int, int, int]:
    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    return action_horizon, input_w, input_h


def _run_one_plus_task(
    *,
    cfg: DictConfig,
    task_suite: LiberoPlusBenchmark,
    task_id: int,
    model,
    processor,
    model_device: str,
    action_horizon: int,
    input_w: int,
    input_h: int,
    category_filter: str | None,
    resolved_plus_root: Path,
):
    from experiments.libero.eval_libero_single import (
        NumpyEncoder,
        run_single_task,
    )

    task = task_suite.get_task(task_id)
    initial_states = _repeat_initial_states(
        task_suite.get_task_init_states(task_id),
        int(cfg.EVALUATION.num_trials),
    )

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    suite_output_name = task.problem_folder
    video_dir = local_log_dir / suite_output_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / suite_output_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    rollout_cfg = cfg.copy()
    rollout_cfg.EVALUATION.task_suite_name = suite_output_name
    rollout_cfg.EVALUATION.task_id = task_id

    results = {
        "task_suite": suite_output_name,
        "task_id": task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
        "plus_category": task.plus_category,
        "plus_category_filter": category_filter,
        "plus_category_short": short_plus_category(task.plus_category),
        "plus_task_name": task.name,
        "plus_original_id": task.plus_original_id,
        "plus_difficulty_level": task.plus_difficulty_level,
        "plus_root": str(resolved_plus_root),
    }

    logging.info(
        "Running LIBERO-Plus evaluation: suite=%s filtered_task_id=%d original_task_id=%d task=%s",
        suite_output_name,
        task_id,
        task.plus_original_id,
        task.name,
    )
    start_time = time.time()
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=rollout_cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)
    results["duration"] = time.time() - start_time

    output_dir = Path(cfg.EVALUATION.output_dir) / suite_output_name / short_plus_category(task.plus_category)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"LIBERO-Plus task {task_id} ({task.name}) completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


def run_plus_task_ids(
    cfg: DictConfig,
    *,
    task_ids: list[int],
) -> list[dict]:
    from accelerate import PartialState

    from experiments.libero.eval_libero_single import _validate_visualize_future_video_cfg
    from fastwam.utils.pytorch_utils import set_global_seed

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_plus.py. "
            "Use persistent multi-GPU worker parallelism instead of vectorized envs."
        )

    partial_state = PartialState()
    partial_state.config = cfg

    resolved_plus_root, _, _, category = _prepare_plus_runtime(cfg)
    model, processor, model_device, _ = _load_eval_components(cfg)
    action_horizon, input_w, input_h = _get_rollout_dimensions(cfg)
    task_suite = _build_plus_task_suite(
        cfg,
        suite_name=str(cfg.EVALUATION.task_suite_name),
        plus_root=resolved_plus_root,
        category=category,
    )

    results = []
    for task_id in task_ids:
        if task_id < 0 or task_id >= task_suite.n_tasks:
            raise IndexError(
                f"task_id={task_id} out of range for suite={cfg.EVALUATION.task_suite_name} "
                f"category={category or '<all>'}; valid range is [0, {task_suite.n_tasks - 1}]"
            )
        results.append(
            _run_one_plus_task(
                cfg=cfg,
                task_suite=task_suite,
                task_id=task_id,
                model=model,
                processor=processor,
                model_device=model_device,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                category_filter=category,
                resolved_plus_root=resolved_plus_root,
            )
        )
    return results


def run_plus_task_plan(
    cfg: DictConfig,
    task_specs: list[tuple[str, int]],
) -> list[dict]:
    from accelerate import PartialState

    from experiments.libero.eval_libero_single import _validate_visualize_future_video_cfg
    from fastwam.utils.pytorch_utils import set_global_seed

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_plus.py. "
            "Use persistent multi-GPU worker parallelism instead of vectorized envs."
        )

    partial_state = PartialState()
    partial_state.config = cfg

    resolved_plus_root, _, _, category = _prepare_plus_runtime(cfg)
    model, processor, model_device, _ = _load_eval_components(cfg)
    action_horizon, input_w, input_h = _get_rollout_dimensions(cfg)

    task_suite_cache: dict[str, LiberoPlusBenchmark] = {}
    outputs: list[dict] = []
    for suite_name, task_id in task_specs:
        task_suite = task_suite_cache.get(suite_name)
        if task_suite is None:
            task_suite = _build_plus_task_suite(
                cfg,
                suite_name=suite_name,
                plus_root=resolved_plus_root,
                category=category,
            )
            task_suite_cache[suite_name] = task_suite

        if task_id < 0 or task_id >= task_suite.n_tasks:
            raise IndexError(
                f"task_id={task_id} out of range for suite={suite_name} "
                f"category={category or '<all>'}; valid range is [0, {task_suite.n_tasks - 1}]"
            )

        outputs.append(
            _run_one_plus_task(
                cfg=cfg,
                task_suite=task_suite,
                task_id=task_id,
                model=model,
                processor=processor,
                model_device=model_device,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                category_filter=category,
                resolved_plus_root=resolved_plus_root,
            )
        )
    return outputs


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    task_plan_file = cfg.EVALUATION.get("task_plan_file", None)
    if task_plan_file not in (None, "", "null", "None"):
        plan_path = Path(os.path.expanduser(os.path.expandvars(str(task_plan_file))))
        task_specs = []
        for line in plan_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            suite_name, task_id = line.split(",", 1)
            task_specs.append((suite_name, int(task_id)))
        return run_plus_task_plan(cfg, task_specs)

    return run_plus_task_ids(cfg, task_ids=[int(cfg.EVALUATION.task_id)])


if __name__ == "__main__":
    eval_single_process()
