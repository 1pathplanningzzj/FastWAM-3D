#!/usr/bin/env bash
set -euo pipefail

cd /data/zijianzhang/FastWAM

OUT_DIR="/data/zijianzhang/FastWAM/evaluate_results/robotwin/robotwin_gaussianwam_stage2_focus3_fullft_current_3cam_384_1e-4_2026-06-08_20-11-33_gpus3-4-5-7_tmux/20260611_3tasks_step40000_gpu6"
CKPT="/data/zijianzhang/FastWAM/runs/robotwin_gaussianwam_stage2_focus3_fullft_current_3cam_384_1e-4/2026-06-08_20-11-33_gpus3-4-5-7_tmux/checkpoints/weights/step_040000.pt"
STATS="/data/zijianzhang/FastWAM/runs/robotwin_gaussianwam_stage2_focus3_fullft_current_3cam_384_1e-4/2026-06-08_20-11-33_gpus3-4-5-7_tmux/dataset_stats.json"
PYTHON_BIN="/data/miniconda3/envs/fastwam/bin/python"
LOG="$OUT_DIR/launch_step40000_gpu6.log"

mkdir -p "$OUT_DIR"

printf '[%s] start step_040000 eval on gpu=6\n' "$(date '+%F %T')" >> "$LOG"

"$PYTHON_BIN" experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt="$CKPT" \
  gpu_id=6 \
  EVALUATION.task_name=turn_switch \
  EVALUATION.task_config=demo_randomized \
  EVALUATION.eval_num_episodes=100 \
  EVALUATION.output_dir="$OUT_DIR" \
  EVALUATION.dataset_stats_path="$STATS" \
  >> "$LOG" 2>&1

"$PYTHON_BIN" experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt="$CKPT" \
  gpu_id=6 \
  EVALUATION.task_name=open_microwave \
  EVALUATION.task_config=demo_clean \
  EVALUATION.eval_num_episodes=100 \
  EVALUATION.output_dir="$OUT_DIR" \
  EVALUATION.dataset_stats_path="$STATS" \
  >> "$LOG" 2>&1

printf '[%s] finished step_040000 eval on gpu=6\n' "$(date '+%F %T')" >> "$LOG"
