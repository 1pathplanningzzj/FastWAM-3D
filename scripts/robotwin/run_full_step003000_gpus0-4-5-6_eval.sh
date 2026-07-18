#!/usr/bin/env bash
set -euo pipefail

cd /data/zijianzhang/FastWAM

export GAUSSIANWAM_ROOT="${GAUSSIANWAM_ROOT:-/data/zijianzhang/gaussianwam_data}"
export RUNS_ROOT="${RUNS_ROOT:-${GAUSSIANWAM_ROOT}/runs}"
export PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/fastwam-libero/bin/python}"
export ROBOTWIN_GPU_IDS="${ROBOTWIN_GPU_IDS:-0,4,5,6}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,4,5,6}"
export PYTHONUNBUFFERED=1

RUN_NAME="robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4"
RUN_ID="2026-07-11_21-01-56_no_release_gpus0-5-6-7_globalbs1024_gradacc32_save1k_max15k_tmux"
CKPT="${RUNS_ROOT}/${RUN_NAME}/${RUN_ID}/checkpoints/weights/step_003000.pt"
STATS="${RUNS_ROOT}/${RUN_NAME}/${RUN_ID}/dataset_stats.json"
OUT_DIR="/data/zijianzhang/FastWAM/evaluate_results/robotwin/${RUN_NAME}_${RUN_ID}/20260713_full_step003000_gpus0-4-5-6"
LOG="${OUT_DIR}/launch.log"

mkdir -p "${OUT_DIR}"

{
  printf '[%s] start RoboTwin full eval\n' "$(date '+%F %T')"
  printf 'ckpt=%s\n' "${CKPT}"
  printf 'stats=%s\n' "${STATS}"
  printf 'ROBOTWIN_GPU_IDS=%s\n' "${ROBOTWIN_GPU_IDS}"

  "${PYTHON_BIN}" experiments/robotwin/run_robotwin_manager.py \
    task=robotwin_uncond_3cam_384_1e-4 \
    ckpt="${CKPT}" \
    EVALUATION.dataset_stats_path="${STATS}" \
    EVALUATION.output_dir="${OUT_DIR}" \
    MULTIRUN.num_gpus=4 \
    MULTIRUN.max_tasks_per_gpu=2

  printf '[%s] finished RoboTwin full eval\n' "$(date '+%F %T')"
} 2>&1 | tee -a "${LOG}"
