#!/usr/bin/env bash
set -euo pipefail

cd /data/zijianzhang/FastWAM

export GAUSSIANWAM_ROOT="${GAUSSIANWAM_ROOT:-/data/zijianzhang/gaussianwam_data}"
export RUNS_ROOT="${RUNS_ROOT:-${GAUSSIANWAM_ROOT}/runs}"
export PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/fastwam-libero/bin/python}"
export ROBOTWIN_GPU_IDS="${ROBOTWIN_GPU_IDS:-1,2,3,4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
export PYTHONUNBUFFERED=1

RUN_NAME="robotwin_uncond_3cam_384_1e-4"
RUN_ID="2026-07-15_fastwam_uncond_clean13_first50_cleanstats_gpus0-5-6-7_globalbs64_gradacc2_save3k_max20k"
CKPT="${RUNS_ROOT}/${RUN_NAME}/${RUN_ID}/checkpoints/weights/step_012000.pt"
STATS="${RUNS_ROOT}/${RUN_NAME}/${RUN_ID}/dataset_stats.json"
TASKS_FILE="/data/zijianzhang/FastWAM/data/robotwin2.0/subsets/clean_13tasks_eval_tasks.txt"
OUT_DIR="/data/zijianzhang/FastWAM/evaluate_results/robotwin/${RUN_NAME}_${RUN_ID}/20260716_step012000_clean13_clean_random_gpus1-2-3-4"
LOG="${OUT_DIR}/launch.log"

mkdir -p "${OUT_DIR}"

{
  printf '[%s] start RoboTwin clean13 clean+random eval\n' "$(date '+%F %T')"
  printf 'ckpt=%s\n' "${CKPT}"
  printf 'stats=%s\n' "${STATS}"
  printf 'tasks_file=%s\n' "${TASKS_FILE}"
  printf 'ROBOTWIN_GPU_IDS=%s\n' "${ROBOTWIN_GPU_IDS}"

  "${PYTHON_BIN}" experiments/robotwin/run_robotwin_manager.py \
    task=robotwin_uncond_3cam_384_1e-4 \
    ckpt="${CKPT}" \
    EVALUATION.dataset_stats_path="${STATS}" \
    +EVALUATION.task_names_file="${TASKS_FILE}" \
    EVALUATION.output_dir="${OUT_DIR}" \
    MULTIRUN.num_gpus=4 \
    MULTIRUN.max_tasks_per_gpu=2

  printf '[%s] finished RoboTwin clean13 clean+random eval\n' "$(date '+%F %T')"
} 2>&1 | tee -a "${LOG}"
