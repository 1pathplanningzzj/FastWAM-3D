#!/usr/bin/env bash
set -euo pipefail

cd /data/zijianzhang/FastWAM

export PATH="/data/miniconda3/envs/fastwam/bin:${PATH}"
export GAUSSIANWAM_ROOT="${GAUSSIANWAM_ROOT:-/data/zijianzhang/gaussianwam_data}"
export RUNS_ROOT="${RUNS_ROOT:-${GAUSSIANWAM_ROOT}/runs}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${GAUSSIANWAM_ROOT}/checkpoints}"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MASTER_PORT="${MASTER_PORT:-29627}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

RUN_ID="2026-07-16_fastwam_uncond_clean15table_first50_cleanstats_gpus0-7_globalbs64_gradacc2_save3k_max20k_nostate_v2"
export RUN_ID

LOG="${GAUSSIANWAM_ROOT}/runs/robotwin_fastwam_uncond_clean15table_stats20k_gradacc2_gpus0_7_20260716_nostate_v2.log"

bash scripts/train_zero1.sh 8 \
  task=robotwin_uncond_3cam_384_1e-4 \
  data.train.episode_subset_manifest=data/robotwin2.0/subsets/clean_15tasks_table_first50.jsonl \
  data.val.episode_subset_manifest=data/robotwin2.0/subsets/clean_15tasks_table_first50.jsonl \
  data.train.pretrained_norm_stats=null \
  data.val.pretrained_norm_stats=null \
  batch_size=4 \
  gradient_accumulation_steps=2 \
  max_steps=20000 \
  save_every=3000 \
  eval_every=500 \
  resume=null \
  save_training_state=false \
  wandb.enabled=false \
  2>&1 | tee -a "${LOG}"
