#!/usr/bin/env bash
set -euo pipefail

cd /data/zijianzhang/FastWAM

export PATH="/data/miniconda3/envs/fastwam/bin:${PATH}"
export PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/fastwam-libero/bin/python}"
export GAUSSIANWAM_ROOT="${GAUSSIANWAM_ROOT:-/data/zijianzhang/gaussianwam_data}"
export RUNS_ROOT="${RUNS_ROOT:-${GAUSSIANWAM_ROOT}/runs}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${GAUSSIANWAM_ROOT}/checkpoints}"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

TASK_CONFIG="robotwin_uncond_3cam_384_1e-4"
SUBSET_DIR="/data/zijianzhang/FastWAM/data/robotwin2.0/subsets/clean_15tasks_table_per_task"
EVAL_ROOT="/data/zijianzhang/FastWAM/evaluate_results/robotwin_per_task_clean15_10epoch"
PIPELINE_LOG="${GAUSSIANWAM_ROOT}/runs/robotwin_clean15_per_task_pipeline_20260716.log"
GLOBAL_BATCH=64
EPOCHS=10
BATCH_SIZE=4
GRAD_ACC=2
NPROC=8

TASKS=(
  adjust_bottle
  beat_block_hammer
  blocks_ranking_rgb
  blocks_ranking_size
  click_alarmclock
  click_bell
  grab_roller
  move_playingcard_away
  pick_diverse_bottles
  pick_dual_bottles
  place_a2b_left
  place_a2b_right
  place_container_plate
  place_fan
  put_bottles_dustbin
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${PIPELINE_LOG}"
}

max_steps_for_task() {
  local task="$1"
  python - "$task" "$GLOBAL_BATCH" "$EPOCHS" <<'PY'
import json, math, sys
from pathlib import Path
import numpy as np

task = sys.argv[1]
global_batch = int(sys.argv[2])
epochs = int(sys.argv[3])
subset = Path(f'/data/zijianzhang/FastWAM/data/robotwin2.0/subsets/clean_15tasks_table_per_task/{task}_first50.jsonl')
episodes_meta = Path('/data/zijianzhang/gaussianwam_data/data/robotwin2.0/robotwin2.0/meta/episodes.jsonl')
lengths = {}
with episodes_meta.open('r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            lengths[int(row['episode_index'])] = int(row['length'])
eps = []
with subset.open('r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            eps.append(int(json.loads(line)['episode_index']))
rng = np.random.default_rng(42)
rng.shuffle(eps)
split = int(len(eps) * 0.99)
train_eps = eps[:split]
train_size = sum(lengths[i] for i in train_eps)
max_steps = math.ceil(train_size / global_batch) * epochs
print(max_steps)
PY
}

run_eval_pair() {
  local task="$1"
  local ckpt="$2"
  local stats="$3"
  local out_name="$4"
  local eval_dir="${EVAL_ROOT}/${task}/${out_name}"
  mkdir -p "${eval_dir}"
  log "eval start task=${task} ckpt=${ckpt} out=${eval_dir}"

  # RoboTwin writes episode*.mp4 under evaluate_results/robotwin/<ckpt_tag>/<output_dir_basename>/<task>.
  # Use different basenames for clean/random so two parallel evals do not race on episode0.mp4.
  "${PYTHON_BIN}" experiments/robotwin/eval_robotwin_single.py \
    task="${TASK_CONFIG}" \
    ckpt="${ckpt}" \
    gpu_id=0 \
    EVALUATION.task_name="${task}" \
    EVALUATION.task_config=demo_clean \
    EVALUATION.eval_num_episodes=100 \
    EVALUATION.dataset_stats_path="${stats}" \
    EVALUATION.output_dir="${eval_dir}/${out_name}_clean_100" \
    > "${eval_dir}/eval_clean.log" 2>&1 &
  local clean_pid=$!

  "${PYTHON_BIN}" experiments/robotwin/eval_robotwin_single.py \
    task="${TASK_CONFIG}" \
    ckpt="${ckpt}" \
    gpu_id=1 \
    EVALUATION.task_name="${task}" \
    EVALUATION.task_config=demo_randomized \
    EVALUATION.eval_num_episodes=100 \
    EVALUATION.dataset_stats_path="${stats}" \
    EVALUATION.output_dir="${eval_dir}/${out_name}_random_100" \
    > "${eval_dir}/eval_random.log" 2>&1 &
  local random_pid=$!

  wait "${clean_pid}"
  wait "${random_pid}"
  log "eval done task=${task} out=${eval_dir}"
}

mkdir -p "${EVAL_ROOT}"
python scripts/robotwin/build_clean15_per_task_subsets.py | tee -a "${PIPELINE_LOG}"

start_from="${START_FROM:-}"
skipping=0
if [[ -n "${start_from}" ]]; then
  skipping=1
fi

for task in "${TASKS[@]}"; do
  if [[ ${skipping} -eq 1 ]]; then
    if [[ "${task}" == "${start_from}" ]]; then
      skipping=0
    else
      log "skip before START_FROM task=${task}"
      continue
    fi
  fi

  subset="${SUBSET_DIR}/${task}_first50.jsonl"
  max_steps="$(max_steps_for_task "${task}")"
  run_id="2026-07-16_fastwam_uncond_${task}_first50_cleanstats_8gpu_globalbs64_10epoch_nostate"
  run_dir="${RUNS_ROOT}/${TASK_CONFIG}/${run_id}"
  train_log="${GAUSSIANWAM_ROOT}/runs/robotwin_per_task_${task}_10epoch_8gpu_20260716.log"
  final_ckpt="${run_dir}/checkpoints/weights/step_$(printf '%06d' "${max_steps}").pt"
  stats="${run_dir}/dataset_stats.json"

  if [[ -f "${final_ckpt}" && -f "${stats}" ]]; then
    log "train skip existing task=${task} ckpt=${final_ckpt}"
  else
    log "train start task=${task} max_steps=${max_steps} run_id=${run_id} subset=${subset}"
    export RUN_ID="${run_id}"
    export MASTER_PORT="$((29700 + (${RANDOM} % 400)))"

    bash scripts/train_zero1.sh "${NPROC}" \
      task="${TASK_CONFIG}" \
      data.train.episode_subset_manifest="${subset}" \
      data.val.episode_subset_manifest="${subset}" \
      data.train.pretrained_norm_stats=null \
      data.val.pretrained_norm_stats=null \
      batch_size="${BATCH_SIZE}" \
      gradient_accumulation_steps="${GRAD_ACC}" \
      max_steps="${max_steps}" \
      save_every=0 \
      eval_every=500 \
      resume=null \
      save_training_state=false \
      wandb.enabled=false \
      2>&1 | tee -a "${train_log}"
  fi

  if [[ -d "${run_dir}" && ! -e "${run_dir}/launch.log" ]]; then
    ln -s "${train_log}" "${run_dir}/launch.log"
  fi

  if [[ ! -f "${final_ckpt}" ]]; then
    log "ERROR final ckpt missing task=${task} ckpt=${final_ckpt}"
    exit 1
  fi
  if [[ ! -f "${stats}" ]]; then
    log "ERROR stats missing task=${task} stats=${stats}"
    exit 1
  fi

  log "train done task=${task} ckpt=${final_ckpt}"
  run_eval_pair "${task}" "${final_ckpt}" "${stats}" "${run_id}"
done

log "all 15 per-task train/eval jobs finished"
