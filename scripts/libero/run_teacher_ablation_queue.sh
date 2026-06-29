#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/zijianzhang/FastWAM}"
CONDA_BIN="${CONDA_BIN:-/data/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-fastwam}"
GPU_LIST="${GPU_LIST:-0,5,6,7}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/data/zijianzhang/libero_mujoco3.3.2}"
TEXT_CACHE_ROOT="${TEXT_CACHE_ROOT:-/data/zijianzhang/gaussianwam_data/data/text_embeds_cache/libero}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-2}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
EVAL_EVERY="${EVAL_EVERY:-500}"

TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
  TASKS=(
    "libero_gaussianwam_stage2_fullft_firstframe_2cam224_no_dense3d_1e-4"
    "libero_gaussianwam_stage2_fullft_firstframe_2cam224_no_depth_1e-4"
    "libero_gaussianwam_stage2_fullft_firstframe_2cam224_no_alpha_1e-4"
  )
fi

cd "${ROOT_DIR}"

for TASK in "${TASKS[@]}"; do
  RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_gpus${GPU_LIST//,/-}_bs${BATCH_SIZE}_tmux"
  LOG="/data/zijianzhang/gaussianwam_data/runs/${TASK}/${RUN_ID}/launch.log"
  mkdir -p "$(dirname "${LOG}")"

  echo "[ablation] task=${TASK}"
  echo "[ablation] run_id=${RUN_ID}"
  echo "[ablation] log=${LOG}"

  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  RUN_ID="${RUN_ID}" \
  LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
  bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
    "task=${TASK}" \
    "batch_size=${BATCH_SIZE}" \
    "gradient_accumulation_steps=${GRAD_ACC}" \
    "save_every=${SAVE_EVERY}" \
    "eval_every=${EVAL_EVERY}" \
    "save_training_state=false" \
    "data.libero_text_cache_root=${TEXT_CACHE_ROOT}" 2>&1 | tee "${LOG}"
done
