#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/gaussianwam/stage1_libero.yaml}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"
START_IDX="${START_IDX:-0}"

export REPO_ROOT
export CONFIG
export GAUSSIANWAM_ROOT="${GAUSSIANWAM_ROOT:-/data/zijianzhang/gaussianwam_data}"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${GAUSSIANWAM_ROOT}/data/libero_mujoco3.3.2}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

readarray -t CFG_INFO < <(
  "${PYTHON_BIN}" - <<'PY'
import os
import sys
repo_root = os.environ["REPO_ROOT"]
sys.path.insert(0, os.path.join(repo_root, "src"))
from gaussianwam.cache import cache_root
from gaussianwam.config import load_config
from gaussianwam.data import build_raw_dataset

cfg = load_config(os.environ["CONFIG"])
dataset = build_raw_dataset(cfg.source)
root = cache_root(cfg.output_dir, int(cfg.cache.version), str(cfg.cache.namespace), str(cfg.split))
print(len(dataset))
print(root)
print(cfg.cache.namespace)
PY
)

TOTAL="${TOTAL:-${CFG_INFO[0]}}"
END_IDX="${END_IDX:-${TOTAL}}"
CACHE_ROOT="${CFG_INFO[1]}"
CACHE_NAMESPACE="${CFG_INFO[2]}"

if (( END_IDX > TOTAL )); then
  END_IDX="${TOTAL}"
fi
if (( START_IDX < 0 || START_IDX >= END_IDX )); then
  echo "Invalid shard bounds: START_IDX=${START_IDX} END_IDX=${END_IDX} TOTAL=${TOTAL}" >&2
  exit 1
fi

TOTAL_WORK=$(( END_IDX - START_IDX ))
CHUNK=$(( (TOTAL_WORK + NUM_WORKERS - 1) / NUM_WORKERS ))
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${GAUSSIANWAM_ROOT}/data/libero_teacher_cache/logs/${CACHE_NAMESPACE}_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

echo "CONFIG=${CONFIG}" | tee "${LOG_DIR}/manager.log"
echo "CACHE_ROOT=${CACHE_ROOT}" | tee -a "${LOG_DIR}/manager.log"
echo "TOTAL=${TOTAL} START_IDX=${START_IDX} END_IDX=${END_IDX} TOTAL_WORK=${TOTAL_WORK} CHUNK=${CHUNK}" | tee -a "${LOG_DIR}/manager.log"
echo "LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT}" | tee -a "${LOG_DIR}/manager.log"
echo "HF_HOME=${HF_HOME}" | tee -a "${LOG_DIR}/manager.log"

PIDS=()
WORKERS=()

for (( i = 0; i < NUM_WORKERS; i++ )); do
  SHARD_START=$(( START_IDX + i * CHUNK ))
  if (( SHARD_START >= END_IDX )); then
    break
  fi
  SHARD_END=$(( SHARD_START + CHUNK ))
  if (( SHARD_END > END_IDX )); then
    SHARD_END="${END_IDX}"
  fi
  GPU=$(( GPU_OFFSET + i ))
  LOG="${LOG_DIR}/worker_${i}_gpu${GPU}.log"
  echo "[$(date '+%F %T')] launch worker=${i} gpu=${GPU} start=${SHARD_START} end=${SHARD_END}" | tee -a "${LOG_DIR}/manager.log"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/gaussianwam/precompute_teacher_cache.py \
    --config "${CONFIG}" \
    --override "source.start_idx=${SHARD_START}" \
    --override "source.end_idx=${SHARD_END}" \
    "$@" \
    > "${LOG}" 2>&1 &
  PIDS+=("$!")
  WORKERS+=("${i}")
done

FAIL=0
for idx in "${!PIDS[@]}"; do
  PID="${PIDS[idx]}"
  WORKER="${WORKERS[idx]}"
  if wait "${PID}"; then
    echo "[$(date '+%F %T')] worker=${WORKER} pid=${PID} finished" | tee -a "${LOG_DIR}/manager.log"
  else
    echo "[$(date '+%F %T')] worker=${WORKER} pid=${PID} failed" | tee -a "${LOG_DIR}/manager.log"
    FAIL=1
  fi
done

if (( FAIL != 0 )); then
  echo "[$(date '+%F %T')] one or more workers failed" | tee -a "${LOG_DIR}/manager.log"
  exit 1
fi

echo "[$(date '+%F %T')] all workers finished" | tee -a "${LOG_DIR}/manager.log"
