#!/usr/bin/env bash
set -euo pipefail

cd /data/zijianzhang/FastWAM

export HF_HOME="${HF_HOME:-/tmp/fastwam_hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/fastwam_hf_datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fastwam_mplconfig}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/fastwam/bin/python}"
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

NAMESPACE="gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1_initstride4_ep6057_idx0_47"
SUBSET="data/robotwin2.0/subsets/teacher_vis_ep6057_idx000000_000047.jsonl"
OUT_DIR="artifacts/robotwin_teacher_demo_ep6057_idx0_47_initstride4_20260713"
LOG="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"

{
  printf '[%s] precompute init_stride=4 continuous demo segment\n' "$(date '+%F %T')"
  "${PYTHON_BIN}" scripts/gaussianwam/precompute_teacher_cache.py \
    --config configs/gaussianwam/stage1_robotwin_fullclean_first50_firstframe_all.yaml \
    --overwrite \
    --override "device=cuda" \
    --override "source.subset_manifest=${SUBSET}" \
    --override "source.start_idx=0" \
    --override "source.end_idx=null" \
    --override "source.max_samples=null" \
    --override "gaussian.init_stride=4" \
    --override "cache.namespace=${NAMESPACE}"

  MANIFEST="/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/v4/${NAMESPACE}/train/manifest.jsonl"

  printf '[%s] visualize continuous demo segment\n' "$(date '+%F %T')"
  "${PYTHON_BIN}" scripts/gaussianwam/visualize_robotwin_teacher_cache.py \
    --manifest "${MANIFEST}" \
    --start-indices 0 \
    --frames 48 \
    --stride 1 \
    --include-feature-pca \
    --make-video \
    --fps 8 \
    --output-dir "${OUT_DIR}"

  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio

d = Path("artifacts/robotwin_teacher_demo_ep6057_idx0_47_initstride4_20260713")
pngs = [d / f"robotwin_teacher_idx{i:08d}.png" for i in range(48)]
frames = []
for p in pngs:
    img = Image.open(p).convert("RGB")
    w, h = img.size
    new_w = 1280
    new_h = int(round(h * new_w / w))
    new_h += new_h % 2
    frames.append(img.resize((new_w, new_h), Image.Resampling.LANCZOS))
out = d / "robotwin_teacher_demo_ep6057_idx0_47_initstride4_1280w_fps8.mp4"
imageio.mimsave(out, frames, fps=8, codec="libx264", quality=8, macro_block_size=1)
print(out)
PY

  printf '[%s] done\n' "$(date '+%F %T')"
} 2>&1 | tee -a "${LOG}"
