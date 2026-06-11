#!/usr/bin/env bash
set -euo pipefail

cd /data/zijianzhang/FastWAM

TOTAL=535116
CHUNK=$(( (TOTAL + 7) / 8 ))
LOG_DIR="/data/zijianzhang/FastWAM/data/robotwin2.0/gaussian_teacher_cache/logs/gaussian_vggt256text_3d_focus3_firstframe_all_v3_clean_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "TOTAL=$TOTAL CHUNK=$CHUNK LOG_DIR=$LOG_DIR" | tee "$LOG_DIR/manager.log"

for i in 0 1 2 3 4 5 6 7; do
  START=$(( i * CHUNK ))
  if [ "$START" -ge "$TOTAL" ]; then
    break
  fi
  END=$(( START + CHUNK ))
  if [ "$END" -gt "$TOTAL" ]; then
    END=$TOTAL
  fi
  GPU=$i
  LOG="$LOG_DIR/worker_${i}.log"
  echo "[$(date '+%F %T')] launch worker=$i gpu=$GPU start=$START end=$END" | tee -a "$LOG_DIR/manager.log"
  CUDA_VISIBLE_DEVICES=$GPU /data/miniconda3/envs/fastwam/bin/python scripts/gaussianwam/precompute_teacher_cache.py \
    --config configs/gaussianwam/stage1_robotwin_focus3_firstframe_all.yaml \
    --override source.start_idx=$START \
    --override source.end_idx=$END \
    > "$LOG" 2>&1 &
done

wait
echo "[$(date '+%F %T')] all workers finished" | tee -a "$LOG_DIR/manager.log"
