# RoboTwin FastWAM New Server Reproduction

This note records what to copy and how to restart the current RoboTwin GaussianWAM workflow on a new server.

## 0. Assumptions

Default target layout:

```bash
export NEW_ROOT=/data/zijianzhang
export REPO=${NEW_ROOT}/FastWAM
export GAUSSIANWAM_ROOT=${NEW_ROOT}/gaussianwam_data
export ROBOTWIN_DATA_ROOT=${GAUSSIANWAM_ROOT}/data/robotwin2.0
```

If the new server uses a different base path, keep these environment variables set and avoid editing configs where possible.

## 1. Copy From Old Server

The current RoboTwin GaussianWAM code depends on local uncommitted files, so copy the working tree, not only a clean git clone.

Required for training from existing teacher cache:

```text
/data/zijianzhang/FastWAM
/data/zijianzhang/gaussianwam_data/data/robotwin2.0/robotwin2.0
/data/zijianzhang/gaussianwam_data/data/robotwin2.0/dataset_stats.json
/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/v4/gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1
/data/zijianzhang/gaussianwam_data/checkpoints
```

Required if you want to re-brush RoboTwin teacher data:

```text
/data/zijianzhang/VGGT-Omega/vggt_omega_1b_256_text.pt
/data/zijianzhang/clip-vit-base-patch16
```

Optional but useful:

```text
/data/zijianzhang/gaussianwam_data/data/text_embeds_cache/robotwin
/data/zijianzhang/gaussianwam_data/runs/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4/<run_id>
```

Notes:

- `text_embeds_cache/robotwin` is large and can be regenerated, but copying it avoids first-run text encoder work.
- The current `full_clean_heuristic_first50.jsonl` subset lives inside the repo at `FastWAM/data/robotwin2.0/subsets`, so copying the working tree is enough for that file.
- Copying an old `runs/.../<run_id>` is only needed for resume/eval of existing checkpoints.
- Current important teacher cache is about `349G`; raw RoboTwin dataset is about `75G`; `checkpoints` is about `96G`.
- `FastWAM/evaluate_results`, `FastWAM/artifacts`, `FastWAM/.hf_cache`, and `FastWAM/data/lingbot_va` are not needed for RoboTwin reproduction unless you want old logs/videos/caches.

Example rsync commands from the old server:

```bash
rsync -aH --info=progress2 \
  --exclude '.hf_cache/' \
  --exclude 'artifacts/' \
  --exclude 'evaluate_results/' \
  --exclude 'data/lingbot_va/' \
  /data/zijianzhang/FastWAM NEW_HOST:/data/zijianzhang/

rsync -aH --info=progress2 /data/zijianzhang/gaussianwam_data/data/robotwin2.0/robotwin2.0 NEW_HOST:/data/zijianzhang/gaussianwam_data/data/robotwin2.0/
rsync -aH --info=progress2 /data/zijianzhang/gaussianwam_data/data/robotwin2.0/dataset_stats.json NEW_HOST:/data/zijianzhang/gaussianwam_data/data/robotwin2.0/
rsync -aH --info=progress2 /data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/v4/gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1 NEW_HOST:/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/v4/
rsync -aH --info=progress2 /data/zijianzhang/gaussianwam_data/checkpoints NEW_HOST:/data/zijianzhang/gaussianwam_data/

rsync -aH --info=progress2 /data/zijianzhang/VGGT-Omega NEW_HOST:/data/zijianzhang/
rsync -aH --info=progress2 /data/zijianzhang/clip-vit-base-patch16 NEW_HOST:/data/zijianzhang/
```

## 2. Create Environment

FastWAM training environment:

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
cd /data/zijianzhang/FastWAM
pip install -e .
```

Evaluation uses the RoboTwin simulator. On this machine it uses a separate env:

```bash
conda create -n fastwam-libero python=3.10 -y
conda activate fastwam-libero
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
cd /data/zijianzhang/FastWAM
pip install -e .
```

Then install RoboTwin dependencies/assets following `third_party/RoboTwin/README.md` or upstream RoboTwin 2.0 docs. Keep the local FastWAM policy link:

```bash
cd /data/zijianzhang/FastWAM
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

## 3. Export Common Variables

Put this in the shell or tmux before training/eval/teacher generation:

```bash
export GAUSSIANWAM_ROOT=/data/zijianzhang/gaussianwam_data
export RUNS_ROOT=${GAUSSIANWAM_ROOT}/runs
export ROBOTWIN_DATA_ROOT=${GAUSSIANWAM_ROOT}/data/robotwin2.0
export DIFFSYNTH_MODEL_BASE_PATH=${GAUSSIANWAM_ROOT}/checkpoints
export HF_HOME=/data/zijianzhang/FastWAM/.hf_cache
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The current teacher config has two absolute paths:

```yaml
vggt_omega.checkpoint_path: /data/zijianzhang/VGGT-Omega/vggt_omega_1b_256_text.pt
clip.model_name: /data/zijianzhang/clip-vit-base-patch16
```

If those are copied elsewhere, pass overrides when brushing teacher:

```bash
--override "vggt_omega.checkpoint_path=/new/path/vggt_omega_1b_256_text.pt" \
--override "clip.model_name=/new/path/clip-vit-base-patch16"
```

## 4. Smoke Checks

```bash
cd /data/zijianzhang/FastWAM
conda activate fastwam

python - <<'PY'
from pathlib import Path
root = Path("/data/zijianzhang/gaussianwam_data")
repo = Path("/data/zijianzhang/FastWAM")
checks = [
    root/"data/robotwin2.0/robotwin2.0/meta/tasks.jsonl",
    root/"data/robotwin2.0/dataset_stats.json",
    repo/"data/robotwin2.0/subsets/full_clean_heuristic_first50.jsonl",
    root/"data/robotwin2.0/gaussian_teacher_cache/v4/gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1/train/manifest.jsonl",
    root/"checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt",
    Path("/data/zijianzhang/VGGT-Omega/vggt_omega_1b_256_text.pt"),
    Path("/data/zijianzhang/clip-vit-base-patch16/model.safetensors"),
]
for p in checks:
    print(("OK   " if p.exists() else "MISS "), p)
PY
```

Teacher dry run:

```bash
PYTHON_BIN=/data/miniconda3/envs/fastwam/bin/python \
CONFIG=configs/gaussianwam/stage1_robotwin_fullclean_first50_firstframe_all.yaml \
NUM_WORKERS=1 GPU_LIST=0 START_IDX=0 END_IDX=2 \
bash scripts/gaussianwam/run_robotwin_firstframe_cache_8gpu.sh --dry-run
```

Training config resolve check:

```bash
python scripts/train.py \
  task=robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4 \
  max_steps=1 \
  save_every=1000 \
  eval_every=1000 \
  batch_size=1 \
  gradient_accumulation_steps=1 \
  num_workers=0
```

Use the command above only as a short smoke test; stop after it starts successfully if you do not want to create a run.

## 5. Brush RoboTwin Teacher Cache

Current training uses the full-clean first-50 first-frame cache:

```bash
configs/gaussianwam/stage1_robotwin_fullclean_first50_firstframe_all.yaml
```

Run on 8 GPUs:

```bash
cd /data/zijianzhang/FastWAM
conda activate fastwam

tmux new-session -s robotwin_teacher_fullclean_first50_v4 \
  'PYTHON_BIN=/data/miniconda3/envs/fastwam/bin/python \
   CONFIG=configs/gaussianwam/stage1_robotwin_fullclean_first50_firstframe_all.yaml \
   NUM_WORKERS=8 GPU_LIST=0,1,2,3,4,5,6,7 \
   bash scripts/gaussianwam/run_robotwin_firstframe_cache_8gpu.sh'
```

Useful overrides:

```bash
# only brush a shard
START_IDX=0 END_IDX=10000 ...

# change gaussian initialization point stride from default 8 to 4
bash scripts/gaussianwam/run_robotwin_firstframe_cache_8gpu.sh \
  --override "gaussian.init_stride=4" \
  --override "cache.namespace=gaussian_vggt256text_3d_fullclean_first50_firstframe_all_initstride4_v1"
```

Generated manifest expected by the current training config:

```text
/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/v4/gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1/train/manifest.jsonl
```

## 6. Train RoboTwin GaussianWAM

Current run recipe:

```bash
cd /data/zijianzhang/FastWAM
conda activate fastwam

export CUDA_VISIBLE_DEVICES=0,1,2,3
export RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_gpus0-1-2-3_globalbs1024_gradacc32_save1k_max15k"

tmux new-session -s robotwin_gaussianwam_train_${RUN_ID} \
  "cd /data/zijianzhang/FastWAM && \
   export GAUSSIANWAM_ROOT=/data/zijianzhang/gaussianwam_data && \
   export ROBOTWIN_DATA_ROOT=/data/zijianzhang/gaussianwam_data/data/robotwin2.0 && \
   export DIFFSYNTH_MODEL_BASE_PATH=/data/zijianzhang/gaussianwam_data/checkpoints && \
   export HF_HOME=/data/zijianzhang/FastWAM/.hf_cache && \
   export HF_DATASETS_CACHE=/data/zijianzhang/FastWAM/.hf_cache/datasets && \
   export HUGGINGFACE_HUB_CACHE=/data/zijianzhang/FastWAM/.hf_cache/hub && \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   export RUN_ID=${RUN_ID} && \
   bash scripts/train_zero1.sh 4 \
     task=robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4 \
     resume=null \
     batch_size=8 \
     gradient_accumulation_steps=32 \
     save_every=1000 \
     eval_every=500 \
     max_steps=15000 \
     2>&1 | tee /data/zijianzhang/gaussianwam_data/runs/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4/${RUN_ID}/launch.log"
```

Output:

```text
${GAUSSIANWAM_ROOT}/runs/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4/${RUN_ID}
```

Checkpoints:

```text
.../checkpoints/weights/step_001000.pt
.../checkpoints/weights/step_002000.pt
.../checkpoints/weights/step_003000.pt
```

## 7. Evaluate A Checkpoint

Template:

```bash
cd /data/zijianzhang/FastWAM
conda activate fastwam-libero

export GAUSSIANWAM_ROOT=/data/zijianzhang/gaussianwam_data
export RUNS_ROOT=${GAUSSIANWAM_ROOT}/runs
export PYTHON_BIN=/data/miniconda3/envs/fastwam-libero/bin/python
export ROBOTWIN_GPU_IDS=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1

RUN_NAME=robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4
RUN_ID=<your_train_run_id>
STEP=003000
CKPT=${RUNS_ROOT}/${RUN_NAME}/${RUN_ID}/checkpoints/weights/step_${STEP}.pt
STATS=${RUNS_ROOT}/${RUN_NAME}/${RUN_ID}/dataset_stats.json
OUT_DIR=/data/zijianzhang/FastWAM/evaluate_results/robotwin/${RUN_NAME}_${RUN_ID}/$(date +%Y%m%d)_full_step${STEP}_gpus0-1-2-3

tmux new-session -s robotwin_full_step${STEP}_gpus0_1_2_3 \
  "${PYTHON_BIN} experiments/robotwin/run_robotwin_manager.py \
     task=robotwin_uncond_3cam_384_1e-4 \
     ckpt=${CKPT} \
     EVALUATION.dataset_stats_path=${STATS} \
     EVALUATION.output_dir=${OUT_DIR} \
     MULTIRUN.num_gpus=4 \
     MULTIRUN.max_tasks_per_gpu=2 \
     2>&1 | tee -a ${OUT_DIR}/launch.log"
```

Monitor:

```bash
tail -f ${OUT_DIR}/manager.log
find ${OUT_DIR} -maxdepth 2 -name '_result_clean.txt' -o -name '_result_random.txt'
```

When complete, look for:

```text
${OUT_DIR}/summary.csv
${OUT_DIR}/summary.json
```

## 8. Visualize Teacher Cache

Example:

```bash
cd /data/zijianzhang/FastWAM
conda activate fastwam

python scripts/gaussianwam/visualize_robotwin_teacher_cache.py \
  --manifest /data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/v4/gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1/train/manifest.jsonl \
  --output-dir artifacts/robotwin_teacher_cache_sample_feature \
  --num-samples 8 \
  --include-feature-pca
```

For a continuous demo-style clip, use the existing helper scripts:

```bash
bash scripts/gaussianwam/run_teacher_vis_12cases_init_stride4.sh
bash scripts/gaussianwam/run_teacher_ep6057_init_stride4_video.sh
```

## 9. Common Failure Points

- Missing `full_clean_heuristic_first50.jsonl`: copy the current `FastWAM` working tree, including `FastWAM/data/robotwin2.0/subsets`.
- Missing teacher manifest: either copy the `gaussian_teacher_cache/v4/...fullclean_first50...` directory or run the teacher brushing step.
- Missing VGGT/CLIP paths: copy `/data/zijianzhang/VGGT-Omega` and `/data/zijianzhang/clip-vit-base-patch16`, or pass config overrides.
- Evaluation starts but simulator fails: finish RoboTwin 2.0 environment/assets installation under `third_party/RoboTwin`.
- `ROBOTWIN_GPU_IDS` length must equal `MULTIRUN.num_gpus`.
- If using `CUDA_VISIBLE_DEVICES`, this code still passes physical `gpu_id` values to RoboTwin workers, so keep `ROBOTWIN_GPU_IDS` aligned with the visible/physical GPU choice used on that server.
