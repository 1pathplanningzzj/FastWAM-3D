# LIBERO migration notes

This note documents how to move the LIBERO teacher-cache workflow to
`115.190.238.113:/home/zijianzhang/FastWAM` without removing the copy on the
current machine.

## What to copy

The current LIBERO migration bundle includes:

- `README.md`
- `README_zh.md`
- `Train_Log.md`
- `configs/data/libero_2cam.yaml`
- `configs/data/libero_gaussianwam.yaml`
- `configs/gaussianwam/stage1_libero.yaml`
- `configs/task/libero_gaussianwam_stage2_current_2cam224_1e-4.yaml`
- `scripts/precompute_text_embeds.py`
- `scripts/gaussianwam/precompute_teacher_cache.py`
- `scripts/gaussianwam/run_libero_firstframe_cache_8gpu.sh`
- `src/gaussianwam/__init__.py`
- `src/gaussianwam/cache.py`
- `src/gaussianwam/clip_teacher.py`
- `src/gaussianwam/compact_feature.py`
- `src/gaussianwam/config.py`
- `src/gaussianwam/data.py`
- `src/gaussianwam/debug.py`
- `src/gaussianwam/fitting.py`
- `src/gaussianwam/gaussian_field.py`
- `src/gaussianwam/geometry.py`
- `src/gaussianwam/precompute.py`
- `src/gaussianwam/renderer.py`
- `src/gaussianwam/validate.py`
- `src/gaussianwam/vggt_omega.py`
- `docs/libero_migration.md`

For the teacher-cache workflow, it is safest to sync the full
`src/gaussianwam/` package because `precompute.py` imports several sibling
modules at runtime.

## Transfer options

### Option 1: rsync files directly

Run this on the source machine:

```bash
cd /data/zijianzhang/FastWAM
rsync -avR -e "ssh -o StrictHostKeyChecking=accept-new" \
  README.md \
  README_zh.md \
  Train_Log.md \
  configs/data/libero_2cam.yaml \
  configs/data/libero_gaussianwam.yaml \
  configs/gaussianwam/stage1_libero.yaml \
  configs/task/libero_gaussianwam_stage2_current_2cam224_1e-4.yaml \
  scripts/precompute_text_embeds.py \
  scripts/gaussianwam/precompute_teacher_cache.py \
  scripts/gaussianwam/run_libero_firstframe_cache_8gpu.sh \
  src/gaussianwam/*.py \
  docs/libero_migration.md \
  zijianzhang@115.190.238.113:/home/zijianzhang/FastWAM/
```

### Option 2: ship a tarball

Create a tarball on the source machine:

```bash
cd /data/zijianzhang/FastWAM
tar czf /tmp/fastwam_libero_migration_20260612.tgz \
  README.md \
  README_zh.md \
  Train_Log.md \
  configs/data/libero_2cam.yaml \
  configs/data/libero_gaussianwam.yaml \
  configs/gaussianwam/stage1_libero.yaml \
  configs/task/libero_gaussianwam_stage2_current_2cam224_1e-4.yaml \
  scripts/precompute_text_embeds.py \
  scripts/gaussianwam/precompute_teacher_cache.py \
  scripts/gaussianwam/run_libero_firstframe_cache_8gpu.sh \
  src/gaussianwam/*.py \
  docs/libero_migration.md
```

Upload and unpack it on the remote machine:

```bash
scp /tmp/fastwam_libero_migration_20260612.tgz \
  zijianzhang@115.190.238.113:/home/zijianzhang/

ssh zijianzhang@115.190.238.113
cd /home/zijianzhang/FastWAM
tar xzf /home/zijianzhang/fastwam_libero_migration_20260612.tgz
```

## Remote prerequisites

Prepare the following items on `115.190.238.113`:

- A Python environment that can run FastWAM and LIBERO dependencies.
- LIBERO dataset under `/home/zijianzhang/FastWAM/data/libero_mujoco3.3.2`
  or another path exported through `LIBERO_DATA_ROOT`.
- A local VGGT-Omega checkpoint.
- A local CLIP model directory.
- The Stage 2 resume checkpoint if you also want to train with
  `configs/task/libero_gaussianwam_stage2_current_2cam224_1e-4.yaml`.

The Stage 1 config now supports environment overrides for model paths:

- `VGGT_OMEGA_CKPT`
- `CLIP_MODEL_PATH`

## Recommended remote launch flow

After the files are in place:

```bash
cd /home/zijianzhang/FastWAM
conda activate fastwam

export PYTHON_BIN=python
export LIBERO_DATA_ROOT=/home/zijianzhang/FastWAM/data/libero_mujoco3.3.2
export VGGT_OMEGA_CKPT=/path/to/vggt_omega_1b_256_text.pt
export CLIP_MODEL_PATH=/path/to/clip-vit-base-patch16
export HF_HOME=/home/zijianzhang/FastWAM/.hf_cache
export HF_DATASETS_CACHE=/home/zijianzhang/FastWAM/.hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=/home/zijianzhang/FastWAM/.hf_cache/hub

python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
bash scripts/gaussianwam/run_libero_firstframe_cache_8gpu.sh
```

## tmux launch

```bash
cd /home/zijianzhang/FastWAM
conda activate fastwam

tmux new-session -d -s libero_cache \
  'cd /home/zijianzhang/FastWAM && \
   export PYTHON_BIN=python && \
   export LIBERO_DATA_ROOT=/home/zijianzhang/FastWAM/data/libero_mujoco3.3.2 && \
   export VGGT_OMEGA_CKPT=/path/to/vggt_omega_1b_256_text.pt && \
   export CLIP_MODEL_PATH=/path/to/clip-vit-base-patch16 && \
   export HF_HOME=/home/zijianzhang/FastWAM/.hf_cache && \
   export HF_DATASETS_CACHE=/home/zijianzhang/FastWAM/.hf_cache/datasets && \
   export HUGGINGFACE_HUB_CACHE=/home/zijianzhang/FastWAM/.hf_cache/hub && \
   bash scripts/gaussianwam/run_libero_firstframe_cache_8gpu.sh'
```

Attach and monitor:

```bash
tmux attach -t libero_cache
tail -f /home/zijianzhang/FastWAM/data/libero_teacher_cache/logs/<run_dir>/manager.log
```

## Notes

- The transfer flow uses copy operations only. The source files remain on the
  current machine.
- `scripts/gaussianwam/run_libero_firstframe_cache_8gpu.sh` now resolves the
  repository root dynamically, so it can run from `/home/zijianzhang/FastWAM`
  on the remote server.
- If direct `rsync` fails with `Permission denied (publickey,password)`, use a
  terminal that already has working SSH credentials or upload the tarball with
  an interactive password prompt.
