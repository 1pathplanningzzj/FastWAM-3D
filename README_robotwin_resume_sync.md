# RobotWin Cache Resume Sync

This note records the minimum paths and commands needed to move the current RoboTwin Gaussian teacher cache job to another server and resume safely.

## Current local run

- Config: `configs/gaussianwam/stage1_robotwin_fullclean_first50_firstframe_all.yaml`
- Cache namespace: `gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1`
- Cache root:
  `/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/v4/gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1/train`
- Dataset root:
  `/data/zijianzhang/gaussianwam_data/data/robotwin2.0/robotwin2.0`
- Text cache path baked into existing cache payloads:
  `/data/zijianzhang/gaussianwam_data/data/text_embeds_cache/robotwin`

Existing cache validation checks `text.text_cache_path` inside each saved `.pt`. To keep completed samples skippable after migration, the new server must expose the same path, either as a real directory or a symlink.

## What must exist on the new server

- Code under `/home/zijianzhang/FastWAM`
- Teacher cache data under `/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache`
- Raw RoboTwin dataset under `/data/zijianzhang/gaussianwam_data/data/robotwin2.0/robotwin2.0`
- Text cache reachable at:
  `/data/zijianzhang/gaussianwam_data/data/text_embeds_cache/robotwin`
- VGGT checkpoint:
  `/data/zijianzhang/VGGT-Omega/vggt_omega_1b_256_text.pt`
- Local CLIP model:
  `/data/zijianzhang/clip-vit-base-patch16`
- Repo third-party VGGT code:
  `/home/zijianzhang/FastWAM/third_party/vggt-omega`

## Remote layout

Recommended layout on the destination server:

- Code: `/home/zijianzhang/FastWAM`
- Data root: `/data/zijianzhang/gaussianwam_data`
- If text cache already exists only in home, expose it with:

```bash
mkdir -p /data/zijianzhang/gaussianwam_data/data/text_embeds_cache
ln -sfn /home/zijianzhang/FastWAM/data/text_embeds_cache/robotwin \
  /data/zijianzhang/gaussianwam_data/data/text_embeds_cache/robotwin
```

## Resume command

Run from `/home/zijianzhang/FastWAM`:

```bash
export GAUSSIANWAM_ROOT=/data/zijianzhang/gaussianwam_data
export ROBOTWIN_DATA_ROOT=/data/zijianzhang/gaussianwam_data/data/robotwin2.0
export HF_HOME=/home/zijianzhang/FastWAM/.hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

bash scripts/gaussianwam/run_robotwin_firstframe_cache_8gpu.sh \
  --config configs/gaussianwam/stage1_robotwin_fullclean_first50_firstframe_all.yaml
```

If the remote machine has different GPU ids, set `GPU_LIST` or `NUM_WORKERS` before launching.

## Sync policy

- Sync code into `/home/zijianzhang/FastWAM`
- Sync only the RoboTwin cache namespace being resumed, not the whole `gaussian_teacher_cache`
- Sync the raw RoboTwin dataset if the remote machine does not already have it
- Reuse remote text cache if present; otherwise sync the missing files or whole directory
- Keep the same config path and the same resolved data paths to avoid a new config hash
