# Train Log

## 2026-06-09

### Current focus

- Main line: `robotwin_gaussianwam_stage2_focus3_fullft_current_3cam_384_1e-4`
- Current run: `runs/robotwin_gaussianwam_stage2_focus3_fullft_current_3cam_384_1e-4/2026-06-08_20-11-33_gpus3-4-5-7_tmux`
- Subset: `switch / microwave / mug`
- Eval loading config: `task=robotwin_uncond_3cam_384_1e-4`

### Available checkpoints

- `step_002500.pt`
- `step_005000.pt`
- `step_007500.pt`
- `step_010000.pt`

Checkpoint directory:

`runs/robotwin_gaussianwam_stage2_focus3_fullft_current_3cam_384_1e-4/2026-06-08_20-11-33_gpus3-4-5-7_tmux/checkpoints/weights`

Dataset stats used for eval:

`runs/robotwin_gaussianwam_stage2_focus3_fullft_current_3cam_384_1e-4/2026-06-08_20-11-33_gpus3-4-5-7_tmux/dataset_stats.json`

### Related runs

- `runs/robotwin_gaussianwam_stage2_current_3cam_384_1e-4/2026-06-08_12-08-11_gpus3-4-5-7_tmux`
- `runs/robotwin_gaussianwam_stage2_current_3cam_384_1e-4/2026-06-08_13-37-37_gpus3-4-5-7_tmux`
- `runs/robotwin_gaussianwam_stage2_current_3cam_384_1e-4/2026-06-08_18-59-04_gpus3-4-5-7_tmux`
- `runs/robotwin_gaussianwam_stage2_focus3_current_3cam_384_1e-4/2026-06-08_19-41-45_gpus3-4-5-7_tmux`
- `runs/robotwin_gaussianwam_stage2_focus3_fullft_current_3cam_384_1e-4/2026-06-08_20-04-33_gpus3-4-5-7_tmux`
- `runs/robotwin_gaussianwam_stage2_focus3_fullft_current_3cam_384_1e-4/2026-06-08_20-11-33_gpus3-4-5-7_tmux`

### Evaluation snapshot

3-task eval target:

`evaluate_results/robotwin/robotwin_gaussianwam_stage2_focus3_fullft_current_3cam_384_1e-4_2026-06-08_20-11-33_gpus3-4-5-7_tmux/20260609_3tasks_step10000_gpu6`

Snapshot at record time:

- `turn_switch` clean phase in progress
- Progress snapshot: `33 / 87 = 37.93%`
- This is an intermediate number, not the final clean result
- Eval currently uses `step_010000.pt`
- Eval currently runs on physical GPU `6`

### Release baseline eval status

Release eval directory:

`evaluate_results/robotwin/robotwin_uncond_3cam_384/full_sapien300b1_gpu6_20260606_172214`

Status snapshot at record time:

- Release checkpoint: `checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`
- Full benchmark over `50` RoboTwin tasks
- Eval setting is `clean 100 + random 100` per task
- Progress snapshot: `clean=23`, `random=21`
- This run is not a fully clean final benchmark record anymore
- `place_bread_basket / random` was interrupted by `SIGTERM (-15)` and recorded in `failed_tasks.txt`
- The session itself continued to later tasks after that interruption

### Notes

- Current codepath explicitly supports evaluating a Stage 2 GaussianWAM checkpoint with `task=robotwin_uncond_3cam_384_1e-4`
- In this path, GaussianWAM auxiliary heads may be ignored at eval load time, but the finetuned policy backbone in `mot.state_dict()` is still used
- For RoboTwin here, `EVALUATION.eval_num_episodes` controls the true episode count and overrides the `episode_num: 50` inside `third_party/RoboTwin/task_config/*.yml`
- Therefore the current default manager behavior is `100` episodes for `demo_clean` and `100` episodes for `demo_randomized` unless explicitly overridden

### First-frame teacher refresh

- Goal: test `focus3 (switch / microwave / mug)` with `first-frame` Gaussian teacher alignment
- Training-side alignment config prepared:
  - `configs/task/robotwin_gaussianwam_stage2_focus3_fullft_firstframe_current_3cam_384_1e-4.yaml`
  - `gaussianwam.target_tokens=video_out_first_frame`
  - `expected_target_offset=0`
- Previous frame-level subset attempt was discarded because its indexing did not align with the filtered training dataset
- A second issue was then found in the old `focus3` subset:
  - keyword-based `switch` matching also pulled in `switch-hand / transfer` tasks
  - this polluted the subset with unrelated prompts such as `microphone`
- Clean episode subset rebuilt from true prompt semantics:
  - `data/robotwin2.0/subsets/stage2_focus3_switch_microwave_mug_clean.jsonl`
  - episode counts: `switch=550`, `microwave=550`, `mug=550`
- Training and teacher configs now both point to the clean subset
- Current teacher cache namespace:
  - `gaussian_vggt256text_3d_focus3_firstframe_all_v3_clean`
- Current 8-GPU precompute tmux session:
  - `gaussian_focus3_firstframe_cache_v3_clean`
- Current log dir:

`data/robotwin2.0/gaussian_teacher_cache/logs/gaussian_vggt256text_3d_focus3_firstframe_all_v3_clean_20260609_103620`

- Current clean-subset train split size used for teacher precompute:
  - `episodes_train=1633`
  - `frames_train=535116`

## 2026-06-11

### LIBERO first-frame teacher cache

- Goal: generate LIBERO Gaussian teacher cache aligned with Stage 2 `first frame` distillation
- Teacher precompute config:
  - `configs/gaussianwam/stage1_libero.yaml`
  - `source.target_frame_policy=first_video_frame`
- Training-side alignment config:
  - `configs/data/libero_gaussianwam.yaml`
  - `expected_target_offset=0`
  - `expected_camera_keys=[image, wrist_image]`
  - `expected_mosaic.layout=horizontal`
  - `expected_mosaic.grid_size=[14, 28]`
- Cache manifest target:
  - `data/libero_teacher_cache/v1/gaussian_vggt256text_3d_libero_firstframe_all_v1/train/manifest.jsonl`
- 8-GPU launcher script:
  - `scripts/gaussianwam/run_libero_firstframe_cache_8gpu.sh`

### Restart command

Use the `fastwam` env and restart LIBERO cache generation later with:

```bash
cd /data/zijianzhang/FastWAM
export LIBERO_DATA_ROOT=/data/zijianzhang/libero_mujoco3.3.2
export HF_HOME=/data/zijianzhang/FastWAM/.hf_cache
export HF_DATASETS_CACHE=/data/zijianzhang/FastWAM/.hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=/data/zijianzhang/FastWAM/.hf_cache/hub
tmux -S /data/zijianzhang/FastWAM/.tmux-fastwam.sock new-session -d -s libero_cache \
  'cd /data/zijianzhang/FastWAM && \
   export LIBERO_DATA_ROOT=/data/zijianzhang/libero_mujoco3.3.2 && \
   export HF_HOME=/data/zijianzhang/FastWAM/.hf_cache && \
   export HF_DATASETS_CACHE=/data/zijianzhang/FastWAM/.hf_cache/datasets && \
   export HUGGINGFACE_HUB_CACHE=/data/zijianzhang/FastWAM/.hf_cache/hub && \
   bash scripts/gaussianwam/run_libero_firstframe_cache_8gpu.sh'
```

### Monitoring

```bash
tmux -S /data/zijianzhang/FastWAM/.tmux-fastwam.sock attach -t libero_cache
tail -f data/libero_teacher_cache/logs/<run_dir>/manager.log
```
