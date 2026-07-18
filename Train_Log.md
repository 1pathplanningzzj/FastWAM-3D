# Train Log

## 2026-06-17

### LIBERO fullft first-frame launch

- New task config:

`configs/task/libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4.yaml`

- Purpose:
  - keep LIBERO Stage 2 teacher alignment on `video_out_first_frame`
  - switch training from partial finetune to full finetune

- Fullft freeze settings:
  - `action_expert=false`
  - `mot=false`
  - `video_expert_train_last_n_layers=0`

- Launch GPUs:
  - `2,3`

- Launch command:

```bash
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_gpus2-3_tmux"
SESSION="libero_gaussianwam_fullft_gpus2_3_$(date +%H%M%S)"
LOG="/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4/${RUN_ID}/launch.log"
mkdir -p "$(dirname "$LOG")"

tmux new-session -d -s "$SESSION" \
  "cd /data/zijianzhang/FastWAM && \
   export CUDA_VISIBLE_DEVICES=2,3 RUN_ID=$RUN_ID LIBERO_DATA_ROOT=/data/zijianzhang/libero_mujoco3.3.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   /data/miniconda3/bin/conda run --no-capture-output -n fastwam \
   bash scripts/train_zero1.sh 2 \
   task=libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4 \
   data.libero_text_cache_root=/data/zijianzhang/gaussianwam_data/data/text_embeds_cache/libero 2>&1 | tee $LOG"
```

### Note on alignment target

- Current GaussianWAM teacher loss still aligns against `tokens_out["video"]`
- With this LIBERO fullft config, the selected frame slice is `video_out_first_frame`
- So this is still final MoT video output token alignment, not an intermediate MoT block hidden state

### 2-GPU launch result

- The 2-GPU `2,3` launch hit CUDA OOM on `optimizer.step()`
- Failure point:
  - Adam multi-tensor update allocated an extra `11.21 GiB`
  - the active GPU had only about `10.26 GiB` free
- This is a full-finetune memory issue, not a dataset/cache issue
- Likely next step:
  - retry with more GPUs or a smaller batch / less optimizer memory pressure

### Follow-up adjustment

- For the next retry, `configs/task/libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4.yaml`
  was updated from `batch_size=16` to `batch_size=8`
- In this trainer, `batch_size` is per-process / per-GPU, not global batch size

### Additional retry adjustment

- After the `4-GPU + batch_size=8` retry still hit OOM during the first forward pass,
  the same task config was reduced again from `batch_size=8` to `batch_size=4`
- Next retry target:
  - GPUs `2,3`

### Robotwin-style alignment

- To match the working `robotwin fullft` recipe, the LIBERO fullft config was then
  restored to `batch_size=8`, `gradient_accumulation_steps=2`, and `mot_checkpoint_mixed_attn=true`
- This keeps the per-step memory-saving attention checkpointing and the same effective batch behavior as robotwin

### Current temporary setting

- Because GPUs are currently tight, the same LIBERO fullft config is now temporarily set back to `batch_size=4`
- `mot_checkpoint_mixed_attn=true` and `gradient_accumulation_steps=2` remain enabled

### `bs=4` retry result

- Run:
  - `libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4/2026-06-17_07-49-17_gpus2-3_bs4_retry_tmux`
- Result:
  - initialization completed
  - dataset stats completed
  - training entered the first step and then OOM'd at `optimizer.step()`
- Failure detail:
  - `torch._foreach_sqrt` in Adam attempted to allocate about `11.21 GiB`
  - at the failure point the device had only about `10.44 GiB` free
- Conclusion:
  - `gradient_accumulation_steps=2` does not reduce this optimizer-step peak enough
  - the next retry reduces per-GPU `batch_size` from `4` to `2` while keeping the same checkpointing settings

### `bs=2` retry result

- Run:
  - `libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4/2026-06-17_07-53-53_gpus2-3_bs2_retry_tmux`
- Result:
  - initialization completed
  - dataset stats completed
  - training reached `Starting training`
  - the first `optimizer.step()` still OOM'd at the same Adam multi-tensor update path
- Failure detail:
  - `torch._foreach_sqrt` again attempted to allocate about `11.21 GiB`
  - the failing device still had only about `10.51 GiB` free at that instant
- Environment note:
  - `nvidia-smi` showed physical GPU `3` had an existing non-FastWAM compute process `PID 125736` occupying about `7.96 GiB`
- Conclusion:
  - reducing per-GPU batch from `4` to `2` does not fix this fullft optimizer-step memory peak on the current `2,3` setup
  - next practical options are to free the resident GPU memory, switch to a more aggressive memory strategy such as optimizer offload / another ZeRO mode, or move back to more GPUs

### Single-GPU `GPU 2` retry result

- Run:
  - `libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4/2026-06-17_08-01-49_gpu2_single_retry_tmux`
- Config:
  - `CUDA_VISIBLE_DEVICES=2`
  - `nproc_per_node=1`
  - current task config still uses `batch_size=2` and `gradient_accumulation_steps=2`
- Result:
  - initialization completed
  - dataset stats completed
  - training reached `Starting training`
  - then OOM'd during the first optimizer / ZeRO step
- Failure detail:
  - attempted allocation was about `22.43 GiB`
  - failure happened in `deepspeed/runtime/zero/stage_1_and_2.py`
  - single-card fullft clearly has higher optimizer-state peak than the 2-GPU case
- Conclusion:
  - this LIBERO Stage 2 fullft job is not viable on a single A100-80GB with the current Zero-1 setup

### 4-GPU successful retry

- Run:
  - `libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4/2026-06-17_12-12-49_gpus0-5-6-7_bs2_tmux`
- Launch GPUs:
  - `0,5,6,7`
- Config at launch:
  - `batch_size=2`
  - `gradient_accumulation_steps=2`
  - `mot_checkpoint_mixed_attn=true`
- Result:
  - optimizer state initialization succeeded
  - training started successfully
  - run passed the earlier `optimizer.step()` OOM point and reached at least `step=10`
- Snapshot:
  - `epoch=0 step=10/173580`
  - `loss=2.1134`
  - `speed=0.41 step/s`
  - `3.32 samples/s`

### LIBERO full-teacher `bs=4` reference run

- Reference run:
  - `libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4/2026-06-17_12-19-41_gpus0-5-6-7_bs4_tmux`
- Launch GPUs:
  - `0,5,6,7`
- Config snapshot:
  - `batch_size=4`
  - `gradient_accumulation_steps=2`
  - `teacher_targets=[dense_3d, depth, alpha, valid_mask]`
- Progress before failure:
  - reached `epoch=3 step=30000/86790`
  - validation at `step=30000` completed with `val_loss=0.1629`
- Latest usable weight checkpoint:
  - `checkpoints/weights/step_030000.pt`
- Failure mode:
  - training itself continued through the earlier optimizer OOM barrier
  - crash happened while saving DeepSpeed / Accelerate training state
  - repeated `PytorchStreamWriter` / `inline_container.cc` write failures under `checkpoints/state/step_030000`
- Practical conclusion:
  - weight checkpoints are usable
  - for the next ablation runs, disable trainer state snapshots and keep weight-only checkpoints

### LIBERO teacher ablation plan

- Goal:
  - isolate which teacher branch contributes most on LIBERO training + LIBERO-Plus evaluation
- Ablation rule:
  - set the matching lambda to `0`
  - remove that branch from `gaussianwam.teacher_targets`
  - mirror the same list into `data.train.gaussian_teacher.targets` to avoid loading unused tensors
- Prepared configs:
  - full teacher: `configs/task/libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4.yaml`
  - no dense: `configs/task/libero_gaussianwam_stage2_fullft_firstframe_2cam224_no_dense3d_1e-4.yaml`
  - no depth: `configs/task/libero_gaussianwam_stage2_fullft_firstframe_2cam224_no_depth_1e-4.yaml`
  - no alpha: `configs/task/libero_gaussianwam_stage2_fullft_firstframe_2cam224_no_alpha_1e-4.yaml`
- Shared launch settings for the first ablation sweep:
  - GPUs `0,5,6,7`
  - conda env `fastwam`
  - `batch_size=4`
  - `gradient_accumulation_steps=2`
  - `save_every=5000`
  - `eval_every=500`
  - `save_training_state=false`
- Evaluation target:
  - train on LIBERO
  - evaluate selected checkpoints on LIBERO-Plus with the `fastwam-libero` environment

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

## 2026-06-14

### LIBERO Stage 2 current run

- Training task: `libero_gaussianwam_stage2_current_2cam224_1e-4`
- Original run dir:

`/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_current_2cam224_1e-4/2026-06-13_11-05-18_gpus3-4-5-7_tmux`

- Original launch GPUs: `3,4,5,7`
- Current eval dir:

`/data/zijianzhang/FastWAM/evaluate_results/libero/libero_gaussianwam_stage2_current_2cam224_1e-4/20260614_full_step010000_gpus0-5-6-7`

### Training status snapshot

- Config-estimated total training length: `43400` steps
- Full resumable training state currently available at:

`/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_current_2cam224_1e-4/2026-06-13_11-05-18_gpus3-4-5-7_tmux/checkpoints/state/step_010000`

- Latest saved weight checkpoint:

`/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_current_2cam224_1e-4/2026-06-13_11-05-18_gpus3-4-5-7_tmux/checkpoints/weights/step_010000.pt`

- `trainer_state.json` at `step_010000` records:
  - `global_step=10000`
  - `epoch=2`
  - `batch_in_epoch=1320`
- The run actually continued past the save point and reached about `step=11930`, but crashed before the next `step_012000` save, so the recoverable resume point is still `10000`

### Crash cause

- The run failed because text embedding cache files were missing from the path used during training:

`/data/zijianzhang/FastWAM/data/text_embeds_cache/libero`

- That repo-local cache directory is currently empty
- The populated cache is under:

`/data/zijianzhang/gaussianwam_data/data/text_embeds_cache/libero`

- Current count there: `40` `.pt` files
- Resume should explicitly point `data.libero_text_cache_root` to the cache under `gaussianwam_data`

### Evaluation result at `step_010000`

- Eval target checkpoint:

`/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_current_2cam224_1e-4/2026-06-13_11-05-18_gpus3-4-5-7_tmux/checkpoints/weights/step_010000.pt`

- Eval GPUs: `0,5,6,7`
- All `40 / 40` LIBERO tasks completed
- `failed_tasks.txt` is empty
- Auto summary generation failed only because `pandas` was missing in the eval env; raw task JSON outputs are complete

Manual success-rate summary from result JSONs:

- Overall: `1924 / 2000 = 96.2%`
- `libero_object`: `497 / 500 = 99.4%`
- `libero_spatial`: `479 / 500 = 95.8%`
- `libero_10`: `478 / 500 = 95.6%`
- `libero_goal`: `470 / 500 = 94.0%`

Lowest tasks in this eval:

- `libero_goal task3`: `42 / 50 = 84.0%`
- `libero_goal task9`: `43 / 50 = 86.0%`
- `libero_10 task0`: `44 / 50 = 88.0%`
- `libero_spatial task4`: `44 / 50 = 88.0%`
- `libero_10 task6`: `45 / 50 = 90.0%`
- `libero_10 task8`: `45 / 50 = 90.0%`

Known result inconsistency:

- `libero_10 task4`: JSON `successes=49`, but `success_episodes` has `44` items
- `libero_10 task6`: JSON `successes=45`, but `success_episodes` has `43` items
- The summary above follows the current JSON `successes` field, matching the existing eval output convention

### Planned resume

- Resume target GPUs: `0,5,6,7`
- Resume from full training state at `step_010000`, not from the plain `.pt` weights
- Keep a new run directory, but load state from:

`/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_current_2cam224_1e-4/2026-06-13_11-05-18_gpus3-4-5-7_tmux/checkpoints/state/step_010000`

- Important override for resume:

`data.libero_text_cache_root=/data/zijianzhang/gaussianwam_data/data/text_embeds_cache/libero`

## 2026-06-15

### LIBERO-Plus eval integration

- Goal: run full `LIBERO-Plus` eval for `libero_gaussianwam_stage2_current_2cam224_1e-4`
- Chosen checkpoint:

`/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_current_2cam224_1e-4/2026-06-14_12-41-35_gpus0-5-6-7_resume_from10k_v2/checkpoints/weights/step_016000.pt`

- GPUs: `0,5,6,7`
- `LIBERO-Plus` runtime root:

`/data/zijianzhang/libero_datasets/LIBERO-plus/libero/libero`

### Standard LIBERO comparison

Among these 4 previous standard-LIBERO eval runs:

- `20260614_step022000_gpus0-5-6-7`
- `20260615_step040000_gpus0-5-6-7`
- `20260614_step016000_gpus0-5-6-7`
- `20260614_full_step010000_gpus0-5-6-7`

Best overall success rate was tied:

- `20260614_step016000_gpus0-5-6-7`: `1924 / 2000 = 96.2%`
- `20260614_full_step010000_gpus0-5-6-7`: `1924 / 2000 = 96.2%`

Selected checkpoint for `LIBERO-Plus`: `step_016000.pt`

### LIBERO-Plus code / runtime fixes

Added or updated:

- `experiments/libero/libero_plus_benchmark.py`
- `experiments/libero/eval_libero_plus.py`
- `experiments/libero/run_libero_plus_manager.py`
- `experiments/libero/summarize_libero_plus_results.py`
- `configs/sim_libero.yaml`

Main fixes applied:

- Added a dedicated `LIBERO-Plus` benchmark wrapper and manager entrypoint
- Corrected BDDL resolution for `view / language / table / add / level` perturbation task names
- Made `plus_root / plus_config_dir` tolerate Hydra `null -> "None"` cases
- Replaced duplicate OmegaConf resolver registration with idempotent registration
- Ensured worker processes import the `LIBERO-Plus` Python package tree instead of the old `openpi` `libero` package
- Installed a local `wand` fallback stub because the current eval env does not ship `wand`
- Fixed `bddl_file_name` type mismatch by converting `Path` to `str` before env construction

### Known LIBERO-Plus runtime issues observed and fixed

Observed failures during bring-up:

- `task_classification.json` path resolved to `.../None/...`
- `ValueError: resolver 'eval' is already registered`
- `KeyError: 'libero_tabletop_manipulation_tabletop_table_cobblestone01_gloss_6k'`
- `ModuleNotFoundError: No module named 'wand'`
- `TypeError: argument of type 'PosixPath' is not iterable`

All of the above were fixed in repo code before relaunching.

### Trial protocol decision

Initial full run was started with:

- `EVALUATION.num_trials=50`

This was then stopped and archived after partial progress because `LIBERO-Plus`
should be compared against the VLA-JEPA protocol, which uses:

- `1` rollout per perturbation task

Archived partial directory:

`/data/zijianzhang/FastWAM/evaluate_results/libero/libero_gaussianwam_stage2_current_2cam224_1e-4/20260615_plus_step016000_gpus0-5-6-7_partial_50trials_aborted`

### Current LIBERO-Plus launch command

Use this command for the current protocol:

```bash
cd /data/zijianzhang/FastWAM
MPLCONFIGDIR=/tmp/matplotlib-fastwam-libero \
CUDA_VISIBLE_DEVICES=0,5,6,7 \
LIBERO_PLUS_ROOT=/data/zijianzhang/libero_datasets/LIBERO-plus/libero/libero \
/data/miniconda3/envs/fastwam-libero/bin/python experiments/libero/run_libero_plus_manager.py \
  task=libero_gaussianwam_stage2_current_2cam224_1e-4 \
  ckpt=/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_current_2cam224_1e-4/2026-06-14_12-41-35_gpus0-5-6-7_resume_from10k_v2/checkpoints/weights/step_016000.pt \
  EVALUATION.output_dir=/data/zijianzhang/FastWAM/evaluate_results/libero/libero_gaussianwam_stage2_current_2cam224_1e-4/20260615_plus_step016000_gpus0-5-6-7_1trial \
  MULTIRUN.num_gpus=4 \
  MULTIRUN.max_tasks_per_gpu=2 \
  MULTIRUN.task_suite_names='[libero_spatial,libero_object,libero_goal,libero_10]' \
  EVALUATION.num_trials=1
```

### Current output directory

Current active `1-trial` run:

`/data/zijianzhang/FastWAM/evaluate_results/libero/libero_gaussianwam_stage2_current_2cam224_1e-4/20260615_plus_step016000_gpus0-5-6-7_1trial`

### Category counts

`LIBERO-Plus` categories are:

- `Camera`
- `Robot`
- `Language`
- `Light`
- `Background`
- `Noise`
- `Layout`
- `Avg` = average across the 7 categories above

Task counts for `libero_spatial`:

- `Camera=376`
- `Robot=350`
- `Language=390`
- `Light=292`
- `Background=258`
- `Noise=351`
- `Layout=385`
- `Total=2402`

Task counts for all 4 suites:

- `libero_spatial=2402`
- `libero_object=2518`
- `libero_goal=2591`
- `libero_10=2519`
- `Total=10030`

### Result summary command

Use:

```bash
python experiments/libero/summarize_libero_plus_results.py \
  --output_dir /data/zijianzhang/FastWAM/evaluate_results/libero/libero_gaussianwam_stage2_current_2cam224_1e-4/20260615_plus_step016000_gpus0-5-6-7_1trial
```

Expected output header:

```text
Camera Robot Language Light Background Noise Layout Avg
```

Result JSON layout after the latest fix:

```text
<output_dir>/<suite>/<category_short>/gpu<gpu>_task<task_id>_results.json
```

Example:

```text
/data/zijianzhang/FastWAM/evaluate_results/libero/libero_gaussianwam_stage2_current_2cam224_1e-4/20260615_plus_step016000_gpus0-5-6-7_1trial/libero_spatial/Camera/gpu0_task4_results.json
```

This avoids mixing standard-LIBERO-style task files with the perturbation categories and makes per-category debugging easier.

### 2026-06-30 LIBERO-Plus env alignment and official release rerun

- Goal: align my local `LIBERO-Plus` eval env with my friend's `libero_plus_eval_env_pip_freeze.txt`, then rerun the official released FastWAM `LIBERO-Plus` benchmark with the aligned env
- Base env used for patching: `/data/miniconda3/envs/fastwam-libero`
- Preserved patched env:

`/data/miniconda3/envs/fastwam-libero-plus-eval-patched`

- Local `LIBERO-Plus` checkout used by eval:

`/data/zijianzhang/LIBERO-plus`

- Runtime root passed to eval:

`/data/zijianzhang/LIBERO-plus/libero/libero`

Environment alignment artifacts saved in repo:

- `libero_plus_eval_env_from_fastwam_libero.summary.txt`
- `libero_plus_eval_env_from_fastwam_libero.stage1.txt`
- `libero_plus_eval_env_from_fastwam_libero.stage2-heavy.txt`
- `libero_plus_eval_env_pip_freeze.local.txt`

What was aligned:

- `pip` package set now matches `181 / 181` comparable entries from the friend's freeze
- critical runtime packages are aligned, including:
  - `torch==2.4.1+cu121`
  - `torchvision==0.19.1+cu121`
  - `mujoco==3.3.2`
  - `numpy==2.2.6`
  - `robosuite==1.4.0`
  - `hydra-core==1.2.0`
  - `Wand==0.6.13`
  - `deepspeed==0.15.4`
  - `nvidia-cublas-cu12==12.1.3.1`

Remaining freeze-level differences after alignment:

- friend freeze installs `LIBERO-plus` from git:

`-e git+https://github.com/sylvestf/LIBERO-plus@4976dc30028e805ff8094b55501d532c48fec182#egg=libero`

- local env installs the same package from the downloaded editable checkout:

`-e /data/zijianzhang/LIBERO-plus`

- friend freeze also includes a prebuilt `flash-attn` wheel, but the current FastWAM `LIBERO` / `LIBERO-Plus` eval path uses PyTorch `scaled_dot_product_attention` and does not import `flash_attn`

Patched env bring-up / verification:

- fixed `wand.api` dynamic library loading; `libMagickWand` now resolves from:

`/data/miniconda3/envs/fastwam-libero-plus-eval-patched/lib/libMagickWand-7.Q16HDRI.so.10`

- verified this imports successfully in the patched env:
  - `import robosuite`
  - `import libero.libero.benchmark`
- `NUMBA_DISABLE_JIT=1` is required on this machine; otherwise `robosuite` import fails with a `numba` cache error

Validation command:

```bash
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/matplotlib-fastwam-libero \
LIBERO_PLUS_ROOT=/data/zijianzhang/LIBERO-plus/libero/libero \
/data/miniconda3/envs/fastwam-libero-plus-eval-patched/bin/python - <<'PY'
from experiments.libero.libero_plus_benchmark import configure_libero_plus_runtime
configure_libero_plus_runtime('/data/zijianzhang/LIBERO-plus/libero/libero')
import robosuite
import libero.libero.benchmark
print("robosuite OK")
print("libero benchmark OK")
PY
```

Additional code fixes applied while rerunning `LIBERO-Plus`:

- `experiments/libero/eval_libero_plus.py`
  - prepare the `LIBERO-Plus` runtime before importing helpers from `eval_libero_single.py`, so worker processes do not accidentally import the wrong `libero`
- `experiments/libero/eval_libero_single.py`
  - changed `ActionEnsembler` import to `from experiments.libero.action_ensembler import ActionEnsembler`
- `experiments/libero/run_libero_plus_manager.py`
  - worker tmux launch now propagates `NUMBA_DISABLE_JIT`

Observed failure before the fix:

- first launch created worker plans but worker panes exited immediately with:
  - `ModuleNotFoundError: No module named 'libero'`
  - then `ModuleNotFoundError: No module named 'action_ensembler'`

Official released FastWAM checkpoint rerun on the aligned env:

- GPUs: `1,2,3,4`
- output dir:

`/data/zijianzhang/FastWAM/evaluate_results/libero/libero_uncond_2cam224_1e-4/20260630_plus_official_gpus1-2-3-4_1trial_patchedenv`

- checkpoint:

`/data/zijianzhang/gaussianwam_data/checkpoints/fastwam_release/libero_uncond_2cam224.pt`

- dataset stats:

`/data/zijianzhang/gaussianwam_data/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`

Launch command:

```bash
cd /data/zijianzhang/FastWAM
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/matplotlib-fastwam-libero \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
LIBERO_PLUS_ROOT=/data/zijianzhang/LIBERO-plus/libero/libero \
PYTHON_BIN=/data/miniconda3/envs/fastwam-libero-plus-eval-patched/bin/python \
/data/miniconda3/envs/fastwam-libero-plus-eval-patched/bin/python \
experiments/libero/run_libero_plus_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/data/zijianzhang/gaussianwam_data/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=/data/zijianzhang/gaussianwam_data/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.output_dir=/data/zijianzhang/FastWAM/evaluate_results/libero/libero_uncond_2cam224_1e-4/20260630_plus_official_gpus1-2-3-4_1trial_patchedenv \
  EVALUATION.num_trials=1 \
  MULTIRUN.num_gpus=4 \
  MULTIRUN.max_tasks_per_gpu=2 \
  MULTIRUN.task_suite_names='[libero_10,libero_goal,libero_spatial,libero_object]'
```

Runtime status after the fix:

- `tmux` session:

`libero_plus_worker`

- worker logs now pass the previous import failure point and enter model load
- result files and rollout videos are already being written under:

`evaluate_results/libero/libero_uncond_2cam224_1e-4/20260630_plus_official_gpus1-2-3-4_1trial_patchedenv`

Monitoring / summary commands:

```bash
tmux attach -t libero_plus_worker
```

```bash
/data/miniconda3/envs/fastwam-libero-plus-eval-patched/bin/python \
experiments/libero/summarize_libero_plus_results.py \
  --output_dir /data/zijianzhang/FastWAM/evaluate_results/libero/libero_uncond_2cam224_1e-4/20260630_plus_official_gpus1-2-3-4_1trial_patchedenv
```

### 2026-06-30 Friend-aligned LIBERO-Plus protocol patch

Goal:

- align only the parts that can change evaluation numbers between my local FastWAM `LIBERO-Plus` run and my friend's script / environment

Main result-affecting differences identified before this patch:

- local default `seed` came from training config (`42`), while friend script uses `seed=0`
- local default step budget followed FastWAM (`400/400/400/700`), while friend / VLA-JEPA style protocol uses `250/280/300/520/400`
- local FastWAM `LIBERO-Plus` path did not support excluding `Sensor Noise`, while friend script explicitly sets:

`exclude_categories = "Sensor Noise"`

- one earlier official rerun used:

`/data/zijianzhang/gaussianwam_data/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`

but the older reference run used:

`/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_current_2cam224_1e-4/2026-06-14_12-41-35_gpus0-5-6-7_resume_from10k_v2/dataset_stats.json`

Code changes added for friend alignment:

- `experiments/libero/libero_plus_benchmark.py`
  - added `normalize_plus_categories(...)`
  - added `exclude_categories` support inside `LiberoPlusBenchmark`
- `experiments/libero/eval_libero_plus.py`
  - propagate `EVALUATION.exclude_categories`
  - write `plus_exclude_categories` into each result json
- `experiments/libero/run_libero_plus_manager.py`
  - build worker plans after removing excluded categories
  - write excluded categories into `worker_plan_summary.txt`
- `configs/sim_libero.yaml`
  - added `EVALUATION.exclude_categories`

Sanity check:

- `python -m py_compile` passes for:
  - `experiments/libero/libero_plus_benchmark.py`
  - `experiments/libero/eval_libero_plus.py`
  - `experiments/libero/run_libero_plus_manager.py`
  - `experiments/libero/eval_libero_single.py`

Additional bug found while launching the friend-aligned rerun:

- `src/fastwam/utils/pytorch_utils.py`
  - `set_global_seed(...)` incorrectly rejected `seed=0` because it asserted:

`np.iinfo(np.uint32).min < seed`

  - fixed to allow the valid lower bound:

`np.iinfo(np.uint32).min <= seed`

- this matters because the friend script explicitly uses `seed=0`, and the first 8-GPU relaunch failed before rollout started with:

`AssertionError: Seed outside the np.uint32 bounds!`

Task-count impact after excluding `Sensor Noise`:

- total perturbation-expanded tasks across `libero_10/libero_goal/libero_spatial/libero_object`:
  - before exclusion: `10030`
  - after exclusion: `8429`
  - excluded tasks: `1601`

Recommended friend-aligned rerun command:

```bash
cd /data/zijianzhang/FastWAM
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/matplotlib-fastwam-libero \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LIBERO_PLUS_ROOT=/data/zijianzhang/LIBERO-plus/libero/libero \
PYTHON_BIN=/data/miniconda3/envs/fastwam-libero-plus-eval-patched/bin/python \
/data/miniconda3/envs/fastwam-libero-plus-eval-patched/bin/python \
experiments/libero/run_libero_plus_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/data/zijianzhang/gaussianwam_data/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  seed=0 \
  EVALUATION.dataset_stats_path=/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_current_2cam224_1e-4/2026-06-14_12-41-35_gpus0-5-6-7_resume_from10k_v2/dataset_stats.json \
  EVALUATION.exclude_categories='Sensor Noise' \
  EVALUATION.step_budget_protocol=vlajepa \
  EVALUATION.output_dir=/data/zijianzhang/FastWAM/evaluate_results/libero/libero_uncond_2cam224_1e-4/20260630_plus_official_gpus0-1-2-3-4-5-6-7_1trial_friend_aligned \
  EVALUATION.num_trials=1 \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2 \
  MULTIRUN.task_suite_names='[libero_10,libero_goal,libero_spatial,libero_object]'
```

Correction made immediately after:

- for this comparison we should keep the full perturbation-expanded benchmark
- `Sensor Noise` should NOT be excluded
- corrected 8-GPU output dir:

`/data/zijianzhang/FastWAM/evaluate_results/libero/libero_uncond_2cam224_1e-4/20260630_plus_official_gpus0-1-2-3-4-5-6-7_1trial_friend_aligned_full10030`

- corrected worker-plan summary:
  - total tasks: `10030`
  - excluded categories: `<none>`
  - per-GPU tasks: `1254,1254,1254,1254,1254,1254,1253,1253`

Remaining alignment note before switching to LingBot-VA eval:

- current FastWAM LIBERO-Plus eval is already close at the protocol level:
  - `seed=0`
  - `num_trials=1`
  - `num_steps_wait=30`
  - `replan_steps=10`
  - `use_action_ensembler=false`
  - `step_budget_protocol=fastwam`
  - full `10030` tasks with `Noise` kept
- but it is still not a perfect code-path match to the friend's script
- the largest remaining difference is model construction:
  - friend script instantiates the model through a hand-written `create_fastwam_cosmos(...)` path with explicit `base_ckpt / vae / coupling / hidden_dim` arguments
  - current repo eval instantiates through Hydra `cfg.model` with `_target_: fastwam.runtime.create_fastwam`
- practical implication:
  - current numbers are already useful for comparison
  - but any remaining discrepancy versus the friend's local run may still come from this model-build path difference rather than only from eval protocol flags

## 2026-07-01

### RobotWin full-data first-frame teacher + Stage 2 wiring

- Added full-data RobotWin first-frame Stage 1 config:

`configs/gaussianwam/stage1_robotwin_firstframe_all.yaml`

- Added full-data RobotWin first-frame Stage 2 full-finetune config:

`configs/task/robotwin_gaussianwam_stage2_fullft_firstframe_current_3cam_384_1e-4.yaml`

- Added 8-worker launch script for full-data teacher precompute:

`scripts/gaussianwam/run_robotwin_firstframe_cache_8gpu.sh`

- Main wiring:
  - teacher cache uses `target_frame_policy: first_video_frame`
  - Stage 2 aligns against `target_tokens: video_out_first_frame`
  - no `episode_subset_manifest`, so training and teacher cache both target the full RobotWin train split
  - teacher manifest target path:

`/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/v4/gaussian_vggt256text_3d_firstframe_all_v1/train/manifest.jsonl`

### Full 8-GPU teacher launch attempt

- Requested action:
  - start full-data RobotWin teacher precompute in a detached `tmux` session with the new script

- Attempted tmux session:

`robotwin_teacher_full_firstframe_20260701_084921`

- Launch command used inside tmux:

```bash
cd /data/zijianzhang/FastWAM
export GAUSSIANWAM_ROOT=/data/zijianzhang/gaussianwam_data
export ROBOTWIN_DATA_ROOT=/data/zijianzhang/gaussianwam_data/data/robotwin2.0
export PYTHON_BIN=/data/miniconda3/envs/fastwam/bin/python

bash scripts/gaussianwam/run_robotwin_firstframe_cache_8gpu.sh
```

### Launch status and stop reason

- The tmux launcher started and began dataset initialization.
- During the launch check, GPUs `1,2,3,4` were already occupied by another user's Python jobs:
  - `gpu1`: ~`10.9 GiB`
  - `gpu2`: ~`10.9 GiB`
  - `gpu3`: ~`14.7 GiB`
  - `gpu4`: ~`14.7 GiB`
- Free / mostly idle GPUs at the check time were:
  - `gpu0`
  - `gpu5`
  - `gpu6`
  - `gpu7`
- To avoid colliding with those existing jobs, the pending 8-GPU tmux session was stopped before worker processes were spawned onto all GPUs.

### Ready next step

- When all 8 GPUs are available again, rerun:

```bash
cd /data/zijianzhang/FastWAM
export GAUSSIANWAM_ROOT=/data/zijianzhang/gaussianwam_data
export ROBOTWIN_DATA_ROOT=/data/zijianzhang/gaussianwam_data/data/robotwin2.0
export PYTHON_BIN=/data/miniconda3/envs/fastwam/bin/python

tmux new-session -d -s robotwin_teacher_full_firstframe \
  'cd /data/zijianzhang/FastWAM && \
   export GAUSSIANWAM_ROOT=/data/zijianzhang/gaussianwam_data ROBOTWIN_DATA_ROOT=/data/zijianzhang/gaussianwam_data/data/robotwin2.0 PYTHON_BIN=/data/miniconda3/envs/fastwam/bin/python && \
   bash scripts/gaussianwam/run_robotwin_firstframe_cache_8gpu.sh'
```

- If immediate progress is preferred before 8 GPUs free up, a fallback option is to shard the same config over the currently idle GPUs `0,5,6,7`.

### 4-GPU fallback teacher launch started

- Confirmed alignment with the LIBERO first-frame teacher recipe:
  - same teacher targets consumed at Stage 2: `dense_3d`, `depth`, `alpha`, `valid_mask`
  - same first-frame supervision idea:
    - Stage 1 cache target: `first_video_frame`
    - Stage 2 student token target: `video_out_first_frame`
  - difference is dataset / camera layout only, not the distillation target family

- Updated launcher:
  - `scripts/gaussianwam/run_robotwin_firstframe_cache_8gpu.sh`
  - now accepts `GPU_LIST=...` so the same full-data sharded cache job can run on a sparse GPU set

- Active fallback tmux session:

`robotwin_teacher_full_firstframe_gpus0_5_6_7_20260701_085616`

- Active manager log:

`/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/logs/gaussian_vggt256text_3d_firstframe_all_v1_20260701_085828/manager.log`

- Launch command:

```bash
cd /data/zijianzhang/FastWAM
tmux new-session -d -s robotwin_teacher_full_firstframe_gpus0_5_6_7_20260701_085616 \
  'cd /data/zijianzhang/FastWAM && \
   export GAUSSIANWAM_ROOT=/data/zijianzhang/gaussianwam_data ROBOTWIN_DATA_ROOT=/data/zijianzhang/gaussianwam_data/data/robotwin2.0 PYTHON_BIN=/data/miniconda3/envs/fastwam/bin/python GPU_LIST=0,5,6,7 && \
   bash scripts/gaussianwam/run_robotwin_firstframe_cache_8gpu.sh'
```

- Current shard assignment:
  - `gpu0`: `0 -> 1502894`
  - `gpu5`: `1502894 -> 3005788`
  - `gpu6`: `3005788 -> 4508682`
  - `gpu7`: `4508682 -> 6011575`

### RobotWin clean-only heuristic subset + teacher relaunch plan

- Goal:
  - avoid the full `27500`-episode teacher sweep
  - train on clean-only RobotWin data first, then evaluate on both `clean` and `random`

- Constraint found:
  - the packed FastWAM RobotWin dataset does not retain explicit `demo_clean` / `demo_randomized` labels
  - so exact clean recovery cannot be done from `meta/episodes.jsonl` alone

- Heuristic adopted:
  - treat the packed dataset as `50` task blocks of `550` episodes
  - slice the first `50` episodes from each block as the clean subset
  - keep a matching `last50` contrast manifest for manual sanity checks

- New helper script:

`scripts/gaussianwam/build_robotwin_clean_heuristic_subset.py`

- Generated manifests:
  - `data/robotwin2.0/subsets/full_clean_heuristic_first50.jsonl`
  - `data/robotwin2.0/subsets/full_clean_heuristic_first50.report.json`
  - `data/robotwin2.0/subsets/full_clean_heuristic_last50.jsonl`
  - `data/robotwin2.0/subsets/full_clean_heuristic_last50.report.json`

- Manifest sizes:
  - `first50`: `2500` episodes
  - `last50`: `2500` episodes

- Spot-check result from extracted head-camera first frames:
  - `block0`: `ep0` is clean white-table scene, `ep500` is cluttered randomized scene
  - `block10`: `ep5500` is clean white-table scene, `ep6000` is cluttered randomized scene
  - `block49`: `ep26950` is clean white-table scene, `ep27450` is cluttered randomized scene
  - conclusion: `first50` is the correct clean heuristic, `last50` behaves like randomized

- Extracted check images:
  - `/tmp/robotwin_clean_check/ep_0_cam_high.png`
  - `/tmp/robotwin_clean_check/ep_500_cam_high.png`
  - `/tmp/robotwin_clean_check/ep_5500_cam_high.png`
  - `/tmp/robotwin_clean_check/ep_6000_cam_high.png`
  - `/tmp/robotwin_clean_check/ep_26950_cam_high.png`
  - `/tmp/robotwin_clean_check/ep_27450_cam_high.png`

- New Stage 1 clean-cache config:

`configs/gaussianwam/stage1_robotwin_fullclean_first50_firstframe_all.yaml`

- New Stage 2 clean-only train config:

`configs/task/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4.yaml`

- Stage 1 clean-cache wiring:
  - `episode_subset_manifest: data/robotwin2.0/subsets/full_clean_heuristic_first50.jsonl`
  - `target_frame_policy: first_video_frame`
  - cache namespace:

`gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1`

- Stage 2 clean-only wiring:
  - train/val both point to `data/robotwin2.0/subsets/full_clean_heuristic_first50.jsonl`
  - teacher manifest target:

`/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/v4/gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1/train/manifest.jsonl`

- Requested operational change:
  - stop the previous full-data teacher tmux run
  - relaunch the cache job on all `8` GPUs using the new clean-only config
  - after cache completion, train and then evaluate on both `clean` and `random`

- Current dataset sizing from manager log:
  - total train samples: `6011575`
  - 4-way chunk size: `1502894`

## 2026-07-04

### RobotWin clean-only first-frame Stage 2 launch

- Teacher cache completion check:
  - `/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/logs/gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1_20260701_111940/manager.log`
  - recorded `all workers finished` at `2026-07-04 08:00:36`

- Launch target:
  - task: `configs/task/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4.yaml`
  - GPUs: `0,5,6,7`
  - env: `fastwam`
  - extra runtime override: `save_every=5000`

- Clean-only data confirmation:
  - `data.train.episode_subset_manifest=data/robotwin2.0/subsets/full_clean_heuristic_first50.jsonl`
  - `data.val.episode_subset_manifest=data/robotwin2.0/subsets/full_clean_heuristic_first50.jsonl`
  - this run is restricted to the heuristic clean subset rather than the full packed RobotWin train split

- Gaussian teacher supervision confirmation:
  - `data.train.gaussian_teacher.manifest_path=/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/v4/gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1/train/manifest.jsonl`
  - `gaussianwam.enabled=true`
  - `gaussianwam.target_tokens=video_out_first_frame`
  - `gaussianwam.teacher_targets=[dense_3d, depth, alpha, valid_mask]`
  - launch log later reported `Loaded Gaussian teacher manifest ... ok_unique=545062`, matching the clean-subset train sample count

- Active run metadata:
  - tmux session: `robotwin_fullclean_ff_083412`
  - run id: `2026-07-04_08-34-12_gpus0-5-6-7_save5k_tmux`
  - launch log:
    `/data/zijianzhang/gaussianwam_data/runs/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4/2026-07-04_08-34-12_gpus0-5-6-7_save5k_tmux/launch.log`

- Early runtime status:
  - dataset size reported by trainer: `545062/4725` train/val
  - training entered the main loop successfully with `max_steps=42585`
  - first visible progress snapshot: `epoch=0 step=10/42585 loss=2.3144`
  - first expected weight checkpoint under this launch setting: `checkpoints/weights/step_005000.pt`

## 2026-07-05

### RobotWin clean-only Stage 2 stop and full benchmark eval

- Stopped the clean-only Stage 2 training run after the `step_040000.pt` weight checkpoint was available.
- Stopped run:

`/data/zijianzhang/gaussianwam_data/runs/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4/2026-07-04_08-34-12_gpus0-5-6-7_save5k_tmux`

- Stop command used:

```bash
pkill -TERM -f '/data/zijianzhang/gaussianwam_data/runs/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4/2026-07-04_08-34-12_gpus0-5-6-7_save5k_tmux'
```

- Latest available weight checkpoint selected for eval:

`/data/zijianzhang/gaussianwam_data/runs/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4/2026-07-04_08-34-12_gpus0-5-6-7_save5k_tmux/checkpoints/weights/step_040000.pt`

- Dataset stats selected for eval:

`/data/zijianzhang/gaussianwam_data/runs/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4/2026-07-04_08-34-12_gpus0-5-6-7_save5k_tmux/dataset_stats.json`

- Reference protocol to align with:

`/data/zijianzhang/FastWAM/evaluate_results/robotwin/robotwin_uncond_3cam_384/full_sapien300b1_gpu6_20260606_172214`

- Alignment check against the reference eval:
  - full RobotWin task list: `50` tasks
  - phases: `demo_clean` and `demo_randomized`
  - `EVALUATION.eval_num_episodes=100`
  - `EVALUATION.instruction_type=unseen`
  - `seed=42`
  - `EVALUATION.replan_steps=24`
  - `EVALUATION.num_inference_steps=10`
  - `mixed_precision=bf16`
  - `EVALUATION.skip_get_obs_within_replan=true`
  - `MULTIRUN.num_gpus=8`
  - `MULTIRUN.max_tasks_per_gpu=2`

- Important protocol note:
  - training data for this checkpoint is clean-only (`full_clean_heuristic_first50`)
  - eval remains the full RobotWin benchmark, matching the reference layout with both clean and randomized phases

- Current eval is running in tmux in the `fastwam` environment:
  - tmux socket: `/data/zijianzhang/FastWAM/.tmux-robotwin-eval.sock`
  - tmux session: `robotwin_full_step040000_8gpu_cleantrained`
  - attach command:

```bash
tmux -S /data/zijianzhang/FastWAM/.tmux-robotwin-eval.sock attach -t robotwin_full_step040000_8gpu_cleantrained
```

- Output directory:

`/data/zijianzhang/FastWAM/evaluate_results/robotwin/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4_2026-07-04_08-34-12_gpus0-5-6-7_save5k_tmux/20260705_full_step040000_8gpu_cleantrained`

- Logs:
  - `manager.log`
  - `manager.launch.log`

- Launch command:

```bash
tmux -S /data/zijianzhang/FastWAM/.tmux-robotwin-eval.sock new-session -d \
  -s robotwin_full_step040000_8gpu_cleantrained \
  bash -lc 'cd /data/zijianzhang/FastWAM && \
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
    export GAUSSIANWAM_ROOT=/data/zijianzhang/gaussianwam_data && \
    export DIFFSYNTH_MODEL_BASE_PATH=/data/zijianzhang/gaussianwam_data/checkpoints && \
    export MPLCONFIGDIR=/tmp/matplotlib-fastwam-robotwin && \
    export MUJOCO_GL=egl && \
    export PYOPENGL_PLATFORM=egl && \
    /data/miniconda3/envs/fastwam/bin/python experiments/robotwin/run_robotwin_manager.py \
      task=robotwin_uncond_3cam_384_1e-4 \
      ckpt=/data/zijianzhang/gaussianwam_data/runs/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4/2026-07-04_08-34-12_gpus0-5-6-7_save5k_tmux/checkpoints/weights/step_040000.pt \
      EVALUATION.dataset_stats_path=/data/zijianzhang/gaussianwam_data/runs/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4/2026-07-04_08-34-12_gpus0-5-6-7_save5k_tmux/dataset_stats.json \
      EVALUATION.output_dir=/data/zijianzhang/FastWAM/evaluate_results/robotwin/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4_2026-07-04_08-34-12_gpus0-5-6-7_save5k_tmux/20260705_full_step040000_8gpu_cleantrained \
      MULTIRUN.num_gpus=8 \
      MULTIRUN.max_tasks_per_gpu=2 \
    > /data/zijianzhang/FastWAM/evaluate_results/robotwin/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4_2026-07-04_08-34-12_gpus0-5-6-7_save5k_tmux/20260705_full_step040000_8gpu_cleantrained/manager.launch.log 2>&1'
```

### RobotWin clean-only failure analysis and next experiment options

- Observation from current RobotWin Stage 2 clean-only experiments:
  - release baseline remains strong on full RobotWin eval: clean about `92.8%`, randomized about `91.5%`
  - `step_040000` clean-only Stage 2 eval drops to about `43.5%` overall, with clean about `77.9%` and randomized about `1.6%`
  - `step_085170` is worse than `step_040000`, with overall about `33%`, clean about `58%`, randomized about `2%`
  - this suggests long clean-only finetuning is causing overfitting / forgetting rather than improving the full RobotWin benchmark

- Current hypothesis for why LIBERO improves while RobotWin does not:
  - LIBERO Stage 2 training and LIBERO plus eval are better aligned; the training data is not restricted to a tiny clean-only slice in the same way
  - RobotWin Stage 2 here trains only on the heuristic clean subset, but eval is the full benchmark with both `demo_clean` and `demo_randomized`
  - RobotWin randomized eval is much more out-of-distribution relative to the clean-only subset: randomized backgrounds, cluttered tables, lighting, and table height shifts are enabled in `demo_randomized.yml`
  - RobotWin is also more sensitive because it uses 3 cameras, a larger mosaic, and 14-D dual-arm action/state, while LIBERO is 2-camera and 7-D action/state
  - continuing full finetune on clean-only data appears to wash out the release policy's original randomized robustness

- Recommended experiment changes if we want RobotWin to reproduce the kind of gain seen on LIBERO:
  1. Use full RobotWin or clean+random mixed teacher cache/training data instead of clean-only.
  2. If teacher cache cost is too high, at least mix per-task clean and randomized subsets, e.g. `first50 + last50`, rather than only `first50`.
  3. Reduce forgetting by lowering LR or freezing more policy layers; try training only GaussianWAM projection/depth/alpha/proprio heads first.
  4. Evaluate early checkpoints (`2.5k`, `5k`, `7.5k`, `10k`) because current evidence suggests later clean-only checkpoints degrade RobotWin.
  5. Report clean-only and randomized-only eval separately to distinguish clean imitation degradation from randomized OOD failure.

- Current retrain launched on `2026-07-08` follows the early-checkpoint plan:
  - tmux session: `robotwin_fullclean_retrain_early_20260708`
  - GPUs: `0,5,6,7`
  - starts from release checkpoint rather than old `step_040000`
  - `save_every=2500`, `max_steps=15000`
  - intended checkpoints: `step_002500`, `step_005000`, `step_007500`, `step_010000`, `step_012500`, `step_015000`
