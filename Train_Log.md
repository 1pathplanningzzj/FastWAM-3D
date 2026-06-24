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
