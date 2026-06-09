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
