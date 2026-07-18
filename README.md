# FastWAM

Official codebase for **Fast-WAM: Do World Action Models Need Test-time Future Imagination?**

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2603.16666-b31b1b.svg)](https://arxiv.org/abs/2603.16666)
[![Project Page](https://img.shields.io/badge/Project_Page-Fast--WAM-2ea44f.svg)](https://yuantianyuan01.github.io/FastWAM/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/yuanty/fastwam)
[![Hugging Face Dataset - LIBERO](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20LIBERO-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![Hugging Face Dataset - RoboTwin](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20RoboTwin-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

This repository contains the training and evaluation code for FastWAM on LIBERO / RoboTwin.

## Index

- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Dataset Download](#dataset-download)
- [Inference with Released Checkpoints](#inference-with-released-checkpoints)
- [Training](#training)
- [Inference with Your Trained Checkpoints](#inference-with-your-trained-checkpoints)
- [Acknowledgements](#acknowledgements)
- [BibTeX](#bibtex)

## File Structure

```text
FastWAM/
├── configs/
│   ├── data/                 # Dataset configs (LIBERO, RoboTwin, etc.)
│   ├── model/                # Model architecture and component configs
│   └── task/                 # Task-level configs (training task names)
├── scripts/
│   ├── train.py
│   ├── train_zero1.sh        # Deepspeed zero1 training entrypoint
│   ├── preprocess_action_dit_backbone.py  # Preprocess ActionDiT backbone before training
│   └── precompute_text_embeds.py  # Precompute T5 text embedding cache before training
├── experiments/
│   ├── libero/
│   │   └── run_libero_manager.py
│   └── robotwin/
│       └── run_robotwin_manager.py
├── src/fastwam/              # Core code
├── runs/                     # Training outputs (ckpt, logs)
├── checkpoints/              # Pretrained or external checkpoints
├── data/                     # Data directory
└── evaluate_results/         # Inference / evaluation results
```

## Environment Setup

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Model Preparation

This step is required before both training and inference.

Step 1: set the Wan model directory first (opional, default `./checkpoints`):

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Step 2: pre-generate the ActionDiT backbone (interpolated from Wan22 DiT):

```bash
# uncond (fastwam)
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## Dataset Download

### LIBERO

The preprocessed LIBERO dataset used by Fast-WAM is available at:

- https://huggingface.co/datasets/yuanty/LIBERO-fastwam

Download all compressed files first, then extract them all:

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2

# Run after downloading all 4 tar.gz files
for f in *.tar.gz; do
  tar -xzf "$f"
done
```

The extracted directory structure should be:

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin

The preprocessed RoboTwin dataset used by Fast-WAM is available at:

- https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

Download all split archive files first, then concatenate and extract:

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0

# Run after downloading all robotwin2.0.tar.gz.part-* files
cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

The extracted directory structure should be:

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

If you also keep:

```text
data/robotwin2.0/dataset_stats.json
```

in the root directory, it can be used directly as the statistics file for the current configs in this repo. You can also recompute it.

## Inference with Released Checkpoints

The released checkpoints and their corresponding dataset stats are available on [Hugging Face](https://huggingface.co/yuanty/fastwam).

Optional: download released checkpoints and dataset stats from Hugging Face:

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

After downloading, the local directory is expected to contain:

```text
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
├── libero_uncond_2cam224_dataset_stats.json
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

Before running the `LIBERO` benchmark, install the official LIBERO environment first
from the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO).
Then run this final step:

```bash
pip install mujoco==3.3.2
```

The `mujoco` environment should ideally stay consistent with the LIBERO data version.

We have already copied the `RoboTwin` evaluation-related code into `third_party/RoboTwin`.
You still need to follow the official RoboTwin instructions from the
[RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin) to finish environment installation and download the required assets, then create the policy symlink:

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

Optional: evaluate released LIBERO checkpoint:

The released `LIBERO` / `RoboTwin` evaluation managers default to `8` GPUs
(`MULTIRUN.num_gpus=8` in `configs/sim_libero.yaml` and `configs/sim_robotwin.yaml`).
If you want to evaluate with fewer GPUs, pass a smaller value such as
`MULTIRUN.num_gpus=4`.

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

### LIBERO-Plus evaluation

`LIBERO-Plus` is evaluated with a separate entrypoint and uses perturbation-expanded task suites.
Compared with standard `LIBERO`, each suite contains many more tasks:

- `libero_spatial`: `2402`
- `libero_object`: `2518`
- `libero_goal`: `2591`
- `libero_10`: `2519`

The current code supports both:

- single-task eval: [`experiments/libero/eval_libero_plus.py`](./experiments/libero/eval_libero_plus.py)
- tmux multi-GPU manager: [`experiments/libero/run_libero_plus_manager.py`](./experiments/libero/run_libero_plus_manager.py)

Required runtime root:

```bash
export LIBERO_PLUS_ROOT=/data/zijianzhang/LIBERO-plus/libero/libero
```

Verified patched eval env on this machine:

```bash
export PYTHON_BIN=/data/miniconda3/envs/fastwam-libero-plus-eval-patched/bin/python
export MPLCONFIGDIR=/tmp/matplotlib-fastwam-libero
export NUMBA_DISABLE_JIT=1
```

This patched env was aligned against `libero_plus_eval_env_pip_freeze.txt`.
The current local freeze is saved in `libero_plus_eval_env_pip_freeze.local.txt`.
The remaining freeze-level differences are intentional / benign for the current FastWAM eval path:

- friend freeze installs `LIBERO-plus` from a git URL, while this machine uses an editable local checkout at `/data/zijianzhang/LIBERO-plus`
- friend freeze includes `flash-attn`, but the current FastWAM `LIBERO` / `LIBERO-Plus` eval path uses PyTorch `scaled_dot_product_attention` and does not import `flash_attn`

Rebuild the patched env from `fastwam-libero`:

```bash
conda create -y -n fastwam-libero-plus-eval-patched --clone fastwam-libero
/data/miniconda3/envs/fastwam-libero-plus-eval-patched/bin/pip install \
  -r libero_plus_eval_env_from_fastwam_libero.stage1.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
/data/miniconda3/envs/fastwam-libero-plus-eval-patched/bin/pip install \
  -r libero_plus_eval_env_from_fastwam_libero.stage2-heavy.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121
/data/miniconda3/envs/fastwam-libero-plus-eval-patched/bin/pip freeze \
  > libero_plus_eval_env_pip_freeze.local.txt
```

Quick runtime validation:

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

Recommended protocol for comparing against VLA-JEPA:

- use `1` rollout per perturbation task
- evaluate all perturbation-expanded tasks
- aggregate results by perturbation category:
  `Camera / Robot / Language / Light / Background / Noise / Layout / Avg`

Example full 4-GPU eval:

```bash
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/matplotlib-fastwam-libero \
CUDA_VISIBLE_DEVICES=0,5,6,7 \
LIBERO_PLUS_ROOT=/data/zijianzhang/LIBERO-plus/libero/libero \
$PYTHON_BIN experiments/libero/run_libero_plus_manager.py \
  task=libero_gaussianwam_stage2_current_2cam224_1e-4 \
  ckpt=/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_current_2cam224_1e-4/2026-06-14_12-41-35_gpus0-5-6-7_resume_from10k_v2/checkpoints/weights/step_016000.pt \
  EVALUATION.output_dir=/data/zijianzhang/FastWAM/evaluate_results/libero/libero_gaussianwam_stage2_current_2cam224_1e-4/20260615_plus_step016000_gpus0-5-6-7_1trial \
  MULTIRUN.num_gpus=4 \
  MULTIRUN.max_tasks_per_gpu=2 \
  MULTIRUN.task_suite_names='[libero_spatial,libero_object,libero_goal,libero_10]' \
  EVALUATION.num_trials=1
```

Official released FastWAM checkpoint on `LIBERO-Plus` with the patched env:

```bash
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

Friend-aligned comparison protocol for the released FastWAM checkpoint:

- keep the patched env above
- explicitly use `seed=0`
- switch to `EVALUATION.step_budget_protocol=vlajepa`
- exclude `Sensor Noise`
- use the original training-run stats file instead of the repackaged release copy:
  `/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_current_2cam224_1e-4/2026-06-14_12-41-35_gpus0-5-6-7_resume_from10k_v2/dataset_stats.json`

Example 8-GPU launch:

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

Monitor the workers with:

```bash
tmux attach -t libero_plus_worker
```

Single-task debug example:

```bash
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/matplotlib-fastwam-libero \
LIBERO_PLUS_ROOT=/data/zijianzhang/LIBERO-plus/libero/libero \
$PYTHON_BIN experiments/libero/eval_libero_plus.py \
  task=libero_gaussianwam_stage2_current_2cam224_1e-4 \
  ckpt=/path/to/step.pt \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1
```

You can optionally restrict eval to one perturbation category:

```bash
EVALUATION.plus_category='Background Textures'
```

Or exclude one or more perturbation categories from the full run:

```bash
EVALUATION.exclude_categories='Sensor Noise'
```

Category summary helper:

```bash
$PYTHON_BIN experiments/libero/summarize_libero_plus_results.py \
  --output_dir /path/to/libero_plus_eval_dir
```

This prints:

```text
Camera Robot Language Light Background Noise Layout Avg
```

Result JSONs are stored by suite and perturbation category, for example:

```text
<output_dir>/libero_spatial/Camera/gpu0_task4_results.json
```

Each JSON also records:

- `plus_category`: full category name such as `Camera Viewpoints`
- `plus_category_short`: short bucket such as `Camera`

Notes for this repo:

- `LIBERO-Plus` must import its own Python package tree, not only its data root.
- The runtime helper in [`experiments/libero/libero_plus_benchmark.py`](./experiments/libero/libero_plus_benchmark.py)
  injects the correct repo root into `sys.path` and clears stale `libero.*` imports.
- The patched eval env now ships `Wand==0.6.13`; the fallback stub still exists as a last-resort safeguard.
- `NUMBA_DISABLE_JIT=1` is required in the patched env because `robosuite` hits a `numba` cache failure during import on this machine.
- `LIBERO-Plus` environment code expects `bddl_file_name` as `str`, so
  [`experiments/libero/libero_utils.py`](./experiments/libero/libero_utils.py)
  now converts `Path` to `str` before constructing the env.

Optional: evaluate released RoboTwin checkpoint:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

For faster RoboTwin evaluation, we have enabled `EVALUATION.skip_get_obs_within_replan=true` in [`configs/sim_robotwin.yaml`](./configs/sim_robotwin.yaml).
This skips RGB rendering while consecutively executing an action chunk within one replan window, which speeds up evaluation but makes the saved video look very low-FPS.
Set it to `false` if you want to save a fully rendered video.

**Note:** We evaluate with **unseen** instructions, following Motus. [Lingbot-VA](https://github.com/Robbyant/lingbot-va/blob/661d52a59dc634a650efcd10a79d06bbb17ea81f/evaluation/robotwin/eval_polict_client_openpi.py#L308) uses **seen** instructions instead. You can try `EVALUATION.instruction_type=seen` to use **seen** instructions, which should theoretically improve performance by one or two points.

## Training

### 1) Precompute T5 embedding cache before training

Use `scripts/precompute_text_embeds.py` to precompute embeddings for each training task:

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

For multi-GPU:

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

### 2) Training (using `fastwam` as an example)

When running a new task for the first time, set `pretrained_norm_stats` in the corresponding `configs/data/*.yaml` to `null` first.
After one training run, a `dataset_stats.json` file will be generated in the current run directory (for example, `runs/{task_name}/{run_id}/dataset_stats.json`).
You can then update `pretrained_norm_stats` to that file path for subsequent runs.

```bash
# LIBERO
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# RoboTwin
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

For LIBERO, we train on a single node with 8 GPUs. For RoboTwin, we use 64 GPUs to accelerate training. You can try reducing the GPU count or training epochs.

## Inference with Your Trained Checkpoints

The `mujoco` environment should ideally stay consistent with the LIBERO data version. Then run LIBERO evaluation:

```bash
# LIBERO
python experiments/libero/run_libero_manager.py task={task_name} ckpt={ckpt_path}
```

We have already copied the `RoboTwin` evaluation-related code into `third_party/RoboTwin`.
You still need to follow the official RoboTwin instructions from the
[RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin).
Finish installation and download the required assets, then create the policy symlink:

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

Then run RoboTwin evaluation:

```bash
python experiments/robotwin/run_robotwin_manager.py task={task_name} ckpt={ckpt_path}
```

Common `task_name` examples:

```text
libero_uncond_2cam224_1e-4
robotwin_uncond_3cam_384_1e-4
```

## Acknowledgements

The RoboTwin evaluation code in this repository is adapted from the official [RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin). We thank the RoboTwin team for releasing their codebase and assets.

## BibTeX

If you find our work helpful, please consider citing:

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```

## Stage 2 Launch Example

Example command to launch the current GaussianWAM Stage 2 distillation training on GPUs `3,4,5,7` in a detached `tmux` session:

```bash
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_gpus3-4-5-7_tmux"
SESSION="fastwam_stage2_gpus3_4_5_7_$(date +%H%M%S)"
LOG="./runs/robotwin_gaussianwam_stage2_current_3cam_384_1e-4/${RUN_ID}/launch.log"
mkdir -p "$(dirname "$LOG")"

tmux new-session -d -s "$SESSION" \
  "cd $(pwd) && \
   export CUDA_VISIBLE_DEVICES=3,4,5,7 RUN_ID=$RUN_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   /data/miniconda3/bin/conda run --no-capture-output -n fastwam \
   bash scripts/train_zero1.sh 4 task=robotwin_gaussianwam_stage2_current_3cam_384_1e-4 2>&1 | tee $LOG"
```

Useful follow-up commands:

```bash
tmux ls
tmux attach -t "$SESSION"
tail -f "$LOG"
```

Focused 3-task Stage 2 training example (`switch` / `microwave` / `mug` subset only):

```bash
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_gpus3-4-5-7_tmux"
SESSION="fastwam_stage2_focus3_gpus3_4_5_7_$(date +%H%M%S)"
LOG="./runs/robotwin_gaussianwam_stage2_focus3_current_3cam_384_1e-4/${RUN_ID}/launch.log"
mkdir -p "$(dirname "$LOG")"

tmux new-session -d -s "$SESSION" \
  "cd $(pwd) && \
   export CUDA_VISIBLE_DEVICES=3,4,5,7 RUN_ID=$RUN_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   /data/miniconda3/bin/conda run --no-capture-output -n fastwam \
   bash scripts/train_zero1.sh 4 task=robotwin_gaussianwam_stage2_focus3_current_3cam_384_1e-4 2>&1 | tee $LOG"
```

This focused config uses:

- task config: `configs/task/robotwin_gaussianwam_stage2_focus3_current_3cam_384_1e-4.yaml`
- subset manifest: `data/robotwin2.0/subsets/stage2_focus3_switch_microwave_mug.jsonl`
- selected episodes matched by keywords: `switch`, `microwave`, `mug`

RobotWin clean-only first-frame teacher recipe (`50` clean episodes per `550`-episode task block):

- subset builder: `scripts/gaussianwam/build_robotwin_clean_heuristic_subset.py`
- selected manifest: `data/robotwin2.0/subsets/full_clean_heuristic_first50.jsonl`
- inspection report: `data/robotwin2.0/subsets/full_clean_heuristic_first50.report.json`
- contrast manifest: `data/robotwin2.0/subsets/full_clean_heuristic_last50.jsonl`
- Stage 1 cache config: `configs/gaussianwam/stage1_robotwin_fullclean_first50_firstframe_all.yaml`
- Stage 2 train config: `configs/task/robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4.yaml`

The `first50` choice is a heuristic reconstructed from the packed `27500`-episode FastWAM dataset:

- the dataset is organized as `50` task blocks of `550` episodes each
- official RoboTwin `demo_clean` is `50` episodes per task, while randomized data is larger
- spot-checking the head-camera first frames shows the first `50` episodes in a block are clean white-table scenes, while the last `50` are cluttered randomized scenes

Generate or refresh the manifests:

```bash
python scripts/gaussianwam/build_robotwin_clean_heuristic_subset.py \
  --output data/robotwin2.0/subsets/full_clean_heuristic_first50.jsonl \
  --report data/robotwin2.0/subsets/full_clean_heuristic_first50.report.json

python scripts/gaussianwam/build_robotwin_clean_heuristic_subset.py \
  --position last \
  --output data/robotwin2.0/subsets/full_clean_heuristic_last50.jsonl \
  --report data/robotwin2.0/subsets/full_clean_heuristic_last50.report.json
```

Launch the clean-only 8-GPU teacher cache job:

```bash
cd /data/zijianzhang/FastWAM
export GAUSSIANWAM_ROOT=/data/zijianzhang/gaussianwam_data
export ROBOTWIN_DATA_ROOT=/data/zijianzhang/gaussianwam_data/data/robotwin2.0
export PYTHON_BIN=/data/miniconda3/envs/fastwam/bin/python
export CONFIG=configs/gaussianwam/stage1_robotwin_fullclean_first50_firstframe_all.yaml
export GPU_LIST=0,1,2,3,4,5,6,7
export NUM_WORKERS=8

tmux new-session -d -s robotwin_teacher_fullclean_first50_firstframe \
  'cd /data/zijianzhang/FastWAM && \
   export GAUSSIANWAM_ROOT=/data/zijianzhang/gaussianwam_data ROBOTWIN_DATA_ROOT=/data/zijianzhang/gaussianwam_data/data/robotwin2.0 PYTHON_BIN=/data/miniconda3/envs/fastwam/bin/python CONFIG=configs/gaussianwam/stage1_robotwin_fullclean_first50_firstframe_all.yaml GPU_LIST=0,1,2,3,4,5,6,7 NUM_WORKERS=8 && \
   bash scripts/gaussianwam/run_robotwin_firstframe_cache_8gpu.sh'
```

Train Stage 2 on the same clean-only subset after the cache finishes:

```bash
bash scripts/train_zero1.sh 8 task=robotwin_gaussianwam_stage2_fullclean_first50_fullft_firstframe_current_3cam_384_1e-4
```

Inference with a saved Stage 2 weight checkpoint:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/robotwin_gaussianwam_stage2_current_3cam_384_1e-4/<run_id>/checkpoints/weights/step_002500.pt
```

Important note for the current codepath:

- `FastWAM.save_checkpoint()` saves `mot`, `proprio_encoder`, and optional GaussianWAM heads.
- The saved `mot.state_dict()` already includes the finetuned `video_expert` / `action_expert` weights under `mixtures.video.*` and `mixtures.action.*`, so the `.pt` file preserves the Stage 2 policy backbone for direct evaluation.
- If you evaluate this checkpoint with a non-GaussianWAM config such as `task=robotwin_uncond_3cam_384_1e-4`, the loader will warn that GaussianWAM heads are ignored. This is expected and only affects the auxiliary distillation heads, not the main policy weights.

## LIBERO Ablations

For LIBERO GaussianWAM teacher ablations, the simplest and cleanest protocol is:

- set the matching loss weight to `0`
- remove that teacher branch from `gaussianwam.teacher_targets`
- mirror the same target list in `data.train.gaussian_teacher.targets` to avoid loading unused teacher tensors

Prepared task configs:

- full teacher: `configs/task/libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4.yaml`
- no `dense_3d`: `configs/task/libero_gaussianwam_stage2_fullft_firstframe_2cam224_no_dense3d_1e-4.yaml`
- no `depth`: `configs/task/libero_gaussianwam_stage2_fullft_firstframe_2cam224_no_depth_1e-4.yaml`
- no `alpha`: `configs/task/libero_gaussianwam_stage2_fullft_firstframe_2cam224_no_alpha_1e-4.yaml`

Current full-teacher reference run:

- run dir:
  `/data/zijianzhang/gaussianwam_data/runs/libero_gaussianwam_stage2_fullft_firstframe_2cam224_1e-4/2026-06-17_12-19-41_gpus0-5-6-7_bs4_tmux`
- latest weight checkpoint before the DeepSpeed state-save failure:
  `checkpoints/weights/step_030000.pt`

Recommended 4-GPU ablation launch template on GPUs `0,5,6,7`:

```bash
TASK=libero_gaussianwam_stage2_fullft_firstframe_2cam224_no_dense3d_1e-4
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_gpus0-5-6-7_bs4_tmux"
SESSION="${TASK}_$(date +%H%M%S)"
LOG="/data/zijianzhang/gaussianwam_data/runs/${TASK}/${RUN_ID}/launch.log"
mkdir -p "$(dirname "$LOG")"

tmux new-session -d -s "$SESSION" \
  "cd /data/zijianzhang/FastWAM && \
   export CUDA_VISIBLE_DEVICES=0,5,6,7 RUN_ID=$RUN_ID LIBERO_DATA_ROOT=/data/zijianzhang/libero_mujoco3.3.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   /data/miniconda3/bin/conda run --no-capture-output -n fastwam \
   bash scripts/train_zero1.sh 4 \
   task=${TASK} \
   batch_size=4 \
   gradient_accumulation_steps=2 \
   save_every=5000 \
   eval_every=500 \
   save_training_state=false \
   data.libero_text_cache_root=/data/zijianzhang/gaussianwam_data/data/text_embeds_cache/libero 2>&1 | tee $LOG"
```

Notes:

- `save_training_state=false` disables DeepSpeed/Accelerate state snapshots and only keeps the weight checkpoints. This avoids the state-save failure that interrupted the full-teacher `step_030000` run.
- Use the `fastwam` conda env for training and the `fastwam-libero` Python env for `LIBERO-Plus` evaluation.
