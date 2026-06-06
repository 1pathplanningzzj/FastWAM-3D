# FastWAM + RoboTwin 运行指南

本文记录在当前服务器上安装 FastWAM、补齐 RoboTwin 评测环境、下载数据/权重并启动最小评测的常用命令。

## 1. 创建并进入环境

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
```

## 2. 安装 PyTorch CUDA 版本

```bash
pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --timeout 120 \
  --resume-retries 20
```

安装后检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

期望能看到 CUDA 可用，例如 `True`。

## 3. 安装 FastWAM

在仓库根目录执行：

```bash
cd /data/zijianzhang/FastWAM

pip install -e . \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --timeout 120 \
  --resume-retries 20
```

如果因为 `torch==2.7.1+cu128` 解析失败，可以先单独安装 PyTorch，再把 `pyproject.toml` 中的 torch/torchvision 临时改成不带 `+cu128` 的版本号后重试。

## 4. 下载并解压 RoboTwin 数据

```bash
cd /data/zijianzhang/FastWAM
mkdir -p data/robotwin2.0
cd data/robotwin2.0

HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 hf download yuanty/robotwin2.0-fastwam \
  --repo-type dataset \
  --local-dir . \
  --max-workers 1
```

如果下载中断，直接在同一目录重跑同一条命令续传，不要删除已下载文件，也不要加 `--force-download`。

下载完后解压：

```bash
cd /data/zijianzhang/FastWAM/data/robotwin2.0
cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

检查目录结构：

```bash
ls -lh /data/zijianzhang/FastWAM/data/robotwin2.0/robotwin2.0
```

理想情况下应包含：

```text
data  meta  videos
```

## 5. 下载官方 release 权重

```bash
cd /data/zijianzhang/FastWAM
mkdir -p checkpoints/fastwam_release

HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 hf download yuanty/fastwam \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

检查：

```bash
ls -lh checkpoints/fastwam_release
```

## 6. 安装 RoboTwin 评测依赖

RoboTwin 代码已经 vendored 在：

```text
third_party/RoboTwin
```

在同一个 `fastwam` 环境中补充仿真依赖：

```bash
conda activate fastwam

pip install sapien gymnasium toppra transforms3d mplib h5py zarr opencv-python imageio \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120 \
  --resume-retries 20
```

评测过程中如果缺 `open3d`：

```bash
conda activate fastwam
conda install open3d -y -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
```

或者用 pip：

```bash
pip install open3d --only-binary=:all: \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 300 \
  --resume-retries 50
```

验证：

```bash
python -c "import open3d as o3d; print(o3d.__version__)"
```

## 7. 修复/确认 SAPIEN 离屏渲染权限

如果运行 RoboTwin 渲染测试时报：

```text
RuntimeError: failed to find a rendering device
```

先检查当前用户是否有 `video` 和 `render` 组：

```bash
id
```

如果没有，用 root 执行：

```bash
usermod -aG render,video zijianzhang
```

然后必须重新登录 `zijianzhang` 用户，或在当前终端临时进入带组权限的 shell：

```bash
exec sg video "sg render 'bash -l'"
```

确认：

```bash
id
for d in /dev/dri/renderD*; do [ -r "$d" ] && [ -w "$d" ] && echo OK "$d" || echo NO_ACCESS "$d"; done
```

如果仍然只有 CPU Vulkan `llvmpipe`，先用 root 检查当前 NVIDIA/Vulkan 状态：

```bash
dpkg -l | grep -E 'nvidia-driver|libnvidia-gl|nvidia-utils|nvidia-fabricmanager|vulkan'
nvidia-smi
cat /proc/driver/nvidia/version
ls -l /etc/vulkan/icd.d /usr/share/vulkan/icd.d
vulkaninfo --summary | grep -E "GPU|deviceName|driverName|driverInfo"
```

本机 A100 节点已验证可用的修复方式是把 NVIDIA 驱动栈统一升级到 apt 管理的 `580.126.20-server`。不要只安装单独的 `libnvidia-gl-*` 或 `nvidia-utils-*`，否则容易出现 kernel driver 与 NVML/graphics userspace 版本不一致，导致 `nvidia-smi` 损坏。

如果当前是 550 runfile 驱动，先停止占用 NVIDIA 的服务并卸载旧驱动：

```bash
sudo -i
systemctl stop cloud-monitor-agent.service || true
systemctl stop nvidia-fabricmanager.service || true
systemctl stop nvidia-persistenced.service || true

env -i HOME=/root USER=root LOGNAME=root SHELL=/bin/bash TERM="$TERM" \
  PATH=/usr/sbin:/usr/bin:/sbin:/bin \
  sh /root/NVIDIA-Linux-x86_64-550.144.03.run --uninstall
```

清理旧的 NVIDIA apt 残留包后安装完整 580 server 栈：

```bash
dpkg -l | awk '/^(ii|rc)/ && ($2 ~ /^(nvidia-(driver|dkms|utils|kernel|firmware|compute|fabricmanager|settings)|libnvidia|xserver-xorg-video-nvidia)/) {print $2}' \
  | xargs -r apt purge -y

apt autoremove -y
apt update
apt install --no-install-recommends -y \
  nvidia-driver-580-server=580.126.20-0ubuntu0.22.04.1 \
  nvidia-fabricmanager-580=580.126.20-0ubuntu0.22.04.1
```

安装完成后 hold 住 580 包，避免后续 apt 自动升级/降级导致再次混装：

```bash
dpkg-query -W -f='${binary:Package}\n' \
  | grep -E '^(nvidia|libnvidia|xserver-xorg-video-nvidia).*580' \
  | xargs -r apt-mark hold

ldconfig
reboot
```

重启后确认 NVIDIA kernel driver、`nvidia-smi`、Fabric Manager 和 Vulkan 都是 580 并能看到 A100：

```bash
nvidia-smi
cat /proc/driver/nvidia/version
systemctl status nvidia-fabricmanager.service --no-pager
vulkaninfo --summary | grep -E "GPU|deviceName|driverName|driverInfo"
```

580 安装后的 NVIDIA ICD 路径是 `/usr/share/vulkan/icd.d/nvidia_icd.json`；如果需要手动指定，使用：

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

测试 RoboTwin/SAPIEN 渲染：

```bash
conda activate fastwam
cd /data/zijianzhang/FastWAM/third_party/RoboTwin
python script/test_render.py
```

看到以下输出说明基础渲染可用：

```text
Render Well
```

## 8. 创建 FastWAM policy 软链接

```bash
cd /data/zijianzhang/FastWAM
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
ls -l third_party/RoboTwin/policy/fastwam_policy
```

## 9. 启动 RoboTwin 最小评测

建议先用单任务、单 episode 测通链路：

```bash
cd /data/zijianzhang/FastWAM
conda activate fastwam

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name=beat_block_hammer \
  EVALUATION.eval_num_episodes=1 \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

结果会保存到：

```text
evaluate_results/robotwin/
```

如果最小评测跑通，再扩大 episode 数量或去掉 `EVALUATION.task_name` 跑全量任务。

## 10. 全量 RoboTwin release 评测

默认配置会跑多任务，耗时较长。按实际 GPU 数量设置 `MULTIRUN.num_gpus`：

```bash
cd /data/zijianzhang/FastWAM
conda activate fastwam

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2
```

如果只想减少评测规模，可以加：

```bash
EVALUATION.eval_num_episodes=1
```

## 11. 常见问题

### pip 下载中断

直接重跑原命令，通常会复用缓存或续传。可加：

```bash
--timeout 120 --resume-retries 20
```

### Hugging Face 下载大文件中断

不要删除已下载文件，直接在同一目录重跑同一条 `hf download` 命令。推荐：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 hf download ... --max-workers 1
```

### `ModuleNotFoundError: No module named 'open3d'`

安装：

```bash
conda install open3d -y -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
```

### `Render Error`

`script/test_render.py` 会吞掉真实异常。可以用下面命令打印 traceback：

```bash
python - <<'PY'
import traceback
try:
    import sapien.core as sapien
    from sapien.render import set_global_config
    engine = sapien.Engine()
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")
    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    print("Render Well")
except Exception:
    traceback.print_exc()
PY
```



tail -f /data/zijianzhang/FastWAM/evaluate_results/robotwin/robotwin_uncond_3cam_384/20260602_131144/manager.log


cd /data/zijianzhang/FastWAM
conda activate fastwam

nohup python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2 \
  > robotwin_eval_$(date +%Y%m%d_%H%M%S).log 2>&1 &




tmux new -s robotwin_eval

cd /data/zijianzhang/FastWAM
conda activate fastwam

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2
