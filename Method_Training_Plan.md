# GaussianWAM 两阶段训练计划

## 1. 目标

目标是在不增加 FastWAM 推理延迟的前提下，把 3D-aware + semantic-aware representation 蒸馏进 FastWAM 的 video/world tokens，让 action expert 通过 MoT 读到更有几何、多视角一致性和语义辨识能力的视觉状态特征。

推理路径保持原始 FastWAM：

```text
image + instruction + proprio
    -> Wan VAE encode
    -> video_expert.pre_dit
    -> MoT video KV cache
    -> action_expert denoising
    -> action chunk
```

训练期额外引入三层 teacher/student 栈：

```text
Layer 1: foundation teachers
    VGGT-Omega geometry / camera / register
    CLIP or future DINO/SAM dense semantic features
        -> distill / fit
Layer 2: Gaussian feature field teacher
    per-sample feature_z + camera-aware render/cache
        -> distill
Layer 3: FastWAM student
    video/world tokens learn cached Gaussian teacher representation
```

核心原则：

```text
Train with 3D Gaussian teacher, deploy 2D FastWAM student.
```

Gaussian field 只作为训练期 teacher 的中间表示；推理时不运行 VGGT-Omega、CLIP、Gaussian construction、Gaussian renderer 或 teacher projection heads。

## 2. 参考方法分工

| 来源 | 借鉴内容 | 在 GaussianWAM 中的角色 |
| --- | --- | --- |
| REPA | projector + normalized cosine representation alignment | Stage 2 中把 FastWAM video tokens 对齐到 teacher feature space |
| Feature4X | Gaussian 中存 compact feature，并 render 成 feature map | Stage 1 中拟合 view-consistent Gaussian feature field |
| VGGT-Omega | camera / depth / point map / register tokens | 为 Gaussian field 提供 3D 几何、相机和全局 scene prior |
| CLIP | image/text 语义空间 | 给 Gaussian feature field 注入 object / language-aware semantic signal |

第一版不让 FastWAM 学显式 Gaussian 参数，也不在 FastWAM training loop 中优化 Gaussian field。Gaussian field 的作用是把 VGGT-Omega 的多视角 3D 信息和 CLIP 的语义信息组织成可 render 的 dense teacher target。

## 3. 两阶段总览

第一版只做两个实现阶段，但概念上有三层：

```text
Stage 1: Foundation teachers -> Gaussian feature field teacher + cache
Stage 2: Gaussian feature field teacher -> FastWAM student
```

Stage 1 会有训练/优化，但训练对象是 **离线 Gaussian feature field teacher**，不是 FastWAM。它类似 `data/text_embeds_cache/` 的思想：把训练中昂贵、可复用的 teacher 输出提前算好；区别是 text cache 只需要 frozen T5 forward，而 Gaussian teacher cache 需要 per-sample / per-clip 地把 VGGT-Omega 几何和 CLIP/DINO/SAM-like dense semantic features 蒸馏进 Gaussian `feature_z`。

## 4. 关键监督位置

FastWAM 的 action expert 不是直接看 RGB，也不是看 `video_expert.post_dit(...)` 的最终 video latent prediction，而是通过 MoT attention 读取 video branch 的 token / K/V。

当前路径可以理解为：

```text
video_expert.pre_dit(...)
    -> video_pre["tokens"]
    -> MoT mixed attention / video KV
    -> tokens_out["video"]
    -> video_expert.post_dit(...)
```

```text
action_expert.pre_dit(...)
    -> action_pre["tokens"]
    -> MoT mixed attention reads video K/V
    -> tokens_out["action"]
    -> action_expert.post_dit(...)
```

第一版 Stage 2 最小实验只监督：

```text
video_pre["tokens"] 的 first-frame video/world tokens
```

原因是这些 token 最直接服务于 action branch 能看到的当前视觉状态。后续如果这个方向有效，再考虑 MoT output video tokens 或中间层 K/V。

不建议第一版主要监督：

```text
video_expert.post_dit(...) output
```

因为它是 video latent-space prediction，主要服务 video loss，不是 action branch 直接读取的表示。

## 5. Stage 1: Foundation Teachers -> Gaussian Feature Field Teacher

### 5.1 目标

Stage 1 目标是先把 foundation teachers 的几何和语义信号蒸馏进离线 Gaussian feature field，再生成可复用 teacher cache：

```text
RoboTwin original multi-view frames
    -> VGGT-Omega frozen forward: camera / depth / confidence / register
    -> CLIP or future DINO/SAM-like dense feature extraction
    -> initialize Gaussian geometry / feature_z
    -> optimize Gaussian feature field against dense teacher maps
    -> camera-aware render feature / depth / alpha / semantic map
    -> save teacher cache for FastWAM Stage 2
```

这个阶段和 FastWAM 训练代码尽量解耦，方便独立调试 VGGT-Omega、CLIP/dense teacher features、camera convention、Gaussian renderer、optimization 和 cache key 对齐。PNG/debug 图只用于人眼检查；真正用于 Stage 2 的是 `.pt` cache 中的 tensor targets。

### 5.2 输入

RoboTwin 每个样本优先使用原始多视角输入，而不是 FastWAM 拼接后的 2D 图：

```text
multi-view images / video frames
camera metadata if available
instruction
action trajectory
proprio
```

如果当前 camera metadata 不完整，第一版可以先用 VGGT-Omega 估计 camera / depth，再构建 Gaussian teacher。

### 5.3 VGGT-Omega teacher

VGGT-Omega 权重位置：

```text
/data/zijianzhang/VGGT-Omega/vggt_omega_1b_512.pt
/data/zijianzhang/VGGT-Omega/vggt_omega_1b_256_text.pt
```

第一版默认使用：

```text
/data/zijianzhang/VGGT-Omega/vggt_omega_1b_512.pt
```

VGGT-Omega forward 提供：

```text
pose / camera encoding
depth / depth confidence
point map optional
camera tokens
register / global tokens
```

这些输出用于：

1. 初始化 Gaussian `xyz` / camera / depth；
2. 约束 rendered depth；
3. 提供 global scene / geometry register teacher。

如果后续需要 VGGT dense patch tokens，可以在 wrapper 中从 aggregator final tokens 取 patch tokens，而不是直接修改 third_party 源码。

### 5.4 CLIP semantic teacher

CLIP 用来给 Gaussian feature field 注入语义信息，补足纯几何 teacher 对 object identity / language-conditioned manipulation 的不足。

第一版 CLIP 可以提供两类信号：

```text
CLIP image patch feature / dense feature    -> object / visual semantic signal
CLIP text feature from instruction          -> language-conditioned semantic signal
```

对于 RoboTwin 任务，CLIP 语义特征主要帮助：

- 多物体 identity；
- 颜色 / 类别 / 属性；
- instruction 中的目标物体；
- object boundary / region relevance；
- `blocks_ranking_rgb`、`blocks_ranking_size`、`pick_diverse_bottles`、`hanging_mug` 等任务。

第一版 Stage 1 应优先使用 frozen CLIP image encoder 的 dense/patch feature 作为 per-view feature map。如果本地 CLIP patch token 不可用，fallback 也应该是空间变化的 dense feature（例如 RGB + xy + depth/confidence + image stats），而不是把图像级 CLIP feature 简单 broadcast 到整张图。broadcast 只能作为 ablation，不能作为正式 Gaussian teacher 质量依据。

### 5.5 Gaussian Feature Field

每个 Gaussian 不只存 RGB/SH，也存 compact feature：

```text
g_i = {
  xyz,
  scale,
  rotation,
  opacity,
  color / SH optional,
  feature_z
}
```

`feature_z` 是核心。第一版建议用 32 / 64 维，承载几何 + 语义混合信息：

```text
feature_z = compact(
  geometry signal,
  VGGT register / dense signal,
  CLIP semantic signal
)
```

渲染后得到：

```text
G_feature_map   view-consistent 3D feature map
G_depth         rendered / fused depth
G_alpha         visibility / confidence
G_semantic      rendered semantic feature map
G_register      pooled global 3D scene feature optional
```

再用轻量 projection / decoder 对齐到 teacher spaces：

```text
D_vggt(G_feature_map) -> VGGT geometry / dense space
D_clip(G_feature_map) -> CLIP semantic space
D_reg(pool(G_feature_map)) -> VGGT register space
D_depth(G_feature_map) -> depth / point space optional
```

这样 `T_gaussian_feature` 既有 3D 几何，又有 CLIP 语义。

### 5.6 Gaussian fitting loss

Stage 1 拟合 Gaussian teacher 时，可以使用：

```text
L_gaussian = lambda_depth * L_depth_render
           + lambda_clip  * L_clip_feature
           + lambda_reg   * L_register
           + lambda_alpha * L_alpha_reg
           + optional L_rgb
```

其中：

```text
L_depth_render = masked_l1(rendered_depth, stopgrad(VGGT_depth))
L_clip_feature = cosine_loss(D_clip(rendered_feature), stopgrad(CLIP_feature))
L_register     = cosine_loss(D_reg(pool(rendered_feature)), stopgrad(VGGT_register))
L_alpha_reg    = visibility / opacity regularization
```

第一版推荐从简单版本开始：

```text
L_depth_render + L_clip_feature + L_alpha_reg
```

如果 `VGGT_register` 对齐稳定，再加 `L_register`。

### 5.7 Cache 保存内容

第一版建议保存：

```text
sample_id
trajectory_id
frame_index
latent_step_index
camera_id / view_id
T_gaussian_feature
T_depth
T_alpha
T_valid_mask
T_clip_feature optional
T_register optional
camera_info
vggt_confidence
gaussian_meta
```

其中最核心的是：

```text
T_gaussian_feature
T_valid_mask
```

Stage 2 会把 `T_gaussian_feature` resize / patchify 到 FastWAM token grid，然后和 `video_pre["tokens"]` 做 REPA-style distillation。

### 5.8 时间和空间对齐

FastWAM video latent 有 temporal downsample，例如：

```text
raw frames:   I0 I1 I2 I3 I4 I5 I6 I7 I8
vae latents:  z0          z1          z2
```

第一版采用简单对齐：

```text
z0 -> I0 teacher
z1 -> I4 teacher
z2 -> I8 teacher
```

最小版本只做 current / first-frame 对齐，可以先缓存最后一个 target frame 或当前 frame，具体由实现配置决定。

空间上，需要把 rendered Gaussian feature map resize / patchify 到 FastWAM student token 网格：

```text
student first-frame tokens: [B, Hs*Ws, D]
teacher feature tokens:     [B, Hs*Ws, Dt]
valid mask tokens:          [B, Hs*Ws]
```

如果 FastWAM 输入是三相机拼接图，teacher 需要从原始三视角构建 Gaussian，再 render/compose 到和 FastWAM 输入对应的 mosaic 或 token grid。

### 5.9 Stage 1 验证标准

进入 Stage 2 前，先确认：

- VGGT-Omega forward 能稳定跑完；
- CLIP feature 能稳定抽取；
- Gaussian 初始化点数 > 0；
- Gaussian fitting loss 不 NaN，并且有下降趋势；
- renderer 输出 feature / depth / alpha；
- rendered feature / depth / alpha 和目标视角对齐；
- teacher cache 的 sample id、frame index、latent step index 能和 FastWAM dataset 对上；
- `T_gaussian_feature`、`T_valid_mask` 能 patchify 到 student token grid；
- debug 可视化中的 depth、alpha、feature PCA 非空且合理。

## 6. Stage 2: FastWAM + REPA-style 3D/Semantic Distillation

### 6.1 目标

训练 FastWAM student，让 video/world tokens 吸收 Stage 1 teacher cache 中的 3D-aware + semantic-aware Gaussian feature。

训练时仍保留原始 FastWAM objectives：

```text
L_video  = video latent diffusion target MSE
L_action = action diffusion target MSE
```

额外加入 REPA-style alignment：

```text
S0 = video_pre["tokens"] first-frame tokens
T0 = cached rendered Gaussian feature tokens

L_repa_3d_sem = - mean(
  normalize(P_student(S0)) · normalize(P_teacher(T0))
)
```

整体 loss 第一版建议：

```text
L_total = L_video
        + L_action
        + lambda_gaussian * L_repa_3d_sem
```

可选再加：

```text
+ lambda_depth * L_depth
+ lambda_reg   * L_reg
+ lambda_clip  * L_clip_semantic
```

但建议第一版 Stage 2 先只开 `L_repa_3d_sem`，等 action eval 不退化后再加 depth/register/clip auxiliary loss。

### 6.2 Projection heads

借鉴 REPA，不直接强迫 FastWAM token 等于 teacher feature，而是加 projection heads：

```text
P_student(S_video) -> d_align
P_teacher(T_gaussian_feature) -> d_align
```

teacher feature stop-gradient：

```text
loss(P_student(S), stopgrad(P_teacher(T)))
```

第一版 trainable 部分建议：

```text
student projection head
teacher projection head optional
video_expert last 2 layers or adapter optional
```

不建议第一版 full fine-tune 全模型。

### 6.3 最小训练实验

最小实验只做：

```text
Target token: video_pre["tokens"] first-frame tokens
Teacher: current-frame rendered Gaussian feature map with VGGT geometry + CLIP semantic signal
Loss: REPA-style normalized cosine alignment
Original losses: keep L_video + L_action
Frozen: action_expert frozen, most video_expert frozen
Trainable: projection head + video_expert last 2 layers or adapter
```

伪代码：

```python
video_pre = video_expert.pre_dit(...)
action_pre = action_expert.pre_dit(...)

tokens_out = mot(...)

pred_video = video_expert.post_dit(tokens_out["video"], video_pre)
pred_action = action_expert.post_dit(tokens_out["action"], action_pre)

loss_video = mse(pred_video, target_video)
loss_action = mse(pred_action, target_action)

S0 = video_pre["tokens"][:, :tokens_per_frame]
T0 = gaussian_feature_current_aligned
M0 = gaussian_valid_mask_current_aligned

S_proj = normalize(P_student(S0), dim=-1)
T_proj = normalize(P_teacher(T0).detach(), dim=-1)
loss_repa_3d_sem = -masked_mean((S_proj * T_proj).sum(dim=-1), M0)

loss = loss_video + loss_action + lambda_gaussian * loss_repa_3d_sem
```

## 7. 当前不做的事情

第一版明确不做：

```text
future 3D distillation
online Gaussian optimization inside FastWAM training
inference-time Gaussian branch
action-region weighted loss
joint/idm GaussianWAM variants
large-scale action_expert fine-tuning
```

注意：Stage 1 会离线优化 Gaussian teacher，但这个优化不发生在 FastWAM training loop 里。

## 8. 需要记录的指标

Stage 1 指标：

```text
gaussian_loss_depth
gaussian_loss_clip
gaussian_loss_alpha
gaussian_loss_total
rendered alpha valid ratio
rendered depth error
CLIP feature cosine
cache file size
cache coverage
```

Stage 2 训练指标：

```text
loss_video
loss_action
loss_repa_3d_sem
loss_total
student-teacher cosine similarity
valid_mask coverage
```

如果启用辅助 loss，再记录：

```text
loss_reg
loss_depth
loss_clip_semantic
current depth error
register cosine similarity
```

任务指标：

```text
RoboTwin success rate overall
success rate by task
especially weak / 3D-heavy / semantic-heavy tasks
```

建议特别关注：

```text
handover_block
handover_mic
blocks_ranking_rgb
blocks_ranking_size
open_microwave
open_laptop
pick_diverse_bottles
hanging_mug
```

## 9. 决策标准

继续推进的信号：

- Stage 1 Gaussian fitting loss 能下降；
- rendered depth / alpha / feature PCA 看起来合理；
- CLIP feature cosine 有提升；
- Stage 2 `loss_repa_3d_sem` 明显下降；
- student-teacher cosine similarity 上升；
- `loss_video` / `loss_action` 没有明显恶化；
- action eval 不下降或小幅提升；
- 3D-heavy / semantic-heavy tasks 有改善迹象。

需要调整的信号：

如果 Stage 1 Gaussian fitting 不稳定：

```text
减少 Gaussian 点数
降低 lr_feature / lr_xyz
先只优化 depth + alpha
暂时关闭 CLIP feature loss
检查 camera / depth convention
```

如果 Stage 2 distillation 下降但 action eval 下降：

```text
降低 lambda_gaussian
冻结更多 video/action 参数
只训练 projection head 或 adapter
延后解冻 mot/action_expert
```

如果 Stage 2 `loss_repa_3d_sem` 不下降：

```text
teacher target 和 student token 网格没对齐
projection head 太弱
teacher feature normalization 有问题
valid mask / resize / patchify 有问题
CLIP / Gaussian feature scale 不匹配
```

## 10. 简短总结

GaussianWAM 第一版仍然只分两个阶段：

```text
Stage 1: Fit Gaussian teacher cache with VGGT-Omega geometry + CLIP semantics
Stage 2: Train FastWAM with REPA-style Gaussian feature distillation
```

Stage 1 使用 VGGT-Omega 前向提供 3D 几何和相机信息，用 CLIP 提供语义信息，再拟合 Feature4X-style Gaussian feature field，生成 view-consistent 的 `T_gaussian_feature` cache。Stage 2 借鉴 REPA，用 projector + normalized cosine alignment 把 FastWAM 的 first-frame video/world tokens 对齐到这个 3D+semantic Gaussian feature。推理时仍然只使用原始 FastWAM 路径。