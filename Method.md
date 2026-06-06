# 方法草案：面向 FastWAM 的 3D-Aware Video World Model

## 1. 动机

FastWAM 当前使用一个 2D video/world branch 为 action generation 提供视觉上下文。在 RoboTwin 推理中，三路相机会被 resize 后拼成一张 RGB 图像，然后经过 Wan Video VAE 编码成 video latent，再经过 Wan 2.2 video DiT / video expert 转成 video tokens，最后 action expert 通过 MoT attention 读取这些 video tokens 来生成 action chunk。

这个设计推理效率很高，而且 Wan 2.2 video DiT 本身具备学习未来 video latent dynamics 的能力。但它默认学习的是 2D VAE latent 空间中的未来演变，不一定显式包含机器人操作需要的 3D 信息，例如：

- 深度和物体空间关系；
- 多视角一致性；
- 遮挡情况下的 object permanence；
- 机械手和物体之间的接触几何；
- 可开合物体、关节物体的结构；
- 多物体场景中的 object identity。

因此目标是：**让 FastWAM 的 video/world branch 在保持未来视频预测能力的同时，学习更 3D-aware 的 world representation，从而更好地辅助 action generation。**

## 2. 核心想法

使用三层 teacher/student 栈，把 foundation teachers 的几何和语义信号先蒸馏进 Feature4X-style Gaussian feature field，再把这个 Gaussian teacher 蒸馏到 FastWAM 的 video/world tokens 中。

这里的核心分工是：

```text
Layer 1 foundation teachers：VGGT-Omega / CLIP / future DINO or SAM-like features
Layer 2 Gaussian field teacher：把 foundation signals 融合成 view-consistent 3D/semantic feature_z
Layer 3 FastWAM student：学习 cached Gaussian teacher representation，推理时只保留 2D FastWAM
Wan 2.2 video DiT：继续负责学习 future latent dynamics
```

也就是说，第一版不显式建模 Gaussian velocity 或 motion field。我们不额外让 Gaussian 预测速度，而是让 3D teacher 提供每个时间步的 3D state target，让 Wan video DiT 继续通过原本的 video latent prediction 学习动态演变。

推理时不运行 3D branch，部署路径仍然保持原始 FastWAM 的低延迟形式：

```text
image + instruction + proprio
    -> Wan VAE / video tokens
    -> MoT / action expert
    -> action chunk
```

训练时额外使用较重的 foundation-teacher-to-Gaussian-teacher 路径：

```text
multi-view images / video
    -> VGGT-Omega camera / depth / confidence / register
    -> CLIP or future DINO/SAM-like dense semantic features
    -> distill / fit Gaussian feature field feature_z
    -> camera-aware rendered Gaussian feature / depth / alpha / valid masks
    -> cached Gaussian teacher targets
    -> distillation targets for FastWAM video/world tokens
```

最终希望形成：

```text
训练阶段: 2D FastWAM student + 3D teacher branch
推理阶段: 只保留 2D FastWAM student
```

也就是：**训练时借助 3D teacher，推理时不增加额外 latency。**

## 3. 和 FastWAM 架构的衔接

当前 FastWAM 的 RoboTwin 推理路径大致是：

```text
RoboTwin observation
    -> 拼接 head / left / right cameras 成一张图
    -> VAE encode image
    -> first_frame_latents
    -> video_expert.pre_dit(...)
    -> video tokens / video KV cache
    -> action_expert denoising with MoT attention
    -> action chunk
```

action expert 不是直接看原始图像，而是通过 MoT 读取 video/world tokens。因此，最有价值的 3D 蒸馏位置应该是：**action branch 实际能看到的 video/world tokens。**

训练目标是在保留 FastWAM 原始 video/action loss 的基础上，额外对 video/world branch 加 3D state distillation loss。

## 4. Teacher Branch 设计

Teacher branch 更准确地分成 foundation teachers 和中间 Gaussian teacher 两层。

### 4.1 VGGT-Omega Teacher

VGGT-Omega 提供强 3D prior，主要包括：

- camera pose / intrinsics；
- depth 或 point map；
- dense local visual-geometric tokens；
- global / register tokens；
- multi-view geometry consistency。

其中 register token 不应该作为唯一 teacher signal。它更适合表示全局场景 layout、相机和整体几何关系；但机器人操作需要局部细节，所以还需要 dense tokens、depth、point map 等局部监督。

### 4.2 Gaussian Feature Field Teacher

参考 Feature4X 的思路，每个 Gaussian 不只存几何和颜色，还存一个 compact latent feature。这个 Gaussian field 是 Layer 2 teacher：它先被 VGGT/CLIP 等 foundation teachers 训练出来，再作为 FastWAM Stage 2 的 teacher。

```text
g_i = {
  xyz,
  scale,
  rotation,
  opacity,
  color / SH,
  feature_z
}
```

`feature_z` 可以设为比较小的维度，例如 32 / 64 / 128。Gaussian renderer 可以渲染出：

```text
rendered RGB
rendered depth
rendered alpha
rendered feature_map
```

然后使用 decoder heads 把 compact rendered feature 映射到不同 teacher feature space：

```text
D_vae(feature_map)   -> Wan VAE token space
D_vggt(feature_map)  -> VGGT dense feature space
D_sem(feature_map)   -> SAM / DINO / object feature space
D_reg(pool(feature)) -> VGGT register / global space
```

Feature4X 的核心思想是：**Gaussian 里存 compact feature，渲染到 2D view 后，再通过 head decode 到目标 2D feature space。**

这里可以把 Feature4X 原来对齐的 SAM2 / InternVideo / LangSeg 特征，替换或扩展成：

- Wan VAE token；
- VGGT dense feature；
- VGGT register token；
- depth / point feature；
- object / semantic feature。

### 4.3 可选 Semantic / Object Teacher

对于机器人操作，object identity 和 object boundary 很重要。因此可以考虑额外引入：

- SAM / SAM2 mask 或 feature；
- DINO feature；
- object-centric feature；
- language-grounded feature。

这些特征可能对以下任务特别有帮助：

- `blocks_ranking_rgb`；
- `blocks_ranking_size`；
- `pick_diverse_bottles`；
- `hanging_mug`；
- 多物体 pick/place 任务。

## 5. Student Branch 设计

Student 就是 FastWAM 本身。

Student path 保持原样：

```text
image
    -> Wan VAE latent
    -> Wan 2.2 video DiT / video expert
    -> video tokens
    -> MoT / action expert
```

训练时，从 FastWAM 中抽取 video/world tokens，并通过 projection heads 与 3D teacher tokens 对齐：

```text
S = FastWAM video tokens
T = rendered Gaussian / VGGT teacher tokens

P_s(S) ~= P_t(T)
```

其中：

- `P_s` 是 student projection head；
- `P_t` 是 teacher projection head；
- teacher features 应该 stop-gradient；
- 主要训练 student video/world branch，让它吸收 teacher 的 3D state representation。

## 6. 当前与未来 3D State Distillation

第一版不显式建模 velocity。更清晰的设定是：

```text
3D teacher 蒸馏 state；
Wan video DiT 学 transition。
```

因此，3D 蒸馏分成两个部分。

### 6.1 Current 3D State Distillation

让当前 video/world tokens 对齐当前帧的 3D teacher：

```text
S_t ~= T_3D_t
```

这里 `T_3D_t` 可以来自：

- VGGT dense tokens；
- VGGT register tokens；
- rendered Gaussian feature map；
- depth / point map；
- object / semantic feature。

这个 loss 让当前 video representation 更懂当前 3D scene。

### 6.2 Future 3D State Distillation

FastWAM 的 video branch 本来就学习未来 video latent。为了让它预测出来的未来 world representation 也具备 3D consistency，可以对未来时间步也加 3D teacher supervision：

```text
S_{t+1} ~= T_3D_{t+1}
S_{t+2} ~= T_3D_{t+2}
...
S_{t+K} ~= T_3D_{t+K}
```

其中 `T_3D_{t+k}` 由真实未来帧离线跑 VGGT-Omega / Gaussian teacher 得到。

这样做的含义是：

```text
video branch 仍然通过 Wan DiT 学 latent dynamics；
3D teacher 只约束每个时间步的 predicted state 应该是 3D-consistent。
```

也就是说，不直接监督速度，而是监督未来每个状态本身。只要模型在多个时间步都对齐了正确的 3D state，动态变化关系就由 video latent dynamics 自己学习。

### 6.3 时间对齐

FastWAM 的 VAE 可能有 temporal downsample，因此真实视频帧和 VAE latent step 不一定一一对应。例如：

```text
原始帧:      I0  I1  I2  I3  I4  I5  I6  I7  I8
VAE latent:  z0          z1          z2
```

因此 teacher 也需要对齐到 latent 时间尺度。第一版可以采用简单策略：

```text
每个 latent step 对齐其覆盖窗口中的最后一帧 3D teacher
```

例如：

```text
z1 对齐 I4 的 3D teacher
z2 对齐 I8 的 3D teacher
```

后续可以改成 temporal pooling。

## 7. Loss 设计

整体 loss 可以写成：

```text
L_total = L_video
        + L_action
        + lambda_now    * L_3d_current
        + lambda_future * L_3d_future
        + lambda_reg    * L_reg_3d
        + lambda_depth  * L_depth
        + lambda_obj    * L_object
```

### 7.1 原始 FastWAM Loss

```text
L_video  = 原始 future video latent denoising / reconstruction loss
L_action = 原始 action denoising loss
```

这两个 loss 用来保持 FastWAM 原本的未来预测和 action 生成能力。

### 7.2 Current / Future Dense 3D Feature Distillation

让 FastWAM 的 spatial video tokens 对齐 rendered Gaussian / VGGT dense tokens：

```text
L_3d_current = cosine_loss(P_s(S_t), stopgrad(P_t(T_3D_t)))

L_3d_future = sum_k cosine_loss(
    P_s(S_{t+k}),
    stopgrad(P_t(T_3D_{t+k}))
)
```

这个 loss 主要希望迁移：

- 局部 3D geometry；
- 物体位置；
- 手和物体的空间关系；
- 遮挡后的 object consistency；
- 未来状态的 3D consistency。

### 7.3 Global / Register Distillation

让 pooled student video tokens 对齐 VGGT register / global tokens：

```text
L_reg_3d = cosine_loss(pool(P_s(S)), stopgrad(P_reg(T_register)))
```

这个 loss 主要希望迁移：

- 全局 scene layout；
- camera / geometry prior；
- 多视角整体一致性。

### 7.4 Depth / Point Distillation

从 student video tokens 预测 depth 或 point 信息：

```text
depth_pred = DepthHead(S_spatial)
L_depth = |depth_pred - stopgrad(depth_vggt)|
```

这个 loss 鼓励 video branch 内部显式学习 3D geometry。

### 7.5 Object / Semantic Distillation

使用 object 或 semantic feature 作为辅助监督：

```text
obj_pred = ObjectHead(S_spatial)
L_object = cosine_loss(obj_pred, stopgrad(object_teacher_feature))
```

这个 loss 对多物体场景和 language-conditioned manipulation 可能很重要。

## 8. 为什么不显式建模 Velocity

第一版建议直接去掉 velocity / motion field，不把它作为主线设计。

原因是：

- FastWAM 的 Wan 2.2 video DiT 本来就在学习 future video latent dynamics；
- 显式 per-Gaussian velocity 需要跨时间 Gaussian correspondence，工程上不稳定；
- object-level velocity 又需要额外 object tracking / rigid assignment；
- velocity loss 可能和原本的 video latent denoising objective 重复或冲突；
- 当前目标是低延迟部署，训练期蒸馏 3D state 更直接、更稳。

因此主线设定为：

```text
不显式监督速度；
只蒸馏当前和未来的 3D state；
动态演变交给 Wan 2.2 video DiT 的 latent dynamics 学。
```

如果后续发现 contact-rich 或 articulated object 任务仍然明显不足，再考虑把 motion / velocity 作为 optional extension。

## 9. Action-Relevant Weighting

对于机器人操作，均匀地对整张图做蒸馏不一定最有效。3D distillation loss 可以对 action-relevant 区域加权：

- gripper / hand 附近区域；
- 目标物体区域；
- 接触区域；
- object boundary；
- depth discontinuity；
- 根据语言指令定位到的 task-relevant object。

这样可以让蒸馏进来的 3D 信息更直接服务于 action generation。

## 10. 训练流程建议

### Stage 1: Foundation Teachers -> Gaussian Field Teacher

先离线运行 foundation teachers，并把它们的输出蒸馏/拟合进 Gaussian feature field：

```text
images / multi-view frames
    -> VGGT-Omega camera / depth / confidence / register
    -> CLIP/DINO/SAM-like dense feature maps
    -> optimize per-sample Gaussian xyz / scale / opacity / feature_z
    -> camera-aware render T_gaussian_feature / T_depth / T_alpha / valid masks
    -> save Gaussian teacher cache
```

这样训练 FastWAM 时不需要每个 step 都跑重型 3D branch，可以显著降低训练成本。旧的 broadcast-feature cache 只能作为 pipeline smoke test；正式 Stage 2 应使用 dense Feature4X-style Gaussian teacher cache。

### Stage 2: Current 3D State Distillation

第一阶段不要大改 FastWAM。可以先 freeze 或 mostly preserve 原始 FastWAM，只在当前 video/world tokens 上加 3D state distillation loss。

训练目标：

```text
L_video + L_action + small lambda_now * L_3d_current
```

`lambda_now` 建议从比较小的值开始，避免破坏原本已经学好的 video/action 能力。

### Stage 3: Future 3D State Distillation

在 current 3D state distillation 稳定后，再加入未来时间步的 3D state distillation：

```text
L_video + L_action + lambda_now * L_3d_current + lambda_future * L_3d_future
```

这一步的目标是让 FastWAM 预测出来的未来 video/world tokens 也保持 3D-consistent。

### Stage 4: Action-Aware Fine-Tuning

如果 video tokens 已经变得更 3D-aware，但 action success rate 提升不明显，可以进一步加入 action-aware 的约束：

- 对 gripper / object / contact 区域加大 3D loss 权重；
- 在 bad-case task 上 targeted fine-tune；
- 观察是否需要让 action tokens 也对齐 3D-aware video representation。

重点任务可以包括：

- handover：`handover_block`, `handover_mic`；
- multi-object reasoning：`blocks_ranking_rgb`, `blocks_ranking_size`；
- contact-rich pushing：`move_stapler_pad`；
- articulated object：`open_microwave`, `open_laptop`；
- diverse grasping：`pick_diverse_bottles`, `hanging_mug`。

### Stage 5: Evaluation

评测时重点看 3D state distillation 是否改善之前分析出的弱项：

- 双臂交接；
- 多物体排序；
- 推移和接触控制；
- 关节物体操作；
- 遮挡、多样物体、复杂抓取。

## 11. 推理阶段

部署时保持低延迟路径：

```text
observation
    -> FastWAM image preprocessing
    -> Wan VAE encode
    -> video tokens
    -> action expert
    -> action
```

推理时不运行：

- VGGT-Omega；
- Gaussian construction；
- Gaussian renderer；
- teacher projection heads；
- velocity / motion branch。

目标是：**训练时让 video branch 内化 3D state representation，推理时仍然只用原来的 FastWAM 路径。**

## 12. 为什么可能有效

之前的 bad-case 分析显示，FastWAM 在简单 click、lift、单物体操作上已经比较强，但在以下任务上比较弱：

- 准确的 3D hand-object 关系；
- 多物体 identity 和 ordering；
- contact-rich dynamics；
- articulated object geometry；
- 遮挡和 object permanence；
- diverse object grasp pose。

如果 video/world branch 通过 3D teacher 学到了更好的几何、物体和多视角一致性，action branch 通过 MoT attention 读取到的 world representation 就会更强，因此可能提升这些任务上的 action generation。

这里不需要第一版显式建 velocity，因为 Wan video DiT 已经负责学习 latent dynamics。3D teacher 的作用是让每个时间步的 latent state 更接近真实 3D world state。

## 13. 关键设计选择

核心设计不是部署一个重型 3D branch，也不是重新设计一个显式 3D dynamics model，而是：

```text
Train with 3D state teacher.
Deploy 2D FastWAM student.
Let Wan video DiT learn the dynamics.
```

这样可以同时满足：

- 训练时利用 VGGT-Omega 的强 3D 能力；
- 利用 Gaussian feature field 提供 view-consistent 3D state；
- 推理时不增加额外 latency；
- 保持 FastWAM 原始部署路径；
- 让 video branch 变成更 3D-aware 的 world representation；
- 避免显式 velocity / motion field 带来的 correspondence 和 tracking 复杂度。

## 14. 待思考问题

1. 3D 蒸馏应该加在哪一层？
   - VAE latent；
   - video expert input tokens；
   - video expert intermediate tokens；
   - MoT video output tokens。

2. 哪种 teacher target 对 action 最有帮助？
   - VGGT dense tokens；
   - VGGT register tokens；
   - rendered Gaussian feature maps；
   - depth / point maps；
   - object / semantic features。

3. Future 3D state teacher 如何和 VAE latent temporal step 对齐？
   - 使用窗口最后一帧；
   - 使用 temporal pooling；
   - 使用 action-relevant frame selection。

4. Wan VAE 是否需要保持 frozen？
   - Frozen 更安全，兼容原来的 latent space；
   - fine-tune VAE encoder 可能塞入更多 3D 信息，但有破坏原 video latent space 的风险。

5. 蒸馏应该全图做，还是 action-region weighted？
   - 全图更简单；
   - action-region weighted 可能对 manipulation 更有效。

6. teacher 应该基于拼接后的 FastWAM image，还是原始多视角图像？
   - 原始多视角图像更适合 3D；
   - 拼接图像更接近 FastWAM 当前输入。

7. Gaussian feature field 是只作为 teacher 离线预计算，还是训练时端到端更新？
   - 离线预计算更省；
   - 端到端更新可能更强，但训练复杂度更高。

## 15. 简短总结

GaussianWAM 的核心是三层蒸馏：先用 VGGT-Omega、CLIP 和后续可选 DINO/SAM-like foundation teachers 训练每个样本的 Gaussian feature field teacher，再把 cached `T_gaussian_feature` 蒸馏到 FastWAM 的 video/world tokens 中。第一版不显式建模 velocity / motion field，而是让 Wan 2.2 video DiT 继续负责 future latent dynamics。推理时保持原 FastWAM 路径不变，从而在不增加推理 latency 的情况下，让 video branch 更具备 3D 世界理解能力，并更好地辅助 action generation。


REPA:       提供 representation alignment / projector / cosine distillation 的训练范式
Feature4X:  提供 Gaussian feature field 的表示和 feature rendering 思路
VGGT-Omega: 提供 3D geometry prior：camera / depth / register / dense 3D tokens
FastWAM:    作为 student，把这些 3D-aware teacher signal 蒸馏进 video/world tokens
