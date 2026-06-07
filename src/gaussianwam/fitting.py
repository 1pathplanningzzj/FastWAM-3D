from __future__ import annotations

import torch
import torch.nn.functional as F

from .gaussian_field import GaussianFeatureField
from .renderer import SoftPointRenderer


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return x[mask].mean()
    return x.mean() * 0.0


def _feature_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_feat = F.normalize(pred, dim=-1)
    tgt_feat = F.normalize(target, dim=-1)
    cos = (pred_feat * tgt_feat).sum(dim=-1)
    return _masked_mean(cos, mask)


def fit_gaussian_field(
    field: GaussianFeatureField,
    renderer: SoftPointRenderer,
    target_depth: torch.Tensor,
    target_feature: torch.Tensor,
    valid_mask: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    cfg,
    loss_cfg,
) -> tuple[dict[str, torch.Tensor], dict[str, float | bool | str]]:
    params = []
    if bool(cfg.get("optimize_xyz", True)):
        params.append({"params": [field.xyz], "lr": float(cfg.lr_xyz), "name": "xyz"})
    if bool(cfg.get("optimize_scale", True)):
        params.append({"params": [field.log_scale], "lr": float(cfg.lr_scale), "name": "scale"})
    if bool(cfg.get("optimize_opacity", True)):
        params.append({"params": [field.opacity_logit], "lr": float(cfg.lr_opacity), "name": "opacity"})
    if bool(cfg.get("optimize_feature", bool(field.feature_z.requires_grad))) and field.feature_z.requires_grad:
        params.append({"params": [field.feature_z], "lr": float(cfg.lr_feature), "name": "feature"})
    if bool(cfg.get("optimize_color", False)) and field.color.requires_grad:
        params.append({"params": [field.color], "lr": float(cfg.get("lr_color", cfg.lr_feature)), "name": "color"})
    if not params:
        raise ValueError("No Gaussian parameters selected for optimization")

    opt = torch.optim.Adam(params)
    steps = int(cfg.optimize_steps)
    metrics: dict[str, float | bool | str] = {
        "initial_loss": 0.0,
        "final_loss": 0.0,
        "initial_compact_cosine": 0.0,
        "final_compact_cosine": 0.0,
        "quality_ok": True,
        "rejection_reason": "",
    }
    alpha_threshold = float(cfg.get("alpha_valid_threshold", 1e-4))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    min_scale = float(cfg.get("min_scale", 1e-4))
    max_scale = float(cfg.get("max_scale", 0.2))
    lambda_compact = float(loss_cfg.get("lambda_compact", loss_cfg.get("lambda_clip", 0.1)))
    lambda_depth = float(loss_cfg.get("lambda_depth", 1.0))
    lambda_alpha = float(loss_cfg.get("lambda_alpha", 0.01))
    lambda_alpha_outside = float(loss_cfg.get("lambda_alpha_outside", 0.01))
    lambda_scale = float(loss_cfg.get("lambda_scale", 0.0))
    lambda_xyz = float(loss_cfg.get("lambda_xyz", 0.0))
    lambda_anchor = float(loss_cfg.get("lambda_feature_anchor", 0.0))

    for step in range(max(steps, 1)):
        opt.zero_grad(set_to_none=True)
        render = renderer.render(field, intrinsics=intrinsics, extrinsics=extrinsics)
        if render["dep"].shape != target_depth.shape:
            raise ValueError(f"Rendered depth shape {tuple(render['dep'].shape)} does not match target {tuple(target_depth.shape)}")
        if render["feature_map"].shape != target_feature.shape:
            raise ValueError(f"Rendered feature shape {tuple(render['feature_map'].shape)} does not match target {tuple(target_feature.shape)}")
        if render["alpha"].shape != valid_mask.shape:
            raise ValueError(f"Rendered alpha shape {tuple(render['alpha'].shape)} does not match mask {tuple(valid_mask.shape)}")
        render_mask = render["alpha"] > alpha_threshold
        finite_mask = torch.isfinite(target_depth) & torch.isfinite(target_feature).all(dim=-1) & render.get("finite_mask", torch.ones_like(valid_mask, dtype=torch.bool))
        teacher_mask = valid_mask & finite_mask
        overlap_mask = teacher_mask & render_mask
        loss_depth = _masked_mean((render["dep"] - target_depth).abs(), teacher_mask)
        compact_cos = _feature_cosine(render["feature_map"], target_feature, teacher_mask)
        loss_feat = 1.0 - compact_cos
        loss_alpha = _masked_mean((1.0 - render["alpha"].clamp(0, 1)).abs(), teacher_mask)
        outside = (~valid_mask) & torch.isfinite(render["alpha"])
        loss_alpha_outside = _masked_mean(render["alpha"].clamp(0, 1).square(), outside)
        loss_scale = field.scale.square().mean()
        loss_xyz = field.xyz_drift_loss()
        loss_anchor = field.feature_anchor_loss() if field.feature_z.requires_grad else loss_scale * 0.0
        loss = (
            lambda_depth * loss_depth
            + lambda_compact * loss_feat
            + lambda_alpha * loss_alpha
            + lambda_alpha_outside * loss_alpha_outside
            + lambda_scale * loss_scale
            + lambda_xyz * loss_xyz
            + lambda_anchor * loss_anchor
        )
        if step == 0:
            metrics["initial_loss"] = float(loss.detach().item())
            metrics["initial_compact_cosine"] = float(compact_cos.detach().item())
        if not torch.isfinite(loss):
            metrics["quality_ok"] = False
            metrics["rejection_reason"] = f"non-finite loss at step {step}"
            break
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for group in params for p in group["params"]], grad_clip)
        opt.step()
        field.clamp_parameters(min_scale=min_scale, max_scale=max_scale)
        metrics["final_loss"] = float(loss.detach().item())
        metrics["final_compact_cosine"] = float(compact_cos.detach().item())
        metrics["loss_depth"] = float(loss_depth.detach().item())
        metrics["loss_compact"] = float(loss_feat.detach().item())
        metrics["loss_alpha"] = float(loss_alpha.detach().item())
        metrics["loss_alpha_outside"] = float(loss_alpha_outside.detach().item())
        metrics["loss_scale"] = float(loss_scale.detach().item())
        metrics["loss_xyz"] = float(loss_xyz.detach().item())
        metrics["loss_feature_anchor"] = float(loss_anchor.detach().item())
        metrics["teacher_render_overlap_ratio"] = float(overlap_mask.float().mean().detach().item())

    render = renderer.render(field, intrinsics=intrinsics, extrinsics=extrinsics)
    render_valid_ratio = float((render["alpha"] > alpha_threshold).float().mean().item())
    teacher_valid_ratio = float(valid_mask.float().mean().item())
    final_teacher_mask = valid_mask & torch.isfinite(target_depth) & torch.isfinite(target_feature).all(dim=-1)
    final_overlap_mask = final_teacher_mask & (render["alpha"] > alpha_threshold)
    final_cos = _feature_cosine(render["feature_map"], target_feature, final_teacher_mask)
    final_depth = _masked_mean((render["dep"] - target_depth).abs(), final_teacher_mask)
    metrics.update(
        {
            "valid_ratio": render_valid_ratio,
            "render_valid_ratio": render_valid_ratio,
            "teacher_valid_ratio": teacher_valid_ratio,
            "final_compact_cosine": float(final_cos.detach().item()),
            "final_depth_error": float(final_depth.detach().item()),
            "feature_z_optimized": bool(field.feature_z.requires_grad),
            "teacher_render_overlap_ratio": float(final_overlap_mask.float().mean().item()) if final_overlap_mask.numel() else 0.0,
        }
    )
    min_render_valid_ratio = float(cfg.get("min_render_valid_ratio", 0.0))
    min_teacher_valid_ratio = float(cfg.get("min_teacher_valid_ratio", 0.0))
    min_compact_cosine = float(cfg.get("min_compact_cosine", -1.0))
    min_overlap_ratio = float(cfg.get("min_teacher_render_overlap_ratio", 0.0))
    max_cosine_drop = float(cfg.get("max_compact_cosine_drop", 1.0))
    reasons = []
    if render_valid_ratio < min_render_valid_ratio:
        reasons.append(f"render_valid_ratio {render_valid_ratio:.4f} < {min_render_valid_ratio:.4f}")
    if teacher_valid_ratio < min_teacher_valid_ratio:
        reasons.append(f"teacher_valid_ratio {teacher_valid_ratio:.4f} < {min_teacher_valid_ratio:.4f}")
    final_cos_value = float(final_cos.detach().item())
    initial_cos_value = float(metrics.get("initial_compact_cosine", final_cos_value))
    overlap_ratio = float(metrics.get("teacher_render_overlap_ratio", 0.0))
    if final_cos_value < min_compact_cosine:
        reasons.append(f"final_compact_cosine {final_cos_value:.4f} < {min_compact_cosine:.4f}")
    if initial_cos_value - final_cos_value > max_cosine_drop:
        reasons.append(f"compact cosine dropped by {initial_cos_value - final_cos_value:.4f} > {max_cosine_drop:.4f}")
    if overlap_ratio < min_overlap_ratio:
        reasons.append(f"teacher_render_overlap_ratio {overlap_ratio:.4f} < {min_overlap_ratio:.4f}")
    if reasons and bool(metrics.get("quality_ok", True)):
        metrics["quality_ok"] = False
        metrics["rejection_reason"] = "; ".join(reasons)
    return render, metrics
