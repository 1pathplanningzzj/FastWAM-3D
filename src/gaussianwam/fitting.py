from __future__ import annotations

import torch
import torch.nn.functional as F

from .gaussian_field import GaussianFeatureField
from .renderer import SoftPointRenderer


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
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    params = [
        {"params": [field.xyz], "lr": float(cfg.lr_xyz)},
        {"params": [field.log_scale], "lr": float(cfg.lr_scale)},
        {"params": [field.opacity_logit], "lr": float(cfg.lr_opacity)},
        {"params": [field.feature_z], "lr": float(cfg.lr_feature)},
    ]
    opt = torch.optim.Adam(params)
    steps = int(cfg.optimize_steps)
    views = int(target_depth.shape[0])
    metrics = {"initial_loss": 0.0, "final_loss": 0.0}
    for step in range(max(steps, 1)):
        opt.zero_grad(set_to_none=True)
        render = renderer.render(field, intrinsics=intrinsics, extrinsics=extrinsics)
        assert render["dep"].shape == target_depth.shape
        assert render["feature_map"].shape == target_feature.shape
        assert render["alpha"].shape == valid_mask.shape
        mask = valid_mask & torch.isfinite(target_depth) & torch.isfinite(target_feature).all(dim=-1)
        if mask.any():
            loss_depth = (render["dep"][mask] - target_depth[mask]).abs().mean()
            pred_feat = F.normalize(render["feature_map"], dim=-1)
            tgt_feat = F.normalize(target_feature, dim=-1)
            loss_feat = -(pred_feat[mask] * tgt_feat[mask]).sum(dim=-1).mean()
            loss_alpha = (1.0 - render["alpha"][mask].clamp(0, 1)).abs().mean()
        else:
            loss_depth = render["dep"].mean() * 0.0
            loss_feat = render["feature_map"].mean() * 0.0
            loss_alpha = render["alpha"].mean() * 0.0
        loss = float(loss_cfg.lambda_depth) * loss_depth + float(loss_cfg.lambda_clip) * loss_feat + float(loss_cfg.lambda_alpha) * loss_alpha
        if step == 0:
            metrics["initial_loss"] = float(loss.detach().item())
        loss.backward()
        opt.step()
        metrics["final_loss"] = float(loss.detach().item())
    render = renderer.render(field, intrinsics=intrinsics, extrinsics=extrinsics)
    render_valid_ratio = float((render["alpha"] > 1e-4).float().mean().item())
    metrics["valid_ratio"] = render_valid_ratio
    metrics["render_valid_ratio"] = render_valid_ratio
    metrics["teacher_valid_ratio"] = float(valid_mask.float().mean().item())
    return render, metrics
