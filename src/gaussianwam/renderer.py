from __future__ import annotations

import torch

from .gaussian_field import GaussianFeatureField


class SoftPointRenderer:
    def __init__(
        self,
        height: int,
        width: int,
        grid_size: tuple[int, int],
        *,
        radius_px: int = 2,
        sigma_min: float = 0.35,
        sigma_max: float = 2.0,
        alpha_eps: float = 1e-6,
        depth_weight: float = 1.0,
    ):
        self.height = int(height)
        self.width = int(width)
        self.grid_size = tuple(int(x) for x in grid_size)
        self.radius_px = int(radius_px)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.alpha_eps = float(alpha_eps)
        self.depth_weight = float(depth_weight)
        self.backend = "soft_point_depth_aware"

    def render(self, field: GaussianFeatureField, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = self.grid_size
        device = field.xyz.device
        dtype = field.xyz.dtype
        feat_dim = field.feature_z.shape[-1]
        xyz = field.xyz
        if intrinsics.ndim == 4:
            intrinsics = intrinsics[0]
        if extrinsics.ndim == 4:
            extrinsics = extrinsics[0]
        intrinsics = intrinsics.to(device=device, dtype=dtype)
        extrinsics = extrinsics.to(device=device, dtype=dtype)
        views = int(intrinsics.shape[0])
        if xyz.numel() == 0:
            zeros_feat = torch.zeros(views, h, w, feat_dim, device=device, dtype=dtype)
            zeros = torch.zeros(views, h, w, device=device, dtype=dtype)
            return {"feature_map": zeros_feat, "dep": zeros, "alpha": zeros, "weight_sum": zeros, "finite_mask": zeros.bool()}

        r = extrinsics[:, :3, :3]
        t = extrinsics[:, :3, 3]
        xyz_cam = torch.matmul(xyz[None], r.transpose(-1, -2)) + t[:, None]
        z = xyz_cam[..., 2]
        z_safe = z.clamp_min(1e-6)

        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        u = fx[:, None] * xyz_cam[..., 0] / z_safe + cx[:, None]
        v = fy[:, None] * xyz_cam[..., 1] / z_safe + cy[:, None]
        valid = torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(z) & (z > 1e-6)

        opacity = field.opacity.squeeze(-1).clamp_min(self.alpha_eps)
        scale = field.scale.squeeze(-1)
        focal = 0.5 * (fx.abs() + fy.abs())
        sigma = (0.5 + focal[:, None] * scale[None] / z_safe).clamp(self.sigma_min, self.sigma_max)
        ix0 = u.round().long()
        iy0 = v.round().long()
        base = torch.arange(views, device=device, dtype=torch.long)[:, None] * (h * w)

        denom = torch.zeros(views * h * w, device=device, dtype=dtype)
        depth_num = torch.zeros(views * h * w, device=device, dtype=dtype)
        feat_num = torch.zeros(views * h * w, feat_dim, device=device, dtype=dtype)

        depth_bias = torch.exp(-self.depth_weight * z_safe)
        radius = max(int(self.radius_px), 0)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ix = ix0 + dx
                iy = iy0 + dy
                in_bounds = valid & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
                dist2 = (ix.to(dtype) - u).square() + (iy.to(dtype) - v).square()
                spatial = torch.exp(-0.5 * dist2 / sigma.square().clamp_min(1e-6))
                weight = torch.where(in_bounds, opacity[None] * spatial * depth_bias, torch.zeros_like(spatial))
                flat = (base + iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1)).reshape(-1)
                weight_flat = weight.reshape(-1)
                denom = denom.scatter_add(0, flat, weight_flat)
                depth_num = depth_num.scatter_add(0, flat, (weight * z_safe).reshape(-1))
                feat_weight = (field.feature_z[None] * weight[..., None]).reshape(-1, feat_dim)
                feat_num = feat_num.scatter_add(0, flat[:, None].expand(-1, feat_dim), feat_weight)

        depth = depth_num / denom.clamp_min(self.alpha_eps)
        feat = feat_num / denom[:, None].clamp_min(self.alpha_eps)
        alpha = denom.clamp(0.0, 1.0)
        finite_mask = torch.isfinite(depth) & torch.isfinite(alpha) & torch.isfinite(feat).all(dim=-1)
        return {
            "feature_map": feat.reshape(views, h, w, feat_dim),
            "dep": depth.reshape(views, h, w),
            "alpha": alpha.reshape(views, h, w),
            "weight_sum": denom.reshape(views, h, w),
            "finite_mask": finite_mask.reshape(views, h, w),
        }
