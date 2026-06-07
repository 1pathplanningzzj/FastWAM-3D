from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFeatureField(nn.Module):
    def __init__(
        self,
        xyz: torch.Tensor,
        color: torch.Tensor,
        feature: torch.Tensor,
        init_scale: float,
        init_opacity: float,
        *,
        optimize_feature: bool = True,
        optimize_color: bool = False,
    ):
        super().__init__()
        self.xyz = nn.Parameter(xyz.float())
        self.color = nn.Parameter(color.float().clamp(0.0, 1.0), requires_grad=bool(optimize_color))
        self.feature_z = nn.Parameter(feature.float(), requires_grad=bool(optimize_feature))
        self.register_buffer("feature_z_init", feature.float().clone())
        self.register_buffer("xyz_init", xyz.float().clone())
        self.log_scale = nn.Parameter(torch.full((xyz.shape[0], 1), float(init_scale), device=xyz.device).log())
        init_opacity = min(max(float(init_opacity), 1e-4), 1 - 1e-4)
        self.opacity_logit = nn.Parameter(torch.full((xyz.shape[0], 1), init_opacity, device=xyz.device).logit())

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp_min(1e-6)

    @property
    def opacity(self) -> torch.Tensor:
        return self.opacity_logit.sigmoid()

    @torch.no_grad()
    def clamp_parameters(self, min_scale: float = 1e-4, max_scale: float = 0.2, min_opacity: float = 1e-4, max_opacity: float = 0.999) -> None:
        scale_bounds = torch.tensor([float(min_scale), float(max_scale)], device=self.log_scale.device).log()
        opacity_bounds = torch.tensor([float(min_opacity), float(max_opacity)], device=self.opacity_logit.device).logit()
        self.log_scale.clamp_(float(scale_bounds[0].item()), float(scale_bounds[1].item()))
        self.opacity_logit.clamp_(float(opacity_bounds[0].item()), float(opacity_bounds[1].item()))
        self.color.clamp_(0.0, 1.0)
        if not self.feature_z.requires_grad:
            self.feature_z.copy_(self.feature_z_init)

    def feature_anchor_loss(self) -> torch.Tensor:
        return 1.0 - (F.normalize(self.feature_z, dim=-1) * F.normalize(self.feature_z_init, dim=-1)).sum(dim=-1).mean()

    def xyz_drift_loss(self) -> torch.Tensor:
        return (self.xyz - self.xyz_init).square().sum(dim=-1).mean()

    def state_payload(self) -> dict[str, torch.Tensor]:
        return {
            "xyz": self.xyz.detach().cpu(),
            "scale": self.scale.detach().cpu(),
            "opacity": self.opacity.detach().cpu(),
            "color": self.color.detach().cpu(),
            "feature_z": self.feature_z.detach().cpu(),
            "feature_z_requires_grad": torch.tensor(bool(self.feature_z.requires_grad)),
        }


def build_feature_vectors(xyz: torch.Tensor, clip_feature: torch.Tensor, feature_dim: int) -> torch.Tensor:
    geom = torch.cat([xyz, xyz.norm(dim=-1, keepdim=True)], dim=-1)
    clip = F.normalize(clip_feature.float(), dim=-1)
    if clip.ndim == 2:
        clip = clip.mean(dim=0, keepdim=True).expand(xyz.shape[0], -1)
    feat = torch.cat([geom, clip], dim=-1)
    repeats = (int(feature_dim) + feat.shape[-1] - 1) // feat.shape[-1]
    return feat.repeat(1, repeats)[:, : int(feature_dim)].contiguous()
