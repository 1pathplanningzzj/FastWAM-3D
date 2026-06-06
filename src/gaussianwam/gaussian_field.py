from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFeatureField(nn.Module):
    def __init__(self, xyz: torch.Tensor, color: torch.Tensor, feature: torch.Tensor, init_scale: float, init_opacity: float):
        super().__init__()
        self.xyz = nn.Parameter(xyz.float())
        self.color = nn.Parameter(color.float().clamp(0.0, 1.0))
        self.feature_z = nn.Parameter(feature.float())
        self.log_scale = nn.Parameter(torch.full((xyz.shape[0], 1), float(init_scale), device=xyz.device).log())
        init_opacity = min(max(float(init_opacity), 1e-4), 1 - 1e-4)
        self.opacity_logit = nn.Parameter(torch.full((xyz.shape[0], 1), init_opacity, device=xyz.device).logit())

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp_min(1e-6)

    @property
    def opacity(self) -> torch.Tensor:
        return self.opacity_logit.sigmoid()

    def state_payload(self) -> dict[str, torch.Tensor]:
        return {
            "xyz": self.xyz.detach().cpu(),
            "scale": self.scale.detach().cpu(),
            "opacity": self.opacity.detach().cpu(),
            "color": self.color.detach().cpu(),
            "feature_z": self.feature_z.detach().cpu(),
        }


def build_feature_vectors(xyz: torch.Tensor, clip_feature: torch.Tensor, feature_dim: int) -> torch.Tensor:
    geom = torch.cat([xyz, xyz.norm(dim=-1, keepdim=True)], dim=-1)
    clip = F.normalize(clip_feature.float(), dim=-1)
    if clip.ndim == 2:
        clip = clip.mean(dim=0, keepdim=True).expand(xyz.shape[0], -1)
    feat = torch.cat([geom, clip], dim=-1)
    repeats = (int(feature_dim) + feat.shape[-1] - 1) // feat.shape[-1]
    return feat.repeat(1, repeats)[:, : int(feature_dim)].contiguous()
