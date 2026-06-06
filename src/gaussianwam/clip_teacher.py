from __future__ import annotations

from typing import Any
import math

import torch
import torch.nn.functional as F


def _fit_feature_dim(feat: torch.Tensor, feature_dim: int) -> torch.Tensor:
    repeats = (int(feature_dim) + feat.shape[-1] - 1) // feat.shape[-1]
    return feat.repeat_interleave(repeats, dim=-1)[..., : int(feature_dim)]


class ClipSemanticTeacher:
    def __init__(self, cfg, device: str | torch.device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.enabled = bool(cfg.get("enabled", True))
        self.feature_dim = int(cfg.get("feature_dim", 512))
        self.mode = "fallback"
        self.model = None
        self.processor = None
        if not self.enabled:
            return
        try:
            from transformers import CLIPImageProcessor, CLIPVisionModel
            model_name = str(cfg.get("model_name", "openai/clip-vit-base-patch16"))
            self.processor = CLIPImageProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = CLIPVisionModel.from_pretrained(model_name, local_files_only=True).to(self.device).eval()
            self.feature_dim = int(self.model.config.hidden_size)
            self.mode = "transformers_clip"
        except Exception:
            self.mode = str(cfg.get("fallback", "image_stats"))

    @torch.no_grad()
    def encode_views(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if not self.enabled:
            return torch.zeros(images.shape[0], self.feature_dim, device=self.device, dtype=torch.float32)
        if self.mode == "transformers_clip" and self.model is not None and self.processor is not None:
            imgs = images.detach().cpu().permute(0, 2, 3, 1).numpy()
            inputs = self.processor(images=list(imgs), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            out = self.model(pixel_values=pixel_values)
            return out.pooler_output.float()
        mean = images.mean(dim=(2, 3))
        std = images.std(dim=(2, 3), unbiased=False)
        feat = torch.cat([mean, std, mean * std, mean.square()], dim=-1)
        feat = _fit_feature_dim(feat, self.feature_dim)
        return feat.float()

    def _dense_fallback(
        self,
        images: torch.Tensor,
        grid_size: tuple[int, int],
        depth: torch.Tensor | None,
        confidence: torch.Tensor | None,
        feature_dim: int,
    ) -> torch.Tensor:
        images_grid = F.interpolate(images, size=grid_size, mode="bilinear", align_corners=False)
        views, _, h, w = images_grid.shape
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=images.device, dtype=images.dtype),
            torch.linspace(-1.0, 1.0, w, device=images.device, dtype=images.dtype),
            indexing="ij",
        )
        xy = torch.stack([xs, ys], dim=0)[None].expand(views, -1, -1, -1)
        chunks = [images_grid, xy]
        if depth is not None:
            dep = depth.to(device=images.device, dtype=images.dtype)
            dep = dep[:, None] if dep.ndim == 3 else dep
            dep_min = dep.amin(dim=(2, 3), keepdim=True)
            dep_max = dep.amax(dim=(2, 3), keepdim=True)
            chunks.append((dep - dep_min) / (dep_max - dep_min).clamp_min(1e-6))
        if confidence is not None:
            conf = confidence.to(device=images.device, dtype=images.dtype)
            conf = conf[:, None] if conf.ndim == 3 else conf
            conf_min = conf.amin(dim=(2, 3), keepdim=True)
            conf_max = conf.amax(dim=(2, 3), keepdim=True)
            chunks.append((conf - conf_min) / (conf_max - conf_min).clamp_min(1e-6))
        mean = images_grid.mean(dim=(2, 3), keepdim=True).expand(-1, -1, h, w)
        std = images_grid.std(dim=(2, 3), unbiased=False, keepdim=True).expand(-1, -1, h, w)
        chunks.extend([mean, std, images_grid.square()])
        feat = torch.cat(chunks, dim=1).permute(0, 2, 3, 1).contiguous()
        feat = _fit_feature_dim(feat, feature_dim)
        return F.normalize(feat.float(), dim=-1)

    @torch.no_grad()
    def encode_dense_views(
        self,
        images: torch.Tensor,
        grid_size: tuple[int, int],
        *,
        depth: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        feature_dim: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        images = images.to(self.device)
        feature_dim = int(feature_dim or self.feature_dim)
        dense_enabled = bool(self.cfg.get("dense_enabled", True))
        meta: dict[str, Any] = {"dense_enabled": dense_enabled, "dense_feature_dim": feature_dim}
        if not self.enabled or not dense_enabled:
            feat = self._dense_fallback(images, grid_size, depth, confidence, feature_dim)
            return feat, {**meta, "dense_mode": "image_stats_dense"}
        if self.mode == "transformers_clip" and self.model is not None and self.processor is not None:
            imgs = images.detach().cpu().permute(0, 2, 3, 1).numpy()
            inputs = self.processor(images=list(imgs), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            out = self.model(pixel_values=pixel_values)
            patch = out.last_hidden_state[:, 1:, :].float()
            side = int(math.sqrt(int(patch.shape[1])))
            if side * side == int(patch.shape[1]):
                patch = patch.reshape(patch.shape[0], side, side, patch.shape[-1]).permute(0, 3, 1, 2)
                patch = F.interpolate(patch, size=grid_size, mode="bilinear", align_corners=False)
                feat = patch.permute(0, 2, 3, 1).contiguous()
                feat = _fit_feature_dim(feat, feature_dim)
                return F.normalize(feat.float(), dim=-1), {**meta, "dense_mode": "clip_patch_tokens", "clip_patch_grid": [side, side]}
        feat = self._dense_fallback(images, grid_size, depth, confidence, feature_dim)
        return feat, {**meta, "dense_mode": "image_stats_dense"}
