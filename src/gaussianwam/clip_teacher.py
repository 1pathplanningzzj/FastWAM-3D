from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import torch
import torch.nn.functional as F

from .compact_feature import PCACompactProjector, file_sha256


def _fit_feature_dim(feat: torch.Tensor, feature_dim: int) -> torch.Tensor:
    repeats = (int(feature_dim) + feat.shape[-1] - 1) // feat.shape[-1]
    return feat.repeat_interleave(repeats, dim=-1)[..., : int(feature_dim)]


class ClipSemanticTeacher:
    def __init__(self, cfg, device: str | torch.device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.enabled = bool(cfg.get("enabled", True))
        self.require_model = bool(cfg.get("require_model", False))
        self.allow_fallback = bool(cfg.get("allow_fallback", True))
        self.local_files_only = bool(cfg.get("local_files_only", True))
        self.model_name = str(cfg.get("model_name", "openai/clip-vit-base-patch16"))
        self.feature_dim = int(cfg.get("feature_dim", 512))
        self.mode = "fallback"
        self.fallback_reason = None
        self.model = None
        self.processor = None
        self.projector: PCACompactProjector | None = None
        self.compact_checkpoint = ""
        self.compact_hash = ""
        if not self.enabled:
            if self.require_model and not self.allow_fallback:
                raise RuntimeError("CLIP teacher is disabled but clip.require_model=true and clip.allow_fallback=false")
            return
        try:
            from transformers import CLIPImageProcessor, CLIPVisionModel

            self.processor = CLIPImageProcessor.from_pretrained(self.model_name, local_files_only=self.local_files_only)
            self.model = CLIPVisionModel.from_pretrained(self.model_name, local_files_only=self.local_files_only).to(self.device).eval()
            self.feature_dim = int(self.model.config.hidden_size)
            self.mode = "transformers_clip"
        except Exception as exc:
            self.fallback_reason = repr(exc)
            if self.require_model and not self.allow_fallback:
                raise RuntimeError(f"Failed to load required CLIP model from {self.model_name}: {exc}") from exc
            self.mode = str(cfg.get("fallback", "image_stats_dense"))

        compact_cfg = cfg.get("compact", {}) or {}
        checkpoint = str(compact_cfg.get("checkpoint_path", "") or cfg.get("compact_checkpoint_path", ""))
        require_compact = bool(compact_cfg.get("require", False))
        if checkpoint:
            path = Path(checkpoint).expanduser()
            if not path.exists():
                if require_compact:
                    raise FileNotFoundError(f"Compact feature checkpoint not found: {path}")
            else:
                self.projector = PCACompactProjector.load(path, map_location="cpu").to(self.device)
                self.compact_checkpoint = str(path)
                self.compact_hash = file_sha256(path)
        elif require_compact:
            raise ValueError("clip.compact.require=true but no checkpoint_path was provided")

    def metadata(self) -> dict[str, Any]:
        return {
            "clip_model_name": self.model_name,
            "clip_mode": self.mode,
            "clip_feature_dim": int(self.feature_dim),
            "fallback_reason": self.fallback_reason,
            "compact_type": "pca" if self.projector is not None else "none",
            "compact_dim": self.projector.output_dim if self.projector is not None else None,
            "compact_checkpoint": self.compact_checkpoint,
            "compact_hash": self.compact_hash,
        }

    def _require_fallback_allowed(self, reason: str) -> None:
        if not self.allow_fallback:
            raise RuntimeError(f"CLIP fallback is disabled, but fallback was requested: {reason}")

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

    def _pixel_values(self, images: torch.Tensor) -> torch.Tensor:
        if self.model is None or self.processor is None:
            raise RuntimeError(f"CLIP model is not available; current mode is {self.mode}")
        imgs = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt", do_rescale=False)
        return inputs["pixel_values"].to(self.device)

    @torch.no_grad()
    def encode_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if self.mode != "transformers_clip" or self.model is None:
            raise RuntimeError(f"CLIP patch tokens require transformers_clip mode, got {self.mode}")
        out = self.model(pixel_values=self._pixel_values(images))
        return out.last_hidden_state[:, 1:, :].float()

    @torch.no_grad()
    def encode_views(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if not self.enabled:
            self._require_fallback_allowed("CLIP teacher is disabled")
            return torch.zeros(images.shape[0], self.feature_dim, device=self.device, dtype=torch.float32)
        if self.mode == "transformers_clip" and self.model is not None:
            out = self.model(pixel_values=self._pixel_values(images))
            return out.pooler_output.float()
        self._require_fallback_allowed("pooled CLIP feature fallback was requested")
        mean = images.mean(dim=(2, 3))
        std = images.std(dim=(2, 3), unbiased=False)
        feat = torch.cat([mean, std, mean * std, mean.square()], dim=-1)
        feat = _fit_feature_dim(feat, self.feature_dim)
        return feat.float()

    @torch.no_grad()
    def encode_dense_views(
        self,
        images: torch.Tensor,
        grid_size: tuple[int, int],
        *,
        depth: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        feature_dim: int | None = None,
        return_raw: bool = False,
    ):
        images = images.to(self.device)
        feature_dim = int(feature_dim or (self.projector.output_dim if self.projector is not None else self.feature_dim))
        dense_enabled = bool(self.cfg.get("dense_enabled", True))
        meta: dict[str, Any] = {**self.metadata(), "dense_enabled": dense_enabled, "dense_feature_dim": feature_dim}
        raw_dense: torch.Tensor | None = None
        if not self.enabled or not dense_enabled:
            self._require_fallback_allowed("CLIP teacher is disabled or dense features are disabled")
            feat = self._dense_fallback(images, grid_size, depth, confidence, feature_dim)
            result = (feat, {**meta, "dense_mode": "image_stats_dense"})
            return (*result, raw_dense) if return_raw else result
        if self.mode == "transformers_clip" and self.model is not None:
            patch = self.encode_patch_tokens(images)
            side = int(math.sqrt(int(patch.shape[1])))
            if side * side == int(patch.shape[1]):
                patch_grid = patch.reshape(patch.shape[0], side, side, patch.shape[-1]).permute(0, 3, 1, 2)
                patch_grid = F.interpolate(patch_grid, size=grid_size, mode="bilinear", align_corners=False)
                raw_dense = patch_grid.permute(0, 2, 3, 1).contiguous().float()
                if self.projector is not None:
                    feat = self.projector.encode(raw_dense)
                    if feat.shape[-1] != feature_dim:
                        raise ValueError(f"Compact feature dim {feat.shape[-1]} does not match requested feature_dim {feature_dim}")
                    result = (feat, {**meta, "dense_mode": "clip_patch_tokens_pca", "clip_patch_grid": [side, side], "raw_dense_feature_dim": int(raw_dense.shape[-1])})
                    return (*result, raw_dense) if return_raw else result
                feat = _fit_feature_dim(raw_dense, feature_dim)
                result = (F.normalize(feat.float(), dim=-1), {**meta, "dense_mode": "clip_patch_tokens", "clip_patch_grid": [side, side], "raw_dense_feature_dim": int(raw_dense.shape[-1])})
                return (*result, raw_dense) if return_raw else result
        self._require_fallback_allowed("CLIP patch tokens were unavailable or not reshapeable as a square grid")
        feat = self._dense_fallback(images, grid_size, depth, confidence, feature_dim)
        result = (feat, {**meta, "dense_mode": "image_stats_dense"})
        return (*result, raw_dense) if return_raw else result
