from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .compact_feature import file_sha256


def tensor_summary(x: torch.Tensor) -> dict[str, Any]:
    xf = x.float()
    finite = torch.isfinite(xf)
    out = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite_ratio": float(finite.float().mean().item()) if x.numel() else 0.0,
    }
    if x.numel() and finite.any():
        vals = xf[finite]
        out.update({"min": float(vals.min().item()), "max": float(vals.max().item()), "mean": float(vals.mean().item())})
    return out


def _cfg_get(cfg: Any | None, path: str, default=None):
    if cfg is None:
        return default
    cur = cfg
    for part in path.split("."):
        if cur is None or not hasattr(cur, "get"):
            return default
        cur = cur.get(part, None)
    return default if cur is None else cur


def _finite_tensor(name: str, value: torch.Tensor, errors: list[str]) -> None:
    if value.numel() and not torch.isfinite(value.float()).all():
        errors.append(f"{name} contains non-finite values")


def validate_payload(payload: dict[str, Any], cfg: Any | None = None) -> list[str]:
    errors: list[str] = []
    targets = payload.get("targets", {})
    required = ["T_gaussian_feature", "T_depth", "T_alpha", "T_valid_mask"]
    for key in required:
        if key not in targets:
            errors.append(f"missing targets.{key}")
    if "T_teacher_valid_mask" not in targets and "T_valid_mask" not in targets:
        errors.append("missing targets.T_teacher_valid_mask or targets.T_valid_mask")
    for key, value in targets.items():
        if torch.is_tensor(value):
            _finite_tensor(f"targets.{key}", value, errors)

    feature = targets.get("T_gaussian_feature")
    depth = targets.get("T_depth")
    alpha = targets.get("T_alpha")
    if torch.is_tensor(feature) and torch.is_tensor(depth):
        if tuple(feature.shape[:3]) != tuple(depth.shape):
            errors.append(f"T_gaussian_feature shape {tuple(feature.shape)} does not match T_depth shape {tuple(depth.shape)}")
        expected_dim = _cfg_get(cfg, "gaussian.feature_dim", None)
        if expected_dim is not None and int(feature.shape[-1]) != int(expected_dim):
            errors.append(f"T_gaussian_feature dim {feature.shape[-1]} does not match expected {expected_dim}")
    if torch.is_tensor(alpha) and torch.is_tensor(depth):
        if tuple(alpha.shape) != tuple(depth.shape):
            errors.append(f"T_alpha shape {tuple(alpha.shape)} does not match T_depth shape {tuple(depth.shape)}")
    for key in ["T_valid_mask", "T_teacher_valid_mask", "T_render_valid_mask"]:
        mask = targets.get(key)
        if torch.is_tensor(mask):
            if torch.is_tensor(depth) and tuple(mask.shape) != tuple(depth.shape):
                errors.append(f"{key} shape {tuple(mask.shape)} does not match T_depth shape {tuple(depth.shape)}")
            if mask.numel() and not bool(mask.any().item()):
                errors.append(f"{key} is all false")
    dense_target = targets.get("T_dense_feature_target")
    if torch.is_tensor(dense_target) and torch.is_tensor(feature) and tuple(dense_target.shape) != tuple(feature.shape):
        errors.append(f"T_dense_feature_target shape {tuple(dense_target.shape)} does not match T_gaussian_feature shape {tuple(feature.shape)}")

    text = payload.get("text", {})
    text_cache_path = text.get("text_cache_path")
    prompt_hash = text.get("prompt_hash")
    if text_cache_path:
        text_path = Path(str(text_cache_path))
        if not text_path.exists():
            errors.append(f"text cache file does not exist: {text_path}")
        if prompt_hash and prompt_hash not in text_path.name:
            errors.append("text cache filename does not contain prompt_hash")
    elif _cfg_get(cfg, "text_cache.dir", None):
        errors.append("missing text.text_cache_path")

    teacher = payload.get("teacher", {})
    if bool(_cfg_get(cfg, "clip.require_model", False)) and teacher.get("clip_mode") != "transformers_clip":
        errors.append(f"expected real CLIP teacher, got clip_mode={teacher.get('clip_mode')}")
    if bool(_cfg_get(cfg, "clip.compact.require", False)):
        if teacher.get("compact_type") != "pca":
            errors.append(f"expected PCA compact feature, got compact_type={teacher.get('compact_type')}")
        if not teacher.get("compact_checkpoint"):
            errors.append("missing teacher.compact_checkpoint")
        if not teacher.get("compact_hash"):
            errors.append("missing teacher.compact_hash")
        cfg_checkpoint = _cfg_get(cfg, "clip.compact.checkpoint_path", None)
        if cfg_checkpoint:
            cfg_path = Path(str(cfg_checkpoint))
            if not cfg_path.exists():
                errors.append(f"configured compact checkpoint does not exist: {cfg_path}")
            else:
                expected_hash = file_sha256(cfg_path)
                if teacher.get("compact_hash") and teacher.get("compact_hash") != expected_hash:
                    errors.append("teacher.compact_hash does not match configured compact checkpoint")

    gaussian = payload.get("gaussian", {})
    if bool(_cfg_get(cfg, "quality.require_frozen_feature_z", True)) and bool(gaussian.get("feature_z_optimized", False)):
        errors.append("feature_z was optimized but quality.require_frozen_feature_z=true")

    if bool(_cfg_get(cfg, "mosaic.enabled", False)):
        mosaic_feature = targets.get("T_gaussian_feature_mosaic")
        mosaic_depth = targets.get("T_depth_mosaic")
        mosaic_alpha = targets.get("T_alpha_mosaic")
        mosaic_mask = targets.get("T_valid_mask_mosaic")
        for key, value in [
            ("T_gaussian_feature_mosaic", mosaic_feature),
            ("T_depth_mosaic", mosaic_depth),
            ("T_alpha_mosaic", mosaic_alpha),
            ("T_valid_mask_mosaic", mosaic_mask),
        ]:
            if not torch.is_tensor(value):
                errors.append(f"missing targets.{key}")
        expected_grid = _cfg_get(cfg, "mosaic.grid_size", None)
        if expected_grid is not None and torch.is_tensor(mosaic_feature):
            expected = tuple(int(x) for x in expected_grid)
            if tuple(mosaic_feature.shape[:2]) != expected:
                errors.append(f"T_gaussian_feature_mosaic grid {tuple(mosaic_feature.shape[:2])} does not match expected {expected}")
        if torch.is_tensor(mosaic_depth) and torch.is_tensor(mosaic_feature) and tuple(mosaic_depth.shape) != tuple(mosaic_feature.shape[:2]):
            errors.append("T_depth_mosaic shape does not match T_gaussian_feature_mosaic grid")
        if torch.is_tensor(mosaic_alpha) and torch.is_tensor(mosaic_feature) and tuple(mosaic_alpha.shape) != tuple(mosaic_feature.shape[:2]):
            errors.append("T_alpha_mosaic shape does not match T_gaussian_feature_mosaic grid")
        if torch.is_tensor(mosaic_mask):
            if torch.is_tensor(mosaic_feature) and tuple(mosaic_mask.shape) != tuple(mosaic_feature.shape[:2]):
                errors.append("T_valid_mask_mosaic shape does not match T_gaussian_feature_mosaic grid")
            min_coverage = float(_cfg_get(cfg, "quality.min_mosaic_valid_ratio", 0.0))
            coverage = float(mosaic_mask.float().mean().item()) if mosaic_mask.numel() else 0.0
            if coverage < min_coverage:
                errors.append(f"mosaic valid coverage {coverage:.4f} < {min_coverage:.4f}")

    min_render = float(_cfg_get(cfg, "quality.min_render_valid_ratio", 0.0))
    min_teacher = float(_cfg_get(cfg, "quality.min_teacher_valid_ratio", 0.0))
    min_cos = float(_cfg_get(cfg, "quality.min_compact_cosine", -1.0))
    min_overlap = float(_cfg_get(cfg, "quality.min_teacher_render_overlap_ratio", 0.0))
    max_cosine_drop = float(_cfg_get(cfg, "quality.max_compact_cosine_drop", 1.0))
    max_feature_anchor_loss = _cfg_get(cfg, "quality.max_feature_anchor_loss", None)
    if float(gaussian.get("render_valid_ratio", 1.0)) < min_render:
        errors.append(f"render_valid_ratio {gaussian.get('render_valid_ratio')} < {min_render}")
    if float(gaussian.get("teacher_valid_ratio", 1.0)) < min_teacher:
        errors.append(f"teacher_valid_ratio {gaussian.get('teacher_valid_ratio')} < {min_teacher}")
    final_cos = float(gaussian.get("final_compact_cosine", 1.0))
    initial_cos = float(gaussian.get("initial_compact_cosine", final_cos))
    if final_cos < min_cos:
        errors.append(f"final_compact_cosine {gaussian.get('final_compact_cosine')} < {min_cos}")
    if initial_cos - final_cos > max_cosine_drop:
        errors.append(f"compact cosine dropped by {initial_cos - final_cos:.4f} > {max_cosine_drop:.4f}")
    if float(gaussian.get("teacher_render_overlap_ratio", 1.0)) < min_overlap:
        errors.append(f"teacher_render_overlap_ratio {gaussian.get('teacher_render_overlap_ratio')} < {min_overlap}")
    if max_feature_anchor_loss is not None and "loss_feature_anchor" in gaussian:
        anchor = float(gaussian.get("loss_feature_anchor", 0.0))
        if anchor > float(max_feature_anchor_loss):
            errors.append(f"loss_feature_anchor {anchor} > {float(max_feature_anchor_loss)}")
    return errors
