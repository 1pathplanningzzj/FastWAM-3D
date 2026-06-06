from __future__ import annotations

from typing import Any

import torch


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


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors = []
    targets = payload.get("targets", {})
    required = ["T_gaussian_feature", "T_depth", "T_alpha", "T_valid_mask"]
    for key in required:
        if key not in targets:
            errors.append(f"missing targets.{key}")
    if "T_teacher_valid_mask" not in targets and "T_valid_mask" not in targets:
        errors.append("missing targets.T_teacher_valid_mask or targets.T_valid_mask")
    for key, value in targets.items():
        if torch.is_tensor(value) and value.numel() and not torch.isfinite(value.float()).all():
            errors.append(f"targets.{key} contains non-finite values")

    feature = targets.get("T_gaussian_feature")
    depth = targets.get("T_depth")
    alpha = targets.get("T_alpha")
    if torch.is_tensor(feature) and torch.is_tensor(depth):
        if tuple(feature.shape[:3]) != tuple(depth.shape):
            errors.append(f"T_gaussian_feature shape {tuple(feature.shape)} does not match T_depth shape {tuple(depth.shape)}")
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
    return errors
