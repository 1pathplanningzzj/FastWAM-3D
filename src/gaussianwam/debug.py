from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def _save_scalar(path: Path, arr: torch.Tensor) -> None:
    arr = arr.detach().cpu().float()
    arr = arr - arr.min()
    arr = arr / arr.max().clamp_min(1e-6)
    Image.fromarray((arr.numpy() * 255).astype("uint8")).save(path)


def _save_feature(path: Path, feature: torch.Tensor) -> None:
    feat = feature.detach().cpu().float()[..., :3]
    feat = feat - feat.amin(dim=(0, 1), keepdim=True)
    feat = feat / feat.amax(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    Image.fromarray((feat.numpy() * 255).astype("uint8")).save(path)


def save_debug_maps(output_dir: str | Path, render: dict[str, torch.Tensor]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save_scalar(out / "alpha.png", render["alpha"][0])
    _save_scalar(out / "depth.png", render["dep"][0])
    _save_feature(out / "feature_rgb.png", render["feature_map"][0])
    if "feature_target" in render:
        _save_feature(out / "feature_target_rgb.png", render["feature_target"][0])
    if "teacher_valid_mask" in render:
        _save_scalar(out / "teacher_valid_mask.png", render["teacher_valid_mask"][0])
    if "render_valid_mask" in render:
        _save_scalar(out / "render_valid_mask.png", render["render_valid_mask"][0])
