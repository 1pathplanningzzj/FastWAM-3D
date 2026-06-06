from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_hw(x: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if x.ndim == 3:
        return F.interpolate(x[:, None], size=size, mode=mode, align_corners=False).squeeze(1)
    if x.ndim == 4:
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    raise ValueError(f"Unsupported tensor rank for resize: {x.ndim}")


def scale_intrinsics(intrinsics: torch.Tensor, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> torch.Tensor:
    source_h, source_w = float(source_hw[0]), float(source_hw[1])
    target_h, target_w = float(target_hw[0]), float(target_hw[1])
    sx = target_w / source_w
    sy = target_h / source_h
    out = intrinsics.clone()
    out[..., 0, 0] = out[..., 0, 0] * sx
    out[..., 1, 1] = out[..., 1, 1] * sy
    out[..., 0, 2] = (out[..., 0, 2] + 0.5) * sx - 0.5
    out[..., 1, 2] = (out[..., 1, 2] + 0.5) * sy - 0.5
    return out


def depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]
    if extrinsics.ndim == 4:
        extrinsics = extrinsics[0]
    v, h, w = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    points_world = []
    for i in range(v):
        z = depth[i]
        fx = intrinsics[i, 0, 0]
        fy = intrinsics[i, 1, 1]
        cx = intrinsics[i, 0, 2]
        cy = intrinsics[i, 1, 2]
        x = (xs - cx) / fx * z
        y = (ys - cy) / fy * z
        pts_cam = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        r = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        pts_world = (pts_cam - t) @ r
        points_world.append(pts_world.reshape(h, w, 3))
    return torch.stack(points_world, dim=0)


def make_valid_mask(depth: torch.Tensor, conf: torch.Tensor | None = None, conf_quantile: float = 0.1) -> torch.Tensor:
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    mask = torch.isfinite(depth) & (depth > 0)
    if conf is not None:
        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        if conf.ndim == 4 and conf.shape[1] == 1:
            conf = conf[:, 0]
        finite = conf[torch.isfinite(conf)]
        if finite.numel():
            threshold = torch.quantile(finite.float(), float(conf_quantile)).to(conf.dtype)
            mask = mask & (conf >= threshold)
    return mask


def sample_points(points: torch.Tensor, colors: torch.Tensor, features: torch.Tensor, mask: torch.Tensor, max_points: int, stride: int):
    v, h, w, _ = points.shape
    keep = torch.zeros((v, h, w), dtype=torch.bool, device=points.device)
    keep[:, :: int(stride), :: int(stride)] = True
    keep = keep & mask
    flat_idx = keep.reshape(-1).nonzero(as_tuple=False).flatten()
    if flat_idx.numel() > int(max_points):
        flat_idx = flat_idx[torch.linspace(0, flat_idx.numel() - 1, int(max_points), device=flat_idx.device).long()]
    pts = points.reshape(-1, 3)[flat_idx]
    cols = colors.permute(0, 2, 3, 1).reshape(-1, 3)[flat_idx]
    feats = features.reshape(-1, features.shape[-1])[flat_idx]
    return pts, cols, feats
