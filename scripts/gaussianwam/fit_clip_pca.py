#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaussianwam.clip_teacher import ClipSemanticTeacher
from gaussianwam.compact_feature import fit_pca, save_projector
from gaussianwam.config import load_config
from gaussianwam.data import build_raw_dataset, get_raw_multiview_sample, select_target_offset


def _device(cfg) -> torch.device:
    requested = str(cfg.device)
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a global PCA compact space for GaussianWAM CLIP patch features.")
    parser.add_argument("--config", default="configs/gaussianwam/clip_pca.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    torch.manual_seed(int(cfg.seed))
    device = _device(cfg)
    source_cfg = cfg.source
    dataset = build_raw_dataset(source_cfg)
    target_offset = select_target_offset(source_cfg)
    camera_keys = list(source_cfg.camera_keys)
    start = int(source_cfg.start_idx)
    end = len(dataset) if source_cfg.end_idx is None else min(int(source_cfg.end_idx), len(dataset))
    max_samples = source_cfg.max_samples if source_cfg.max_samples is not None else args.limit

    clip = ClipSemanticTeacher(cfg.clip, device=device)
    if clip.mode != "transformers_clip":
        raise RuntimeError(f"Expected real CLIP patch features, got clip mode {clip.mode}")

    max_features = int(cfg.pca.max_features)
    max_patches_per_view = int(cfg.clip.get("max_patches_per_view", 64))
    chunks: list[torch.Tensor] = []
    total = 0

    generator = torch.Generator().manual_seed(int(cfg.seed))
    indices = torch.arange(start, end)
    if len(indices) > 0:
        indices = indices[torch.randperm(len(indices), generator=generator)]
    if max_samples is not None:
        indices = indices[: int(max_samples)]

    for idx_tensor in tqdm(indices.tolist(), desc="Collect CLIP patches", unit="sample"):
        idx = int(idx_tensor)
        sample = get_raw_multiview_sample(dataset, idx, camera_keys, target_offset)
        if bool(sample.image_is_pad.item()):
            continue
        raw = clip.encode_patch_tokens(sample.images.to(device))
        if raw.numel() == 0:
            continue
        if raw.shape[1] > max_patches_per_view:
            select = torch.randperm(raw.shape[1], generator=generator)[:max_patches_per_view].to(raw.device)
            raw = raw[:, select]
        flat = raw.reshape(-1, raw.shape[-1]).detach().cpu()
        remaining = max_features - total
        if remaining <= 0:
            break
        if flat.shape[0] > remaining:
            flat = flat[:remaining]
        chunks.append(flat)
        total += int(flat.shape[0])
        if total >= max_features:
            break

    if not chunks:
        raise RuntimeError("No CLIP patch features collected")
    features = torch.cat(chunks, dim=0)
    projector = fit_pca(features, int(cfg.pca.output_dim))
    projector.metadata.update(
        {
            "source_config": str(args.config),
            "clip_model_name": str(cfg.clip.model_name),
            "target_frame_policy": str(source_cfg.target_frame_policy),
            "target_offset": int(target_offset),
            "camera_keys": camera_keys,
        }
    )
    output_path = Path(str(cfg.output_path))
    save_projector(projector, output_path)
    print({"output_path": str(output_path), "features": int(features.shape[0]), "input_dim": int(features.shape[1]), "output_dim": projector.output_dim})


if __name__ == "__main__":
    main()
