#!/usr/bin/env python
"""Visualize RoboTwin GaussianWAM teacher cache samples."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/tmp/fastwam_hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/fastwam_hf_datasets")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fastwam_mplconfig")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaussianwam.data import build_raw_dataset, get_raw_multiview_sample, select_target_offset

DEFAULT_CONFIG = ROOT / "configs/gaussianwam/stage1_robotwin_fullclean_first50_firstframe_all.yaml"
DEFAULT_MANIFEST = Path(
    "/data/zijianzhang/gaussianwam_data/data/robotwin2.0/gaussian_teacher_cache/v4/"
    "gaussian_vggt256text_3d_fullclean_first50_firstframe_all_v1/train/manifest.jsonl"
)
CAMERA_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/robotwin_teacher_cache"))
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Manifest/cache idx values to render.")
    parser.add_argument("--start-indices", type=int, nargs="*", default=None, help="Start idx values for segment rendering.")
    parser.add_argument("--frames", type=int, default=16, help="Frames per segment when --start-indices is used.")
    parser.add_argument("--stride", type=int, default=8, help="Idx stride between frames when --start-indices is used.")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-valid-ratio", type=float, default=0.25)
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--include-feature-pca", action="store_true")
    parser.add_argument("--make-video", action="store_true")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def tensor_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def image_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().float().cpu().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected image [C,H,W], got {arr.shape}")
    arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def resize_map(arr: np.ndarray, shape: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    img = Image.fromarray(arr)
    img = img.resize((shape[1], shape[0]), mode)
    return np.asarray(img)


def overlay_mask(image: np.ndarray, mask_small: np.ndarray) -> np.ndarray:
    mask = resize_map(mask_small.astype(np.uint8) * 255, image.shape[:2], nearest=True) > 127
    base = image.astype(np.float32) / 255.0
    color = np.zeros_like(base)
    color[..., 1] = 1.0
    out = base.copy()
    out[mask] = base[mask] * 0.45 + color[mask] * 0.55
    return np.clip(out, 0.0, 1.0)


def overlay_heat(image: np.ndarray, value_small: np.ndarray, mask_small: np.ndarray, cmap_name: str) -> np.ndarray:
    value = resize_map(value_small.astype(np.float32), image.shape[:2], nearest=False)
    mask = resize_map(mask_small.astype(np.uint8) * 255, image.shape[:2], nearest=True) > 127
    finite = np.isfinite(value) & mask
    norm = np.zeros_like(value, dtype=np.float32)
    if finite.any():
        lo, hi = np.percentile(value[finite], [2, 98])
        if hi > lo:
            norm = np.clip((value - lo) / (hi - lo), 0.0, 1.0)
    heat = plt.get_cmap(cmap_name)(norm)[..., :3]
    base = image.astype(np.float32) / 255.0
    out = base.copy()
    out[mask] = base[mask] * 0.35 + heat[mask] * 0.65
    return np.clip(out, 0.0, 1.0)


def feature_pca_rgb(feature: np.ndarray, valid: np.ndarray) -> np.ndarray:
    h, w, c = feature.shape
    mask = valid.astype(bool)
    x = feature[mask].astype(np.float32)
    if x.shape[0] < 4:
        return np.zeros((h, w, 3), dtype=np.float32)
    if x.shape[0] > 20000:
        stride = max(x.shape[0] // 20000, 1)
        x = x[::stride]
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:3].T.astype(np.float32)
    rgb = ((feature.reshape(h * w, c).astype(np.float32) - mean) @ basis).reshape(h, w, 3)
    for ch in range(3):
        values = rgb[..., ch]
        finite = np.isfinite(values) & mask
        if finite.any():
            lo, hi = np.percentile(values[finite], [2, 98])
            if hi > lo:
                rgb[..., ch] = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
            else:
                rgb[..., ch] = 0.0
        else:
            rgb[..., ch] = 0.0
    rgb[~mask] *= 0.15
    return np.clip(rgb, 0.0, 1.0)


def overlay_feature_rgb(image: np.ndarray, feature_rgb: np.ndarray, valid_small: np.ndarray) -> np.ndarray:
    feat = resize_map((feature_rgb * 255).astype(np.uint8), image.shape[:2], nearest=True).astype(np.float32) / 255.0
    mask = resize_map(valid_small.astype(np.uint8) * 255, image.shape[:2], nearest=True) > 127
    base = image.astype(np.float32) / 255.0
    out = base.copy()
    out[mask] = base[mask] * 0.35 + feat[mask] * 0.65
    return np.clip(out, 0.0, 1.0)


def load_manifest_rows(manifest_path: Path, *, include_rejected: bool) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    allowed = {"ok", "rejected"} if include_rejected else {"ok"}
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") in allowed and "idx" in row:
                rows[int(row["idx"])] = row
    return rows


def choose_indices(rows: dict[int, dict[str, Any]], args: argparse.Namespace) -> list[int]:
    if args.start_indices:
        indices: list[int] = []
        for start in args.start_indices:
            indices.extend(int(start) + frame * int(args.stride) for frame in range(int(args.frames)))
        return indices
    if args.indices:
        return list(args.indices)
    candidates = [
        idx
        for idx, row in rows.items()
        if float(row.get("valid_ratio", 0.0)) >= float(args.min_valid_ratio)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    return sorted(candidates[: int(args.num_samples)])


def load_config(path: Path, overrides: list[str]):
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def load_payload(row: dict[str, Any]) -> dict[str, Any]:
    return torch.load(row["path"], map_location="cpu")


def render_sample(
    *,
    idx: int,
    row: dict[str, Any],
    payload: dict[str, Any],
    raw_images: torch.Tensor,
    output_dir: Path,
    include_feature_pca: bool,
) -> Path:
    targets = payload["targets"]
    depth = tensor_to_numpy(targets["T_depth"])
    alpha = tensor_to_numpy(targets["T_alpha"])
    valid = tensor_to_numpy(targets["T_valid_mask"]).astype(bool)
    feature = tensor_to_numpy(targets["T_gaussian_feature"]) if include_feature_pca else None

    cols = 5 if include_feature_pca else 4
    fig, axes = plt.subplots(3, cols, figsize=(18 if cols == 5 else 15, 9), dpi=120)
    for view_idx, camera in enumerate(CAMERA_KEYS):
        raw = image_tensor_to_uint8(raw_images[view_idx])
        axes[view_idx, 0].imshow(raw)
        axes[view_idx, 0].set_title(f"{camera} raw", fontsize=9)
        axes[view_idx, 1].imshow(overlay_mask(raw, valid[view_idx]))
        axes[view_idx, 1].set_title("valid mask", fontsize=9)
        axes[view_idx, 2].imshow(overlay_heat(raw, alpha[view_idx], valid[view_idx], "viridis"))
        axes[view_idx, 2].set_title("alpha", fontsize=9)
        axes[view_idx, 3].imshow(overlay_heat(raw, depth[view_idx], valid[view_idx], "magma"))
        axes[view_idx, 3].set_title("depth", fontsize=9)
        if include_feature_pca and feature is not None:
            feat_rgb = feature_pca_rgb(feature[view_idx], valid[view_idx])
            axes[view_idx, 4].imshow(overlay_feature_rgb(raw, feat_rgb, valid[view_idx]))
            axes[view_idx, 4].set_title("dense3d PCA", fontsize=9)

    gaussian = payload.get("gaussian", {})
    dataset = payload.get("dataset", {})
    task = str(dataset.get("task", row.get("prompt", "")))
    if len(task) > 120:
        task = task[:117] + "..."
    title = (
        f"idx={idx} global_idx={dataset.get('global_idx')} episode={dataset.get('episode_index')} "
        f"frame={dataset.get('frame_index')} valid={float(row.get('valid_ratio', 0.0)):.3f} "
        f"overlap={float(row.get('teacher_render_overlap_ratio', 0.0)):.3f} "
        f"depth_err={float(row.get('final_depth_error', 0.0)):.3f}\n{task}"
    )
    fig.suptitle(title, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path = output_dir / f"robotwin_teacher_idx{idx:08d}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_rows(args.manifest, include_rejected=args.include_rejected)
    indices = choose_indices(rows, args)
    if not indices:
        raise SystemExit("No teacher-cache rows selected.")

    cfg = load_config(args.config, args.override)
    dataset = build_raw_dataset(cfg.source)
    camera_keys = list(cfg.source.camera_keys)
    if camera_keys != CAMERA_KEYS:
        raise ValueError(f"This visualizer expects {CAMERA_KEYS}, got {camera_keys}")
    target_offset = select_target_offset(cfg.source)

    summaries = []
    frames = []
    for idx in indices:
        if idx not in rows:
            raise KeyError(f"idx={idx} not found in manifest: {args.manifest}")
        row = rows[idx]
        payload = load_payload(row)
        global_idx = int(payload.get("dataset", {}).get("global_idx", idx))
        sample = get_raw_multiview_sample(dataset, global_idx, camera_keys, target_offset)
        out_path = render_sample(
            idx=idx,
            row=row,
            payload=payload,
            raw_images=sample.images,
            output_dir=args.output_dir,
            include_feature_pca=args.include_feature_pca,
        )
        print(out_path)
        summaries.append(
            {
                "idx": idx,
                "global_idx": global_idx,
                "output": str(out_path),
                "status": row.get("status"),
                "valid_ratio": row.get("valid_ratio"),
                "teacher_render_overlap_ratio": row.get("teacher_render_overlap_ratio"),
                "final_depth_error": row.get("final_depth_error"),
                "task": payload.get("dataset", {}).get("task"),
                "cache_path": row.get("path"),
            }
        )
        if args.make_video:
            frames.append(imageio.imread(out_path))

    summary_path = args.output_dir / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as file:
        for summary in summaries:
            file.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"Wrote {len(summaries)} visualizations and summary: {summary_path}")

    if args.make_video and frames:
        video_path = args.output_dir / f"robotwin_teacher_cache_{len(frames)}samples.mp4"
        imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=16)
        print(video_path)


if __name__ == "__main__":
    main()
