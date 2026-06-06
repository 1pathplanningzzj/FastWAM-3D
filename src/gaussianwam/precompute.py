from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import torch
from tqdm import tqdm

from .cache import append_jsonl, atomic_torch_save, cache_root, config_hash, stable_hash
from .clip_teacher import ClipSemanticTeacher
from .data import build_raw_dataset, get_raw_multiview_sample, select_target_offset, video_sample_indices
from .debug import save_debug_maps
from .fitting import fit_gaussian_field
from .gaussian_field import GaussianFeatureField
from .geometry import depth_to_points, make_valid_mask, resize_hw, sample_points, scale_intrinsics
from .renderer import SoftPointRenderer
from .vggt_omega import VGGTOmegaTeacher


def _device(cfg) -> torch.device:
    requested = str(cfg.device)
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _cache_path(root: Path, sample, target_offset: int, cfg_hash: str) -> tuple[str, Path]:
    key_payload = {"idx": sample.global_idx, "task": sample.task, "target_offset": target_offset, "cfg": cfg_hash}
    key = stable_hash(key_payload, length=16)
    path = root / f"idx{sample.global_idx:08d}_off{target_offset:02d}_{key}.pt"
    return key, path


def _text_cache_metadata(cfg, task: str) -> dict[str, Any]:
    text_cfg = cfg.get("text_cache")
    if text_cfg is None:
        return {}
    prompt_template = str(text_cfg.get("prompt_template", "{task}"))
    prompt = prompt_template.format(task=task)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    context_len = int(text_cfg.get("context_len", 128))
    encoder_id = str(text_cfg.get("encoder_id", "wan22ti2v5b"))
    filename = f"{prompt_hash}.t5_len{context_len}.{encoder_id}.pt"
    cache_dir = Path(str(text_cfg.get("dir", "")))
    cache_path = cache_dir / filename if str(cache_dir) else Path(filename)
    return {
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "context_len": context_len,
        "encoder_id": encoder_id,
        "text_cache_dir": str(cache_dir),
        "text_cache_filename": filename,
        "text_cache_path": str(cache_path),
        "text_cache_exists": cache_path.exists(),
    }


def precompute(cfg, *, limit: int | None = None, dry_run: bool = False, overwrite: bool | None = None) -> dict[str, int]:
    torch.manual_seed(int(cfg.seed))
    device = _device(cfg)
    source_cfg = cfg.source
    cache_cfg = cfg.cache
    overwrite = bool(cache_cfg.overwrite if overwrite is None else overwrite)
    cfg_hash = config_hash(cfg)
    split = str(cfg.split)
    root = cache_root(cfg.output_dir, int(cache_cfg.version), str(cache_cfg.namespace), split)
    manifest_path = root / "manifest.jsonl"

    dataset = build_raw_dataset(source_cfg)
    target_offset = select_target_offset(source_cfg)
    camera_keys = list(source_cfg.camera_keys)
    start = int(source_cfg.start_idx)
    end = len(dataset) if source_cfg.end_idx is None else min(int(source_cfg.end_idx), len(dataset))
    max_samples = source_cfg.max_samples if source_cfg.max_samples is not None else limit
    if max_samples is not None:
        end = min(end, start + int(max_samples))
    if limit is not None:
        end = min(end, start + int(limit))

    vggt = None if dry_run else VGGTOmegaTeacher(cfg.vggt_omega, device=device)
    clip = None if dry_run else ClipSemanticTeacher(cfg.clip, device=device)
    renderer = SoftPointRenderer(int(cfg.gaussian.render_height), int(cfg.gaussian.render_width), tuple(cfg.gaussian.target_grid_size))
    counts = {"new": 0, "skip": 0, "error": 0, "dry_run": 0}

    for idx in tqdm(range(start, end), desc="GaussianWAM Stage1", unit="sample"):
        try:
            sample = get_raw_multiview_sample(dataset, idx, camera_keys, target_offset)
            text_meta = _text_cache_metadata(cfg, sample.task)
            cache_key, path = _cache_path(root, sample, target_offset, cfg_hash)
            if path.exists() and not overwrite:
                counts["skip"] += 1
                continue
            if dry_run:
                counts["dry_run"] += 1
                append_jsonl(manifest_path, {"idx": idx, "cache_key": cache_key, "path": str(path), "status": "dry_run", **text_meta})
                continue

            images = sample.images.to(device)
            vggt_out = vggt(images)
            depth = vggt_out["depth"]
            conf = vggt_out["depth_conf"]
            if depth.ndim == 5:
                depth = depth[0]
            if conf.ndim == 5:
                conf = conf[0]
            if depth.shape[-1] == 1:
                depth_v = depth[..., 0]
            elif depth.shape[1] == 1:
                depth_v = depth[:, 0]
            else:
                depth_v = depth.squeeze()
            if conf.shape[-1] == 1:
                conf_v = conf[..., 0]
            elif conf.shape[1] == 1:
                conf_v = conf[:, 0]
            else:
                conf_v = conf.squeeze()
            grid_size = tuple(int(x) for x in cfg.gaussian.target_grid_size)
            intrinsics_grid = scale_intrinsics(vggt_out["intrinsics"], tuple(int(x) for x in depth_v.shape[-2:]), grid_size)
            depth_grid = resize_hw(depth_v, grid_size).contiguous()
            conf_grid = resize_hw(conf_v, grid_size).contiguous()
            teacher_valid_mask = make_valid_mask(depth_grid, conf_grid)
            points = depth_to_points(depth_grid, intrinsics_grid, vggt_out["extrinsics"])
            images_grid = resize_hw(images, grid_size).contiguous()
            clip_feat = clip.encode_views(images)
            target_feature, dense_meta = clip.encode_dense_views(
                images,
                grid_size,
                depth=depth_grid,
                confidence=conf_grid,
                feature_dim=int(cfg.gaussian.feature_dim),
            )
            xyz, color, init_feat = sample_points(
                points,
                images_grid,
                target_feature,
                teacher_valid_mask,
                max_points=int(cfg.gaussian.max_points),
                stride=int(cfg.gaussian.init_stride),
            )
            if xyz.numel() == 0:
                raise ValueError("No valid Gaussian initialization points")
            field = GaussianFeatureField(
                xyz=xyz,
                color=color,
                feature=init_feat,
                init_scale=float(cfg.gaussian.init_scale),
                init_opacity=float(cfg.gaussian.init_opacity),
            ).to(device)
            render, metrics = fit_gaussian_field(field, renderer, depth_grid, target_feature, teacher_valid_mask, intrinsics_grid, vggt_out["extrinsics"], cfg.gaussian, cfg.loss)

            alpha_threshold = float(cache_cfg.get("alpha_valid_threshold", 1e-4))
            render_valid_mask = render["alpha"].detach() > alpha_threshold
            valid_mask = teacher_valid_mask & render_valid_mask
            targets = {
                "T_gaussian_feature": render["feature_map"].detach().to("cpu", dtype=torch.bfloat16),
                "T_depth": render["dep"].detach().to("cpu", dtype=torch.float32),
                "T_alpha": render["alpha"].detach().to("cpu", dtype=torch.float32),
                "T_valid_mask": valid_mask.detach().to("cpu", dtype=torch.bool),
                "T_teacher_valid_mask": teacher_valid_mask.detach().to("cpu", dtype=torch.bool),
                "T_render_valid_mask": render_valid_mask.detach().to("cpu", dtype=torch.bool),
                "T_clip_feature": clip_feat.detach().to("cpu", dtype=torch.float32),
                "T_register": vggt_out["register_tokens"].detach().to("cpu", dtype=torch.bfloat16),
                "extrinsics": vggt_out["extrinsics"].detach().to("cpu", dtype=torch.float32),
                "intrinsics": vggt_out["intrinsics"].detach().to("cpu", dtype=torch.float32),
            }
            if bool(cache_cfg.get("save_dense_feature_target", False)):
                targets["T_dense_feature_target"] = target_feature.detach().to("cpu", dtype=torch.bfloat16)
            payload: dict[str, Any] = {
                "version": int(cache_cfg.version),
                "cache_key": cache_key,
                "dataset": {"global_idx": sample.global_idx, "task": sample.task, **sample.metadata},
                "text": text_meta,
                "alignment": {
                    "num_frames": int(source_cfg.num_frames),
                    "action_video_freq_ratio": int(source_cfg.action_video_freq_ratio),
                    "video_sample_indices": video_sample_indices(source_cfg.num_frames, source_cfg.action_video_freq_ratio),
                    "target_frame_policy": str(source_cfg.target_frame_policy),
                    "target_offset": int(target_offset),
                    "camera_keys": camera_keys,
                },
                "teacher": {"vggt_omega": str(cfg.vggt_omega.checkpoint_path), "clip_mode": clip.mode, **dense_meta},
                "gaussian": {
                    "feature_dim": int(cfg.gaussian.feature_dim),
                    "feature_target": "dense_per_view",
                    "init_feature_source": str(cfg.gaussian.get("init_feature_source", "dense_target")),
                    "num_points": int(field.xyz.shape[0]),
                    **metrics,
                },
                "targets": targets,
            }
            if bool(cache_cfg.save_gaussian_state):
                payload["gaussian_state"] = field.state_payload()
            if bool(cache_cfg.atomic_write):
                atomic_torch_save(payload, path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, str(path))
            if bool(cache_cfg.save_debug_images):
                save_debug_maps(root / "debug" / cache_key, render)
            append_jsonl(manifest_path, {"idx": idx, "cache_key": cache_key, "path": str(path), "status": "ok", **text_meta, **metrics})
            counts["new"] += 1
        except Exception as exc:
            counts["error"] += 1
            append_jsonl(manifest_path, {"idx": idx, "status": "error", "error": repr(exc)})
    return counts
