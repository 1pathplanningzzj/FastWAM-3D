from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import os

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
from .validate import validate_payload
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


def _existing_cache_is_valid(path: Path, cfg) -> bool:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if validate_payload(payload, cfg):
        return False
    gaussian = payload.get("gaussian", {})
    if gaussian.get("validation_errors"):
        return False
    if gaussian.get("quality_ok") is False:
        return False
    return True


def _load_subset_rows(path: str | Path | None) -> list[dict[str, Any]] | None:
    if not path:
        return None
    subset_path = Path(str(path))
    rows = []
    with subset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def _resize_feature_hw(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=size, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).contiguous()


def _resize_mask_hw(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(x[:, None].float(), size=size, mode="nearest").squeeze(1).bool()


def _split_spans(total: int, count: int) -> list[tuple[int, int]]:
    if count <= 0:
        raise ValueError(f"Expected positive tile count, got {count}")
    base = total // count
    remainder = total % count
    spans: list[tuple[int, int]] = []
    start = 0
    for i in range(count):
        extent = base + (1 if i < remainder else 0)
        end = start + extent
        spans.append((start, end))
        start = end
    return spans


def _robotwin_tiles(camera_keys: list[str], mosaic_grid_size: tuple[int, int]) -> dict[str, tuple[slice, slice]]:
    h, w = mosaic_grid_size
    top_h = int(round(h * 2 / 3))
    bottom_h = h - top_h
    left_w = w // 2
    return {
        camera_keys[0]: (slice(0, top_h), slice(0, w)),
        camera_keys[1]: (slice(top_h, h), slice(0, left_w)),
        camera_keys[2]: (slice(top_h, h), slice(left_w, w)),
    }


def _horizontal_tiles(camera_keys: list[str], mosaic_grid_size: tuple[int, int]) -> dict[str, tuple[slice, slice]]:
    h, w = mosaic_grid_size
    tiles = {}
    for key, (start, end) in zip(camera_keys, _split_spans(w, len(camera_keys))):
        tiles[key] = (slice(0, h), slice(start, end))
    return tiles


def _vertical_tiles(camera_keys: list[str], mosaic_grid_size: tuple[int, int]) -> dict[str, tuple[slice, slice]]:
    h, w = mosaic_grid_size
    tiles = {}
    for key, (start, end) in zip(camera_keys, _split_spans(h, len(camera_keys))):
        tiles[key] = (slice(start, end), slice(0, w))
    return tiles


def _compose_mosaic(
    feature: torch.Tensor,
    depth: torch.Tensor,
    alpha: torch.Tensor,
    mask: torch.Tensor,
    camera_keys: list[str],
    mosaic_grid_size: tuple[int, int],
    layout: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    h, w = mosaic_grid_size
    feat_m = torch.zeros(h, w, feature.shape[-1], device=feature.device, dtype=feature.dtype)
    dep_m = torch.zeros(h, w, device=depth.device, dtype=depth.dtype)
    alpha_m = torch.zeros(h, w, device=alpha.device, dtype=alpha.dtype)
    mask_m = torch.zeros(h, w, device=mask.device, dtype=torch.bool)
    layout = str(layout)
    if layout == "robotwin":
        if len(camera_keys) != 3:
            raise ValueError(f"RobotWin mosaic composition expects 3 cameras, got {camera_keys}")
        tiles = _robotwin_tiles(camera_keys, mosaic_grid_size)
    elif layout == "horizontal":
        tiles = _horizontal_tiles(camera_keys, mosaic_grid_size)
    elif layout == "vertical":
        tiles = _vertical_tiles(camera_keys, mosaic_grid_size)
    else:
        raise ValueError(f"Unsupported mosaic layout: {layout}")
    tile_meta = {}
    for view, key in enumerate(camera_keys):
        row_slice, col_slice = tiles[key]
        tile_h = row_slice.stop - row_slice.start
        tile_w = col_slice.stop - col_slice.start
        feat_tile = _resize_feature_hw(feature[view : view + 1], (tile_h, tile_w))[0]
        dep_tile = resize_hw(depth[view : view + 1], (tile_h, tile_w))[0]
        alpha_tile = resize_hw(alpha[view : view + 1], (tile_h, tile_w))[0]
        mask_tile = _resize_mask_hw(mask[view : view + 1], (tile_h, tile_w))[0]
        feat_m[row_slice, col_slice] = feat_tile
        dep_m[row_slice, col_slice] = dep_tile
        alpha_m[row_slice, col_slice] = alpha_tile
        mask_m[row_slice, col_slice] = mask_tile
        tile_meta[key] = {"rows": [row_slice.start, row_slice.stop], "cols": [col_slice.start, col_slice.stop]}
    return {
        "T_gaussian_feature_mosaic": feat_m,
        "T_depth_mosaic": dep_m,
        "T_alpha_mosaic": alpha_m,
        "T_valid_mask_mosaic": mask_m,
    }, {"layout": layout, "grid_size": [h, w], "tiles": tile_meta}


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
    subset_rows = _load_subset_rows(source_cfg.get("subset_manifest", None))
    if subset_rows is not None:
        start = int(source_cfg.get("start_idx", 0))
        end = len(subset_rows) if source_cfg.end_idx is None else min(int(source_cfg.end_idx), len(subset_rows))
        max_samples = source_cfg.max_samples if source_cfg.max_samples is not None else limit
        if max_samples is not None:
            end = min(end, start + int(max_samples))
        if limit is not None:
            end = min(end, start + int(limit))
        work_items = subset_rows[start:end]
    else:
        start = int(source_cfg.start_idx)
        end = len(dataset) if source_cfg.end_idx is None else min(int(source_cfg.end_idx), len(dataset))
        max_samples = source_cfg.max_samples if source_cfg.max_samples is not None else limit
        if max_samples is not None:
            end = min(end, start + int(max_samples))
        if limit is not None:
            end = min(end, start + int(limit))
        work_items = [{"idx": int(idx)} for idx in range(start, end)]

    vggt = None if dry_run else VGGTOmegaTeacher(cfg.vggt_omega, device=device)
    clip = None if dry_run else ClipSemanticTeacher(cfg.clip, device=device)
    grid_size = tuple(int(x) for x in cfg.gaussian.target_grid_size)
    renderer_cfg = cfg.get("renderer", {}) or {}
    renderer = SoftPointRenderer(
        int(cfg.gaussian.render_height),
        int(cfg.gaussian.render_width),
        grid_size,
        radius_px=int(renderer_cfg.get("radius_px", 2)),
        sigma_min=float(renderer_cfg.get("sigma_min", 0.35)),
        sigma_max=float(renderer_cfg.get("sigma_max", 2.0)),
        alpha_eps=float(renderer_cfg.get("alpha_eps", 1e-6)),
        depth_weight=float(renderer_cfg.get("depth_weight", 1.0)),
    )
    counts = {"new": 0, "skip": 0, "error": 0, "dry_run": 0, "rejected": 0}

    for work_item in tqdm(work_items, desc="GaussianWAM Stage1", unit="sample"):
        try:
            idx = int(work_item["idx"])
            sample = get_raw_multiview_sample(dataset, idx, camera_keys, target_offset)
            text_meta = _text_cache_metadata(cfg, sample.task)
            cache_key, path = _cache_path(root, sample, target_offset, cfg_hash)
            if path.exists() and not overwrite:
                if _existing_cache_is_valid(path, cfg):
                    counts["skip"] += 1
                    continue
                stale_path = root / "stale" / path.name
                stale_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, stale_path)
                append_jsonl(manifest_path, {"idx": idx, "cache_key": cache_key, "path": str(stale_path), "status": "stale", "reason": "existing cache failed current validation"})
            if bool(sample.image_is_pad.item()):
                counts["skip"] += 1
                append_jsonl(manifest_path, {"idx": idx, "cache_key": cache_key, "path": str(path), "status": "skip", "reason": "target frame is padded", **text_meta})
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
            intrinsics_grid = scale_intrinsics(vggt_out["intrinsics"], tuple(int(x) for x in depth_v.shape[-2:]), grid_size)
            depth_grid = resize_hw(depth_v, grid_size).contiguous()
            conf_grid = resize_hw(conf_v, grid_size).contiguous()
            geom_cfg = cfg.get("geometry", {}) or {}
            teacher_valid_mask = make_valid_mask(
                depth_grid,
                conf_grid,
                conf_quantile=float(geom_cfg.get("conf_quantile", 0.1)),
                min_depth=geom_cfg.get("min_depth", None),
                max_depth=geom_cfg.get("max_depth", None),
            )
            points = depth_to_points(depth_grid, intrinsics_grid, vggt_out["extrinsics"])
            images_grid = resize_hw(images, grid_size).contiguous()
            clip_feat = clip.encode_views(images)
            dense_result = clip.encode_dense_views(
                images,
                grid_size,
                depth=depth_grid,
                confidence=conf_grid,
                feature_dim=int(cfg.gaussian.feature_dim),
                return_raw=bool(cache_cfg.get("save_raw_clip_feature", False)),
            )
            if bool(cache_cfg.get("save_raw_clip_feature", False)):
                target_feature, dense_meta, raw_clip_dense = dense_result
            else:
                target_feature, dense_meta = dense_result
                raw_clip_dense = None
            xyz, color, init_feat, point_metrics = sample_points(
                points,
                images_grid,
                target_feature,
                teacher_valid_mask,
                max_points=int(cfg.gaussian.max_points),
                stride=int(cfg.gaussian.init_stride),
                per_view_balanced=bool(cfg.gaussian.get("per_view_balanced_sampling", True)),
            )
            if xyz.numel() == 0:
                raise ValueError("No valid Gaussian initialization points")
            field = GaussianFeatureField(
                xyz=xyz,
                color=color,
                feature=init_feat,
                init_scale=float(cfg.gaussian.init_scale),
                init_opacity=float(cfg.gaussian.init_opacity),
                optimize_feature=bool(cfg.gaussian.get("optimize_feature", False)),
                optimize_color=bool(cfg.gaussian.get("optimize_color", False)),
            ).to(device)
            render, metrics = fit_gaussian_field(field, renderer, depth_grid, target_feature, teacher_valid_mask, intrinsics_grid, vggt_out["extrinsics"], cfg.gaussian, cfg.loss)

            alpha_threshold = float(cache_cfg.get("alpha_valid_threshold", cfg.gaussian.get("alpha_valid_threshold", 1e-4)))
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
            if "text_alignment_embedding" in vggt_out:
                targets["T_vggt_text_alignment_embedding"] = vggt_out["text_alignment_embedding"].detach().to("cpu", dtype=torch.bfloat16)
            if "text_alignment_token" in vggt_out:
                targets["T_vggt_text_alignment_token"] = vggt_out["text_alignment_token"].detach().to("cpu", dtype=torch.bfloat16)
            if bool(cache_cfg.get("save_dense_feature_target", False)):
                targets["T_dense_feature_target"] = target_feature.detach().to("cpu", dtype=torch.bfloat16)
            if raw_clip_dense is not None:
                targets["T_clip_dense_feature_raw"] = raw_clip_dense.detach().to("cpu", dtype=torch.bfloat16)
            mosaic_cfg = cfg.get("mosaic", {}) or {}
            if bool(mosaic_cfg.get("enabled", True)):
                mosaic_grid = tuple(int(x) for x in mosaic_cfg.get("grid_size", cfg.gaussian.target_grid_size))
                mosaic_layout = str(mosaic_cfg.get("layout", "robotwin"))
                mosaic_targets, mosaic_meta = _compose_mosaic(
                    render["feature_map"].detach(),
                    render["dep"].detach(),
                    render["alpha"].detach(),
                    valid_mask.detach(),
                    camera_keys,
                    mosaic_grid,
                    mosaic_layout,
                )
                targets.update({key: value.to("cpu", dtype=torch.bfloat16 if value.ndim == 3 else value.dtype) for key, value in mosaic_targets.items()})
            else:
                mosaic_meta = {"enabled": False}
            payload: dict[str, Any] = {
                "version": int(cache_cfg.version),
                "schema": "frozen_feature_z_mosaic_v1",
                "cache_key": cache_key,
                "dataset": {"global_idx": sample.global_idx, "task": sample.task, "subset": work_item if subset_rows is not None else None, **sample.metadata},
                "text": text_meta,
                "alignment": {
                    "num_frames": int(source_cfg.num_frames),
                    "action_video_freq_ratio": int(source_cfg.action_video_freq_ratio),
                    "video_sample_indices": video_sample_indices(source_cfg.num_frames, source_cfg.action_video_freq_ratio),
                    "target_frame_policy": str(source_cfg.target_frame_policy),
                    "target_offset": int(target_offset),
                    "camera_keys": camera_keys,
                    "per_view_grid_size": list(grid_size),
                    "mosaic": mosaic_meta,
                },
                "teacher": {"vggt_omega": str(cfg.vggt_omega.checkpoint_path), **clip.metadata(), **dense_meta},
                "gaussian": {
                    "feature_dim": int(cfg.gaussian.feature_dim),
                    "feature_target": "frozen_compact_clip_dense_per_view",
                    "feature_z_optimized": bool(field.feature_z.requires_grad),
                    "renderer_backend": renderer.backend,
                    "init_feature_source": str(cfg.gaussian.get("init_feature_source", "compact_dense_target")),
                    "num_points": int(field.xyz.shape[0]),
                    **point_metrics,
                    **metrics,
                },
                "targets": targets,
            }
            validation_errors = validate_payload(payload, cfg)
            quality_ok = bool(metrics.get("quality_ok", True)) and not validation_errors
            if validation_errors:
                payload["gaussian"]["validation_errors"] = validation_errors
            if bool(cache_cfg.save_gaussian_state):
                payload["gaussian_state"] = field.state_payload()
            status = "ok" if quality_ok else "rejected"
            output_path = path if quality_ok else root / "rejected" / path.name
            if bool(cache_cfg.atomic_write):
                atomic_torch_save(payload, output_path)
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, str(output_path))
            if bool(cache_cfg.save_debug_images):
                debug_render = {**render, "feature_target": target_feature, "teacher_valid_mask": teacher_valid_mask, "render_valid_mask": render_valid_mask}
                save_debug_maps(root / "debug" / cache_key, debug_render)
            append_jsonl(manifest_path, {"idx": idx, "cache_key": cache_key, "path": str(output_path), "status": status, **text_meta, **metrics, "validation_errors": validation_errors})
            if quality_ok:
                counts["new"] += 1
            else:
                counts["rejected"] += 1
        except Exception as exc:
            counts["error"] += 1
            append_jsonl(manifest_path, {"idx": idx, "status": "error", "error": repr(exc)})
    return counts
