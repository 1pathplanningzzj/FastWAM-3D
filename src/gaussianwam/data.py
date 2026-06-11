from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset


@dataclass
class RawMultiViewSample:
    global_idx: int
    task: str
    images: torch.Tensor
    image_is_pad: torch.Tensor
    metadata: dict[str, Any]


def build_raw_dataset(source_cfg) -> BaseLerobotDataset:
    camera_keys = list(source_cfg.camera_keys)
    shape_meta = {
        "images": [
            {"key": key, "raw_shape": [3, 480, 640], "shape": [3, 240, 320]}
            for key in camera_keys
        ],
        "action": [{"key": "default", "raw_shape": 14, "shape": 14}],
        "state": [{"key": "default", "raw_shape": 14, "shape": 14}],
    }
    return BaseLerobotDataset(
        dataset_dirs=list(source_cfg.dataset_dirs),
        shape_meta=shape_meta,
        obs_size=int(source_cfg.num_frames),
        action_size=int(source_cfg.num_frames) - 1,
        val_set_proportion=float(source_cfg.val_set_proportion),
        is_training_set=bool(source_cfg.is_training_set),
        global_sample_stride=int(source_cfg.global_sample_stride),
        episode_subset_manifest=source_cfg.get("episode_subset_manifest", None),
    )


def video_sample_indices(num_frames: int, action_video_freq_ratio: int) -> list[int]:
    return list(range(0, int(num_frames), int(action_video_freq_ratio)))


def select_target_offset(source_cfg) -> int:
    indices = video_sample_indices(source_cfg.num_frames, source_cfg.action_video_freq_ratio)
    policy = str(source_cfg.target_frame_policy)
    if policy == "last_video_frame":
        return indices[-1]
    if policy == "first_video_frame":
        return indices[0]
    if policy.startswith("offset:"):
        return int(policy.split(":", 1)[1])
    raise ValueError(f"Unsupported target_frame_policy: {policy}")


def get_raw_multiview_sample(dataset: BaseLerobotDataset, idx: int, camera_keys: list[str], target_offset: int) -> RawMultiViewSample:
    sample = dataset[idx]
    views = []
    for key in camera_keys:
        image = sample["images"][key]
        if image.ndim != 4:
            raise ValueError(f"Expected image sequence [T,C,H,W] for {key}, got {tuple(image.shape)}")
        views.append(image[int(target_offset)].float() / 255.0)
    images = torch.stack(views, dim=0).contiguous()
    metadata = {}
    for key, value in sample.items():
        if key in {"images", "action", "state"}:
            continue
        if torch.is_tensor(value):
            if value.numel() == 1:
                metadata[key] = value.item()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
    image_is_pad = torch.as_tensor(sample.get("image_is_pad", torch.zeros(int(target_offset) + 1, dtype=torch.bool)))
    if image_is_pad.ndim > 0:
        image_is_pad = image_is_pad[int(target_offset)]
    image_is_pad = image_is_pad.reshape(())
    return RawMultiViewSample(
        global_idx=int(sample.get("idx", idx)),
        task=str(sample.get("task", metadata.get("task", ""))),
        images=images,
        image_is_pad=image_is_pad.bool(),
        metadata=metadata,
    )
