import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        episode_subset_manifest: Optional[str] = None,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        gaussian_teacher: Optional[Any] = None,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            episode_subset_manifest=episode_subset_manifest,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.gaussian_teacher_cfg = self._normalize_gaussian_teacher_cfg(gaussian_teacher)
        self.gaussian_teacher_enabled = bool(self.gaussian_teacher_cfg.get("enabled", False))
        self.gaussian_teacher_targets = self._enabled_gaussian_teacher_targets(self.gaussian_teacher_cfg)
        self.gaussian_teacher_index = {}
        self.gaussian_teacher_indices = None
        self.gaussian_teacher_zero_targets = None
        if self.gaussian_teacher_enabled:
            self._init_gaussian_teacher_cache()

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    @staticmethod
    def _normalize_gaussian_teacher_cfg(gaussian_teacher: Optional[Any]) -> dict[str, Any]:
        if gaussian_teacher is None:
            return {}
        if isinstance(gaussian_teacher, DictConfig):
            gaussian_teacher = OmegaConf.to_container(gaussian_teacher, resolve=True)
        if not isinstance(gaussian_teacher, dict):
            raise TypeError(f"`gaussian_teacher` must be dict-like, got {type(gaussian_teacher)}")
        return dict(gaussian_teacher)

    @staticmethod
    def _enabled_gaussian_teacher_targets(cfg: dict[str, Any]) -> set[str]:
        targets = cfg.get("targets", ["dense_3d", "depth", "alpha", "valid_mask"])
        if isinstance(targets, DictConfig):
            targets = OmegaConf.to_container(targets, resolve=True)
        if isinstance(targets, dict):
            enabled = {str(key) for key, value in targets.items() if bool(value)}
        else:
            enabled = {str(value) for value in targets}
        enabled.add("valid_mask")
        return enabled

    def _resolve_existing_path(self, path_value: str | os.PathLike, *, base_dir: Optional[Path] = None) -> Path:
        path = Path(path_value)
        candidates = [path]
        if not path.is_absolute():
            candidates.append(Path.cwd() / path)
            if base_dir is not None:
                candidates.append(base_dir / path)
                candidates.append(base_dir / path.name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _gaussian_teacher_manifest_path(self) -> Path:
        cfg = self.gaussian_teacher_cfg
        manifest = cfg.get("manifest_path", None) or cfg.get("manifest", None)
        if manifest is not None:
            path = self._resolve_existing_path(manifest)
            if not path.exists():
                raise FileNotFoundError(f"Gaussian teacher manifest not found: {manifest}")
            return path

        cache_dir = cfg.get("cache_dir", None)
        version = cfg.get("version", None)
        namespace = cfg.get("namespace", None)
        split = cfg.get("split", "train" if self.lerobot_dataset.is_training_set else "val")
        if cache_dir is None or version is None or namespace is None:
            raise ValueError(
                "Gaussian teacher config must provide either `manifest_path` or "
                "`cache_dir`, `version`, and `namespace`."
            )
        version_name = str(version)
        if not version_name.startswith("v"):
            version_name = f"v{version_name}"
        path = self._resolve_existing_path(Path(cache_dir) / version_name / str(namespace) / str(split) / "manifest.jsonl")
        if not path.exists():
            raise FileNotFoundError(f"Gaussian teacher manifest not found: {path}")
        return path

    def _resolve_gaussian_cache_path(self, row: dict[str, Any], manifest_path: Path) -> Path:
        raw_path = row.get("path") or row.get("cache_path") or row.get("pt_path") or row.get("file")
        if raw_path is None:
            raise ValueError(f"Gaussian teacher manifest row is missing a cache path: {row}")
        path = self._resolve_existing_path(raw_path, base_dir=manifest_path.parent)
        if not path.exists():
            raise FileNotFoundError(f"Gaussian teacher cache file not found: {raw_path}")
        return path

    def _init_gaussian_teacher_cache(self):
        manifest_path = self._gaussian_teacher_manifest_path()
        latest_by_idx: dict[int, dict[str, Any]] = {}
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                if "idx" not in row:
                    continue
                sample_idx = int(row["idx"])
                row = dict(row)
                row["_resolved_path"] = self._resolve_gaussian_cache_path(row, manifest_path)
                latest_by_idx[sample_idx] = row

        if not latest_by_idx:
            raise ValueError(f"Gaussian teacher manifest has no ok rows: {manifest_path}")

        self.gaussian_teacher_manifest_path = manifest_path
        self.gaussian_teacher_index = latest_by_idx
        self.gaussian_teacher_indices = sorted(latest_by_idx)
        first_row = latest_by_idx[self.gaussian_teacher_indices[0]]
        payload = torch.load(first_row["_resolved_path"], map_location="cpu")
        targets = payload.get("targets", payload)
        self.gaussian_teacher_zero_targets = {
            "T_valid_mask_mosaic": torch.zeros_like(targets["T_valid_mask_mosaic"], dtype=torch.bool),
        }
        if "dense_3d" in self.gaussian_teacher_targets:
            self.gaussian_teacher_zero_targets["T_gaussian_feature_mosaic"] = torch.zeros_like(targets["T_gaussian_feature_mosaic"])
        if "depth" in self.gaussian_teacher_targets:
            self.gaussian_teacher_zero_targets["T_depth_mosaic"] = torch.zeros_like(targets["T_depth_mosaic"])
        if "alpha" in self.gaussian_teacher_targets:
            self.gaussian_teacher_zero_targets["T_alpha_mosaic"] = torch.zeros_like(targets["T_alpha_mosaic"])
        logger.info(
            "Loaded Gaussian teacher manifest: path=%s ok_unique=%d restrict_to_cache=%s",
            manifest_path,
            len(self.gaussian_teacher_indices),
            bool(self.gaussian_teacher_cfg.get("restrict_to_cache", False)),
        )

    def _dataset_index_for_request(self, idx: int) -> int:
        if self.gaussian_teacher_enabled and bool(self.gaussian_teacher_cfg.get("restrict_to_cache", False)):
            return int(self.gaussian_teacher_indices[int(idx)])
        return int(idx)

    def _random_request_index(self) -> int:
        return int(np.random.randint(len(self)))

    def _zero_gaussian_teacher(self) -> dict[str, torch.Tensor]:
        if self.gaussian_teacher_zero_targets is None:
            return {}
        data = {key: value.clone() for key, value in self.gaussian_teacher_zero_targets.items()}
        data["gaussianwam_has_teacher"] = torch.tensor(False, dtype=torch.bool)
        return data

    @staticmethod
    def _normalize_nested_metadata(value):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return {str(k): RobotVideoDataset._normalize_nested_metadata(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [RobotVideoDataset._normalize_nested_metadata(v) for v in value]
        return value

    def _load_gaussian_teacher(self, sample_idx: int) -> dict[str, torch.Tensor]:
        if not self.gaussian_teacher_enabled:
            return {}
        row = self.gaussian_teacher_index.get(int(sample_idx))
        if row is None:
            return self._zero_gaussian_teacher()

        payload = torch.load(row["_resolved_path"], map_location="cpu")
        dataset_meta = payload.get("dataset", {})
        payload_idx = int(dataset_meta.get("global_idx", dataset_meta.get("idx", sample_idx)))
        if payload_idx != int(sample_idx):
            raise ValueError(
                f"Gaussian teacher idx mismatch: sample_idx={sample_idx}, payload_idx={payload_idx}, path={row['_resolved_path']}"
            )

        alignment = payload.get("alignment", {})
        expected_offset = self.gaussian_teacher_cfg.get("expected_target_offset", None)
        if expected_offset is not None and int(alignment.get("target_offset", -1)) != int(expected_offset):
            raise ValueError(
                "Gaussian teacher target_offset mismatch: "
                f"expected={expected_offset}, got={alignment.get('target_offset')}, path={row['_resolved_path']}"
            )
        expected_camera_keys = self.gaussian_teacher_cfg.get("expected_camera_keys", None)
        if expected_camera_keys is not None:
            actual_camera_keys = list(alignment.get("camera_keys", []))
            if actual_camera_keys != list(expected_camera_keys):
                raise ValueError(
                    "Gaussian teacher camera_keys mismatch: "
                    f"expected={list(expected_camera_keys)}, got={actual_camera_keys}, path={row['_resolved_path']}"
                )
        expected_mosaic = self.gaussian_teacher_cfg.get("expected_mosaic", None)
        if expected_mosaic is not None:
            expected_mosaic = self._normalize_nested_metadata(expected_mosaic)
            actual_mosaic = self._normalize_nested_metadata(alignment.get("mosaic", None))
            if actual_mosaic != expected_mosaic:
                raise ValueError(
                    "Gaussian teacher mosaic metadata mismatch: "
                    f"expected={expected_mosaic}, got={actual_mosaic}, path={row['_resolved_path']}"
                )

        targets = payload.get("targets", payload)
        data = {
            "gaussianwam_has_teacher": torch.tensor(True, dtype=torch.bool),
            "T_valid_mask_mosaic": targets["T_valid_mask_mosaic"].detach().cpu().bool().contiguous(),
        }
        if "dense_3d" in self.gaussian_teacher_targets:
            data["T_gaussian_feature_mosaic"] = targets["T_gaussian_feature_mosaic"].detach().cpu().contiguous()
        if "depth" in self.gaussian_teacher_targets:
            data["T_depth_mosaic"] = targets["T_depth_mosaic"].detach().cpu().float().contiguous()
        if "alpha" in self.gaussian_teacher_targets:
            data["T_alpha_mosaic"] = targets["T_alpha_mosaic"].detach().cpu().float().contiguous()
        return data

    def __len__(self):
        if self.gaussian_teacher_enabled and bool(self.gaussian_teacher_cfg.get("restrict_to_cache", False)):
            return len(self.gaussian_teacher_indices)
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = self._dataset_index_for_request(idx)
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            sample_idx = int(sample.get("idx", sample_idx))

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = self._dataset_index_for_request(self._random_request_index())

        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "idx": torch.tensor(sample_idx, dtype=torch.long),
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        data.update(self._load_gaussian_teacher(sample_idx))
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = self._random_request_index()
            data = self._get(random_idx)
        return data
