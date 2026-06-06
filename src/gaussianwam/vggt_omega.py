from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


class VGGTOmegaTeacher:
    def __init__(self, cfg, device: str | torch.device):
        root = Path(__file__).resolve().parents[2]
        third_party = root / "third_party" / "vggt-omega"
        sys.path.insert(0, str(third_party))
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.pose_enc import encoding_to_camera

        self.encoding_to_camera = encoding_to_camera
        self.cfg = cfg
        self.device = torch.device(device)
        self.image_resolution = int(cfg.image_resolution)
        self.dtype = torch.bfloat16 if str(cfg.dtype).lower() == "bf16" else torch.float32
        checkpoint = Path(str(cfg.checkpoint_path))
        if not checkpoint.exists():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint}")
        self.model = VGGTOmega(enable_alignment=bool(cfg.get("enable_alignment", False))).to(self.device).eval()
        state = torch.load(str(checkpoint), map_location="cpu")
        self.model.load_state_dict(state)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        _, _, h, w = images.shape
        mode = str(self.cfg.get("preprocess_mode", "max_size"))
        if mode == "max_size":
            scale = self.image_resolution / max(h, w)
            new_h = max(1, round(h * scale))
            new_w = max(1, round(w * scale))
        else:
            new_h = new_w = self.image_resolution
        return F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.preprocess(images)
        with torch.inference_mode():
            pred = self.model(x)
        extrinsics, intrinsics = self.encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
        camera_and_register = pred["camera_and_register_tokens"]
        return {
            "images": pred["images"].detach(),
            "pose_enc": pred["pose_enc"].detach(),
            "extrinsics": extrinsics.detach(),
            "intrinsics": intrinsics.detach(),
            "depth": pred["depth"].detach(),
            "depth_conf": pred["depth_conf"].detach(),
            "camera_tokens": camera_and_register[:, :, :1].detach(),
            "register_tokens": camera_and_register[:, :, 1:].detach(),
        }
