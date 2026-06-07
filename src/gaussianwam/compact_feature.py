from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class PCACompactProjector:
    def __init__(self, mean: torch.Tensor, components: torch.Tensor, metadata: dict[str, Any] | None = None):
        if components.ndim != 2:
            raise ValueError(f"Expected PCA components [D,C], got {tuple(components.shape)}")
        if mean.ndim != 1 or mean.shape[0] != components.shape[1]:
            raise ValueError(f"Expected PCA mean [{components.shape[1]}], got {tuple(mean.shape)}")
        self.mean = mean.float()
        self.components = components.float()
        self.metadata = dict(metadata or {})

    @property
    def input_dim(self) -> int:
        return int(self.components.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def to(self, device: str | torch.device) -> "PCACompactProjector":
        device = torch.device(device)
        self.mean = self.mean.to(device)
        self.components = self.components.to(device)
        return self

    def encode(self, features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {features.shape[-1]}")
        compact = (features.float() - self.mean.to(features.device)) @ self.components.to(features.device).t()
        if normalize:
            compact = F.normalize(compact, dim=-1)
        return compact

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "pca",
            "mean": self.mean.detach().cpu(),
            "components": self.components.detach().cpu(),
            "metadata": self.metadata,
        }

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "PCACompactProjector":
        payload = torch.load(str(path), map_location=map_location)
        if payload.get("type") != "pca":
            raise ValueError(f"Unsupported compact projector type: {payload.get('type')}")
        return cls(payload["mean"], payload["components"], payload.get("metadata", {}))


def fit_pca(features: torch.Tensor, output_dim: int) -> PCACompactProjector:
    if features.ndim != 2:
        raise ValueError(f"Expected features [N,C], got {tuple(features.shape)}")
    if features.shape[0] < int(output_dim):
        raise ValueError(f"Need at least {output_dim} feature vectors, got {features.shape[0]}")
    features = features.float()
    mean = features.mean(dim=0)
    centered = features - mean
    cov = centered.t().matmul(centered) / max(int(features.shape[0]) - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    components = eigvecs[:, order[: int(output_dim)]].t().contiguous()
    metadata = {
        "input_dim": int(features.shape[1]),
        "output_dim": int(output_dim),
        "num_features": int(features.shape[0]),
        "explained_variance": eigvals[order[: int(output_dim)]].detach().cpu(),
        "explained_variance_total": float(eigvals.clamp_min(0).sum().item()),
    }
    return PCACompactProjector(mean, components, metadata)


def save_projector(projector: PCACompactProjector, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), str(path))
