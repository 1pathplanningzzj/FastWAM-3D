from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


def stable_hash(payload: Any, length: int = 12) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def cache_root(output_dir: str | Path, version: int, namespace: str, split: str) -> Path:
    return Path(output_dir) / f"v{int(version)}" / str(namespace) / str(split)


def atomic_torch_save(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def config_hash(cfg: Any) -> str:
    container = OmegaConf.to_container(cfg, resolve=True)
    return stable_hash(container)
