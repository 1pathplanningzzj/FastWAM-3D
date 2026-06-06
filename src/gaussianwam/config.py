from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None):
    cfg = OmegaConf.load(str(path))
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def cfg_get(cfg: Any, key: str, default=None):
    value = cfg.get(key, default)
    return default if value is None else value
