"""Helpers for evaluating FastWAM checkpoints on LIBERO-Plus tasks.

This module intentionally avoids importing `libero` at module import time so the
caller can first point `LIBERO_CONFIG_PATH` to a runtime config that references
the LIBERO-Plus assets / BDDL / init-state trees.
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path
from typing import NamedTuple

import yaml


DEFAULT_LIBERO_PLUS_ROOT = Path(
    os.environ.get(
        "LIBERO_PLUS_ROOT",
        "/data/zijianzhang/libero_datasets/LIBERO-plus/libero/libero",
    )
)
DEFAULT_LIBERO_PLUS_CONFIG_DIR = Path(
    os.environ.get(
        "FASTWAM_LIBERO_PLUS_CONFIG_DIR",
        "/tmp/fastwam_libero_plus_config",
    )
)

PLUS_CATEGORY_ALIASES = {
    "background": "Background Textures",
    "background_textures": "Background Textures",
    "camera": "Camera Viewpoints",
    "camera_viewpoints": "Camera Viewpoints",
    "language": "Language Instructions",
    "language_instructions": "Language Instructions",
    "light": "Light Conditions",
    "light_conditions": "Light Conditions",
    "layout": "Objects Layout",
    "objects_layout": "Objects Layout",
    "robot": "Robot Initial States",
    "robot_initial_states": "Robot Initial States",
    "sensor": "Sensor Noise",
    "sensor_noise": "Sensor Noise",
}

PLUS_CATEGORY_TO_SHORT = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}

PLUS_SHORT_ORDER = [
    "Camera",
    "Robot",
    "Language",
    "Light",
    "Background",
    "Noise",
    "Layout",
]

PLUS_EVAL_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


class PlusTask(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    language_bddl_file: str
    init_states_file: str
    plus_category: str
    plus_difficulty_level: int | None
    plus_original_id: int


def normalize_plus_category(category: str | None) -> str | None:
    if category is None:
        return None
    normalized = str(category).strip()
    if normalized == "":
        return None
    key = normalized.lower().replace("-", "_").replace(" ", "_")
    return PLUS_CATEGORY_ALIASES.get(key, normalized)


def normalize_plus_categories(categories) -> tuple[str, ...]:
    if categories is None:
        return ()

    if isinstance(categories, str):
        raw_items = re.split(r"[,;\n]", categories)
    else:
        raw_items = list(categories)

    normalized_items: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        normalized = normalize_plus_category(item)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        normalized_items.append(normalized)
    return tuple(normalized_items)


def short_plus_category(category: str | None) -> str:
    if category is None:
        return "Unknown"
    return PLUS_CATEGORY_TO_SHORT.get(str(category), "Unknown")


def configure_libero_plus_runtime(
    plus_root: str | Path,
    config_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write a runtime LIBERO config that points to LIBERO-Plus assets.

    Returns:
        `(resolved_plus_root, config_file_path)`
    """
    resolved_plus_root = Path(plus_root).expanduser().resolve()
    if not resolved_plus_root.exists():
        raise FileNotFoundError(f"LIBERO-Plus root not found: {resolved_plus_root}")
    _install_wand_stub_if_needed()
    _activate_libero_plus_python(resolved_plus_root)

    config_home = Path(config_dir or DEFAULT_LIBERO_PLUS_CONFIG_DIR).expanduser().resolve()
    config_home.mkdir(parents=True, exist_ok=True)
    config_file = config_home / "config.yaml"

    payload = {
        "benchmark_root": str(resolved_plus_root),
        "bddl_files": str(resolved_plus_root / "bddl_files"),
        "init_states": str(resolved_plus_root / "init_files"),
        # Avoid noisy warnings from LIBERO path checks; this path only needs to exist.
        "datasets": str(resolved_plus_root),
        "assets": str(resolved_plus_root / "assets"),
    }
    config_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_home)
    return resolved_plus_root, config_file


def _activate_libero_plus_python(resolved_plus_root: Path) -> None:
    """Ensure imports resolve to the LIBERO-Plus Python package tree.

    The copied runtime root ends at `.../libero/libero`, but imports use
    `libero.libero...`, so the path we need on `sys.path` is the repository
    root `.../LIBERO-plus`, matching the setup used by other projects such as
    VLA-JEPA.
    """
    repo_root = resolved_plus_root.parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    # If another LIBERO package was imported earlier in this process, force the
    # subsequent import to resolve against the LIBERO-Plus tree we just added.
    stale_modules = [name for name in sys.modules if name == "libero" or name.startswith("libero.")]
    for name in stale_modules:
        del sys.modules[name]

    # `libero` is installed as a namespace package in multiple locations on this
    # machine. If both the upstream OpenPI copy and the LIBERO-Plus copy are
    # present, Python merges them into one package and may import env modules
    # from the wrong tree. Pin the namespace to the LIBERO-Plus path only.
    import importlib

    libero_pkg = importlib.import_module("libero")
    if hasattr(libero_pkg, "__path__"):
        plus_pkg_root = str(repo_root / "libero")
        libero_pkg.__path__[:] = [plus_pkg_root]


def _install_wand_stub_if_needed() -> None:
    """Provide a minimal stub when Wand is unavailable.

    LIBERO-Plus imports Wand unconditionally for motion-blur corruptions. On
    this machine the evaluation env does not have the `wand` package installed.
    A lightweight stub keeps non-noise tasks runnable and degrades the
    motion-blur branch to identity instead of failing during import.
    """
    if "wand" in sys.modules:
        return
    try:
        import wand  # type: ignore  # pragma: no cover
        return
    except ModuleNotFoundError:
        pass

    wand_module = types.ModuleType("wand")
    wand_api_module = types.ModuleType("wand.api")
    wand_image_module = types.ModuleType("wand.image")

    class _StubLibrary:
        def __init__(self):
            self.MagickMotionBlurImage = lambda *args, **kwargs: None

    class _StubImage:
        def __init__(self, blob=None, *args, **kwargs):
            self._blob = blob or b""
            self.wand = object()

        def make_blob(self):
            return self._blob

    wand_api_module.library = _StubLibrary()
    wand_image_module.Image = _StubImage
    wand_module.api = wand_api_module
    wand_module.image = wand_image_module

    sys.modules["wand"] = wand_module
    sys.modules["wand.api"] = wand_api_module
    sys.modules["wand.image"] = wand_image_module


def _resolve_bddl_file(suite_name: str, task_name: str) -> str:
    """Resolve the on-disk BDDL filename used to read task language."""
    filename = f"{task_name}.bddl"
    candidate = DEFAULT_LIBERO_PLUS_ROOT / "bddl_files" / suite_name / filename
    if candidate.exists():
        return filename

    if "_view_" in task_name:
        fallback = f"{task_name.split('_view_')[0]}.bddl"
        fallback_path = DEFAULT_LIBERO_PLUS_ROOT / "bddl_files" / suite_name / fallback
        if fallback_path.exists():
            return fallback

    return filename


def _resolve_env_bddl_file(task_name: str) -> str:
    """Return the env-facing BDDL name.

    Camera variants must keep the synthetic `_view_..._initstate_...` suffix and
    intentionally omit the `.bddl` extension so LIBERO can parse the perturbation.
    """
    return task_name


def _grab_language_from_filename(suite_name: str, filename: str) -> str:
    import libero.libero.envs.bddl_utils as BDDLUtils
    from libero.libero import get_libero_path

    bddl_file_path = os.path.join(get_libero_path("bddl_files"), suite_name, filename)
    if not os.path.exists(bddl_file_path):
        raise FileNotFoundError(f"LIBERO-Plus BDDL not found: {bddl_file_path}")
    problem_info = BDDLUtils.get_problem_info(bddl_file_path)
    return str(problem_info["language_instruction"])


def _resolve_init_states_path(task: PlusTask) -> Path:
    from libero.libero import get_libero_path

    init_root = Path(get_libero_path("init_states"))
    init_name = task.init_states_file
    suite = task.problem_folder

    if "_language_" in init_name:
        resolved = init_root / suite / (init_name.split("_language_")[0] + "." + init_name.split(".")[-1])
    else:
        resolved = init_root / suite / init_name
        if "_view_" in init_name:
            resolved = init_root / suite / (init_name.split("_view_")[0] + "." + init_name.split(".")[-1])
        else:
            if "_table_" in init_name:
                resolved = init_root / suite / re.sub(r"_table_\d+", "", init_name)
            if "_tb_" in init_name:
                resolved = init_root / suite / re.sub(r"_tb_\d+", "", init_name)
            if "_light_" in init_name:
                resolved = init_root / suite / (init_name.split("_light_")[0] + "." + init_name.split(".")[-1])
            if "_add_" in init_name or "_level" in init_name:
                resolved = init_root / "libero_newobj" / suite / init_name
    return resolved


class LiberoPlusBenchmark:
    """Minimal benchmark wrapper with the API expected by FastWAM evaluation."""

    def __init__(
        self,
        suite_name: str,
        *,
        plus_root: str | Path = DEFAULT_LIBERO_PLUS_ROOT,
        category: str | None = None,
        exclude_categories=None,
        task_classification_path: str | Path | None = None,
    ):
        self.name = str(suite_name)
        self.plus_root = Path(plus_root).expanduser().resolve()
        self.category = normalize_plus_category(category)
        self.exclude_categories = set(normalize_plus_categories(exclude_categories))
        self.task_classification_path = Path(
            task_classification_path or self.plus_root / "benchmark" / "task_classification.json"
        ).expanduser().resolve()
        if not self.task_classification_path.exists():
            raise FileNotFoundError(
                f"LIBERO-Plus task_classification.json not found: {self.task_classification_path}"
            )

        self.task_embs = None
        self.tasks = self._load_tasks()
        self.n_tasks = len(self.tasks)

    def _load_tasks(self) -> list[PlusTask]:
        classification = json.loads(self.task_classification_path.read_text(encoding="utf-8"))
        suites = PLUS_EVAL_SUITES if self.name == "libero_mix" else (self.name,)
        tasks: list[PlusTask] = []

        for suite_name in suites:
            if suite_name not in classification:
                raise ValueError(
                    f"Suite {suite_name!r} missing from {self.task_classification_path}. "
                    f"Available: {sorted(classification.keys())}"
                )
            records = classification[suite_name]

            for record in records:
                record_category = normalize_plus_category(record.get("category"))
                if self.category is not None and record_category != self.category:
                    continue
                if record_category in self.exclude_categories:
                    continue
                task_name = str(record["name"])
                language_filename = _resolve_bddl_file(suite_name, task_name)
                tasks.append(
                    PlusTask(
                        name=task_name,
                        language=_grab_language_from_filename(suite_name, language_filename),
                        problem="Libero",
                        problem_folder=suite_name,
                        bddl_file=_resolve_env_bddl_file(task_name)
                        if "_view_" in task_name
                        else language_filename,
                        language_bddl_file=language_filename,
                        init_states_file=f"{task_name}.pruned_init",
                        plus_category=record_category or str(record.get("category", "")),
                        plus_difficulty_level=record.get("difficulty_level"),
                        plus_original_id=int(record.get("id", 0)),
                    )
                )
        return tasks

    def get_num_tasks(self) -> int:
        return self.n_tasks

    def get_task_names(self) -> list[str]:
        return [task.name for task in self.tasks]

    def get_task(self, i: int) -> PlusTask:
        return self.tasks[i]

    def get_task_init_states(self, i: int):
        import torch

        task = self.tasks[i]
        path = _resolve_init_states_path(task)
        if not path.exists():
            raise FileNotFoundError(
                f"Failed to locate LIBERO-Plus init states for {task.name}: {path}"
            )
        init_states = torch.load(path, weights_only=False)
        if "_add_" in task.init_states_file or "_level" in task.init_states_file:
            init_states = init_states.reshape(1, -1)
        return init_states
