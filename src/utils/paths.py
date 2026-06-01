"""
src/utils/paths.py
==================
paths.yaml loader.

Returns a namespace object so callers can use attribute access:
    from src.utils.paths import load_paths
    p = load_paths()
    p.data.ff_plus_plus       # -> Path
    p.output.root             # -> Path
    p.weights.noiseprint      # -> Path

Environment variables override values from paths.yaml.
Relative paths are resolved relative to the repo root (the directory
containing paths.yaml).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Repo root = directory containing this __file__'s grandparent (src/utils/)
_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
_PATHS_YAML = _REPO_ROOT / "paths.yaml"

# Environment variable names (override paths.yaml values)
_ENV_MAP = {
    ("data", "ff_plus_plus"): "THESIS_FFPP_ROOT",
    ("data", "celeb_df"):     "THESIS_CELEBDF_ROOT",
    ("data", "dfdc"):         "THESIS_DFDC_ROOT",
    ("data", "dff"):          "THESIS_DFF_ROOT",
    ("output", "root"):       "THESIS_OUTPUT_ROOT",
    ("output", "cache"):      "THESIS_CACHE_ROOT",
    ("weights", "noiseprint"):"THESIS_NOISEPRINT_WEIGHTS",
    ("weights", "sbi_baseline"):"THESIS_SBI_WEIGHTS",
    ("tiny", "root"):         "THESIS_TINY_ROOT",
    ("tiny", "manifest"):     "THESIS_TINY_ROOT",  # derived
}


class _Namespace:
    """Thin wrapper so dict values are accessible as attributes."""
    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, _Namespace(v))
            else:
                setattr(self, k, v)

    def __repr__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"Namespace({attrs})"


def _resolve(value: Any, root: Path) -> Any:
    """Convert a string path to a resolved Path object."""
    if not isinstance(value, str):
        return value
    p = Path(value)
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def load_paths(yaml_path: str | Path | None = None) -> _Namespace:
    """Load paths.yaml and apply environment-variable overrides.

    Parameters
    ----------
    yaml_path : path to paths.yaml. Defaults to <repo_root>/paths.yaml.

    Returns
    -------
    _Namespace with nested attributes matching the paths.yaml structure.
    All path values are resolved Path objects.
    """
    yaml_path = Path(yaml_path) if yaml_path else _PATHS_YAML
    if not yaml_path.exists():
        raise FileNotFoundError(f"paths.yaml not found at {yaml_path}. "
                                 "Run from the repo root or pass yaml_path=.")

    with open(yaml_path) as f:
        raw: dict = yaml.safe_load(f) or {}

    # Apply environment variable overrides
    for (section, key), env_var in _ENV_MAP.items():
        if env_var in os.environ:
            raw.setdefault(section, {})[key] = os.environ[env_var]

    # Resolve all string values to Paths
    repo_root = yaml_path.parent
    resolved: dict = {}
    for section, sub in raw.items():
        if isinstance(sub, dict):
            resolved[section] = {
                k: _resolve(v, repo_root) for k, v in sub.items()
            }
        else:
            resolved[section] = _resolve(sub, repo_root)

    return _Namespace(resolved)
