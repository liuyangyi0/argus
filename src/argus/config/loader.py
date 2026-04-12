"""YAML configuration loader with validation and environment variable overrides.

Environment variable override format::

    ARGUS__<SECTION>__<KEY>=<VALUE>

Examples::

    ARGUS__DASHBOARD__PORT=9090        -> config.dashboard.port = 9090
    ARGUS__LOG_LEVEL=DEBUG             -> config.log_level = "DEBUG"
    ARGUS__ALERTS__WEBHOOK__ENABLED=true -> config.alerts.webhook.enabled = True

Double-underscore (``__``) separates nesting levels.  Values are parsed
as JSON first (for numbers, booleans, lists); if that fails, they are
kept as plain strings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from argus.config.schema import ArgusConfig

_ENV_PREFIX = "ARGUS__"


def _apply_env_overrides(raw: dict) -> dict:
    """Overlay ``ARGUS__*`` environment variables onto the raw config dict."""
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        parts = key[len(_ENV_PREFIX):].lower().split("__")
        if not parts:
            continue

        # Parse value: try JSON first (handles int, float, bool, list)
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            parsed = value

        # Walk into nested dict, creating intermediate dicts as needed
        target = raw
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = parsed

    return raw


def load_config(config_path: str | Path) -> ArgusConfig:
    """Load and validate configuration from a YAML file.

    After loading the YAML, any ``ARGUS__*`` environment variables are
    applied as overrides before Pydantic validation.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # 核心修复：添加 encoding="utf-8"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    raw = _apply_env_overrides(raw)

    return ArgusConfig.model_validate(raw)


def load_config_layered(
    *config_paths: str | Path,
    apply_env: bool = True,
) -> ArgusConfig:
    """Load and merge multiple YAML files (left-to-right precedence).

    Typical usage::

        config = load_config_layered(
            "configs/default.yaml",      # base
            "configs/local.yaml",        # user overrides (optional)
        )

    Missing files in the list are silently skipped (except the first,
    which must exist).
    """
    merged: dict = {}
    for i, path in enumerate(config_paths):
        p = Path(path)
        if not p.exists():
            if i == 0:
                raise FileNotFoundError(f"Base config file not found: {p}")
            continue
        with open(p) as f:
            layer = yaml.safe_load(f) or {}
        _deep_merge(merged, layer)

    if apply_env:
        merged = _apply_env_overrides(merged)

    return ArgusConfig.model_validate(merged)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def save_config(config: ArgusConfig, config_path: str | Path) -> None:
    """Save configuration to a YAML file (atomic write with retry for Windows)."""
    import time

    path = Path(config_path)
    data = config.model_dump(mode="json")
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Atomic rename — retry on Windows where target may be briefly locked
    for attempt in range(3):
        try:
            tmp.replace(path)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.1)


def load_camera_configs(cameras_dir: str | Path) -> list[dict]:
    """Load all camera config files from a directory."""
    dir_path = Path(cameras_dir)
    if not dir_path.is_dir():
        return []

    configs = []
    for yaml_file in sorted(dir_path.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
            if data:
                configs.append(data)
    return configs
