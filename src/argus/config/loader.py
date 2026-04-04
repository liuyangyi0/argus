"""YAML configuration loader with validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from argus.config.schema import ArgusConfig


def load_config(config_path: str | Path) -> ArgusConfig:
    """Load and validate configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return ArgusConfig.model_validate(raw)


def save_config(config: ArgusConfig, config_path: str | Path) -> None:
    """Save configuration to a YAML file (atomic write)."""
    path = Path(config_path)
    data = config.model_dump(mode="json")
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    tmp.replace(path)


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
