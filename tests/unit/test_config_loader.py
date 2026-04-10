"""Tests for config loader with environment variable overrides."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from argus.config.loader import (
    _apply_env_overrides,
    _deep_merge,
    load_config,
    load_config_layered,
)


class TestEnvOverrides:
    def test_simple_string_override(self):
        raw = {"log_level": "INFO"}
        os.environ["ARGUS__LOG_LEVEL"] = "DEBUG"
        try:
            result = _apply_env_overrides(raw)
            assert result["log_level"] == "DEBUG"
        finally:
            del os.environ["ARGUS__LOG_LEVEL"]

    def test_nested_override(self):
        raw = {"dashboard": {"port": 8080}}
        os.environ["ARGUS__DASHBOARD__PORT"] = "9090"
        try:
            result = _apply_env_overrides(raw)
            assert result["dashboard"]["port"] == 9090
        finally:
            del os.environ["ARGUS__DASHBOARD__PORT"]

    def test_boolean_override(self):
        raw = {"auth": {"enabled": False}}
        os.environ["ARGUS__AUTH__ENABLED"] = "true"
        try:
            result = _apply_env_overrides(raw)
            assert result["auth"]["enabled"] is True
        finally:
            del os.environ["ARGUS__AUTH__ENABLED"]

    def test_deep_nested_override(self):
        raw = {"alerts": {"webhook": {"enabled": False}}}
        os.environ["ARGUS__ALERTS__WEBHOOK__ENABLED"] = "true"
        try:
            result = _apply_env_overrides(raw)
            assert result["alerts"]["webhook"]["enabled"] is True
        finally:
            del os.environ["ARGUS__ALERTS__WEBHOOK__ENABLED"]

    def test_creates_intermediate_dicts(self):
        raw = {}
        os.environ["ARGUS__NEW_SECTION__VALUE"] = "42"
        try:
            result = _apply_env_overrides(raw)
            assert result["new_section"]["value"] == 42
        finally:
            del os.environ["ARGUS__NEW_SECTION__VALUE"]

    def test_non_argus_env_ignored(self):
        raw = {"key": "value"}
        os.environ["OTHER_VAR"] = "ignored"
        try:
            result = _apply_env_overrides(raw)
            assert "other_var" not in result
        finally:
            del os.environ["OTHER_VAR"]

    def test_json_list_value(self):
        raw = {}
        os.environ["ARGUS__PERSON_FILTER__CLASSES"] = "[0, 1, 2]"
        try:
            result = _apply_env_overrides(raw)
            assert result["person_filter"]["classes"] == [0, 1, 2]
        finally:
            del os.environ["ARGUS__PERSON_FILTER__CLASSES"]


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_override_replaces_non_dict(self):
        base = {"x": "string"}
        override = {"x": {"nested": True}}
        result = _deep_merge(base, override)
        assert result == {"x": {"nested": True}}


class TestLoadConfigLayered:
    def test_base_only(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"node_id": "node-1", "cameras": []}))

        config = load_config_layered(str(base))
        assert config.node_id == "node-1"

    def test_base_with_override(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"node_id": "node-1", "log_level": "INFO", "cameras": []}))

        override = tmp_path / "local.yaml"
        override.write_text(yaml.dump({"log_level": "DEBUG"}))

        config = load_config_layered(str(base), str(override))
        assert config.node_id == "node-1"
        assert config.log_level == "DEBUG"

    def test_missing_override_skipped(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"node_id": "node-1", "cameras": []}))

        config = load_config_layered(str(base), str(tmp_path / "nonexistent.yaml"))
        assert config.node_id == "node-1"

    def test_missing_base_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config_layered(str(tmp_path / "missing.yaml"))

    def test_env_override_on_layered(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"node_id": "node-1", "log_level": "INFO", "cameras": []}))

        os.environ["ARGUS__LOG_LEVEL"] = "WARNING"
        try:
            config = load_config_layered(str(base))
            assert config.log_level == "WARNING"
        finally:
            del os.environ["ARGUS__LOG_LEVEL"]


class TestLoadConfig:
    def test_env_override_integration(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({
            "node_id": "test-node",
            "cameras": [],
            "dashboard": {"port": 8080},
        }))

        os.environ["ARGUS__DASHBOARD__PORT"] = "9999"
        try:
            config = load_config(str(cfg_file))
            assert config.dashboard.port == 9999
        finally:
            del os.environ["ARGUS__DASHBOARD__PORT"]
