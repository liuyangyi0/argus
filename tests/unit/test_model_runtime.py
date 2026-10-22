"""Tests for the read-only dashboard model-registry helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from argus.dashboard.model_runtime import (
    find_registered_model_by_path,
    get_registry,
)


def _request(**state):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


def test_get_registry_accepts_database_state() -> None:
    database = SimpleNamespace(get_session=MagicMock())

    registry = get_registry(_request(database=database))

    assert registry is not None


def test_get_registry_returns_none_without_session_factory() -> None:
    assert get_registry(_request()) is None
    assert get_registry(_request(database=object())) is None


def test_find_registered_model_by_directory_is_read_only(tmp_path) -> None:
    model_dir = tmp_path / "models" / "cam_01" / "v1"
    model_dir.mkdir(parents=True)
    record = SimpleNamespace(model_path=str(model_dir))
    registry = MagicMock()
    registry.list_models.return_value = [record]

    with patch("argus.dashboard.model_runtime.get_registry", return_value=registry):
        found = find_registered_model_by_path(
            _request(),
            model_dir,
            camera_id="cam_01",
        )

    assert found is record
    registry.list_models.assert_called_once_with(camera_id="cam_01")
    assert not hasattr(registry, "method_calls") or all(
        call[0] == "list_models" for call in registry.method_calls
    )


def test_find_registered_model_by_artifact_matches_registered_parent(tmp_path) -> None:
    model_dir = tmp_path / "models" / "cam_01" / "v1"
    model_dir.mkdir(parents=True)
    artifact = model_dir / "model.pt"
    artifact.write_bytes(b"model")
    record = SimpleNamespace(model_path=str(model_dir))
    registry = MagicMock()
    registry.list_models.return_value = [record]

    with patch("argus.dashboard.model_runtime.get_registry", return_value=registry):
        found = find_registered_model_by_path(_request(), artifact)

    assert found is record
