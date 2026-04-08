"""Helpers for syncing registry-backed model state to running camera pipelines."""

from __future__ import annotations

from pathlib import Path

from fastapi import Request

from argus.core.model_discovery import resolve_runtime_model_path
from argus.storage.model_registry import ModelRegistry
from argus.storage.models import ModelRecord


def get_registry(request: Request) -> ModelRegistry | None:
    """Build a ModelRegistry from app state when the database is available."""
    db = getattr(request.app.state, "database", None) or getattr(request.app.state, "db", None)
    if db is None:
        return None
    session_factory = getattr(db, "get_session", None)
    if session_factory is None:
        return None
    return ModelRegistry(session_factory=session_factory)


def _resolve_model_file(model_path: str, camera_id: str) -> str:
    """Resolve a model path to an actual model file if it's a directory."""
    resolved = resolve_runtime_model_path(model_path, camera_id)
    return str(resolved) if resolved is not None else model_path


def sync_model_record_runtime(request: Request, record: ModelRecord) -> bool:
    """Hot-reload a registered model into the running camera pipeline when possible."""
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is None or not record.model_path:
        return False

    resolved_path = _resolve_model_file(record.model_path, record.camera_id)
    if Path(resolved_path).suffix.lower() not in {".xml", ".pt"}:
        return False
    return camera_manager.reload_model(
        record.camera_id,
        resolved_path,
        version_tag=record.model_version_id,
    )


def activate_model_version(
    request: Request,
    version_id: str,
    *,
    triggered_by: str = "dashboard",
) -> tuple[ModelRecord, bool]:
    """Activate a model version in the registry and sync runtime if the camera is running."""
    registry = get_registry(request)
    if registry is None:
        raise ValueError("Database not available")

    registry.activate(version_id, triggered_by=triggered_by, allow_bypass=True)
    record = registry.get_by_version_id(version_id)
    if record is None:
        raise ValueError(f"Model version not found: {version_id}")
    return record, sync_model_record_runtime(request, record)


def rollback_camera_model(
    request: Request,
    camera_id: str,
    *,
    triggered_by: str = "dashboard",
) -> tuple[ModelRecord | None, bool]:
    """Rollback a camera to its previous registered model and sync runtime."""
    registry = get_registry(request)
    if registry is None:
        raise ValueError("Database not available")

    record = registry.rollback(camera_id, triggered_by=triggered_by)
    if record is None:
        return None, False
    return record, sync_model_record_runtime(request, record)


def sync_active_camera_model(request: Request, camera_id: str) -> bool:
    """Sync the currently active registered model for a camera into runtime."""
    registry = get_registry(request)
    if registry is None:
        return False
    record = registry.get_active(camera_id)
    if record is None:
        return False
    return sync_model_record_runtime(request, record)


def find_registered_model_by_path(
    request: Request,
    model_path: str | Path,
    *,
    camera_id: str | None = None,
) -> ModelRecord | None:
    """Find a registry record matching a model directory path."""
    registry = get_registry(request)
    if registry is None:
        return None

    resolved_target = Path(model_path).resolve()
    candidate_paths = {resolved_target}
    if resolved_target.is_file():
        candidate_paths.update(resolved_target.parents)
    for record in registry.list_models(camera_id=camera_id):
        if not record.model_path:
            continue
        if Path(record.model_path).resolve() in candidate_paths:
            return record
    return None