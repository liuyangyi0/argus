"""Helpers for syncing registry-backed model state to running camera pipelines."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any

from fastapi import Request
import structlog

from argus.core.model_discovery import resolve_runtime_model_path
from argus.storage.model_registry import ModelRegistry
from argus.storage.models import ModelRecord

logger = structlog.get_logger()


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


def _jsonable_runtime_state(value: Any) -> Any:
    """Convert common runtime-state return types to JSON-safe structures."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable_runtime_state(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable_runtime_state(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _get_release_state_applier(camera_manager: object):
    """Return apply_model_release_state only when it exists on the object/class.

    MagicMock creates arbitrary attributes on demand, so a normal getattr would
    make routes think every mock manager supports this newer API. Static lookup
    avoids that while still accepting real methods and explicitly assigned mocks.
    """
    if inspect.getattr_static(camera_manager, "apply_model_release_state", None) is None:
        return None
    method = getattr(camera_manager, "apply_model_release_state", None)
    return method if callable(method) else None


def sync_model_release_state(
    request: Request,
    camera_id: str,
) -> tuple[bool, Any, bool]:
    """Apply registry release state to the running camera if supported.

    Returns ``(runtime_synced, runtime_state, attempted)``. ``attempted`` lets
    older managers fall back to the legacy production-only hot reload path
    without masking failures from a manager that does implement release-state
    sync.
    """
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is None:
        return False, None, False

    apply_release_state = _get_release_state_applier(camera_manager)
    if apply_release_state is None:
        return False, None, False

    try:
        result = apply_release_state(camera_id)
    except Exception as exc:
        logger.warning(
            "model_runtime.release_state_sync_failed",
            camera_id=camera_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return False, None, True

    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
        return result[0], _jsonable_runtime_state(result[1]), True
    if isinstance(result, bool):
        return result, None, True

    runtime_state = _jsonable_runtime_state(result)
    runtime_synced = True
    if isinstance(runtime_state, dict):
        for key in ("runtime_synced", "synced", "applied"):
            if isinstance(runtime_state.get(key), bool):
                runtime_synced = runtime_state[key]
                break
        else:
            errors = runtime_state.get("errors") or []
            changed = any(
                bool(runtime_state.get(key))
                for key in ("primary_reloaded", "shadow_attached", "shadow_detached")
            )
            runtime_synced = changed or not bool(errors)
    return runtime_synced, runtime_state, True


def activate_model_version(
    request: Request,
    version_id: str,
    *,
    triggered_by: str = "dashboard",
    allow_bypass: bool = False,
) -> tuple[ModelRecord, bool]:
    """Activate a model version in the registry and sync runtime.

    P1 fix (2026-05): default no longer bypasses the candidate→shadow→canary→
    production stage gate. Callers wanting an emergency forced activation must
    opt in explicitly with ``allow_bypass=True`` from an internal path.
    """
    registry = get_registry(request)
    if registry is None:
        raise ValueError("Database not available")

    registry.activate(
        version_id, triggered_by=triggered_by, allow_bypass=allow_bypass,
    )
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


def apply_camera_release_state(
    request: Request,
    camera_id: str,
    *,
    reason: str = "release_transition",
) -> tuple[bool, dict]:
    """Ask the camera manager to apply registry release state to runtime."""
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is None or not hasattr(camera_manager, "apply_model_release_state"):
        return False, {
            "camera_id": camera_id,
            "reason": reason,
            "running": False,
            "primary_reloaded": False,
            "shadow_attached": False,
            "shadow_detached": False,
            "errors": ["camera_manager_unavailable"],
        }

    state = camera_manager.apply_model_release_state(camera_id, reason=reason)
    errors = state.get("errors") or []
    synced = bool(
        not errors
        or any(
            key in state and state[key]
            for key in ("primary_reloaded", "shadow_attached", "shadow_detached")
        )
    )
    return synced, state


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
