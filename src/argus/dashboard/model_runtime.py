"""Read-only helpers for registry-backed model metadata.

Model stage changes and runtime swaps deliberately do not live in this module.
They are coordinated by :class:`argus.storage.model_release_service.ModelReleaseService`
so applied registry state cannot get ahead of the verified runtime identity.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import Request

from argus.storage.model_registry import ModelRegistry
from argus.storage.models import ModelRecord

_REGISTRY_LOCK = threading.Lock()


def get_registry(request: Request) -> ModelRegistry | None:
    """Share one registry per application/database, including derived training."""
    db = getattr(request.app.state, "database", None) or getattr(request.app.state, "db", None)
    if db is None:
        return None
    session_factory = getattr(db, "get_session", None)
    if session_factory is None:
        return None
    with _REGISTRY_LOCK:
        state = request.app.state
        cached = getattr(state, "_model_registry", None)
        if cached is not None and getattr(state, "_model_registry_database", None) is db:
            return cached
        registry = ModelRegistry(session_factory=session_factory)
        state._model_registry = registry
        state._model_registry_database = db
        return registry


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
