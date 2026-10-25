"""Read-only authorization for restoring a previously applied production model.

Retired is not a promotion stage. A rollback must name the production revision
it replaces and retain the original server-owned evidence source identity.
"""

from __future__ import annotations

import json

from argus.storage.models import ModelRecord, ModelReleaseOperation, ModelVersionEvent


def rollback_context(session, model_version_id: str) -> tuple[ModelRecord, dict]:
    target = session.query(ModelRecord).filter_by(model_version_id=model_version_id).first()
    if target is None:
        raise ValueError(f"Model not found: {model_version_id}")
    if target.stage != "retired" or target.is_active:
        raise ValueError("Rollback target must be an inactive retired production model")
    current = session.query(ModelRecord).filter_by(
        camera_id=target.camera_id, stage="production", is_active=True,
    ).one_or_none()
    if current is None or current.model_version_id == model_version_id:
        raise ValueError("Rollback requires another active production model on this camera")
    if session.query(ModelRecord).filter_by(camera_id=target.camera_id, stage="canary").first():
        raise ValueError("Withdraw the camera's canary before restoring production")
    history = session.query(ModelReleaseOperation).filter_by(
        model_version_id=model_version_id, camera_id=target.camera_id,
        target_stage="production", status="applied",
    ).order_by(ModelReleaseOperation.id.desc()).first()
    if (
        history is None or history.from_stage not in {"canary", "retired"}
        or history.applied_revision is None
        or int(history.applied_revision) != int(history.expected_revision) + 1
        or int(history.applied_revision) >= int(target.release_revision or 0)
    ):
        raise ValueError("Rollback target has no verified two-phase production history")
    if session.query(ModelVersionEvent).filter_by(
        camera_id=target.camera_id, to_version=model_version_id, to_stage="production",
    ).first() is None:
        raise ValueError("Rollback target has no production audit history")
    command = json.loads(history.desired_state_json or "{}")
    sources = command.get("release_evidence_sources") if isinstance(command, dict) else None
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Original production evidence sources are unavailable")
    if not target.release_evidence_sha256 or not target.release_evidence_json:
        raise ValueError("Persisted production evidence is unavailable")
    return target, {
        "source_model_version_id": current.model_version_id,
        "source_release_revision": int(current.release_revision or 0),
        "history_operation_id": history.operation_id,
        "evidence_sha256": target.release_evidence_sha256,
        "release_evidence_sources": sources,
    }


def validate_rollback_context(session, model_version_id: str, expected: dict):
    target, actual = rollback_context(session, model_version_id)
    if actual != expected:
        raise ValueError("Rollback identity changed; reload readiness and submit a new request")
    return target
