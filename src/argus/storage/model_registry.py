"""Model version registration and read model for MLOps tracking (C4-1).

``ModelRegistry`` may create candidates and read applied release state.  Stage
mutations are deliberately fail-closed here; ``ModelReleaseService`` and its
``ReleasePipeline`` transaction are the only supported release boundary.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from argus.preprocessing.scene_profile import runtime_artifact_sha256
from argus.storage.models import (
    AuditLog,
    ModelRecord,
    ModelReleaseOperation,
    ModelStage,
    ModelVersionEvent,
    ReleaseOperationStatus,
)
from argus.storage.release_evidence import (
    ReleaseEvidenceValidation,
    canonical_json,
    registry_training_data_hash,
    validate_release_evidence,
)

logger = structlog.get_logger()


class ModelStageMutationDisabledError(RuntimeError):
    """Raised when legacy registry code tries to mutate release stage."""


class ModelDeletionConflictError(RuntimeError):
    """Raised when deleting a model would race with an applied/desired release."""


def _invalid_release_evidence(reason: str) -> ReleaseEvidenceValidation:
    return ReleaseEvidenceValidation(
        valid=False,
        reasons=(reason,),
        canonical_json=None,
        sha256=None,
    )


def validate_persisted_release_evidence(
    record: ModelRecord,
) -> ReleaseEvidenceValidation:
    """Validate stored production evidence against the current artifact tree.

    This is the runtime trust boundary.  Merely having ``stage=production`` or
    ``is_active=True`` is insufficient: legacy rows, modified evidence JSON,
    and model/sidecar drift all fail closed.  The stage itself is deliberately
    checked by callers so the same evidence check can be used before restoring
    a retired model during rollback.
    """

    evidence_json = record.release_evidence_json
    evidence_sha256 = record.release_evidence_sha256
    if not isinstance(evidence_json, str) or not evidence_json:
        return _invalid_release_evidence("persisted release evidence is missing")
    if not isinstance(evidence_sha256, str) or not evidence_sha256:
        return _invalid_release_evidence("persisted release evidence digest is missing")

    actual_evidence_sha256 = hashlib.sha256(
        evidence_json.encode("utf-8")
    ).hexdigest()
    if actual_evidence_sha256 != evidence_sha256:
        return _invalid_release_evidence(
            "persisted release evidence digest does not match evidence JSON"
        )

    try:
        evidence = json.loads(evidence_json)
    except (TypeError, ValueError) as exc:
        return _invalid_release_evidence(
            f"persisted release evidence is not valid JSON: {exc}"
        )
    try:
        if canonical_json(evidence) != evidence_json:
            return _invalid_release_evidence(
                "persisted release evidence JSON is not canonical"
            )
    except (TypeError, ValueError) as exc:
        return _invalid_release_evidence(
            f"persisted release evidence is not canonical JSON: {exc}"
        )

    if not record.model_path:
        return _invalid_release_evidence("registry model_path is missing")
    try:
        artifact_sha256 = runtime_artifact_sha256(record.model_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _invalid_release_evidence(
            f"runtime artifact fingerprint unavailable: {exc}"
        )

    return validate_release_evidence(
        evidence,
        model_version_id=record.model_version_id,
        camera_id=record.camera_id,
        registry_model_hash=record.model_hash,
        runtime_artifact_sha256=artifact_sha256,
    )


class ModelRegistry:
    """Register candidates and expose applied model state without mutating it."""

    _REGISTER_ATTEMPTS = 5

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def uses_same_sessions_as(self, other: object) -> bool:
        """Whether two registry facades belong to the same database owner."""
        return isinstance(other, ModelRegistry) and (
            self._session_factory == other._session_factory
        )

    def register(
        self,
        model_path: str | Path,
        baseline_dir: str | Path,
        camera_id: str,
        model_type: str,
        training_params: dict | None = None,
        component_type: str = "full",
        backbone_ref: str | None = None,
        registered_by: str = "model_registry",
    ) -> str:
        """Register a candidate with an atomic version event and audit row."""
        if not isinstance(registered_by, str) or not registered_by.strip():
            raise ValueError("registered_by is required for audit compliance")
        registered_by = registered_by.strip()
        model_path = Path(model_path)
        baseline_dir = Path(baseline_dir)

        # New registrations name an exact runtime file.  Its identity includes
        # OpenVINO companion weights and score/input sidecars, rather than
        # silently treating changed calibration as the same trained model.
        # Preserve the legacy directory hash for existing registration clients.
        if model_path.is_file() and model_path.suffix.lower() in {
            ".pt", ".pth", ".xml", ".bin", ".onnx",
        }:
            model_hash = runtime_artifact_sha256(model_path)[:16]
        else:
            model_hash = self._compute_dir_hash(model_path)
        data_hash = self._compute_dir_hash(baseline_dir)
        code_version = self._get_git_hash()

        last_error: IntegrityError | None = None
        for attempt in range(1, self._REGISTER_ATTEMPTS + 1):
            model_version_id = self._new_model_version_id(camera_id, model_type)
            record = ModelRecord(
                model_version_id=model_version_id,
                camera_id=camera_id,
                model_type=model_type,
                model_hash=model_hash,
                data_hash=data_hash,
                code_version=code_version,
                training_params=json.dumps(training_params) if training_params else None,
                stage=ModelStage.CANDIDATE.value,
                component_type=component_type,
                model_path=str(model_path),
                backbone_version_id=backbone_ref,
            )

            with self._session_factory() as session:
                session.add(record)
                session.add(ModelVersionEvent(
                    camera_id=camera_id,
                    from_version=None,
                    to_version=model_version_id,
                    from_stage=None,
                    to_stage=ModelStage.CANDIDATE.value,
                    triggered_by=registered_by,
                    reason="model registered as candidate",
                ))
                session.add(AuditLog(
                    user=registered_by,
                    action="model_candidate_registered",
                    target_type="model",
                    target_id=model_version_id,
                    detail=f"camera_id={camera_id}; model_type={model_type}",
                ))
                try:
                    session.commit()
                except IntegrityError as exc:
                    session.rollback()
                    last_error = exc
                    logger.warning(
                        "model_registry.version_id_collision",
                        model_version_id=model_version_id,
                        attempt=attempt,
                        max_attempts=self._REGISTER_ATTEMPTS,
                    )
                    continue
            break
        else:
            raise RuntimeError(
                "Failed to generate a unique model_version_id after "
                f"{self._REGISTER_ATTEMPTS} attempts"
            ) from last_error

        logger.info(
            "model_registry.registered",
            model_version_id=model_version_id,
            camera_id=camera_id,
            model_type=model_type,
            stage=ModelStage.CANDIDATE.value,
        )
        return model_version_id

    @staticmethod
    def _new_model_version_id(camera_id: str, model_type: str) -> str:
        """Generate a DB-safe model version ID across registry instances."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")
        suffix = uuid.uuid4().hex[:8]
        return f"{camera_id}-{model_type}-{ts}-{suffix}"

    def get_active(self, camera_id: str) -> ModelRecord | None:
        """Get the runtime-authorized active production model for a camera.

        Legacy ``is_active`` rows without canonical release evidence and rows
        whose runtime artifact tree changed after promotion are intentionally
        hidden from runtime consumers.
        """
        with self._session_factory() as session:
            record = (
                session.query(ModelRecord)
                .filter_by(
                    camera_id=camera_id,
                    stage=ModelStage.PRODUCTION.value,
                    is_active=True,
                )
                .first()
            )
            if record is None:
                return None
            validation = validate_persisted_release_evidence(record)
            if not validation.valid:
                logger.error(
                    "model_registry.active_model_rejected",
                    camera_id=camera_id,
                    model_version_id=record.model_version_id,
                    reasons=list(validation.reasons),
                )
                return None
            return record

    def get_runtime_approved_production(
        self,
        camera_id: str,
    ) -> list[ModelRecord]:
        """Return production rows whose persisted evidence is still valid."""

        approved: list[ModelRecord] = []
        with self._session_factory() as session:
            records = list(
                session.query(ModelRecord)
                .filter_by(
                    camera_id=camera_id,
                    stage=ModelStage.PRODUCTION.value,
                )
                .order_by(ModelRecord.created_at.desc())
                .all()
            )
            for record in records:
                validation = validate_persisted_release_evidence(record)
                if validation.valid:
                    approved.append(record)
                else:
                    logger.error(
                        "model_registry.production_model_rejected",
                        camera_id=camera_id,
                        model_version_id=record.model_version_id,
                        reasons=list(validation.reasons),
                    )
        return approved

    def get_by_stage(self, camera_id: str, stage: str) -> list[ModelRecord]:
        """Get models for a camera at a specific stage."""
        with self._session_factory() as session:
            return list(
                session.query(ModelRecord)
                .filter_by(camera_id=camera_id, stage=stage)
                .order_by(ModelRecord.created_at.desc())
                .all()
            )

    def get_by_version_id(self, model_version_id: str) -> ModelRecord | None:
        """Get a model by its version ID."""
        with self._session_factory() as session:
            return (
                session.query(ModelRecord)
                .filter_by(model_version_id=model_version_id)
                .first()
            )

    def promote(
        self,
        model_version_id: str,
        target_stage: str,
        triggered_by: str,
        reason: str | None = None,
        canary_camera_id: str | None = None,
        *,
        release_evidence: object | None = None,
    ) -> ModelRecord:
        """Fail closed; release stage writes belong to ModelReleaseService."""

        raise ModelStageMutationDisabledError(
            "ModelRegistry.promote is disabled; use "
            "ModelReleaseService.request_transition"
        )

    def activate(
        self,
        model_version_id: str,
        triggered_by: str = "system",
    ) -> None:
        """Fail closed; activation is part of a two-phase release command."""

        raise ModelStageMutationDisabledError(
            "ModelRegistry.activate is disabled; use "
            "ModelReleaseService.request_transition"
        )

    def retire(
        self,
        model_version_id: str,
        reason: str | None = None,
        triggered_by: str = "system",
    ) -> ModelRecord | None:
        """Fail closed; retirement must use the release command service."""

        raise ModelStageMutationDisabledError(
            "ModelRegistry.retire is disabled; use "
            "ModelReleaseService.request_transition"
        )

    def delete_model(
        self,
        model_version_id: str,
        *,
        requested_by: str = "model_registry",
        expected_revision: int | None = None,
    ) -> dict[str, object]:
        """Remove an inactive registry entry and defer artifact reclamation.

        Filesystem paths are references, not ownership boundaries.  A legacy
        row may name a directory containing another model's exact artifact,
        or two paths may resolve through the same symlink.  Deleting a row
        therefore never unlinks files.  An atomic audit tombstone preserves
        the removed reference for a separately reviewed garbage-collection
        pass, after all remaining model references have been checked.

        The conditional DELETE is the final arbiter against a simultaneous
        release: either it removes the still-inactive revision, causing a
        later release CAS to fail and restore runtime, or a release wins and
        this command cannot delete its newly applied model.
        """
        requested_by = requested_by.strip()
        if not requested_by:
            raise ValueError("requested_by is required")
        with self._session_factory() as session:
            record = (
                session.query(ModelRecord)
                .filter_by(model_version_id=model_version_id)
                .first()
            )
            if record is None:
                raise ValueError(f"Model not found: {model_version_id}")
            if record.is_active or record.stage not in {
                ModelStage.CANDIDATE.value,
                ModelStage.RETIRED.value,
            }:
                raise ModelDeletionConflictError(
                    "Only inactive candidate/retired models can be deleted"
                )
            revision = int(record.release_revision or 0)
            if expected_revision is not None and revision != expected_revision:
                raise ModelDeletionConflictError("Model release revision changed")
            pending = (
                session.query(ModelReleaseOperation.id)
                .filter_by(
                    model_version_id=model_version_id,
                    status=ReleaseOperationStatus.PENDING.value,
                )
                .first()
            )
            if pending is not None:
                raise ModelDeletionConflictError(
                    "Model has a pending release operation"
                )

            tombstone = {
                "schema_version": "argus.model-artifact-tombstone.v1",
                "model_version_id": model_version_id,
                "camera_id": record.camera_id,
                "model_path": record.model_path,
                "registry_model_hash": record.model_hash,
                "release_revision": revision,
                "stage": record.stage,
                "garbage_collection": "deferred",
                "reason": "artifact references require an independent retention review",
            }
            removed = session.execute(
                delete(ModelRecord)
                .where(
                    ModelRecord.id == record.id,
                    ModelRecord.model_version_id == model_version_id,
                    ModelRecord.stage == record.stage,
                    ModelRecord.is_active.is_(False),
                    ModelRecord.release_revision == revision,
                    ~select(ModelReleaseOperation.id).where(
                        ModelReleaseOperation.model_version_id == model_version_id,
                        ModelReleaseOperation.status == ReleaseOperationStatus.PENDING.value,
                    ).exists(),
                )
                .execution_options(synchronize_session=False)
            )
            if removed.rowcount != 1:
                session.rollback()
                raise ModelDeletionConflictError(
                    "Model changed while deletion was being prepared"
                )
            session.add(AuditLog(
                user=requested_by,
                action="model_artifacts_deferred",
                target_type="model",
                target_id=model_version_id,
                detail=json.dumps(tombstone, ensure_ascii=False, sort_keys=True),
            ))
            session.commit()
        logger.info("model_registry.deleted", model_version_id=model_version_id)
        return {
            "deleted": model_version_id,
            "artifacts_retained": True,
            "garbage_collection": "deferred",
        }

    def list_models(self, camera_id: str | None = None) -> list[ModelRecord]:
        """List all models, optionally filtered by camera_id."""
        with self._session_factory() as session:
            query = session.query(ModelRecord).order_by(ModelRecord.created_at.desc())
            if camera_id:
                query = query.filter_by(camera_id=camera_id)
            return list(query.all())

    def rollback(self, camera_id: str, triggered_by: str = "system") -> ModelRecord | None:
        """Fail closed until rollback is represented by ModelReleaseService."""

        raise ModelStageMutationDisabledError(
            "ModelRegistry.rollback is disabled; rollback requires a "
            "ModelReleaseService two-phase transaction"
        )

    def compensate_failed_production_sync(
        self,
        promoted_version_id: str,
        previous_version_id: str | None,
        *,
        triggered_by: str,
        reason: str,
    ) -> ModelRecord | None:
        """Fail closed; compensation belongs inside the release transaction."""

        raise ModelStageMutationDisabledError(
            "ModelRegistry.compensate_failed_production_sync is disabled; "
            "ModelReleaseService owns runtime/DB compensation"
        )

    def get_version_events(
        self,
        camera_id: str | None = None,
        model_version_id: str | None = None,
        limit: int = 50,
    ) -> list[ModelVersionEvent]:
        """Query version transition events."""
        with self._session_factory() as session:
            query = session.query(ModelVersionEvent).order_by(
                ModelVersionEvent.timestamp.desc()
            )
            if camera_id:
                query = query.filter_by(camera_id=camera_id)
            if model_version_id:
                query = query.filter(
                    (ModelVersionEvent.from_version == model_version_id)
                    | (ModelVersionEvent.to_version == model_version_id)
                )
            return list(query.limit(limit).all())

    @staticmethod
    def _compute_dir_hash(path: Path) -> str:
        """Compute SHA256 hash of all files in a directory (chunked to avoid OOM)."""
        return registry_training_data_hash(path)

    @staticmethod
    def _get_git_hash() -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            logger.warning("model_registry.git_hash_unavailable", exc_info=True)
            return None
