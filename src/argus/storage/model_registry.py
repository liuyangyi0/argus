"""Model version registry for MLOps tracking (C4-1).

Tracks trained model versions with hashes, parameters, and activation state.
Supports the four-stage release pipeline: candidate → shadow → canary → production.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import structlog
from sqlalchemy.orm import Session

from argus.storage.models import ModelRecord, ModelStage, ModelVersionEvent
from argus.storage.release_pipeline import VALID_TRANSITIONS

logger = structlog.get_logger()


class ModelRegistry:
    """Manages model version registration and activation."""

    def __init__(self, session_factory):
        self._session_factory = session_factory
        self._counter = 0
        self._counter_lock = threading.Lock()

    def register(
        self,
        model_path: str | Path,
        baseline_dir: str | Path,
        camera_id: str,
        model_type: str,
        training_params: dict | None = None,
        component_type: str = "full",
        backbone_ref: str | None = None,
    ) -> str:
        """Register a new model version as candidate. Returns model_version_id."""
        model_path = Path(model_path)
        baseline_dir = Path(baseline_dir)

        model_hash = self._compute_dir_hash(model_path)
        data_hash = self._compute_dir_hash(baseline_dir)
        code_version = self._get_git_hash()

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        with self._counter_lock:
            self._counter += 1
            counter_val = self._counter
        model_version_id = f"{camera_id}-{model_type}-{ts}-{counter_val:04d}"

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
            session.commit()

        logger.info(
            "model_registry.registered",
            model_version_id=model_version_id,
            camera_id=camera_id,
            model_type=model_type,
            stage=ModelStage.CANDIDATE.value,
        )
        return model_version_id

    def get_active(self, camera_id: str) -> ModelRecord | None:
        """Get the active model for a camera."""
        with self._session_factory() as session:
            return (
                session.query(ModelRecord)
                .filter_by(camera_id=camera_id, is_active=True)
                .first()
            )

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
    ) -> ModelRecord:
        """Promote a model to the next stage in the release pipeline.

        Validates that the transition is allowed by the state machine.
        If target is 'production', retires the current production model
        and sets is_active=True.  Every promotion generates a
        ModelVersionEvent for audit.

        Args:
            model_version_id: Model to promote.
            target_stage: Target stage (must be a valid ModelStage value).
            triggered_by: Who initiated the promotion (required, non-empty).
            reason: Optional reason for the transition.
            canary_camera_id: Required when target_stage is "canary".

        Returns:
            The updated ModelRecord.

        Raises:
            ValueError: If the model is not found, transition is invalid,
                        or triggered_by is empty.
        """
        if not triggered_by:
            raise ValueError("triggered_by is required for audit compliance")

        valid_stages = {s.value for s in ModelStage}
        if target_stage not in valid_stages:
            raise ValueError(f"Invalid target stage: {target_stage}")

        if target_stage == ModelStage.CANARY.value and not canary_camera_id:
            raise ValueError("canary_camera_id is required for canary stage")

        with self._session_factory() as session:
            record = (
                session.query(ModelRecord)
                .filter_by(model_version_id=model_version_id)
                .first()
            )
            if record is None:
                raise ValueError(f"Model version not found: {model_version_id}")

            current_stage = record.stage
            allowed = VALID_TRANSITIONS.get(current_stage, set())
            if target_stage not in allowed:
                raise ValueError(
                    f"Invalid transition: {current_stage} → {target_stage}. "
                    f"Allowed targets: {sorted(allowed) or 'none (terminal state)'}"
                )

            from_version = record.model_version_id
            from_stage = current_stage

            # If promoting to production, retire current production model
            if target_stage == ModelStage.PRODUCTION.value:
                current_prod = (
                    session.query(ModelRecord)
                    .filter_by(camera_id=record.camera_id, is_active=True)
                    .first()
                )
                if current_prod and current_prod.model_version_id != model_version_id:
                    current_prod.is_active = False
                    current_prod.stage = ModelStage.RETIRED.value
                    # Record retirement event
                    session.add(ModelVersionEvent(
                        camera_id=record.camera_id,
                        from_version=current_prod.model_version_id,
                        to_version=current_prod.model_version_id,
                        from_stage=ModelStage.PRODUCTION.value,
                        to_stage=ModelStage.RETIRED.value,
                        triggered_by=triggered_by,
                        reason=f"Replaced by {model_version_id}",
                    ))
                record.is_active = True
            elif current_stage == ModelStage.PRODUCTION.value:
                # Leaving production
                record.is_active = False

            record.stage = target_stage

            if target_stage == ModelStage.CANARY.value:
                record.canary_camera_id = canary_camera_id
            elif target_stage in (ModelStage.PRODUCTION.value, ModelStage.RETIRED.value):
                record.canary_camera_id = None

            # Record promotion event
            session.add(ModelVersionEvent(
                camera_id=record.camera_id,
                from_version=from_version,
                to_version=model_version_id,
                from_stage=from_stage,
                to_stage=target_stage,
                triggered_by=triggered_by,
                reason=reason,
            ))
            session.commit()
            session.refresh(record)

        logger.info(
            "model_registry.promoted",
            model_version_id=model_version_id,
            from_stage=from_stage,
            to_stage=target_stage,
            triggered_by=triggered_by,
        )
        return record

    def activate(
        self,
        model_version_id: str,
        triggered_by: str = "system",
        allow_bypass: bool = False,
    ) -> None:
        """Set a model as active (deactivates others for same camera).

        Also sets stage to production and records a version event.

        By default, only models already at PRODUCTION or RETIRED stage can be
        activated (for rollback / reactivation). Set ``allow_bypass=True`` to
        skip this check (e.g. scheduler auto-deploy after retraining).
        """
        with self._session_factory() as session:
            record = (
                session.query(ModelRecord)
                .filter_by(model_version_id=model_version_id)
                .first()
            )
            if record is None:
                raise ValueError(f"Model version not found: {model_version_id}")

            if not allow_bypass and record.stage not in (
                ModelStage.PRODUCTION.value,
                ModelStage.RETIRED.value,
            ):
                raise ValueError(
                    f"Cannot activate model at stage '{record.stage}'. "
                    f"Use the release pipeline to promote through "
                    f"shadow → canary → production first."
                )

            # Find current active model for version event
            current = (
                session.query(ModelRecord)
                .filter_by(camera_id=record.camera_id, is_active=True)
                .first()
            )
            from_version = current.model_version_id if current else None
            from_stage = current.stage if current else None

            # Deactivate all other models for this camera
            session.query(ModelRecord).filter_by(
                camera_id=record.camera_id, is_active=True
            ).update({"is_active": False})

            record.is_active = True
            record.stage = ModelStage.PRODUCTION.value

            # Record version event
            event = ModelVersionEvent(
                camera_id=record.camera_id,
                from_version=from_version,
                to_version=model_version_id,
                from_stage=from_stage,
                to_stage=ModelStage.PRODUCTION.value,
                triggered_by=triggered_by,
                reason="direct activation",
            )
            session.add(event)
            session.commit()

        logger.info("model_registry.activated", model_version_id=model_version_id)

    def delete_model(self, model_version_id: str) -> None:
        """Delete a model record from the database."""
        with self._session_factory() as session:
            record = (
                session.query(ModelRecord)
                .filter_by(model_version_id=model_version_id)
                .first()
            )
            if record is not None:
                session.delete(record)
                session.commit()
        logger.info("model_registry.deleted", model_version_id=model_version_id)

    def list_models(self, camera_id: str | None = None) -> list[ModelRecord]:
        """List all models, optionally filtered by camera_id."""
        with self._session_factory() as session:
            query = session.query(ModelRecord).order_by(ModelRecord.created_at.desc())
            if camera_id:
                query = query.filter_by(camera_id=camera_id)
            return list(query.all())

    def rollback(self, camera_id: str, triggered_by: str = "system") -> ModelRecord | None:
        """Reactivate the previous model version for a camera.

        Deactivates the current active model and activates the most recent
        previously registered model. Returns the newly activated model or None
        if no previous model exists.
        """
        with self._session_factory() as session:
            # Find current active model
            current = (
                session.query(ModelRecord)
                .filter_by(camera_id=camera_id, is_active=True)
                .first()
            )

            # Find the previous production model via version event history
            previous = None
            if current is not None:
                # Look for the model that was retired when current was promoted
                prev_event = (
                    session.query(ModelVersionEvent)
                    .filter_by(
                        camera_id=camera_id,
                        to_stage=ModelStage.RETIRED.value,
                    )
                    .filter(ModelVersionEvent.reason.contains(current.model_version_id))
                    .order_by(ModelVersionEvent.timestamp.desc())
                    .first()
                )
                if prev_event:
                    previous = (
                        session.query(ModelRecord)
                        .filter_by(model_version_id=prev_event.from_version)
                        .first()
                    )

            # Fallback: find most recent non-current model that was previously
            # in production (via version events), excluding candidate/retired models
            # that were never deployed.
            if previous is None:
                exclude_id = current.model_version_id if current else ""
                prev_prod_events = (
                    session.query(ModelVersionEvent)
                    .filter_by(camera_id=camera_id, from_stage=ModelStage.PRODUCTION.value)
                    .filter(ModelVersionEvent.from_version != exclude_id)
                    .order_by(ModelVersionEvent.timestamp.desc())
                    .all()
                )
                for evt in prev_prod_events:
                    candidate = (
                        session.query(ModelRecord)
                        .filter_by(model_version_id=evt.from_version)
                        .first()
                    )
                    if candidate is not None:
                        previous = candidate
                        break

            if previous is None:
                logger.warning("model_registry.rollback_no_previous", camera_id=camera_id)
                return None

            from_version = current.model_version_id if current else None
            from_stage = current.stage if current else None

            # Deactivate all models for this camera
            session.query(ModelRecord).filter_by(
                camera_id=camera_id, is_active=True
            ).update({"is_active": False})

            previous.is_active = True
            previous.stage = ModelStage.PRODUCTION.value

            # Record version event
            event = ModelVersionEvent(
                camera_id=camera_id,
                from_version=from_version,
                to_version=previous.model_version_id,
                from_stage=from_stage,
                to_stage=ModelStage.PRODUCTION.value,
                triggered_by=triggered_by,
                reason="rollback",
            )
            session.add(event)
            session.commit()

            logger.info(
                "model_registry.rollback",
                camera_id=camera_id,
                model_version_id=previous.model_version_id,
            )
            return previous

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
        h = hashlib.sha256()
        files = [path] if path.is_file() else sorted(path.rglob("*")) if path.is_dir() else []
        for f in files:
            if not f.is_file():
                continue
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
        return h.hexdigest()[:16]

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
