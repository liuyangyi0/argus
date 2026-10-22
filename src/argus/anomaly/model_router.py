"""Model routing for canary deployments.

Decides which model a camera should use based on the release pipeline stage.
Canary cameras use the canary model; all others use production.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from argus.storage.model_registry import validate_persisted_release_evidence
from argus.storage.models import ModelRecord, ModelStage

logger = structlog.get_logger()


class ModelRouter:
    """Routes cameras to the correct model version based on release stage."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def get_model_for_camera(self, camera_id: str) -> ModelRecord | None:
        """Get the model that should be used for a specific camera.

        Priority:
        1. Canary model targeting this camera
        2. Production model for this camera
        3. None (no model assigned)
        """
        with self._session_factory() as session:
            # Check for canary model targeting this camera
            canaries = (
                session.query(ModelRecord)
                .filter_by(
                    camera_id=camera_id,
                    stage=ModelStage.CANARY.value,
                    canary_camera_id=camera_id,
                )
                .limit(2)
                .all()
            )
            if len(canaries) > 1:
                logger.error(
                    "model_router.ambiguous_canary_assignment",
                    camera_id=camera_id,
                    versions=[model.model_version_id for model in canaries],
                )
                return None
            canary = canaries[0] if canaries else None
            if canary is not None:
                logger.debug(
                    "model_router.canary",
                    camera_id=camera_id,
                    version=canary.model_version_id,
                )
                return canary

            # Fall back to production model
            production = (
                session.query(ModelRecord)
                .filter_by(
                    camera_id=camera_id,
                    stage=ModelStage.PRODUCTION.value,
                    is_active=True,
                )
                .order_by(ModelRecord.created_at.desc())
                .first()
            )
            if production is None:
                return None
            validation = validate_persisted_release_evidence(production)
            if not validation.valid:
                logger.error(
                    "model_router.production_rejected",
                    camera_id=camera_id,
                    version=production.model_version_id,
                    reasons=list(validation.reasons),
                )
                return None
            return production

    def get_shadow_model_for_camera(self, camera_id: str) -> ModelRecord | None:
        """Return the unique applied shadow assignment, or fail closed."""
        with self._session_factory() as session:
            models = (
                session.query(ModelRecord)
                .filter_by(
                    camera_id=camera_id,
                    stage=ModelStage.SHADOW.value,
                )
                .limit(2)
                .all()
            )
            if len(models) > 1:
                logger.error(
                    "model_router.ambiguous_shadow_assignment",
                    camera_id=camera_id,
                    versions=[model.model_version_id for model in models],
                )
                return None
            return models[0] if models else None

    def get_model_path(self, camera_id: str) -> Path | None:
        """Get the model file path for a camera, considering canary routing."""
        record = self.get_model_for_camera(camera_id)
        if record is None or record.model_path is None:
            return None
        return Path(record.model_path)

    def has_model_records(self, camera_id: str) -> bool:
        """Return True when the registry knows about any model for a camera."""
        with self._session_factory() as session:
            return (
                session.query(ModelRecord)
                .filter_by(camera_id=camera_id)
                .first()
            ) is not None

    def is_canary(self, camera_id: str) -> bool:
        """Check if a camera is currently running a canary model."""
        with self._session_factory() as session:
            assignments = (
                session.query(ModelRecord.id)
                .filter_by(
                    camera_id=camera_id,
                    stage=ModelStage.CANARY.value,
                    canary_camera_id=camera_id,
                )
                .limit(2)
                .all()
            )
            return len(assignments) == 1
