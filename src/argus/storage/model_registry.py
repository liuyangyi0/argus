"""Model version registry for MLOps tracking (C4-1).

Tracks trained model versions with hashes, parameters, and activation state.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import structlog
from sqlalchemy.orm import Session

from argus.storage.models import ModelRecord

logger = structlog.get_logger()


class ModelRegistry:
    """Manages model version registration and activation."""

    def __init__(self, session_factory):
        self._session_factory = session_factory
        self._counter = 0

    def register(
        self,
        model_path: str | Path,
        baseline_dir: str | Path,
        camera_id: str,
        model_type: str,
        training_params: dict | None = None,
    ) -> str:
        """Register a new model version. Returns model_version_id."""
        model_path = Path(model_path)
        baseline_dir = Path(baseline_dir)

        model_hash = self._compute_dir_hash(model_path)
        data_hash = self._compute_dir_hash(baseline_dir)
        code_version = self._get_git_hash()

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self._counter += 1
        model_version_id = f"{camera_id}-{model_type}-{ts}-{self._counter:04d}"

        record = ModelRecord(
            model_version_id=model_version_id,
            camera_id=camera_id,
            model_type=model_type,
            model_hash=model_hash,
            data_hash=data_hash,
            code_version=code_version,
            training_params=json.dumps(training_params) if training_params else None,
        )

        with self._session_factory() as session:
            session.add(record)
            session.commit()

        logger.info(
            "model_registry.registered",
            model_version_id=model_version_id,
            camera_id=camera_id,
            model_type=model_type,
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

    def activate(self, model_version_id: str) -> None:
        """Set a model as active (deactivates others for same camera)."""
        with self._session_factory() as session:
            record = (
                session.query(ModelRecord)
                .filter_by(model_version_id=model_version_id)
                .first()
            )
            if record is None:
                raise ValueError(f"Model version not found: {model_version_id}")

            # Deactivate all other models for this camera
            session.query(ModelRecord).filter_by(
                camera_id=record.camera_id, is_active=True
            ).update({"is_active": False})

            record.is_active = True
            session.commit()

        logger.info("model_registry.activated", model_version_id=model_version_id)

    def list_models(self, camera_id: str | None = None) -> list[ModelRecord]:
        """List all models, optionally filtered by camera_id."""
        with self._session_factory() as session:
            query = session.query(ModelRecord).order_by(ModelRecord.created_at.desc())
            if camera_id:
                query = query.filter_by(camera_id=camera_id)
            return list(query.all())

    def rollback(self, camera_id: str) -> ModelRecord | None:
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

            # Find the previous model (most recent non-active, or second most recent)
            query = (
                session.query(ModelRecord)
                .filter_by(camera_id=camera_id)
                .order_by(ModelRecord.created_at.desc())
            )
            candidates = list(query.all())

            previous = None
            for candidate in candidates:
                if current is None or candidate.model_version_id != current.model_version_id:
                    previous = candidate
                    break

            if previous is None:
                logger.warning("model_registry.rollback_no_previous", camera_id=camera_id)
                return None

            # Deactivate all models for this camera
            session.query(ModelRecord).filter_by(
                camera_id=camera_id, is_active=True
            ).update({"is_active": False})

            previous.is_active = True
            session.commit()

            logger.info(
                "model_registry.rollback",
                camera_id=camera_id,
                model_version_id=previous.model_version_id,
            )
            return previous

    @staticmethod
    def _compute_dir_hash(path: Path) -> str:
        """Compute SHA256 hash of all files in a directory."""
        h = hashlib.sha256()
        if path.is_file():
            h.update(path.read_bytes())
        elif path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file():
                    h.update(f.read_bytes())
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
            return None
