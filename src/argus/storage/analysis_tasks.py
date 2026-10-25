"""Task configuration, immutable shadow bundles and legacy release projection.

Managed tasks have separate revisioned configuration and model registrations.
Their bundles authorize shadow execution only. The existing ModelReleaseService
retains exclusive authority over legacy production state.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from argus.core.analysis_task_contracts import (
    LEGACY_AUTHORITY,
    LEGACY_TEMPLATE_ID,
    MANAGED_AUTHORITY,
    TaskModelDescriptor,
    get_default_config,
    legacy_task_id,
    model_role_incompatibilities,
    normalize_task_config,
    template_spec,
    template_specs,
)
from argus.storage.database import Database
from argus.storage.models import (
    AnalysisTaskRecord,
    AnalysisTaskOperationRecord,
    AnalysisTaskRevisionRecord,
    ModelArtifactRecord,
    ModelRecord,
    ModelStage,
    TaskDeploymentRecord,
    TaskDeploymentOperationRecord,
    TaskExecutionBundleRecord,
    TaskModelRecord,
)


class AnalysisTaskIntegrityError(RuntimeError):
    """Persisted task identity/configuration is inconsistent; do not execute it."""


class AnalysisTaskProjectionConflict(RuntimeError):
    """Legacy authority changed while its read-only projection was validated."""


class AnalysisTaskValidationError(ValueError):
    """A data-only task/model command violates the closed platform contract."""


class AnalysisTaskConflictError(RuntimeError):
    """Stale revision, reused idempotency key, or incompatible task authority."""


class AnalysisTaskNotFoundError(LookupError):
    """The addressed task, model or bundle does not exist."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    # SQLite drops tzinfo from DATETIME on reload. All task timestamps are
    # written as UTC; preserve that meaning in receipts and subsequent GETs.
    return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else
            value.astimezone(timezone.utc)).isoformat()


class AnalysisTaskService:
    """Persist task identities without taking over legacy runtime authority."""

    def __init__(self, db: Database, *, models_root: str | Path | None = None):
        self._db = db
        self._models_root = Path(models_root).expanduser().absolute() if models_root is not None else None

    template_specs = staticmethod(template_specs)
    get_default_config = staticmethod(get_default_config)

    @staticmethod
    def _actor(value: str) -> str:
        if (not isinstance(value, str) or not value.strip() or len(value) > 100
                or any(ord(char) < 32 for char in value)):
            raise AnalysisTaskValidationError("requested_by must identify the authenticated actor")
        return value.strip()

    @staticmethod
    def _revision(value: int) -> int:
        if type(value) is not int or value < 1:
            raise AnalysisTaskValidationError("expected_revision must be a positive integer")
        return value

    @staticmethod
    def _name(value: str) -> str:
        if (not isinstance(value, str) or not value.strip() or len(value) > 200
                or any(ord(char) < 32 for char in value)):
            raise AnalysisTaskValidationError("name must contain 1 to 200 characters")
        return value.strip()

    @staticmethod
    def _receipt(session, key, request: dict) -> dict | None:
        if not isinstance(key, str) or not key.strip() or len(key) > 128:
            raise AnalysisTaskValidationError("idempotency_key must contain 1 to 128 characters")
        receipt = session.scalar(select(AnalysisTaskOperationRecord).where(
            AnalysisTaskOperationRecord.idempotency_key == key,
        ))
        if receipt is None:
            return None
        if receipt.request_sha256 != _digest(_canonical(request)):
            raise AnalysisTaskConflictError("idempotency key was used for a different request")
        return json.loads(receipt.result_json)

    @staticmethod
    def _save_receipt(session, key, request, result, *, action, subject_id, actor):
        session.add(AnalysisTaskOperationRecord(
            operation_id=uuid4().hex, idempotency_key=key, action=action, subject_id=subject_id,
            request_sha256=_digest(_canonical(request)), result_json=_canonical(result),
            requested_by=actor,
        ))

    def _task_command_request(self, action, task_id, *, expected_revision, requested_by,
                              name=None, config=None) -> dict:
        """Use the same actor/payload identity for writes and read-only retries."""
        actor = self._actor(requested_by)
        revision = self._revision(expected_revision)
        if action == "update_task":
            if name is None and config is None:
                raise AnalysisTaskValidationError("an updated name or configuration is required")
            name = self._name(name) if name is not None else None
            return {"action": action, "task_id": task_id, "expected_revision": revision,
                    "name": name, "config": config, "requested_by": actor}
        if action == "archive_task":
            if name is not None or config is not None:
                raise AnalysisTaskValidationError("archive requests cannot update task configuration")
            return {"action": action, "task_id": task_id, "expected_revision": revision,
                    "requested_by": actor}
        raise AnalysisTaskValidationError("only task update/archive receipts may be reconciled")

    def get_task_command_receipt(self, action, task_id, *, expected_revision, requested_by,
                                 idempotency_key, name=None, config=None) -> dict | None:
        """Reconcile an acknowledged/lost task command without authorizing a write.

        This is deliberately SELECT-only. A missing receipt still requires the
        caller's normal runtime/lifecycle gate before attempting any mutation.
        Actor, command and complete normalized request identity must all match.
        """
        request = self._task_command_request(
            action, task_id, expected_revision=expected_revision, requested_by=requested_by,
            name=name, config=config,
        )
        with self._db.get_session() as session:
            return self._receipt(session, idempotency_key, request)

    @staticmethod
    def _managed(session: Session, task_id: str, *, expected_revision: int | None = None):
        task = session.get(AnalysisTaskRecord, task_id)
        if task is None:
            raise AnalysisTaskNotFoundError("analysis task not found")
        if task.authority != MANAGED_AUTHORITY or task.template_id == LEGACY_TEMPLATE_ID:
            raise AnalysisTaskConflictError("legacy tasks retain their existing configuration authority")
        if task.archived:
            raise AnalysisTaskConflictError("analysis task is archived")
        if expected_revision is not None and task.current_revision != expected_revision:
            raise AnalysisTaskConflictError("analysis task revision changed")
        return task

    def create_task(self, camera_id, template_id, name, config, *, requested_by, idempotency_key):
        actor = self._actor(requested_by)
        name = self._name(name)
        if (not isinstance(camera_id, str) or not camera_id.strip() or len(camera_id) > 50
                or any(ord(char) < 32 for char in camera_id)):
            raise AnalysisTaskValidationError("camera_id is required")
        try:
            config = normalize_task_config(template_id, config)
        except (TypeError, ValueError) as exc:
            raise AnalysisTaskValidationError(str(exc)) from exc
        request = {"action": "create_task", "camera_id": camera_id, "template_id": template_id,
                   "name": name, "config": config, "requested_by": actor}
        with self._db.get_session() as session:
            self._write_lock(session)
            receipt = self._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            task = AnalysisTaskRecord(
                task_id=f"task-{uuid4().hex}", camera_id=camera_id, template_id=template_id,
                authority=MANAGED_AUTHORITY, name=name, current_revision=1, archived=False,
                created_by=actor, updated_at=datetime.now(timezone.utc),
            )
            config_json = _canonical(config)
            session.add(task)
            session.add(AnalysisTaskRevisionRecord(
                task_id=task.task_id, revision=1, config_json=config_json,
                config_sha256=_digest(config_json),
            ))
            session.add(TaskDeploymentRecord(
                task_id=task.task_id, authority=MANAGED_AUTHORITY,
                projection_status="shadow_unconfigured", runtime_status="unknown",
                reason="No shadow execution bundle has been applied",
            ))
            session.flush()
            result = self._task_dict(session, task)
            self._save_receipt(session, idempotency_key, request, result,
                               action="create_task", subject_id=task.task_id, actor=actor)
            session.commit()
            return result

    def update_task(self, task_id, *, expected_revision, name=None, config=None,
                    requested_by, idempotency_key):
        request = self._task_command_request(
            "update_task", task_id, expected_revision=expected_revision, name=name,
            config=config, requested_by=requested_by,
        )
        actor, expected_revision, name = request["requested_by"], request["expected_revision"], request["name"]
        with self._db.get_session() as session:
            self._write_lock(session)
            receipt = self._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            task = self._managed(session, task_id, expected_revision=expected_revision)
            self._require_stopped_intent(session, task_id)
            previous = self._task_dict(session, task)
            try:
                normalized = normalize_task_config(
                    task.template_id, config if config is not None else previous["config"],
                )
            except (TypeError, ValueError) as exc:
                raise AnalysisTaskValidationError(str(exc)) from exc
            config_json = _canonical(normalized)
            task.current_revision += 1
            if name is not None:
                task.name = name
            task.updated_at = datetime.now(timezone.utc)
            session.add(AnalysisTaskRevisionRecord(
                task_id=task_id, revision=task.current_revision, config_json=config_json,
                config_sha256=_digest(config_json),
            ))
            session.flush()
            result = self._task_dict(session, task)
            self._save_receipt(session, idempotency_key, request, result,
                               action="update_task", subject_id=task_id, actor=actor)
            session.commit()
            return result

    def archive_task(self, task_id, *, expected_revision, requested_by, idempotency_key):
        request = self._task_command_request(
            "archive_task", task_id, expected_revision=expected_revision, requested_by=requested_by,
        )
        actor, expected_revision = request["requested_by"], request["expected_revision"]
        with self._db.get_session() as session:
            self._write_lock(session)
            receipt = self._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            task = self._managed(session, task_id, expected_revision=expected_revision)
            self._require_stopped_intent(session, task_id)
            previous = self._task_dict(session, task)
            task.archived = True
            task.current_revision += 1
            task.updated_at = datetime.now(timezone.utc)
            config_json = _canonical(previous["config"])
            session.add(AnalysisTaskRevisionRecord(
                task_id=task_id, revision=task.current_revision, config_json=config_json,
                config_sha256=_digest(config_json),
            ))
            session.flush()
            result = self._task_dict(session, task)
            self._save_receipt(session, idempotency_key, request, result,
                               action="archive_task", subject_id=task_id, actor=actor)
            session.commit()
            return result

    @staticmethod
    def _require_stopped_intent(session: Session, task_id: str) -> None:
        deployment = session.get(TaskDeploymentRecord, task_id)
        if deployment is not None and deployment.desired_state == "running":
            raise AnalysisTaskConflictError("persist a stop intent before editing or archiving the task")

    def _trusted_artifact(self, relative_path: str) -> tuple[Path, str]:
        """Resolve and fingerprint a local file without following symlinks/junctions."""
        if self._models_root is None:
            raise AnalysisTaskValidationError("a trusted models_root must be configured")
        if (not isinstance(relative_path, str) or not relative_path or len(relative_path) > 500
                or "\\" in relative_path or ":" in relative_path or "\x00" in relative_path
                or PureWindowsPath(relative_path).drive or Path(relative_path).is_absolute()
                or any(part in {"", ".", ".."} for part in relative_path.split("/"))):
            raise AnalysisTaskValidationError("model path must be a safe models-root relative path")
        path = self._models_root / relative_path
        if path.suffix.lower() != ".onnx":
            raise AnalysisTaskValidationError("only ONNX artifacts are supported")
        try:
            # Inspect lexical components, not only resolve()'s already-followed target.
            for component in [path, *path.parents]:
                stat = component.lstat()
                if component.is_symlink() or getattr(stat, "st_file_attributes", 0) & 0x400:
                    raise AnalysisTaskValidationError("model path must not traverse links or junctions")
            root = self._models_root.resolve(strict=True)
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            before = path.stat()
            if not path.is_file() or not 0 < before.st_size <= 8 * 1024 ** 3:
                raise AnalysisTaskValidationError("ONNX artifact must be a non-empty bounded file")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                opened = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                    before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                ):
                    raise AnalysisTaskConflictError("model artifact changed while opening")
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                after = os.fstat(stream.fileno())
            current = path.lstat()
            if path.is_symlink() or getattr(current, "st_file_attributes", 0) & 0x400 or (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            ) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) or (
                current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns,
            ) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
                raise AnalysisTaskConflictError("model artifact changed during fingerprinting")
            return resolved, digest.hexdigest()
        except (OSError, ValueError) as exc:
            if isinstance(exc, AnalysisTaskValidationError):
                raise
            raise AnalysisTaskValidationError("model artifact is unavailable under trusted models_root") from exc

    def register_task_model(self, relative_path, descriptor, *, scope_camera_id=None,
                            requested_by, idempotency_key):
        actor = self._actor(requested_by)
        if scope_camera_id is not None and (
            not isinstance(scope_camera_id, str) or not scope_camera_id.strip()
            or len(scope_camera_id) > 50 or any(ord(char) < 32 for char in scope_camera_id)
        ):
            raise AnalysisTaskValidationError("camera-scoped model requires a camera identity")
        try:
            descriptor = TaskModelDescriptor.model_validate(descriptor).model_dump(mode="json")
        except (TypeError, ValueError) as exc:
            raise AnalysisTaskValidationError(str(exc)) from exc
        request = {"action": "register_task_model", "relative_path": relative_path,
                   "descriptor": descriptor,
                   "scope_camera_id": scope_camera_id, "requested_by": actor}
        # A committed registration is its idempotency result even if the source
        # file is later removed or changed. Execution still checks the registered
        # SHA at freeze/load time. Never hold a DB write lock while hashing a
        # potentially large artifact.
        with self._db.get_session() as session:
            receipt = self._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
        _path, artifact_sha256 = self._trusted_artifact(relative_path)
        with self._db.get_session() as session:
            self._write_lock(session)
            receipt = self._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            descriptor_json = _canonical(descriptor)
            record = TaskModelRecord(
                model_id=f"model-{uuid4().hex}", authority=MANAGED_AUTHORITY,
                relative_path=relative_path, artifact_sha256=artifact_sha256,
                scope_camera_id=scope_camera_id, descriptor_json=descriptor_json,
                descriptor_sha256=_digest(descriptor_json), created_by=actor,
            )
            session.add(record)
            session.flush()
            result = self._model_dict(record)
            self._save_receipt(session, idempotency_key, request, result,
                               action="register_task_model", subject_id=record.model_id, actor=actor)
            session.commit()
            return result

    @staticmethod
    def _model_dict(record: TaskModelRecord) -> dict:
        if (record.authority != MANAGED_AUTHORITY
                or not isinstance(record.artifact_sha256, str)
                or len(record.artifact_sha256) != 64
                or any(char not in "0123456789abcdef" for char in record.artifact_sha256)
                or _digest(record.descriptor_json) != record.descriptor_sha256):
            raise AnalysisTaskIntegrityError("model descriptor integrity check failed")
        try:
            descriptor = TaskModelDescriptor.model_validate_json(record.descriptor_json).model_dump(mode="json")
        except ValueError as exc:
            raise AnalysisTaskIntegrityError("model descriptor is invalid") from exc
        return {"model_id": record.model_id, "authority": record.authority,
                "scope_kind": "camera" if record.scope_camera_id is not None else "global",
                "scope_camera_id": record.scope_camera_id, "artifact_sha256": record.artifact_sha256,
                "descriptor": descriptor, "name": descriptor["name"],
                "adapter_id": descriptor["adapter_id"], "labels": descriptor["labels"],
                "created_by": record.created_by, "created_at": _iso(record.created_at)}

    def list_models(self) -> list[dict]:
        with self._db.get_session() as session:
            return [self._model_dict(record) for record in session.scalars(
                select(TaskModelRecord).order_by(TaskModelRecord.created_at, TaskModelRecord.model_id),
            )]

    def list_compatible_models(self, task_id: str) -> list[dict]:
        return [model for model in self.list_model_compatibility(task_id) if model["compatible_roles"]]

    def list_model_compatibility(self, task_id: str) -> list[dict]:
        """Explain every model/role pairing without opening model artifacts."""
        with self._db.get_session() as session:
            task = self._managed(session, task_id)
            task_dict = self._task_dict(session, task)
            roles = template_spec(task.template_id)["model_roles"]
            result = []
            for record in session.scalars(select(TaskModelRecord).order_by(TaskModelRecord.model_id)):
                model = self._model_dict(record)
                assessments = {}
                for role in roles:
                    reasons = model_role_incompatibilities(
                        task.template_id, role, model["descriptor"], task_dict["config"]["rules"]["labels"],
                    )
                    if record.scope_camera_id not in (None, task.camera_id):
                        reasons.insert(0, "CAMERA_SCOPE_MISMATCH")
                    assessments[role] = {"compatible": not reasons, "reasons": reasons}
                model["role_compatibility"] = assessments
                model["compatible_roles"] = [role for role, result in assessments.items() if result["compatible"]]
                result.append(model)
            return result

    def freeze_bundle(self, task_id, *, expected_revision, model_ids, requested_by, idempotency_key):
        actor = self._actor(requested_by)
        expected_revision = self._revision(expected_revision)
        if not isinstance(model_ids, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                 for k, v in model_ids.items()):
            raise AnalysisTaskValidationError("model_ids must map declared roles to registered IDs")
        request = {"action": "freeze_bundle", "task_id": task_id,
                   "expected_revision": expected_revision, "model_ids": model_ids, "requested_by": actor}
        # File fingerprinting may be slow. Validate outside the write lock,
        # then recheck the immutable registrations and task CAS in a fresh
        # transaction before committing the frozen manifest.
        with self._db.get_session() as session:
            receipt = self._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            task = self._managed(session, task_id, expected_revision=expected_revision)
            task_dict = self._task_dict(session, task)
            roles = template_spec(task.template_id)["model_roles"]
            if set(model_ids) != set(roles):
                raise AnalysisTaskValidationError("model bindings must exactly match template roles")
            checked = {}
            for role, model_id in model_ids.items():
                record = session.get(TaskModelRecord, model_id)
                if record is None:
                    raise AnalysisTaskNotFoundError("task model not found")
                model = self._model_dict(record)
                if (record.scope_camera_id not in (None, task.camera_id)
                        or model["adapter_id"] != roles[role]):
                    raise AnalysisTaskValidationError("model adapter or scope is incompatible with task")
                incompatibilities = model_role_incompatibilities(
                    task.template_id, role, model["descriptor"], task_dict["config"]["rules"]["labels"],
                )
                if incompatibilities:
                    required = template_spec(task.template_id)["model_role_requirements"][role]["required_labels"]
                    raise AnalysisTaskValidationError(
                        f"model role {role} label/adapter mismatch: {incompatibilities}; required labels {required}",
                    )
                checked[role] = (record.relative_path, model)
        for relative_path, model in checked.values():
            _path, current_sha = self._trusted_artifact(relative_path)
            if current_sha != model["artifact_sha256"]:
                raise AnalysisTaskConflictError("registered model artifact changed")
        with self._db.get_session() as session:
            self._write_lock(session)
            receipt = self._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            task = self._managed(session, task_id, expected_revision=expected_revision)
            task_dict = self._task_dict(session, task)
            roles = template_spec(task.template_id)["model_roles"]
            if set(model_ids) != set(roles):
                raise AnalysisTaskValidationError("model bindings must exactly match template roles")
            models = {}
            for role, model_id in model_ids.items():
                record = session.get(TaskModelRecord, model_id)
                if record is None:
                    raise AnalysisTaskNotFoundError("task model not found")
                model = self._model_dict(record)
                if (record.authority != MANAGED_AUTHORITY or record.scope_camera_id not in (None, task.camera_id)
                        or model["adapter_id"] != roles[role]):
                    raise AnalysisTaskValidationError("model adapter or scope is incompatible with task")
                if checked[role] != (record.relative_path, model):
                    raise AnalysisTaskConflictError("model registration changed during artifact validation")
                if model_role_incompatibilities(
                    task.template_id, role, model["descriptor"], task_dict["config"]["rules"]["labels"],
                ):
                    raise AnalysisTaskValidationError(f"model role {role} label schema is incompatible")
                models[role] = {"model_id": model_id, "artifact_sha256": record.artifact_sha256,
                                **model["descriptor"]}
            bundle_id = f"bundle-{uuid4().hex}"
            manifest = {
                "schema_version": "argus.task-execution-bundle.v1", "bundle_id": bundle_id,
                "task_id": task_id, "camera_id": task.camera_id, "task_revision": task.current_revision,
                "task_config_sha256": task_dict["config_sha256"], "template_id": task.template_id,
                "authority": MANAGED_AUTHORITY, "stage": "shadow", "config": task_dict["config"],
                "models": models,
            }
            manifest_json = _canonical(manifest)
            record = TaskExecutionBundleRecord(
                bundle_id=bundle_id, task_id=task_id, task_revision=task.current_revision,
                template_id=task.template_id, manifest_json=manifest_json,
                manifest_sha256=_digest(manifest_json), created_by=actor,
            )
            session.add(record)
            session.flush()
            result = self._bundle_dict(record)
            self._save_receipt(session, idempotency_key, request, result,
                               action="freeze_bundle", subject_id=bundle_id, actor=actor)
            session.commit()
            return result

    @staticmethod
    def _bundle_dict(record: TaskExecutionBundleRecord) -> dict:
        if _digest(record.manifest_json) != record.manifest_sha256:
            raise AnalysisTaskIntegrityError("execution bundle integrity check failed")
        try:
            manifest = json.loads(record.manifest_json)
            if not isinstance(manifest, dict) or _canonical(manifest) != record.manifest_json:
                raise ValueError("bundle manifest is not a canonical object")
        except (TypeError, ValueError) as exc:
            raise AnalysisTaskIntegrityError("execution bundle manifest is invalid") from exc
        if (set(manifest) != {
                    "schema_version", "bundle_id", "task_id", "camera_id", "task_revision",
                    "task_config_sha256", "template_id", "authority", "stage", "config", "models",
                }
                or manifest.get("schema_version") != "argus.task-execution-bundle.v1"
                or manifest.get("bundle_id") != record.bundle_id or manifest.get("task_id") != record.task_id
                or manifest.get("task_revision") != record.task_revision
                or manifest.get("template_id") != record.template_id
                or manifest.get("authority") != MANAGED_AUTHORITY or manifest.get("stage") != "shadow"):
            raise AnalysisTaskIntegrityError("execution bundle identity is inconsistent")
        try:
            if (type(manifest["task_revision"]) is not int or manifest["task_revision"] < 1
                    or not isinstance(manifest["camera_id"], str) or not manifest["camera_id"]
                    or not isinstance(manifest["models"], dict)
                    or set(manifest["models"]) != set(template_spec(record.template_id)["model_roles"])
                    or normalize_task_config(record.template_id, manifest["config"]) != manifest["config"]
                    or _digest(_canonical(manifest["config"])) != manifest["task_config_sha256"]):
                raise ValueError("bundle configuration or roles are invalid")
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisTaskIntegrityError("execution bundle configuration is inconsistent") from exc
        return {"bundle_id": record.bundle_id, "task_id": record.task_id,
                "task_revision": record.task_revision, "template_id": record.template_id,
                "manifest_sha256": record.manifest_sha256, "manifest": manifest,
                "stage": "shadow", "production_readiness": {"ready": False, "decision": "NO-GO",
                    "reason": "Template-specific production evaluation and signing policy are not implemented"},
                "created_by": record.created_by, "created_at": _iso(record.created_at)}

    def get_bundle(self, bundle_id: str) -> dict | None:
        with self._db.get_session() as session:
            record = session.get(TaskExecutionBundleRecord, bundle_id)
            return self._bundle_dict(record) if record is not None else None

    def list_bundles(self, task_id: str) -> list[dict]:
        with self._db.get_session() as session:
            # The UI's operation timeline is intentionally bounded, but a
            # historical apply receipt remains authoritative after it leaves
            # that window. This is observational metadata, not manifest input.
            applied = set(session.scalars(select(TaskDeploymentOperationRecord.bundle_id).where(
                TaskDeploymentOperationRecord.task_id == task_id,
                TaskDeploymentOperationRecord.status == "applied",
                TaskDeploymentOperationRecord.action.in_(("deploy", "rollback")),
                TaskDeploymentOperationRecord.bundle_id.is_not(None),
            )))
            return [{**self._bundle_dict(record), "historically_applied": record.bundle_id in applied}
                    for record in session.scalars(
                select(TaskExecutionBundleRecord).where(TaskExecutionBundleRecord.task_id == task_id)
                .order_by(TaskExecutionBundleRecord.created_at, TaskExecutionBundleRecord.bundle_id),
            )]

    def resolve_bundle(self, task_id: str, bundle_id: str) -> dict:
        """Server-only resolver; exact scope/config/revision/bytes are rechecked."""
        with self._db.get_session() as session:
            task = self._managed(session, task_id)
            task_dict = self._task_dict(session, task)
            bundle = session.get(TaskExecutionBundleRecord, bundle_id)
            if bundle is None:
                raise AnalysisTaskNotFoundError("execution bundle not found")
            manifest = self._bundle_dict(bundle)["manifest"]
            if (manifest["task_id"] != task_id or manifest["camera_id"] != task.camera_id
                    or manifest["task_revision"] != task.current_revision
                    or manifest["template_id"] != task.template_id
                    or manifest["task_config_sha256"] != task_dict["config_sha256"]
                    or manifest["config"] != task_dict["config"]):
                raise AnalysisTaskConflictError("bundle is not bound to the current task revision")
            roles = template_spec(task.template_id)["model_roles"]
            if set(manifest["models"]) != set(roles):
                raise AnalysisTaskIntegrityError("execution bundle model roles are invalid")
            for role, model in manifest["models"].items():
                record = session.get(TaskModelRecord, model["model_id"])
                if record is None:
                    raise AnalysisTaskIntegrityError("execution bundle model registration is missing")
                descriptor = self._model_dict(record)["descriptor"]
                if (record.authority != MANAGED_AUTHORITY
                        or record.scope_camera_id not in (None, task.camera_id)
                        or descriptor["adapter_id"] != roles[role]
                        or model_role_incompatibilities(task.template_id, role, descriptor,
                                                        task_dict["config"]["rules"]["labels"])
                        or model != {"model_id": record.model_id,
                                     "artifact_sha256": record.artifact_sha256, **descriptor}):
                    raise AnalysisTaskIntegrityError("execution bundle model binding changed")
                path, digest = self._trusted_artifact(record.relative_path)
                if digest != record.artifact_sha256:
                    raise AnalysisTaskConflictError("execution bundle artifact bytes changed")
                model["artifact_path"] = str(path)
            manifest["manifest_sha256"] = bundle.manifest_sha256
            return manifest

    def restore_bundle_configuration(self, task_id: str, bundle_id: str, *, expected_revision: int,
                                     requested_by: str, idempotency_key: str) -> dict:
        """Explicitly restore historical configuration as a new shadow revision.

        The historical bundle stays immutable and is never authorized against
        a different current revision. A new bundle and an immutable provenance
        receipt are committed with the new task revision. Deployment is a
        separate subsequent command; this operation cannot start a worker.
        """
        request = self._bundle_configuration_restore_request(task_id, bundle_id,
            expected_revision=expected_revision, requested_by=requested_by)
        actor, expected_revision = request["requested_by"], request["expected_revision"]

        def snapshot(session):
            task = self._managed(session, task_id, expected_revision=expected_revision)
            current = self._task_dict(session, task)
            deployment = session.get(TaskDeploymentRecord, task_id)
            if (deployment is None or deployment.authority != MANAGED_AUTHORITY
                    or deployment.desired_state != "stopped" or deployment.runtime_status != "stopped"):
                raise AnalysisTaskConflictError("complete a durable stop before restoring configuration")
            operation = session.get(TaskDeploymentOperationRecord, deployment.current_operation_id)
            if operation is None or operation.action != "stop" or operation.status != "applied":
                raise AnalysisTaskConflictError("confirmed stop receipt is required before restoration")
            historical = session.get(TaskExecutionBundleRecord, bundle_id)
            if historical is None:
                raise AnalysisTaskNotFoundError("historical execution bundle not found")
            manifest = self._bundle_dict(historical)["manifest"]
            proof = session.scalar(select(TaskDeploymentOperationRecord).where(
                TaskDeploymentOperationRecord.task_id == task_id,
                TaskDeploymentOperationRecord.bundle_id == bundle_id,
                TaskDeploymentOperationRecord.status == "applied",
            ))
            historical_revision = session.scalar(select(AnalysisTaskRevisionRecord).where(
                AnalysisTaskRevisionRecord.task_id == task_id,
                AnalysisTaskRevisionRecord.revision == manifest["task_revision"],
            ))
            if (proof is None or manifest["task_id"] != task_id or manifest["camera_id"] != task.camera_id
                    or manifest["template_id"] != task.template_id or historical_revision is None
                    or historical_revision.config_sha256 != manifest["task_config_sha256"]
                    or historical_revision.config_json != _canonical(manifest["config"])):
                raise AnalysisTaskConflictError("historical bundle has no matching applied task authority")
            roles, checked = template_spec(task.template_id)["model_roles"], {}
            for role, model in manifest["models"].items():
                record = session.get(TaskModelRecord, model["model_id"])
                if record is None:
                    raise AnalysisTaskIntegrityError("historical model registration is missing")
                descriptor = self._model_dict(record)["descriptor"]
                if (record.authority != MANAGED_AUTHORITY or record.scope_camera_id not in (None, task.camera_id)
                        or descriptor["adapter_id"] != roles[role]
                        or model_role_incompatibilities(task.template_id, role, descriptor,
                                                        manifest["config"]["rules"]["labels"])
                        or model != {"model_id": record.model_id,
                                     "artifact_sha256": record.artifact_sha256, **descriptor}):
                    raise AnalysisTaskIntegrityError("historical model binding is incompatible")
                checked[role] = (record.relative_path, record.artifact_sha256, descriptor)
            return current, manifest, checked, deployment.deployment_revision

        with self._db.get_session() as session:
            receipt = self._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            original, historical, checked, deployment_revision = snapshot(session)
        for relative, expected_sha, _descriptor in checked.values():
            _path, actual_sha = self._trusted_artifact(relative)
            if actual_sha != expected_sha:
                raise AnalysisTaskConflictError("historical model artifact bytes changed")
        with self._db.get_session() as session:
            self._write_lock(session)
            receipt = self._receipt(session, idempotency_key, request)
            if receipt is not None:
                return receipt
            current, manifest, models, deployment_version = snapshot(session)
            if (current["config_sha256"] != original["config_sha256"] or manifest != historical
                    or models != checked or deployment_version != deployment_revision):
                raise AnalysisTaskConflictError("restoration authority changed during artifact validation")
            revision = expected_revision + 1
            encoded = _canonical(historical["config"])
            task = session.get(AnalysisTaskRecord, task_id)
            task.current_revision, task.updated_at = revision, datetime.now(timezone.utc)
            session.add(AnalysisTaskRevisionRecord(task_id=task_id, revision=revision,
                config_json=encoded, config_sha256=_digest(encoded)))
            restored_manifest = json.loads(_canonical(historical))
            restored_manifest.update(bundle_id=f"bundle-{uuid4().hex}", task_revision=revision,
                                     task_config_sha256=_digest(encoded))
            manifest_json = _canonical(restored_manifest)
            bundle = TaskExecutionBundleRecord(bundle_id=restored_manifest["bundle_id"], task_id=task_id,
                task_revision=revision, template_id=task.template_id, manifest_json=manifest_json,
                manifest_sha256=_digest(manifest_json), created_by=actor)
            session.add(bundle)
            session.flush()
            result = {"task": self._task_dict(session, task), "bundle": self._bundle_dict(bundle),
                      "restore": {"restored_from_bundle_id": bundle_id,
                                  "restored_from_task_revision": historical["task_revision"],
                                  "prior_task_revision": expected_revision, "new_task_revision": revision,
                                  "restored_by": actor, "production_allowed": False}}
            self._save_receipt(session, idempotency_key, request, result,
                action="restore_bundle_configuration", subject_id=task_id, actor=actor)
            session.commit()
            return result

    def _bundle_configuration_restore_request(self, task_id, bundle_id, *, expected_revision, requested_by):
        if not isinstance(bundle_id, str) or not bundle_id:
            raise AnalysisTaskValidationError("historical bundle identity is required")
        return {"action": "restore_bundle_configuration", "task_id": task_id, "bundle_id": bundle_id,
                "expected_revision": self._revision(expected_revision), "requested_by": self._actor(requested_by)}

    def get_bundle_configuration_restore_receipt(self, task_id, bundle_id, *, expected_revision,
                                                requested_by, idempotency_key):
        request = self._bundle_configuration_restore_request(task_id, bundle_id,
            expected_revision=expected_revision, requested_by=requested_by)
        with self._db.get_session() as session:
            return self._receipt(session, idempotency_key, request)

    def commit_if_current(self, task_id: str, revision: int, config_sha256: str, callback) -> bool:
        """Serialize a bounded in-memory result commit with task configuration CAS.

        The caller must fence source/deployment identity separately. ``callback``
        may only perform a short in-memory commit with nonblocking source and
        scene identity checks. It must not do I/O, await, or run computation.
        Callers acquire this DB fence before entering runtime/source locks.
        No runtime-applied state or release authorization is written here.
        Contention drops this shadow result immediately: this method is called
        before the source identity fence and must never wait for a DB writer.
        """
        self._revision(revision)
        with self._db.get_session() as session:
            connection = session.connection()
            if connection.dialect.name != "sqlite":
                raise RuntimeError("analysis-task result commit currently requires SQLite")
            previous_timeout = int(connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one())
            try:
                connection.exec_driver_sql("PRAGMA busy_timeout=0")
                try:
                    self._write_lock(session)
                    task = session.get(AnalysisTaskRecord, task_id)
                    if (task is None or task.authority != MANAGED_AUTHORITY or task.archived
                            or task.template_id == LEGACY_TEMPLATE_ID or task.current_revision != revision):
                        return False
                    record = session.scalar(select(AnalysisTaskRevisionRecord).where(
                        AnalysisTaskRevisionRecord.task_id == task_id,
                        AnalysisTaskRevisionRecord.revision == revision,
                    ))
                    if record is None or _digest(record.config_json) != record.config_sha256:
                        raise AnalysisTaskIntegrityError("Task configuration revision is missing or corrupted")
                    if record.config_sha256 != config_sha256:
                        return False
                except OperationalError as exc:
                    error_code = getattr(exc.orig, "sqlite_errorcode", -1)
                    if (error_code & 0xFF) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                        return False
                    raise
                callback()
                return True
            finally:
                # busy_timeout belongs to the DBAPI connection, not this
                # transaction. Restore it before the session returns a pooled
                # connection, including the contention/error/early-return paths.
                try:
                    connection.exec_driver_sql(f"PRAGMA busy_timeout={previous_timeout}")
                except Exception:
                    connection.invalidate()
                    raise

    @staticmethod
    def _write_lock(session: Session) -> None:
        # A database lock, not a process-local mutex: bootstrap/projection must
        # also be safe when two app instances share a database. It is acquired
        # before any reads, and session exit rolls back every failed write.
        if session.get_bind().dialect.name != "sqlite":
            raise RuntimeError("analysis-task bootstrap currently requires SQLite")
        session.execute(text("BEGIN IMMEDIATE"))

    def bootstrap_legacy(self, camera_id: str, config: dict) -> dict:
        """Create once; subsequent boots never replace the stored configuration."""
        task_id = legacy_task_id(camera_id)
        if not isinstance(config, dict):
            raise ValueError("task configuration must be a JSON object")
        with self._db.get_session() as session:
            self._write_lock(session)
            existing = session.get(AnalysisTaskRecord, task_id)
            if existing is not None:
                self._check_legacy_identity(existing, camera_id)
                return self._task_dict(session, existing)

            config_json = _canonical(config)
            record = AnalysisTaskRecord(
                task_id=task_id,
                camera_id=camera_id,
                template_id=LEGACY_TEMPLATE_ID,
                authority=LEGACY_AUTHORITY,
                name="Legacy anomaly detection",
                current_revision=1,
            )
            session.add(record)
            session.add(AnalysisTaskRevisionRecord(
                task_id=task_id,
                revision=1,
                config_json=config_json,
                config_sha256=_digest(config_json),
            ))
            session.add(TaskDeploymentRecord(
                task_id=task_id,
                authority=LEGACY_AUTHORITY,
                projection_status="not_synced",
                runtime_status="unknown",
                reason="Legacy registry projection has not been synchronized",
            ))
            session.flush()
            result = self._task_dict(session, record)
            session.commit()
            return result

    def list_tasks(self, camera_id: str | None = None) -> list[dict]:
        with self._db.get_session() as session:
            statement = select(AnalysisTaskRecord).order_by(
                AnalysisTaskRecord.camera_id, AnalysisTaskRecord.task_id,
            )
            if camera_id is not None:
                statement = statement.where(AnalysisTaskRecord.camera_id == camera_id)
            return [self._task_dict(session, task) for task in session.scalars(statement)]

    def get_task(self, task_id: str) -> dict | None:
        with self._db.get_session() as session:
            task = session.get(AnalysisTaskRecord, task_id)
            return None if task is None else self._task_dict(session, task)

    def resolve_legacy(self, camera_id: str) -> dict | None:
        with self._db.get_session() as session:
            task = session.get(AnalysisTaskRecord, legacy_task_id(camera_id))
            if task is None:
                return None
            self._check_legacy_identity(task, camera_id)
            return self._task_dict(session, task)

    @staticmethod
    def _check_legacy_identity(task: AnalysisTaskRecord, camera_id: str) -> None:
        if (
            task.task_id != legacy_task_id(camera_id)
            or task.camera_id != camera_id
            or task.template_id != LEGACY_TEMPLATE_ID
            or task.authority != LEGACY_AUTHORITY
        ):
            raise AnalysisTaskIntegrityError("Legacy task identity or authority is inconsistent")

    @staticmethod
    def _active(session: Session, camera_id: str) -> ModelRecord | None:
        return session.scalar(select(ModelRecord).where(
            ModelRecord.camera_id == camera_id,
            ModelRecord.stage == ModelStage.PRODUCTION.value,
            ModelRecord.is_active.is_(True),
        ))

    @staticmethod
    def _model_snapshot(record: ModelRecord | None) -> str | None:
        if record is None:
            return None
        # Snapshot all persisted fields, including evidence and its revision,
        # before validating outside the write lock. Any concurrent authoritative
        # change invalidates this projection attempt.
        return _canonical({
            column.name: (
                _iso(value) if isinstance(value := getattr(record, column.name), datetime) else value
            )
            for column in ModelRecord.__table__.columns
        })

    def sync_legacy_model_projection(self, camera_id: str) -> dict | None:
        """Project a verified legacy desired identity, never runtime-applied state.

        Explicit lifecycle action, not a GET side effect. The existing evidence
        validator hashes the authorized artifact but never loads/runs a model.
        Registry validation occurs outside the SQLite write lock, followed by
        a full authoritative-record comparison inside the transaction.
        """
        task_id = legacy_task_id(camera_id)
        with self._db.get_session() as session:
            if session.get(AnalysisTaskRecord, task_id) is None:
                return None
            model = self._active(session, camera_id)
            snapshot = self._model_snapshot(model)

        valid = False
        if model is not None:
            from argus.storage.model_registry import validate_persisted_release_evidence

            valid = validate_persisted_release_evidence(model).valid

        with self._db.get_session() as session:
            self._write_lock(session)
            task = session.get(AnalysisTaskRecord, task_id)
            if task is None:
                return None
            self._check_legacy_identity(task, camera_id)
            current = self._active(session, camera_id)
            if self._model_snapshot(current) != snapshot:
                raise AnalysisTaskProjectionConflict("Legacy registry changed during projection")
            deployment = session.get(TaskDeploymentRecord, task_id)
            if deployment is None or deployment.authority != LEGACY_AUTHORITY:
                raise AnalysisTaskIntegrityError("Legacy deployment identity is missing or inconsistent")

            artifact = self._project_artifact(session, current) if valid and current else None
            deployment.desired_artifact_id = artifact.artifact_id if artifact else None
            deployment.registry_artifact_id = artifact.artifact_id if artifact else None
            deployment.registry_model_version_id = current.model_version_id if artifact else None
            deployment.registry_release_revision = int(current.release_revision) if artifact else None
            deployment.registry_stage = current.stage if artifact else None
            deployment.projection_status = (
                "authorized" if artifact else "blocked" if current is not None else "unbound"
            )
            # A historical applied release operation is not current runtime
            # observation after a process restart or source/model failure.
            deployment.applied_artifact_id = None
            deployment.runtime_status = "unknown"
            deployment.reason = (
                "Current runtime identity must be observed independently"
                if artifact else "Legacy production evidence validation failed"
                if current is not None else "No active legacy production model"
            )
            deployment.observed_at = datetime.now(timezone.utc)
            session.flush()
            result = self._task_dict(session, task)
            session.commit()
            return result

    @staticmethod
    def _project_artifact(session: Session, model: ModelRecord) -> ModelArtifactRecord:
        # Legacy registry hashes may be abbreviated. Only the verified release
        # evidence supplies the full runtime-artifact SHA-256 identity.
        artifact_sha256 = json.loads(model.release_evidence_json)["artifacts"][
            "runtime_artifact_sha256"
        ]
        if (
            not isinstance(artifact_sha256, str)
            or len(artifact_sha256) != 64
            or any(char not in "0123456789abcdef" for char in artifact_sha256)
        ):
            raise AnalysisTaskIntegrityError("Verified runtime artifact identity is invalid")
        metadata = {
            "authority": LEGACY_AUTHORITY,
            "legacy_model_version_id": model.model_version_id,
            "scope_kind": "camera",
            "scope_id": model.camera_id,
            "model_type": model.model_type,
            "component_type": model.component_type,
            "registry_model_hash": model.model_hash,
            "artifact_sha256": artifact_sha256,
            "model_path": model.model_path,
            "data_hash": model.data_hash,
            "code_version": model.code_version,
            "training_params": model.training_params,
            "calibration_thresholds": model.calibration_thresholds,
            "backbone_version_id": model.backbone_version_id,
        }
        metadata_json = _canonical(metadata)
        artifact_id = _digest(metadata_json)
        artifact = session.get(ModelArtifactRecord, artifact_id)
        if artifact is None:
            artifact = ModelArtifactRecord(
                artifact_id=artifact_id,
                authority=LEGACY_AUTHORITY,
                legacy_model_version_id=model.model_version_id,
                scope_kind="camera",
                scope_id=model.camera_id,
                model_type=model.model_type,
                component_type=model.component_type,
                registry_model_hash=model.model_hash,
                artifact_sha256=artifact_sha256,
                metadata_json=metadata_json,
            )
            session.add(artifact)
            session.flush()
        elif artifact.metadata_json != metadata_json:
            raise AnalysisTaskIntegrityError("Immutable artifact identity is inconsistent")
        return artifact

    @staticmethod
    def _task_dict(session: Session, task: AnalysisTaskRecord) -> dict:
        if task.authority == LEGACY_AUTHORITY:
            AnalysisTaskService._check_legacy_identity(task, task.camera_id)
        elif task.authority != MANAGED_AUTHORITY or task.template_id == LEGACY_TEMPLATE_ID:
            raise AnalysisTaskIntegrityError("Task identity or authority is inconsistent")
        if type(task.current_revision) is not int or task.current_revision < 1:
            raise AnalysisTaskIntegrityError("Task revision is invalid")
        revision = session.scalar(select(AnalysisTaskRevisionRecord).where(
            AnalysisTaskRevisionRecord.task_id == task.task_id,
            AnalysisTaskRevisionRecord.revision == task.current_revision,
        ))
        if revision is None or _digest(revision.config_json) != revision.config_sha256:
            raise AnalysisTaskIntegrityError("Task configuration revision is missing or corrupted")
        try:
            config = json.loads(revision.config_json)
            if not isinstance(config, dict) or _canonical(config) != revision.config_json:
                raise ValueError("configuration is not a canonical JSON object")
            if task.authority == MANAGED_AUTHORITY and normalize_task_config(
                task.template_id, config,
            ) != config:
                raise ValueError("managed configuration is not normalized")
        except (TypeError, ValueError) as exc:
            raise AnalysisTaskIntegrityError("Task configuration revision is invalid") from exc
        deployment = session.get(TaskDeploymentRecord, task.task_id)
        deployment_dict = None
        if deployment is not None:
            deployment_dict = {
                column.name: (
                    _iso(value)
                    if isinstance(value := getattr(deployment, column.name), datetime) else value
                )
                for column in TaskDeploymentRecord.__table__.columns
            }
            artifact_id = deployment.registry_artifact_id
            artifact = session.get(ModelArtifactRecord, artifact_id) if artifact_id else None
            if deployment.authority != task.authority:
                raise AnalysisTaskIntegrityError("Task deployment authority is inconsistent")
            if artifact_id and artifact is None:
                raise AnalysisTaskIntegrityError("Task deployment artifact identity is missing")
            if artifact is not None:
                try:
                    metadata = json.loads(artifact.metadata_json)
                    metadata_matches = isinstance(metadata, dict) and all(
                        metadata.get(field) == getattr(artifact, field)
                        for field in (
                            "authority", "legacy_model_version_id", "scope_kind", "scope_id",
                            "model_type", "component_type", "registry_model_hash", "artifact_sha256",
                        )
                    )
                except (TypeError, ValueError):
                    metadata_matches = False
                if (
                    _digest(artifact.metadata_json) != artifact.artifact_id
                    or not metadata_matches
                    or artifact.authority != task.authority
                    or (artifact.scope_kind == "camera" and artifact.scope_id != task.camera_id)
                    or (task.authority == LEGACY_AUTHORITY and artifact.scope_kind != "camera")
                    or artifact.legacy_model_version_id != deployment.registry_model_version_id
                ):
                    raise AnalysisTaskIntegrityError("Task deployment artifact scope is inconsistent")
            deployment_dict["registry_artifact"] = (
                {
                    "artifact_id": artifact.artifact_id,
                    "authority": artifact.authority,
                    "legacy_model_version_id": artifact.legacy_model_version_id,
                    "scope_kind": artifact.scope_kind,
                    "scope_id": artifact.scope_id,
                    "model_type": artifact.model_type,
                    "component_type": artifact.component_type,
                    "registry_model_hash": artifact.registry_model_hash,
                    "artifact_sha256": artifact.artifact_sha256,
                }
                if artifact else None
            )
            # Read-only projection freshness check. A GET must not synchronize
            # state, nor present a startup snapshot as current authorization.
            if task.authority == LEGACY_AUTHORITY and deployment.projection_status == "authorized":
                current = AnalysisTaskService._active(session, task.camera_id)
                if (
                    current is None or artifact is None
                    or current.model_version_id != deployment.registry_model_version_id
                    or current.release_revision != deployment.registry_release_revision
                    or current.model_hash != artifact.registry_model_hash
                ):
                    deployment_dict["projection_status"] = "stale"
                    deployment_dict["reason"] = "Legacy registry changed since this snapshot"
        return {
            "task_id": task.task_id,
            "camera_id": task.camera_id,
            "template_id": task.template_id,
            "authority": task.authority,
            "name": task.name,
            "archived": task.archived,
            "created_by": task.created_by,
            "updated_at": _iso(task.updated_at),
            "current_revision": task.current_revision,
            "config": config,
            "config_sha256": revision.config_sha256,
            "deployment": deployment_dict,
            "created_at": _iso(task.created_at),
        }
