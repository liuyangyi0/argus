"""Durable, task-scoped shadow deployment intent and observed receipts.

Only this service changes managed deployment authority. It never promotes a
model, starts a source, or infers runtime success from a successful DB commit.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from uuid import uuid4

from sqlalchemy import select, text

from argus.storage.analysis_tasks import AnalysisTaskConflictError, AnalysisTaskValidationError
from argus.storage.models import (
    AnalysisTaskRecord, AnalysisTaskRevisionRecord,
    TaskDeploymentOperationRecord, TaskDeploymentRecord,
)

_ACTIVE = {"pending", "running", "waiting_source", "waiting_scene", "reconcile_required"}
_APPLIED_STATES = {"running", "waiting_source", "blocked", "sampling_insufficient"}


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value).isoformat() if value else None


class TaskDeploymentService:
    def __init__(self, db, task_service):
        self._db, self._tasks = db, task_service

    @staticmethod
    def _begin(session):
        if session.get_bind().dialect.name != "sqlite":
            raise RuntimeError("managed deployment store requires SQLite write serialization")
        session.execute(text("BEGIN IMMEDIATE"))

    def _task(self, task_id, *, stopping=False):
        task = self._tasks.get_task(task_id)
        if task is None:
            raise LookupError("task not found")
        if task.get("authority") != "managed" or (task.get("archived") and not stopping):
            raise AnalysisTaskConflictError("only active managed tasks can be deployed")
        return task

    @staticmethod
    def _operation(row):
        return {"operation_id": row.operation_id, "task_id": row.task_id,
                "action": row.action, "bundle_id": row.bundle_id,
                "previous_bundle_id": row.previous_bundle_id,
                "deployment_revision": row.deployment_revision, "status": row.status,
                "failure_code": row.failure_code, "requested_by": row.requested_by,
                "created_at": _iso(row.created_at), "updated_at": _iso(row.updated_at),
                "stage": "shadow", "production_allowed": False}

    @staticmethod
    def _deployment(row):
        return {"task_id": row.task_id, "authority": row.authority,
                "desired_bundle_id": row.desired_bundle_id, "applied_bundle_id": row.applied_bundle_id,
                "previous_bundle_id": row.previous_bundle_id, "desired_state": row.desired_state,
                "deployment_revision": row.deployment_revision,
                "applied_deployment_revision": row.applied_deployment_revision,
                "current_operation_id": row.current_operation_id,
                "observed_state": row.runtime_status, "failure_code": row.failure_code,
                "operation_id": row.current_operation_id, "state": row.runtime_status,
                "error_code": row.failure_code,
                "observed_at": _iso(row.observed_at), "stage": "shadow", "production_allowed": False}

    def get(self, task_id):
        self._task(task_id, stopping=True)
        with self._db.get_session() as session:
            row = session.get(TaskDeploymentRecord, task_id)
            if row is None or row.authority != "managed":
                raise AnalysisTaskConflictError("managed deployment identity is missing")
            result = self._deployment(row)
            operation = session.get(TaskDeploymentOperationRecord, row.current_operation_id) if row.current_operation_id else None
            result["operation"] = self._operation(operation) if operation else None
            return result

    def get_operation(self, operation_id):
        with self._db.get_session() as session:
            row = session.get(TaskDeploymentOperationRecord, operation_id)
            if row is None:
                raise LookupError("deployment operation not found")
            return self._operation(row)

    def list_operations(self, task_id):
        self._task(task_id, stopping=True)
        with self._db.get_session() as session:
            return [self._operation(row) for row in session.scalars(select(TaskDeploymentOperationRecord).where(
                TaskDeploymentOperationRecord.task_id == task_id,
            ).order_by(TaskDeploymentOperationRecord.deployment_revision.desc()).limit(100))]

    def request(self, task_id, *, action, bundle_id=None, expected_deployment_revision,
                requested_by, idempotency_key):
        if (action not in {"deploy", "stop", "rollback"}
                or type(expected_deployment_revision) is not int or expected_deployment_revision < 0
                or not isinstance(requested_by, str) or not requested_by.strip()
                or requested_by.strip().lower() == "unknown" or len(requested_by) > 100
                or not isinstance(idempotency_key, str) or not 8 <= len(idempotency_key) <= 128):
            raise AnalysisTaskValidationError("invalid managed deployment command")
        if (action == "stop" and bundle_id is not None
                or bundle_id is not None and (not isinstance(bundle_id, str) or len(bundle_id) > 64)):
            raise AnalysisTaskValidationError("invalid deployment bundle")
        requested_by = requested_by.strip()
        payload = {"task_id": task_id, "action": action, "bundle_id": bundle_id,
                   "expected_deployment_revision": expected_deployment_revision, "requested_by": requested_by}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        # Reconcile exact committed requests before inspecting current task or
        # runtime state. A lost acknowledgement must never mint another intent.
        with self._db.get_session() as session:
            existing = session.scalar(select(TaskDeploymentOperationRecord).where(
                TaskDeploymentOperationRecord.idempotency_key == idempotency_key))
            if existing:
                if existing.request_sha256 != digest:
                    raise AnalysisTaskConflictError("idempotency key belongs to another command")
                return self._operation(existing)
        task = self._task(task_id, stopping=action == "stop")
        if action == "rollback" and bundle_id is None:
            bundle_id = self.get(task_id)["previous_bundle_id"]
        if action != "stop":
            if not bundle_id:
                raise AnalysisTaskValidationError("a frozen bundle is required")
            # Exact registered bytes, template, camera, task config and revision
            # are checked again by the executor immediately before loading.
            manifest = self._tasks.resolve_bundle(task_id, bundle_id)
            if manifest["task_revision"] != task["current_revision"]:
                raise AnalysisTaskConflictError("restore historical configuration as a new revision before rollback")
        with self._db.get_session() as session:
            self._begin(session)
            existing = session.scalar(select(TaskDeploymentOperationRecord).where(
                TaskDeploymentOperationRecord.idempotency_key == idempotency_key))
            if existing:
                if existing.request_sha256 != digest:
                    raise AnalysisTaskConflictError("idempotency key belongs to another command")
                return self._operation(existing)
            current_task = session.get(AnalysisTaskRecord, task_id)
            current_config = session.scalar(select(AnalysisTaskRevisionRecord).where(
                AnalysisTaskRevisionRecord.task_id == task_id,
                AnalysisTaskRevisionRecord.revision == current_task.current_revision,
            )) if current_task is not None else None
            if (current_task is None or current_config is None or current_task.authority != "managed"
                    or current_task.current_revision != task["current_revision"]
                    or current_task.camera_id != task["camera_id"]
                    or current_task.template_id != task["template_id"]
                    or current_config.config_sha256 != task["config_sha256"]
                    or current_task.archived and action != "stop"):
                raise AnalysisTaskConflictError("task authority changed while accepting deployment")
            row = session.get(TaskDeploymentRecord, task_id)
            if row is None or row.authority != "managed" or row.deployment_revision != expected_deployment_revision:
                raise AnalysisTaskConflictError("deployment revision changed")
            current = session.get(TaskDeploymentOperationRecord, row.current_operation_id) if row.current_operation_id else None
            if current and current.status in _ACTIVE:
                if action != "stop":
                    raise AnalysisTaskConflictError("reconcile or stop the current deployment operation first")
                current.status, current.updated_at = "superseded", _now()
            if action == "rollback":
                authorized = session.scalar(select(TaskDeploymentOperationRecord).where(
                    TaskDeploymentOperationRecord.task_id == task_id,
                    TaskDeploymentOperationRecord.bundle_id == bundle_id,
                    TaskDeploymentOperationRecord.status == "applied"))
                if authorized is None:
                    raise AnalysisTaskConflictError("rollback target has no applied deployment receipt")
            revision = row.deployment_revision + 1
            operation = TaskDeploymentOperationRecord(
                operation_id=f"deploy-{uuid4().hex}", task_id=task_id, idempotency_key=idempotency_key,
                request_sha256=digest, action=action, bundle_id=bundle_id,
                previous_bundle_id=row.applied_bundle_id, deployment_revision=revision,
                status="pending", requested_by=requested_by, updated_at=_now())
            row.deployment_revision, row.current_operation_id = revision, operation.operation_id
            row.desired_state = "stopped" if action == "stop" else "running"
            row.desired_bundle_id = row.applied_bundle_id if action == "stop" else bundle_id
            row.runtime_status, row.failure_code = "pending", None
            session.add(operation)
            session.flush()
            result = self._operation(operation)
            session.commit()
            return result

    def list_pending(self):
        with self._db.get_session() as session:
            return [self._operation(row) for row in session.scalars(select(TaskDeploymentOperationRecord).where(
                TaskDeploymentOperationRecord.status.in_(_ACTIVE)).order_by(TaskDeploymentOperationRecord.created_at))]

    def list_recoverable(self):
        with self._db.get_session() as session:
            return [self._deployment(row) for row in session.scalars(select(TaskDeploymentRecord).where(
                TaskDeploymentRecord.authority == "managed", TaskDeploymentRecord.desired_state == "running"))]

    def recovery_actor(self, task_id, bundle_id):
        with self._db.get_session() as session:
            operation = session.scalar(select(TaskDeploymentOperationRecord).where(
                TaskDeploymentOperationRecord.task_id == task_id,
                TaskDeploymentOperationRecord.bundle_id == bundle_id,
                TaskDeploymentOperationRecord.status == "applied",
            ).order_by(TaskDeploymentOperationRecord.deployment_revision.desc()))
            if operation is None:
                raise AnalysisTaskConflictError("recovery bundle has no applied command identity")
            return operation.requested_by

    def claim(self, operation_id):
        with self._db.get_session() as session:
            self._begin(session)
            operation = session.get(TaskDeploymentOperationRecord, operation_id)
            if operation is None:
                raise LookupError("deployment operation not found")
            row = session.get(TaskDeploymentRecord, operation.task_id)
            if (row.current_operation_id != operation_id or operation.status not in {"pending", "waiting_source", "waiting_scene"}):
                return None
            operation.status, operation.updated_at = "running", _now()
            row.runtime_status, row.failure_code = "preparing", None
            session.flush()
            result = self._operation(operation)
            session.commit()
            return result

    def observe(self, operation_id, runtime=None, *, failure_code=None, uncertain=False):
        """Commit only an exact runtime receipt, or persist explicit uncertainty."""
        if failure_code is not None and not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", failure_code):
            failure_code = "DEPLOYMENT_FAILED"
        with self._db.get_session() as session:
            self._begin(session)
            operation = session.get(TaskDeploymentOperationRecord, operation_id)
            if operation is None:
                raise LookupError("deployment operation not found")
            row = session.get(TaskDeploymentRecord, operation.task_id)
            if row.current_operation_id != operation_id or operation.status in {"applied", "failed", "superseded"}:
                return self._operation(operation)
            runtime = runtime or {}
            exact = (runtime.get("task_id") == operation.task_id
                     and runtime.get("deployment_operation_id") == operation.operation_id
                     and runtime.get("deployment_revision") == operation.deployment_revision)
            applied = (operation.action == "stop" and runtime.get("task_id") == operation.task_id
                       and runtime.get("state") == "stopped") or (
                operation.action != "stop" and exact and runtime.get("bundle_id") == operation.bundle_id
                and runtime.get("state") in _APPLIED_STATES)
            if applied:
                if operation.action != "stop":
                    if row.applied_bundle_id != operation.bundle_id:
                        row.previous_bundle_id = row.applied_bundle_id
                    row.applied_bundle_id = operation.bundle_id
                row.applied_deployment_revision = operation.deployment_revision
                row.runtime_status, row.failure_code = runtime["state"], None
                operation.status, operation.failure_code = "applied", None
            elif failure_code in {"SOURCE_UNAVAILABLE", "SCENE_BLOCKED"}:
                waiting = "waiting_source" if failure_code == "SOURCE_UNAVAILABLE" else "waiting_scene"
                row.runtime_status, row.failure_code = waiting, failure_code
                operation.status, operation.failure_code = waiting, failure_code
            elif uncertain or not failure_code:
                row.runtime_status, row.failure_code = "reconcile_required", failure_code or "RUNTIME_RECEIPT_UNKNOWN"
                operation.status, operation.failure_code = "reconcile_required", row.failure_code
            else:
                operation.status, operation.failure_code = "failed", failure_code
                row.runtime_status = (runtime.get("state") if runtime.get("bundle_id") == row.applied_bundle_id
                                      and runtime.get("state") in _APPLIED_STATES else "failed")
                row.failure_code = failure_code
            row.observed_at = operation.updated_at = _now()
            session.flush()
            result = self._operation(operation)
            session.commit()
            return result

    def requeue_owned_recovery(self, operation_id):
        """Only the exclusive runtime owner may use this after confirmed absence.

        Reusing the original operation preserves its actor, requested bundle
        and revision. This does not create approval or revise a user's intent.
        """
        with self._db.get_session() as session:
            self._begin(session)
            operation = session.get(TaskDeploymentOperationRecord, operation_id)
            if operation is None:
                raise LookupError("deployment operation not found")
            row = session.get(TaskDeploymentRecord, operation.task_id)
            if row.current_operation_id == operation_id and operation.status in {"running", "reconcile_required"}:
                operation.status, operation.updated_at = "pending", _now()
                row.runtime_status, row.failure_code = "recovering", None
                session.commit()
            return self._operation(operation)

    def observe_runtime(self, task_id, operation_id, runtime):
        """Refresh live availability without rewriting historical apply receipts."""
        with self._db.get_session() as session:
            self._begin(session)
            row = session.get(TaskDeploymentRecord, task_id)
            operation = session.get(TaskDeploymentOperationRecord, operation_id)
            if (row is None or operation is None or row.current_operation_id != operation_id
                    or operation.status != "applied"):
                return
            if row.desired_state == "stopped":
                if runtime is None or runtime.get("state") == "stopped":
                    row.runtime_status, row.failure_code = "stopped", None
                else:
                    row.runtime_status, row.failure_code = "reconcile_required", "UNEXPECTED_RUNTIME"
            elif (runtime is None or runtime.get("deployment_operation_id") != operation_id
                  or runtime.get("deployment_revision") != operation.deployment_revision):
                row.runtime_status, row.failure_code = "unknown", "RUNTIME_RECEIPT_UNKNOWN"
            else:
                state = runtime.get("state")
                row.runtime_status = state if state in {
                    "running", "waiting_source", "blocked", "sampling_insufficient", "stopped", "failed", "cleanup_pending",
                } else "unknown"
                code = runtime.get("error_code")
                row.failure_code = code if isinstance(code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", code) else None
            row.observed_at = _now()
            session.commit()
