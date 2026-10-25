"""Crash-safe two-phase application service for model releases.

The durable ``ModelReleaseOperation`` is the desired state.  ``ModelRecord``
remains the applied state and is never changed merely to make a candidate
visible to runtime code.  Runtime preparation must load and verify an exact
registry-authorized artifact into standby storage; only a verified transaction
may be committed.  The model row, production uniqueness changes, revision and
operation status are then persisted atomically.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol, runtime_checkable

import structlog
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from argus.storage.models import (
    ModelRecord,
    ModelReleaseOperation,
    ModelStage,
    ReleaseOperationStatus,
)
from argus.storage.release_pipeline import (
    ReleasePipeline,
    ReleaseTransitionPlan,
)
from argus.storage.release_execution import ReleaseExecutionLock

logger = structlog.get_logger()

EventPublisher = Callable[[str, dict], None]


@dataclass(frozen=True, slots=True)
class RuntimeReleaseIdentity:
    """Identity proved by a standby runtime before any applied-state write."""

    camera_id: str
    model_version_id: str
    target_stage: str
    runtime_artifact_sha256: str | None


@dataclass(frozen=True, slots=True)
class RuntimeReleaseCommit:
    """Result of atomically switching a previously prepared runtime token."""

    success: bool
    state: object = None
    error: str | None = None


@runtime_checkable
class RuntimeReleaseTransaction(Protocol):
    """Standby runtime transaction retained until DB commit succeeds.

    ``commit`` may atomically expose the prepared engine/runner but must retain
    enough previous state for ``abort`` to restore it if the database CAS or
    commit fails.  ``abort`` is also used to release an unused standby token.
    """

    def commit(self) -> RuntimeReleaseCommit:
        ...

    def abort(self) -> RuntimeReleaseCommit:
        ...

    def finalize(self) -> RuntimeReleaseCommit:
        """Release transaction resources after the applied DB commit."""
        ...


@dataclass(frozen=True, slots=True)
class RuntimeReleasePreparation:
    """Result returned after loading and checking a standby runtime."""

    success: bool
    identity: RuntimeReleaseIdentity | None = None
    transaction: RuntimeReleaseTransaction | None = None
    state: object = None
    error: str | None = None


RuntimePrepare = Callable[[ReleaseTransitionPlan], RuntimeReleasePreparation]


@dataclass(slots=True)
class _NoRuntimeReleaseTransaction:
    """Commit token for an applied-state change that cannot affect runtime.

    An inactive candidate has never been authorized for either the primary or
    shadow slot.  Retiring it therefore has no runtime work to prepare or
    swap.  Keeping the no-op behind ``ModelReleaseService`` still gives the
    cleanup a durable desired operation, CAS, version event and audit entry.
    """

    committed: bool = False

    def commit(self) -> RuntimeReleaseCommit:
        if self.committed:
            return RuntimeReleaseCommit(
                False,
                error="no-runtime release transaction already committed",
            )
        self.committed = True
        return RuntimeReleaseCommit(
            True,
            state={
                "runtime_impact": "none",
                "reason": "inactive_candidate_retirement",
            },
        )

    def abort(self) -> RuntimeReleaseCommit:
        self.committed = False
        return RuntimeReleaseCommit(
            True,
            state={"runtime_impact": "none", "restored": True},
        )

    def finalize(self) -> RuntimeReleaseCommit:
        return RuntimeReleaseCommit(
            True,
            state={"runtime_impact": "none", "released": True},
        )


class ModelReleaseService:
    """Coordinate desired state, runtime prepare and atomic applied state."""

    _locks_guard = threading.Lock()
    _camera_locks: dict[str, threading.RLock] = {}

    def __init__(
        self,
        session_factory,
        pipeline: ReleasePipeline,
        *,
        runtime_prepare: RuntimePrepare,
        event_publisher: EventPublisher | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._pipeline = pipeline
        self._runtime_prepare = runtime_prepare
        self._event_publisher = event_publisher
        self._interrupted_operations: set[str] = set()

    @classmethod
    def _lock_for_camera(cls, camera_id: str) -> threading.RLock:
        with cls._locks_guard:
            return cls._camera_locks.setdefault(camera_id, threading.RLock())

    def get_operation(self, operation_id: str) -> ModelReleaseOperation | None:
        with self._session_factory() as session:
            return (
                session.query(ModelReleaseOperation)
                .filter_by(operation_id=operation_id)
                .first()
            )

    def get_latest_operation(
        self,
        model_version_id: str,
        *,
        target_stage: str | None = None,
    ) -> ModelReleaseOperation | None:
        """Return the latest durable desired state for one model.

        Read-only dashboard routes use this service boundary instead of
        coupling themselves to the release-operation ORM table.  Ordering by
        both request time and the monotonic primary key makes simultaneous
        requests deterministic on databases with coarse timestamp precision.
        """

        with self._session_factory() as session:
            query = session.query(ModelReleaseOperation).filter_by(
                model_version_id=model_version_id,
            )
            if target_stage is not None:
                query = query.filter_by(target_stage=target_stage)
            return query.order_by(
                ModelReleaseOperation.requested_at.desc(),
                ModelReleaseOperation.id.desc(),
            ).first()

    def get_latest_operations(
        self,
        model_version_ids: list[str],
    ) -> dict[str, ModelReleaseOperation]:
        """Fetch one latest durable operation per model without N+1 queries."""
        model_ids = tuple(dict.fromkeys(model_version_ids))
        if not model_ids:
            return {}
        with self._session_factory() as session:
            ranked = session.query(
                ModelReleaseOperation.id.label("operation_row_id"),
                func.row_number().over(
                    partition_by=ModelReleaseOperation.model_version_id,
                    order_by=(
                        ModelReleaseOperation.requested_at.desc(),
                        ModelReleaseOperation.id.desc(),
                    ),
                ).label("operation_rank"),
            ).filter(
                ModelReleaseOperation.model_version_id.in_(model_ids),
            ).subquery()
            operations = session.query(ModelReleaseOperation).join(
                ranked,
                ranked.c.operation_row_id == ModelReleaseOperation.id,
            ).filter(ranked.c.operation_rank == 1).all()
            return {operation.model_version_id: operation for operation in operations}

    def execution_lock(self, scope: str) -> ReleaseExecutionLock:
        return ReleaseExecutionLock(self._session_factory, scope)

    def list_pending_operations(self) -> list[ModelReleaseOperation]:
        with self._session_factory() as session:
            return session.query(ModelReleaseOperation).filter_by(
                status=ReleaseOperationStatus.PENDING.value,
            ).order_by(ModelReleaseOperation.id.asc()).all()

    def rollback_readiness(self, model_version_id: str) -> dict:
        """Recheck historical evidence without creating an operation or approval."""
        from argus.storage.rollback_policy import rollback_context

        with self._session_factory() as session:
            model, context = rollback_context(session, model_version_id)
            history = session.query(ModelReleaseOperation).filter_by(
                operation_id=context["history_operation_id"],
            ).one()
            evidence = self._build_rollback_evidence(
                model, context, history.requested_by, history.requested_at,
            )
            revision = int(model.release_revision or 0)
        self._pipeline.prepare_rollback(
            model_version_id=model_version_id, expected_revision=revision,
            rollback_context=context, release_evidence=evidence,
        )
        return {"expected_revision": revision, **context}

    def submit_rollback(
        self,
        *,
        model_version_id: str,
        source_model_version_id: str,
        source_release_revision: int,
        expected_revision: int,
        requested_by: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> tuple[ModelReleaseOperation, ModelRecord]:
        """Persist a source-revision-bound rollback for the ordinary release worker."""
        from argus.storage.rollback_policy import rollback_context

        requested_by = requested_by.strip()
        reason = reason.strip()
        if not requested_by or not reason:
            raise ValueError("Rollback requester and reason are required")
        key = idempotency_key.strip() if idempotency_key else None
        if key is not None and (not key or len(key) > 128):
            raise ValueError("idempotency_key must contain 1 to 128 characters")
        existing = self._get_operation_by_idempotency(key) if key else None
        if existing is not None:
            desired = json.loads(existing.desired_state_json or "{}")
            previous = desired.get("rollback", {})
            if (
                existing.model_version_id != model_version_id
                or existing.target_stage != ModelStage.PRODUCTION.value
                or existing.requested_by != requested_by
                or existing.reason != reason
                or int(existing.expected_revision) != expected_revision
                or previous.get("source_model_version_id") != source_model_version_id
                or previous.get("source_release_revision") != source_release_revision
            ):
                raise ValueError("idempotency key was already used for another rollback command")
            return self._return_idempotent(
                existing, model_version_id=model_version_id,
                target_stage=ModelStage.PRODUCTION.value, requested_by=requested_by,
                desired_state_json=existing.desired_state_json,
            )
        with self._session_factory() as session:
            model, context = rollback_context(session, model_version_id)
            if (
                int(model.release_revision or 0) != expected_revision
                or context["source_model_version_id"] != source_model_version_id
                or context["source_release_revision"] != source_release_revision
            ):
                raise ValueError("Rollback identity changed; reload readiness before retrying")
        desired = self._canonical_desired_state(
            reason=reason, canary_camera_id=None, release_evidence=None,
            release_evidence_sources=context["release_evidence_sources"],
            rollback=context,
        )
        operation, created = self._create_operation(
            model=model, target_stage=ModelStage.PRODUCTION.value,
            requested_by=requested_by, reason=reason, expected_revision=expected_revision,
            idempotency_key=key, desired_state_json=desired,
        )
        self._validate_idempotent_command(
            operation, model_version_id=model_version_id,
            target_stage=ModelStage.PRODUCTION.value, requested_by=requested_by,
            desired_state_json=desired,
        )
        if created:
            self._publish(operation, model)
        return operation, model

    @staticmethod
    def _build_rollback_evidence(model, context: dict, operator_id: str, approved_time):
        from argus.preprocessing.scene_profile import runtime_artifact_sha256
        from argus.storage.release_evidence import build_release_evidence_from_sources

        if approved_time.tzinfo is None:
            approved_time = approved_time.replace(tzinfo=timezone.utc)
        return build_release_evidence_from_sources(
            context["release_evidence_sources"], model_version_id=model.model_version_id,
            camera_id=model.camera_id, registry_model_hash=model.model_hash,
            registry_data_hash=model.data_hash,
            runtime_artifact_digest=runtime_artifact_sha256(model.model_path),
            operator_id=operator_id, operator_approved=True,
            approved_at=approved_time.isoformat(),
        )

    def submit_transition(
        self,
        *,
        model_version_id: str,
        target_stage: str,
        requested_by: str,
        reason: str | None = None,
        canary_camera_id: str | None = None,
        release_evidence_sources: dict | None = None,
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[ModelReleaseOperation, ModelRecord]:
        """Persist an authorized command; do no artifact I/O or runtime work.

        Evidence source paths and the approval timestamp are frozen with the
        command. They are rebuilt and verified by the worker; pending is never
        a claim that the requested transition has passed the release gates.
        """
        requested_by = requested_by.strip()
        if not requested_by:
            raise ValueError("requested_by is required")
        key = (idempotency_key.strip() or None) if idempotency_key else None
        if key is not None and len(key) > 128:
            raise ValueError("idempotency_key must contain at most 128 characters")
        model = self._get_model(model_version_id)
        if model is None:
            raise ValueError(f"Model not found: {model_version_id}")
        if target_stage not in {stage.value for stage in ModelStage}:
            raise ValueError(f"Invalid target stage: {target_stage}")
        if canary_camera_id is not None and canary_camera_id != model.camera_id:
            raise ValueError("canary_camera_id must match model camera_id")
        if target_stage == ModelStage.CANARY.value and not canary_camera_id:
            raise ValueError("canary_camera_id required for canary stage")
        if (
            target_stage == ModelStage.RETIRED.value
            and model.stage == ModelStage.PRODUCTION.value
            and model.is_active
        ):
            raise ValueError("Active production models cannot be retired directly")
        desired = self._canonical_desired_state(
            reason=reason,
            canary_camera_id=canary_camera_id,
            release_evidence=None,
            release_evidence_sources=release_evidence_sources,
        )
        existing = self._get_operation_by_idempotency(key) if key else None
        if existing is not None:
            if (
                expected_revision is not None
                and int(existing.expected_revision) != int(expected_revision)
            ):
                raise ValueError("idempotency key was already used with another release revision")
            return self._return_idempotent(
                existing,
                model_version_id=model_version_id,
                target_stage=target_stage,
                requested_by=requested_by,
                desired_state_json=desired,
            )
        operation, created = self._create_operation(
            model=model,
            target_stage=target_stage,
            requested_by=requested_by,
            reason=reason,
            expected_revision=(
                int(expected_revision)
                if expected_revision is not None
                else int(model.release_revision or 0)
            ),
            idempotency_key=key,
            desired_state_json=desired,
        )
        self._validate_idempotent_command(
            operation,
            model_version_id=model_version_id,
            target_stage=target_stage,
            requested_by=requested_by,
            desired_state_json=desired,
        )
        if created:
            self._publish(operation, model)
        return operation, model

    def execute_operation(
        self,
        operation_id: str,
    ) -> tuple[ModelReleaseOperation, ModelRecord]:
        """Claim and execute a durable command once, outside request handling."""
        operation = self.get_operation(operation_id)
        if operation is None:
            raise ValueError(f"Release operation not found: {operation_id}")
        with self._lock_for_camera(operation.camera_id), self.execution_lock(
            f"camera:{operation.camera_id}"
        ):
            operation = self.get_operation(operation_id)
            assert operation is not None
            if operation.status != ReleaseOperationStatus.PENDING.value:
                return self._failed_result(operation_id, operation.model_version_id)
            if operation_id in self._interrupted_operations:
                self._finish_interrupted_operation(operation_id)
                return self._failed_result(operation_id, operation.model_version_id)
            claim = json.dumps({"execution_owner": uuid.uuid4().hex}, sort_keys=True)
            with self._session_factory() as session:
                claimed = session.query(ModelReleaseOperation).filter(
                    ModelReleaseOperation.operation_id == operation_id,
                    ModelReleaseOperation.status == ReleaseOperationStatus.PENDING.value,
                    ModelReleaseOperation.runtime_state_json.is_(None),
                ).update({"runtime_state_json": claim}, synchronize_session=False)
                session.commit()
            if claimed != 1:
                # The camera OS lock proves no executor still owns this
                # claim. It survived a failed DB write or interrupted thread;
                # never re-run its approval or leave it pending forever.
                latest = self.get_operation(operation_id)
                if latest is not None and latest.status == ReleaseOperationStatus.PENDING.value:
                    self._finish_interrupted_operation(operation_id)
                return self._failed_result(operation_id, operation.model_version_id)
            try:
                desired = json.loads(operation.desired_state_json or "{}")
                if not isinstance(desired, dict):
                    raise ValueError("release desired command is not an object")
                if desired.get("rollback") is not None and (
                    not isinstance(desired["rollback"], dict)
                    or desired.get("release_evidence_sources")
                    != desired["rollback"].get("release_evidence_sources")
                ):
                    raise ValueError("Rollback command evidence identity is inconsistent")
                evidence = desired.get("release_evidence")
                if operation.target_stage == ModelStage.PRODUCTION.value:
                    from argus.preprocessing.scene_profile import runtime_artifact_sha256
                    from argus.storage.release_evidence import build_release_evidence_from_sources

                    model = self._get_model(operation.model_version_id)
                    if model is None or not model.model_path:
                        raise ValueError("Registry model_path is missing")
                    # This timestamp comes from the durable request, never
                    # from a retry or a worker's current wall clock.
                    approved_time = operation.requested_at
                    if approved_time.tzinfo is None:
                        approved_time = approved_time.replace(tzinfo=timezone.utc)
                    approved_at = approved_time.isoformat()
                    evidence = build_release_evidence_from_sources(
                        desired.get("release_evidence_sources"),
                        model_version_id=model.model_version_id,
                        camera_id=model.camera_id,
                        registry_model_hash=model.model_hash,
                        registry_data_hash=model.data_hash,
                        runtime_artifact_digest=runtime_artifact_sha256(model.model_path),
                        operator_id=operation.requested_by,
                        operator_approved=True,
                        approved_at=approved_at,
                    )
                return self.request_transition(
                    model_version_id=operation.model_version_id,
                    target_stage=operation.target_stage,
                    requested_by=operation.requested_by,
                    reason=operation.reason,
                    canary_camera_id=desired.get("canary_camera_id"),
                    release_evidence=evidence,
                    expected_revision=operation.expected_revision,
                    idempotency_key=operation.idempotency_key,
                    _operation_id=operation_id,
                    _desired_state_json=operation.desired_state_json,
                    _execution_locked=True,
                    _rollback_context=desired.get("rollback"),
                )
            except Exception as exc:
                latest = self.get_operation(operation_id)
                if latest is not None and latest.status == ReleaseOperationStatus.PENDING.value:
                    self._finish_failed_operation(
                        operation_id,
                        error=f"release execution failed: {type(exc).__name__}: {exc}",
                    )
                return self._failed_result(operation_id, operation.model_version_id)

    def reconcile_interrupted_operations(self) -> int:
        """Fail interrupted requests without replaying an operator approval.

        The worker's database lease must be held by the caller. Per-camera
        locks additionally protect synchronous internal release commands.
        """
        from argus.storage.release_execution import ReleaseExecutionBusy

        pending = self.list_pending_operations()
        self._interrupted_operations.update(op.operation_id for op in pending)
        recovered = 0
        for operation in pending:
            camera_lock = self._lock_for_camera(operation.camera_id)
            if not camera_lock.acquire(blocking=False):
                continue
            try:
                with self.execution_lock(f"camera:{operation.camera_id}"):
                    self._finish_interrupted_operation(operation.operation_id)
                    recovered += 1
            except ReleaseExecutionBusy:
                continue
            finally:
                camera_lock.release()
        return recovered

    def _finish_interrupted_operation(self, operation_id: str) -> None:
        self._finish_failed_operation(
            operation_id,
            error=(
                "Release interrupted by runtime restart; applied state was "
                "preserved. Submit a new authorized request with a new "
                "idempotency key; production approvals are never replayed."
            ),
            runtime_state={"recovery": "new_authorization_required"},
        )
        self._interrupted_operations.discard(operation_id)

    def request_transition(
        self,
        *,
        model_version_id: str,
        target_stage: str,
        requested_by: str,
        reason: str | None = None,
        canary_camera_id: str | None = None,
        release_evidence: object | None = None,
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
        _operation_id: str | None = None,
        _desired_state_json: str | None = None,
        _execution_locked: bool = False,
        _rollback_context: dict | None = None,
    ) -> tuple[ModelReleaseOperation, ModelRecord]:
        """Execute a release command without an intermediate applied DB state."""

        requested_by = requested_by.strip()
        if not requested_by:
            raise ValueError("requested_by is required")
        normalized_key = (idempotency_key.strip() or None) if idempotency_key else None
        desired_state_json = _desired_state_json or self._canonical_desired_state(
            reason=reason,
            canary_camera_id=canary_camera_id,
            release_evidence=release_evidence,
        )

        initial = self._get_model(model_version_id)
        if initial is None:
            raise ValueError(f"Model not found: {model_version_id}")

        with self._lock_for_camera(initial.camera_id):
            if not _execution_locked:
                with self.execution_lock(f"camera:{initial.camera_id}"):
                    return self.request_transition(
                        model_version_id=model_version_id,
                        target_stage=target_stage,
                        requested_by=requested_by,
                        reason=reason,
                        canary_camera_id=canary_camera_id,
                        release_evidence=release_evidence,
                        expected_revision=expected_revision,
                        idempotency_key=normalized_key,
                        _operation_id=_operation_id,
                        _desired_state_json=_desired_state_json,
                        _execution_locked=True,
                        _rollback_context=_rollback_context,
                    )
            existing = (
                self.get_operation(_operation_id)
                if _operation_id
                else (
                    self._get_operation_by_idempotency(normalized_key)
                    if normalized_key
                    else None
                )
            )
            if existing is not None:
                if (
                    expected_revision is not None
                    and int(existing.expected_revision) != int(expected_revision)
                ):
                    raise ValueError(
                        "idempotency key was already used with another release revision"
                    )
                self._validate_idempotent_command(
                    existing,
                    model_version_id=model_version_id,
                    target_stage=target_stage,
                    requested_by=requested_by,
                    desired_state_json=desired_state_json,
                )
                if existing.status != ReleaseOperationStatus.PENDING.value:
                    return self._return_idempotent(
                        existing,
                        model_version_id=model_version_id,
                        target_stage=target_stage,
                        requested_by=requested_by,
                        desired_state_json=desired_state_json,
                    )
                if existing.runtime_state_json and _operation_id is None:
                    # An explicit internal retry must not take over a command
                    # already owned by the background worker.
                    return self._return_idempotent(
                        existing,
                        model_version_id=model_version_id,
                        target_stage=target_stage,
                        requested_by=requested_by,
                        desired_state_json=desired_state_json,
                    )
                operation = existing
                requested_revision = int(existing.expected_revision)
            else:
                current = self._get_model(model_version_id)
                if current is None:
                    raise ValueError(f"Model not found: {model_version_id}")
                requested_revision = (
                    int(expected_revision)
                    if expected_revision is not None
                    else int(current.release_revision or 0)
                )
                operation, created = self._create_operation(
                    model=current,
                    target_stage=target_stage,
                    requested_by=requested_by,
                    reason=reason,
                    expected_revision=requested_revision,
                    idempotency_key=normalized_key,
                    desired_state_json=desired_state_json,
                )
                if not created:
                    # A different process won the unique idempotency insert.
                    # Re-enter so the winner's terminal result is returned or
                    # its still-pending desired command is safely resumed.
                    return self.request_transition(
                        model_version_id=model_version_id,
                        target_stage=target_stage,
                        requested_by=requested_by,
                        reason=reason,
                        canary_camera_id=canary_camera_id,
                        release_evidence=release_evidence,
                        expected_revision=expected_revision,
                        idempotency_key=normalized_key,
                        _execution_locked=True,
                        _rollback_context=_rollback_context,
                    )

            # From this point, ordinary failures mark the desired operation
            # failed.  BaseException is intentionally not caught: a process
            # crash leaves a durable pending operation while applied state is
            # unchanged, which a startup reconciler can safely inspect.
            try:
                if _rollback_context is not None:
                    if (
                        _operation_id is None
                        or target_stage != ModelStage.PRODUCTION.value
                        or json.loads(operation.desired_state_json).get("rollback")
                        != _rollback_context
                    ):
                        raise ValueError("Rollback requires its exact durable worker command")
                    plan = self._pipeline.prepare_rollback(
                        model_version_id=model_version_id, expected_revision=requested_revision,
                        rollback_context=_rollback_context, release_evidence=release_evidence,
                    )
                else:
                    plan = self._pipeline.prepare_transition(
                        model_version_id=model_version_id,
                        target_stage=target_stage,
                        canary_camera_id=canary_camera_id,
                        release_evidence=release_evidence,
                        expected_revision=requested_revision,
                    )
            except Exception as exc:
                self._finish_failed_operation(operation.operation_id, error=str(exc))
                raise

            try:
                prepared = self._prepare_runtime(plan)
            except Exception as exc:
                self._finish_failed_operation(
                    operation.operation_id,
                    error=f"runtime prepare failed: {type(exc).__name__}: {exc}",
                )
                return self._failed_result(operation.operation_id, model_version_id)

            preparation_error = self._validate_runtime_preparation(plan, prepared)
            if preparation_error is not None:
                abort_state = self._abort_runtime(getattr(prepared, "transaction", None))
                self._finish_failed_operation(
                    operation.operation_id,
                    error=preparation_error,
                    runtime_state={
                        "prepared": getattr(prepared, "state", None),
                        "abort": abort_state,
                    },
                )
                return self._failed_result(operation.operation_id, model_version_id)

            transaction = prepared.transaction
            assert transaction is not None  # checked above
            try:
                runtime_commit = transaction.commit()
            except BaseException as exc:
                # SystemExit/KeyboardInterrupt still run Python finally blocks.
                # Restore the old runtime, but leave the durable operation
                # pending so startup recovery can distinguish an interrupted
                # command from an ordinary runtime rejection.
                if not isinstance(exc, Exception):
                    self._abort_runtime(transaction)
                    raise
                runtime_commit = RuntimeReleaseCommit(
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )

            if not isinstance(runtime_commit, RuntimeReleaseCommit):
                runtime_commit = RuntimeReleaseCommit(
                    False, error="runtime commit returned an invalid contract",
                )

            if runtime_commit.success is not True:
                abort_state = self._abort_runtime(transaction)
                self._finish_failed_operation(
                    operation.operation_id,
                    error=(
                        runtime_commit.error
                        or "runtime commit did not confirm the prepared identity"
                    ),
                    runtime_state={
                        "prepared": prepared.state,
                        "commit": runtime_commit.state,
                        "abort": abort_state,
                    },
                )
                return self._failed_result(operation.operation_id, model_version_id)

            runtime_state = {
                "prepared": prepared.state,
                "commit": runtime_commit.state,
                "identity": {
                    "camera_id": prepared.identity.camera_id,
                    "model_version_id": prepared.identity.model_version_id,
                    "target_stage": prepared.identity.target_stage,
                    "runtime_artifact_sha256": (
                        prepared.identity.runtime_artifact_sha256
                    ),
                },
            }
            if isinstance(runtime_commit.state, dict):
                for key, value in runtime_commit.state.items():
                    runtime_state.setdefault(key, value)
            commit_interrupt: BaseException | None = None
            try:
                record = self._pipeline.commit_transition(
                    plan,
                    triggered_by=requested_by,
                    reason=reason,
                    operation_id=operation.operation_id,
                    runtime_state=runtime_state,
                )
            except BaseException as exc:
                # COMMIT may have succeeded before a connection/refresh error
                # lost its result. Confirm the durable operation before
                # compensating; otherwise we would restore the old runtime
                # while the registry already authorizes the new one.
                try:
                    confirmed = self.get_operation(operation.operation_id)
                    observed = self._get_model(model_version_id)
                except Exception:
                    confirmed = None
                    observed = None
                    logger.exception(
                        "model_release.commit_confirmation_unavailable",
                        operation_id=operation.operation_id,
                    )
                if (
                    confirmed is not None
                    and confirmed.status == ReleaseOperationStatus.APPLIED.value
                    and confirmed.applied_revision == plan.expected_revision + 1
                    and observed is not None
                    and observed.stage == plan.target_stage
                    and observed.release_revision == confirmed.applied_revision
                ):
                    record = observed
                    if not isinstance(exc, Exception):
                        commit_interrupt = exc
                    logger.warning(
                        "model_release.commit_result_lost_but_confirmed",
                        operation_id=operation.operation_id,
                        error=str(exc),
                    )
                else:
                    abort_state = self._abort_runtime(transaction)
                    if not isinstance(exc, Exception):
                        raise
                    self._finish_failed_operation(
                        operation.operation_id,
                        error=f"applied-state commit failed: {type(exc).__name__}: {exc}",
                        runtime_state={
                            **runtime_state,
                            "abort_after_db_failure": abort_state,
                        },
                    )
                    raise

            try:
                finalize_result = transaction.finalize()
            except Exception as exc:  # applied state is already committed
                finalize_result = RuntimeReleaseCommit(
                    False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if not isinstance(finalize_result, RuntimeReleaseCommit):
                finalize_result = RuntimeReleaseCommit(
                    False, error="runtime finalize returned an invalid contract",
                )
            if finalize_result.success is not True:
                logger.critical(
                    "model_release.runtime_finalize_failed",
                    operation_id=operation.operation_id,
                    camera_id=plan.camera_id,
                    error=finalize_result.error,
                )

            applied = self.get_operation(operation.operation_id)
            if applied is None:  # pragma: no cover - corrupted database
                raise RuntimeError(
                    f"release operation disappeared: {operation.operation_id}"
                )
            self._publish(applied, record)
            if commit_interrupt is not None:
                raise commit_interrupt
            return applied, record

    def _prepare_runtime(
        self,
        plan: ReleaseTransitionPlan,
    ) -> RuntimeReleasePreparation:
        """Prepare runtime, except for a provably inactive candidate cleanup."""

        if (
            plan.from_stage == ModelStage.CANDIDATE.value
            and plan.target_stage == ModelStage.RETIRED.value
        ):
            current = self._get_model(plan.model_version_id)
            if current is None:
                return RuntimeReleasePreparation(
                    success=False,
                    error="candidate disappeared before retirement",
                )
            if (
                current.stage != ModelStage.CANDIDATE.value
                or current.is_active is not False
                or int(current.release_revision or 0) != plan.expected_revision
            ):
                return RuntimeReleasePreparation(
                    success=False,
                    error=(
                        "candidate is no longer provably inactive at the "
                        "validated release revision"
                    ),
                )
            return RuntimeReleasePreparation(
                success=True,
                identity=RuntimeReleaseIdentity(
                    camera_id=plan.camera_id,
                    model_version_id=plan.model_version_id,
                    target_stage=plan.target_stage,
                    runtime_artifact_sha256=plan.runtime_artifact_sha256,
                ),
                transaction=_NoRuntimeReleaseTransaction(),
                state={
                    "runtime_impact": "none",
                    "reason": "inactive_candidate_retirement",
                },
            )
        return self._runtime_prepare(plan)

    @staticmethod
    def _validate_runtime_preparation(
        plan: ReleaseTransitionPlan,
        prepared: RuntimeReleasePreparation,
    ) -> str | None:
        if not isinstance(prepared, RuntimeReleasePreparation):
            return "runtime prepare returned an invalid contract"
        if prepared.success is not True:
            return prepared.error or "runtime prepare did not report literal success"
        if prepared.transaction is None:
            return "runtime prepare returned no standby transaction"
        if not all(
            callable(getattr(prepared.transaction, method, None))
            for method in ("commit", "abort", "finalize")
        ):
            return "runtime prepare returned an incomplete transaction contract"
        identity = prepared.identity
        if identity is None:
            return "runtime prepare returned no verified identity"
        expected = RuntimeReleaseIdentity(
            camera_id=plan.camera_id,
            model_version_id=plan.model_version_id,
            target_stage=plan.target_stage,
            runtime_artifact_sha256=plan.runtime_artifact_sha256,
        )
        if identity != expected:
            return (
                "runtime prepared identity mismatch: "
                f"expected={expected!r}, observed={identity!r}"
            )
        return None

    def _return_idempotent(
        self,
        operation: ModelReleaseOperation,
        *,
        model_version_id: str,
        target_stage: str,
        requested_by: str,
        desired_state_json: str,
    ) -> tuple[ModelReleaseOperation, ModelRecord]:
        self._validate_idempotent_command(
            operation,
            model_version_id=model_version_id,
            target_stage=target_stage,
            requested_by=requested_by,
            desired_state_json=desired_state_json,
        )
        model = self._get_model(operation.model_version_id)
        if model is None:  # pragma: no cover - corrupted database
            raise RuntimeError("release operation model no longer exists")
        return operation, model

    @staticmethod
    def _validate_idempotent_command(
        operation: ModelReleaseOperation,
        *,
        model_version_id: str,
        target_stage: str,
        requested_by: str,
        desired_state_json: str,
    ) -> None:
        if (
            operation.model_version_id != model_version_id
            or operation.target_stage != target_stage
            or operation.requested_by != requested_by
            or (
                operation.desired_state_json is not None
                and operation.desired_state_json != desired_state_json
            )
        ):
            raise ValueError(
                "idempotency key was already used for another release command"
            )

    @staticmethod
    def _canonical_desired_state(
        *,
        reason: str | None,
        canary_camera_id: str | None,
        release_evidence: object | None,
        release_evidence_sources: dict | None = None,
        rollback: dict | None = None,
    ) -> str:
        desired = {
            "canary_camera_id": canary_camera_id,
            "reason": reason,
            "release_evidence": release_evidence,
        }
        if release_evidence_sources is not None:
            desired["release_evidence_sources"] = release_evidence_sources
        if rollback is not None:
            desired["rollback"] = rollback
        return json.dumps(
            desired,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _get_model(self, model_version_id: str) -> ModelRecord | None:
        with self._session_factory() as session:
            return (
                session.query(ModelRecord)
                .filter_by(model_version_id=model_version_id)
                .first()
            )

    def _get_operation_by_idempotency(
        self,
        idempotency_key: str,
    ) -> ModelReleaseOperation | None:
        with self._session_factory() as session:
            return (
                session.query(ModelReleaseOperation)
                .filter_by(idempotency_key=idempotency_key)
                .first()
            )

    def _create_operation(
        self,
        *,
        model: ModelRecord,
        target_stage: str,
        requested_by: str,
        reason: str | None,
        expected_revision: int,
        idempotency_key: str | None,
        desired_state_json: str,
    ) -> tuple[ModelReleaseOperation, bool]:
        operation = ModelReleaseOperation(
            operation_id=f"release-{uuid.uuid4().hex}",
            idempotency_key=idempotency_key,
            model_version_id=model.model_version_id,
            camera_id=model.camera_id,
            from_stage=model.stage,
            target_stage=target_stage,
            expected_revision=expected_revision,
            status=ReleaseOperationStatus.PENDING.value,
            requested_by=requested_by,
            reason=reason,
            desired_state_json=desired_state_json,
        )
        with self._session_factory() as session:
            try:
                # Reserve the exact observed model row in the same SQLite
                # write transaction as the pending insert. A concurrent delete
                # cannot pass its pending-operation guard between this check
                # and the durable command becoming visible.
                reserved = session.query(ModelRecord).filter(
                    ModelRecord.model_version_id == model.model_version_id,
                    ModelRecord.release_revision == int(model.release_revision or 0),
                    ModelRecord.stage == model.stage,
                    ModelRecord.is_active == bool(model.is_active),
                    ModelRecord.model_path == model.model_path,
                ).update(
                    {"release_revision": int(model.release_revision or 0)},
                    synchronize_session=False,
                )
                if reserved != 1:
                    session.rollback()
                    if idempotency_key:
                        existing = session.query(ModelReleaseOperation).filter_by(
                            idempotency_key=idempotency_key,
                        ).first()
                        if existing is not None:
                            return existing, False
                    raise ValueError("Model changed while submitting release; reload and retry")
                session.add(operation)
                session.commit()
                session.refresh(operation)
                return operation, True
            except IntegrityError:
                session.rollback()
                if idempotency_key:
                    existing = (
                        session.query(ModelReleaseOperation)
                        .filter_by(idempotency_key=idempotency_key)
                        .first()
                    )
                    if existing is not None:
                        return existing, False
                raise

    def _finish_failed_operation(
        self,
        operation_id: str,
        *,
        error: str,
        runtime_state: object = None,
    ) -> ModelReleaseOperation:
        with self._session_factory() as session:
            operation = (
                session.query(ModelReleaseOperation)
                .filter_by(operation_id=operation_id)
                .first()
            )
            if operation is None:  # pragma: no cover - corrupted database
                raise RuntimeError(f"release operation not found: {operation_id}")
            # Terminal results are immutable, including the original failure
            # and compensation details returned to an idempotent caller.
            if operation.status != ReleaseOperationStatus.PENDING.value:
                return operation
            operation.status = ReleaseOperationStatus.FAILED.value
            operation.applied_revision = None
            operation.runtime_state_json = json.dumps(
                runtime_state,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            operation.error = error
            operation.completed_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(operation)
        model = self._get_model(operation.model_version_id)
        self._publish(operation, model)
        return operation

    def _failed_result(
        self,
        operation_id: str,
        model_version_id: str,
    ) -> tuple[ModelReleaseOperation, ModelRecord]:
        operation = self.get_operation(operation_id)
        model = self._get_model(model_version_id)
        if operation is None or model is None:  # pragma: no cover - corrupted database
            raise RuntimeError("failed release command state disappeared")
        return operation, model

    @staticmethod
    def _abort_runtime(
        transaction: RuntimeReleaseTransaction | None,
    ) -> object:
        if transaction is None:
            return {"success": True, "state": "nothing_to_abort"}
        try:
            result = transaction.abort()
            return {
                "success": result.success,
                "state": result.state,
                "error": result.error,
            }
        except Exception as exc:  # pragma: no cover - defensive telemetry
            logger.critical(
                "model_release.runtime_abort_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _publish(
        self,
        operation: ModelReleaseOperation,
        record: ModelRecord | None,
    ) -> None:
        if self._event_publisher is None:
            return
        try:
            self._event_publisher("model_release", {
                "type": f"release_operation_{operation.status}",
                "operation": operation.to_dict(),
                "model": record.to_dict() if record is not None else None,
            })
        except Exception:
            logger.warning(
                "model_release.publish_failed",
                operation_id=operation.operation_id,
                exc_info=True,
            )


__all__ = [
    "ModelReleaseService",
    "RuntimeReleaseCommit",
    "RuntimeReleaseIdentity",
    "RuntimeReleasePreparation",
    "RuntimeReleaseTransaction",
]
