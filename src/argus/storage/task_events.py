"""Task-scoped validation events, auditable review and controlled keyframes.

Source callbacks only enqueue references. A dedicated bounded worker performs
encoding and persistence; it never opens a camera or a source video. Database
events and outbox rows commit together. Existing legacy release/dispatch paths
are neither called nor granted additional authority by this service.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
from uuid import uuid4

import cv2
from sqlalchemy import func, select, text

from argus.core.task_events import (
    TaskEventEngine, same_json_identity, validate_event_source_identity, validate_observation,
)
from argus.storage.models import (
    AlertRecord, AnalysisTaskRecord, AnalysisTaskRevisionRecord, OutboxEvent,
    TaskDeploymentOperationRecord, TaskDeploymentRecord, TaskEventRecord,
    TaskExecutionBundleRecord, TaskModelRecord,
)


class TaskEventConflict(ValueError):
    pass


class TaskEventEvidenceUnavailable(ValueError):
    pass


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _iso(value):
    return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value).isoformat() if value else None


_COLLECTION_BINDING_KEYS = ("task_id", "camera_id", "template_id", "task_revision", "task_config_sha256")
_COLLECTION_CONTEXT_KEYS = (
    "task_id", "bundle_id", "manifest_sha256", "execution_id", "deployment_revision",
    "deployment_operation_id", "source_session_id", "source_epoch",
)
_COLLECTION_PAYLOAD_KEYS = (
    "schema_version", "event_id", "event_type", "mode", "binding", "source", "first_source",
    "request_id", "occurred_at", "object", "duration_seconds", "frame_size",
    "production_allowed", "alerts_emitted", "replay", "release",
)


class TaskEventService:
    def __init__(self, db, task_service, *, evidence_root: str | Path,
                 release_service=None, current_runtime=None, encoder=None, broadcaster=None,
                 max_pending: int = 8, max_retries: int = 3,
                 max_evidence_files: int = 20000, max_evidence_bytes: int = 512 * 1024 * 1024):
        if type(max_pending) is not int or not 1 <= max_pending <= 64:
            raise ValueError("max_pending must be between 1 and 64")
        if type(max_retries) is not int or not 1 <= max_retries <= 10:
            raise ValueError("max_retries must be between 1 and 10")
        if (type(max_evidence_files) is not int or max_evidence_files < 1
                or type(max_evidence_bytes) is not int or max_evidence_bytes < 1):
            raise ValueError("event evidence quotas must be positive integers")
        self._db, self._tasks = db, task_service
        self._root = Path(evidence_root).absolute()
        self._release, self._current = release_service, current_runtime
        self._broadcaster = broadcaster
        self._encoder = encoder or self._encode_jpeg
        self._queue = queue.Queue(maxsize=max_pending)
        self._engine = TaskEventEngine()
        self._max_retries = max_retries
        self._max_evidence_files, self._max_evidence_bytes = max_evidence_files, max_evidence_bytes
        self._lock, self._process_lock = threading.RLock(), threading.Lock()
        self._stop, self._thread = threading.Event(), None
        self._closed = False
        self._pending_write = None
        self._accepted = self._dropped = self._persisted = self._failed = 0
        self._last_error = None
        self._continuity_lock = threading.Lock()
        self._continuity: dict[str, object] = {}
        self._processed_continuity: dict[str, object] = {}

    def start(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            state = self._thread_state(self._thread)
            if state == "running":
                return True
            if state == "unknown":
                self._last_error = "EVENT_WORKER_STATE_UNKNOWN"
                return False
            self._stop.clear()
            try:
                self._thread = threading.Thread(target=self._run, name="task-event-evidence", daemon=True)
                self._thread.start()
            except Exception:
                self._last_error = "EVENT_WORKER_START_FAILED"
                # A failed Thread.start can be injected after it created the
                # thread. Never lose a live or uninspectable owner and start a
                # second evidence consumer. Only a proven unstarted handle is
                # disposable; keeping other handles also makes close honest.
                if self._thread_state(self._thread) == "not_started":
                    self._thread = None
                return False
            state = self._thread_state(self._thread)
            if state == "running":
                self._last_error = None
                return True
            self._last_error = "EVENT_WORKER_STATE_UNKNOWN" if state == "unknown" else "EVENT_WORKER_START_FAILED"
            if state == "not_started":
                self._thread = None
            return False

    @staticmethod
    def _thread_state(thread):
        if thread is None:
            return "absent"
        try:
            alive, ident, started = thread.is_alive(), thread.ident, thread._started.is_set()
        except Exception:
            return "unknown"
        if type(alive) is not bool or type(started) is not bool:
            return "unknown"
        if ident is None:
            return "not_started" if alive is False and started is False else "unknown"
        if type(ident) is not int or started is not True:
            return "unknown"
        return "running" if alive else "stopped"

    def close(self, timeout: float = 5.0) -> bool:
        with self._lock:
            # Close admission before the consumer can observe an empty queue
            # and exit. An offer already waiting here must recheck this fence.
            with self._continuity_lock:
                self._closed = True
                self._stop.set()
            thread = self._thread
        state = self._thread_state(thread)
        if state == "unknown":
            self._last_error = "EVENT_WORKER_STATE_UNKNOWN"
            return False
        if state == "not_started":
            with self._lock:
                if self._thread is thread:
                    self._thread = None
        elif state in {"running", "stopped"}:
            try:
                thread.join(timeout=max(0.0, timeout))
            except Exception:
                self._last_error = "EVENT_WORKER_CLEANUP_FAILED"
                return False
            if self._thread_state(thread) != "stopped":
                return False
        return self._pending_write is None and self._queue.empty()

    def snapshot(self) -> dict:
        with self._lock:
            state = self._thread_state(self._thread)
            clean = state in {"absent", "not_started", "stopped"} and self._pending_write is None and self._queue.empty()
            visible = ("stopped" if clean else "cleanup_pending") if self._closed else (
                "running" if state == "running" else "unavailable" if state == "unknown" else "not_started")
            return {"state": visible,
                    "pending": self._queue.qsize() + (len(self._pending_write[0]) if self._pending_write else 0),
                    "accepted": self._accepted,
                    "dropped": self._dropped, "persisted": self._persisted,
                    "failed": self._failed, "last_error": self._last_error,
                    "production_allowed": False}

    def offer(self, manifest, result, frame, *, deployment_revision, operation_id) -> bool:
        """Nonblocking transport only; safe inside the existing source fence."""
        if (self._closed or self._pending_write is not None or self._thread_state(self._thread) != "running"
                or manifest.get("template_id") != "object_detection.v1"
                or type(deployment_revision) is not int or deployment_revision < 1
                or not isinstance(operation_id, str) or not operation_id
                or frame.frame.nbytes > 8 * 1024 * 1024):
            self._dropped += 1
            self._mark_gap(manifest["task_id"])
            return False
        try:
            # Serialize only in-memory admission/invalidations. The evidence
            # worker can also invalidate a task; read-increment-write counters
            # could lose that gap and make an older packet look current.
            with self._continuity_lock:
                if self._closed:
                    self._dropped += 1
                    self._renew_continuity_locked(manifest["task_id"])
                    return False
                continuity = self._continuity.get(manifest["task_id"])
                if continuity is None:
                    continuity = self._renew_continuity_locked(manifest["task_id"])
                try:
                    self._queue.put_nowait((manifest, result, frame, deployment_revision, operation_id, continuity))
                except queue.Full:
                    self._renew_continuity_locked(manifest["task_id"])
                    raise
            self._accepted += 1
            return True
        except queue.Full:
            self._dropped += 1
            self._last_error = "EVENT_QUEUE_FULL"
            return False

    def _renew_continuity_locked(self, task_id):
        # Match the engine's task bound, including repeated failed offers when
        # no worker can sweep inactive tasks. Eviction only invalidates packets;
        # a token is never reused, even if the same task id is later re-created.
        if task_id not in self._continuity and len(self._continuity) >= 128:
            old_task = next(iter(self._continuity))
            self._continuity.pop(old_task, None)
            self._processed_continuity.pop(old_task, None)
        token = object()
        self._continuity[task_id] = token
        return token

    def _mark_gap(self, task_id):
        # This short memory-only lock never acquires a manager/process/DB lock.
        with self._continuity_lock:
            self._renew_continuity_locked(task_id)

    def _prune_continuity(self):
        if not callable(self._current):
            return
        with self._continuity_lock:
            candidates = tuple(self._continuity.items())
        # Never call the runtime while holding the admission lock.
        inactive = [(task_id, token) for task_id, token in candidates if self._current(task_id) is None]
        with self._continuity_lock:
            for task_id, token in inactive:
                if self._continuity.get(task_id) is token:
                    self._continuity.pop(task_id, None)
                    self._processed_continuity.pop(task_id, None)

    def _run(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                packet = self._queue.get(timeout=0.1)
            except queue.Empty:
                try:
                    with self._process_lock:
                        self._prune_continuity()
                        if self._pending_write is not None and not self._stop.is_set():
                            self._flush_pending()
                except Exception:
                    self._stop.wait(0.2)
                self._deliver_outbox()
                continue
            try:
                self.process_observation(packet[0], packet[1], packet[2],
                    deployment_revision=packet[3], operation_id=packet[4], _continuity=packet[5])
            except Exception:
                with self._lock:
                    self._failed += 1
                    self._last_error = "EVENT_PERSISTENCE_FAILED"
                self._mark_gap(packet[0]["task_id"])
            finally:
                self._queue.task_done()
            self._deliver_outbox()

    def _deliver_outbox(self):
        try:
            self.drain_outbox()
        except Exception:
            self._last_error = "EVENT_OUTBOX_UNAVAILABLE"

    def drain_outbox(self, limit: int = 50) -> int:
        """At-least-once task-event broadcast, independent of legacy activity.

        A rejected head remains pending and stops this ordered batch. Another
        dispatcher can publish the same committed event concurrently; clients
        must keep the existing event-id/revision deduplication contract.
        """
        if self._broadcaster is None:
            return 0
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("outbox batch limit must be between 1 and 100")
        with self._db.get_session() as session:
            rows = [row.to_dict() for row in session.scalars(select(OutboxEvent).where(
                OutboxEvent.topic == "task_events", OutboxEvent.status == "pending"
            ).order_by(OutboxEvent.id).limit(limit))]
        delivered = 0
        for row in rows:
            try:
                payload = row["payload"]
                if (not isinstance(payload, dict) or payload.get("event_id") != row["resource_id"]
                        or not isinstance(payload.get("task_id"), str) or not payload["task_id"]
                        or not isinstance(payload.get("camera_id"), str) or not payload["camera_id"]
                        or type(payload.get("revision")) is not int or payload["revision"] < 1):
                    raise ValueError("invalid committed task event outbox envelope")
                accepted = self._broadcaster("task_events", payload,
                    event_type=row["event_type"], resource_id=row["resource_id"],
                    event_id=row["event_id"], occurred_at=row["created_at"])
                if accepted is False:
                    raise RuntimeError("task event broadcaster rejected the event")
                if not self._db.mark_outbox_published(row["event_id"]):
                    raise RuntimeError("task event outbox acknowledgement failed")
                delivered += 1
            except Exception:
                self._db.mark_outbox_failed(row["event_id"], "TASK_EVENT_BROADCAST_FAILED")
                self._last_error = "TASK_EVENT_BROADCAST_FAILED"
                break
        return delivered

    def _runtime_current(self, binding, source) -> bool:
        if not callable(self._current):
            return False
        current = self._current(binding["task_id"])
        expected = {key: binding[key] for key in (
            "task_id", "bundle_id", "manifest_sha256", "execution_id",
            "deployment_revision", "deployment_operation_id")}
        expected.update(source_session_id=source["source_session_id"], source_epoch=source["source_epoch"])
        return isinstance(current, dict) and all(same_json_identity(current.get(key), value) for key, value in expected.items())

    def process_observation(self, manifest, result, frame, *, deployment_revision, operation_id,
                            _continuity=None) -> list[dict]:
        """Synchronous test/worker entry; never call this from a source lock."""
        with self._process_lock:
            previous = self._flush_pending()
            if callable(self._current):
                self._engine.forget_inactive(lambda task_id: self._current(task_id) is not None)
            self._prune_continuity()
            observation = validate_observation(manifest, result, frame,
                deployment_revision=deployment_revision, operation_id=operation_id)
            binding, source = observation["binding"], observation["source"]
            task_id = binding["task_id"]
            with self._continuity_lock:
                continuity = self._continuity.get(task_id)
                if _continuity is None:
                    if continuity is None:
                        continuity = self._renew_continuity_locked(task_id)
                elif _continuity is not continuity:
                    self._dropped += 1
                    return previous
            if not self._runtime_current(binding, source):
                self._dropped += 1
                self._mark_gap(task_id)
                return previous
            try:
                # Runtime can produce a frame before the coordinator records
                # its applied receipt. Rejected admission must not consume
                # the engine's first-event cooldown or duration state.
                with self._db.get_session() as session:
                    self._verify_database_identity(session, {"binding": binding, "source": source})
            except TaskEventConflict:
                self._dropped += 1
                self._mark_gap(task_id)
                return previous
            with self._continuity_lock:
                if self._continuity.get(task_id) is not continuity:
                    self._dropped += 1
                    return previous
                reset = self._processed_continuity.get(task_id) is not continuity
                self._processed_continuity[task_id] = continuity
            if reset:
                self._engine.forget(task_id)
            events = self._engine.observe(manifest, result, frame,
                deployment_revision=deployment_revision, operation_id=operation_id)
            if not events:
                return previous
            try:
                encoded = self._encoder(frame.frame)
                if not isinstance(encoded, bytes) or not encoded or len(encoded) > 8 * 1024 * 1024:
                    raise TaskEventEvidenceUnavailable("invalid keyframe encoding")
                encode_error = None
            except Exception:
                encoded, encode_error = None, "KEYFRAME_ENCODING_FAILED"
            self._pending_write = (events, encoded, encode_error)
            return previous + self._flush_pending()

    def _flush_pending(self):
        if self._pending_write is None:
            return []
        events, encoded, encode_error = self._pending_write
        persisted = []
        for index, event in enumerate(events):
            last_error = None
            for attempt in range(self._max_retries):
                try:
                    persisted.append(self._persist(event, encoded, encode_error))
                    self._persisted += 1
                    self._last_error = None
                    break
                except TaskEventConflict:
                    # A stale identity is never retried as a current event.
                    task_id = event["binding"]["task_id"]
                    self._mark_gap(task_id)
                    self._engine.forget(task_id)
                    self._dropped += len(events) - index
                    self._pending_write = None
                    return persisted
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < self._max_retries:
                        self._stop.wait(0.01 * (attempt + 1))
            else:
                # Keep the exact immutable event id/image for reconciliation;
                # a new source frame must not manufacture a second alert id
                # when commit may have succeeded but its reply was lost.
                self._pending_write = (events[index:], encoded, encode_error)
                raise RuntimeError("event persistence retry exhausted") from last_error
        self._pending_write = None
        return persisted

    @staticmethod
    def _encode_jpeg(pixels) -> bytes:
        # Called solely by this dedicated evidence worker, not a default pool.
        ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise TaskEventEvidenceUnavailable("keyframe encoding failed")
        return encoded.tobytes()

    def _safe_path(self, relative) -> Path:
        candidate = self._root / relative
        if candidate.resolve().parent != self._root.resolve():
            raise TaskEventEvidenceUnavailable("event evidence path is outside the controlled root")
        for path in (self._root, candidate):
            if path.is_symlink() or (path.exists() and getattr(path.lstat(), "st_file_attributes", 0) & 0x400):
                raise TaskEventEvidenceUnavailable("linked event evidence is not permitted")
        return candidate

    def _write_keyframe(self, event_id, content):
        self._root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(content).hexdigest()
        relative = event_id + ".jpg"
        target = self._safe_path(relative)
        if target.exists():
            if not target.is_file() or target.stat().st_size > 8 * 1024 * 1024:
                raise TaskEventEvidenceUnavailable("event evidence file is invalid")
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise TaskEventEvidenceUnavailable("event evidence identity changed")
            return relative, digest
        if len(content) > self._max_evidence_bytes:
            raise TaskEventEvidenceUnavailable("event evidence byte quota exhausted")
        count, total = 0, 0
        for path in self._root.iterdir():
            count += 1
            if count >= self._max_evidence_files:
                raise TaskEventEvidenceUnavailable("event evidence file quota exhausted")
            if path.is_file():
                total += path.stat().st_size
            if total + len(content) > self._max_evidence_bytes:
                raise TaskEventEvidenceUnavailable("event evidence byte quota exhausted")
        temp = self._safe_path("." + event_id + "." + uuid4().hex + ".tmp")
        try:
            with temp.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return relative, digest

    def _verify_database_identity(self, session, event):
        binding = event["binding"]
        task = session.get(AnalysisTaskRecord, binding["task_id"])
        revision = session.scalar(select(AnalysisTaskRevisionRecord).where(
            AnalysisTaskRevisionRecord.task_id == binding["task_id"],
            AnalysisTaskRevisionRecord.revision == binding["task_revision"]))
        bundle = session.get(TaskExecutionBundleRecord, binding["bundle_id"])
        deployment = session.get(TaskDeploymentRecord, binding["task_id"])
        operation = session.get(TaskDeploymentOperationRecord, binding["deployment_operation_id"])
        if (task is None or task.authority != "managed" or task.archived
                or task.template_id != "object_detection.v1" or task.current_revision != binding["task_revision"]
                or task.camera_id != binding["camera_id"] or revision is None
                or revision.config_sha256 != binding["task_config_sha256"]
                or hashlib.sha256(revision.config_json.encode()).hexdigest() != revision.config_sha256
                or bundle is None or bundle.task_id != task.task_id or bundle.manifest_sha256 != binding["manifest_sha256"]
                or deployment is None or deployment.authority != "managed" or deployment.desired_state != "running"
                or deployment.applied_bundle_id != binding["bundle_id"]
                or deployment.applied_deployment_revision != binding["deployment_revision"]
                or deployment.deployment_revision != binding["deployment_revision"]
                or deployment.current_operation_id != binding["deployment_operation_id"]
                or operation is None or operation.task_id != task.task_id or operation.status != "applied"
                or operation.bundle_id != binding["bundle_id"] or operation.action not in {"deploy", "rollback"}
                or operation.deployment_revision != binding["deployment_revision"]
                or not self._runtime_current(binding, event["source"])):
            raise TaskEventConflict("task event no longer belongs to the applied current execution")
        try:
            stored = json.loads(bundle.manifest_json)
            models = {role: {"model_id": model["model_id"], "artifact_sha256": model["artifact_sha256"]}
                      for role, model in stored["models"].items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskEventConflict("stored task execution bundle is malformed") from exc
        if (hashlib.sha256(bundle.manifest_json.encode()).hexdigest() != bundle.manifest_sha256
                or models != binding["models"] or stored.get("task_config_sha256") != binding["task_config_sha256"]):
            raise TaskEventConflict("stored task execution bundle identity changed")
        for role, model in stored["models"].items():
            registered = session.get(TaskModelRecord, model["model_id"])
            if (registered is None or registered.authority != "managed"
                    or registered.scope_camera_id not in (None, task.camera_id)
                    or registered.artifact_sha256 != model["artifact_sha256"]
                    or hashlib.sha256(registered.descriptor_json.encode()).hexdigest() != registered.descriptor_sha256
                    or model != {"model_id": registered.model_id, "artifact_sha256": registered.artifact_sha256,
                                 **json.loads(registered.descriptor_json)}):
                raise TaskEventConflict("stored task model binding changed")

    def _persist(self, event, encoded, encode_error):
        payload = deepcopy(event)
        with self._db.get_session() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            existing = session.get(TaskEventRecord, event["event_id"])
            if existing is not None:
                old = json.loads(existing.payload_json)
                # A review or an old mode may have changed since the first
                # successful write. Immutable event identity must still match.
                if any(not same_json_identity(old.get(key), event.get(key)) for key in ("binding", "source", "object", "request_id")):
                    raise TaskEventConflict("event id already belongs to another observation")
                outbox = session.scalar(select(OutboxEvent).where(OutboxEvent.event_id == event["event_id"] + ":1"))
                if outbox is None or outbox.resource_id != event["event_id"] or outbox.topic != "task_events":
                    raise RuntimeError("task event/outbox atomicity invariant is broken")
                if existing.mode == "production" and (
                    not existing.alert_id or session.scalar(select(AlertRecord).where(
                        AlertRecord.alert_id == existing.alert_id)) is None
                ):
                    raise RuntimeError("task event/alert atomicity invariant is broken")
                return self._serialize(existing)
            self._verify_database_identity(session, payload)
            authorization = {"authorized": False, "reasons": ["NO_APPROVED_TASK_RELEASE"]}
            if self._release is not None:
                authorization = self._release.authorize_event(
                    event["binding"]["task_id"], bundle_id=event["binding"]["bundle_id"],
                    deployment_revision=event["binding"]["deployment_revision"],
                    execution_id=event["binding"]["execution_id"],
                    source_session_id=event["source"]["source_session_id"],
                    source_epoch=event["source"]["source_epoch"], session=session)
            if not self._runtime_current(event["binding"], event["source"]):
                raise TaskEventConflict("task execution changed during the release authorization read")
            # Do not consume evidence quota for an observation already rejected
            # by the durable/live fences, or rewrite a historical retry's JPEG.
            relative, digest = None, None
            evidence_status = "failed"
            if encoded is not None:
                try:
                    relative, digest = self._write_keyframe(event["event_id"], encoded)
                    evidence_status = "ready"
                except TaskEventEvidenceUnavailable:
                    encode_error = "KEYFRAME_STORAGE_UNAVAILABLE"
            payload["replay"].update(ready=evidence_status == "ready", sha256=digest,
                                     reason=encode_error, video_available=False)
            exact_authority = {**event["binding"],
                               "source_session_id": event["source"]["source_session_id"],
                               "source_epoch": event["source"]["source_epoch"]}
            production = (isinstance(authorization, dict)
                          and authorization.get("schema_version") == "argus.task-production-event-authorization.v1"
                          and authorization.get("authorized") is True and authorization.get("production_allowed") is True
                          and authorization.get("applied") is True and authorization.get("applied_stage") == "production"
                          and same_json_identity(authorization.get("binding"), exact_authority)
                          and isinstance(authorization.get("release_id"), str) and bool(authorization["release_id"])
                          and evidence_status == "ready")
            payload["release"] = deepcopy(authorization)
            payload["mode"] = "production" if production else "validation"
            payload["production_allowed"] = production
            payload["alerts_emitted"] = production
            alert_id = "TAL-" + event["event_id"][4:] if production else None
            if production:
                alert = AlertRecord(alert_id=alert_id, timestamp=datetime.fromisoformat(event["occurred_at"]),
                    camera_id=event["binding"]["camera_id"], zone_id="task:" + event["binding"]["task_id"][-40:],
                    severity="medium", anomaly_score=event["object"]["confidence"],
                    detection_type="task_object", evidence_summary="Task object detection: " + event["object"]["label"],
                    evidence_json=_canonical(payload), category="task_object", workflow_status="new",
                    model_version_id=event["binding"]["models"]["detector"]["model_id"])
                session.add(alert)
                session.add(OutboxEvent(event_id=alert_id + ":created", topic="alerts", event_type="created",
                    resource_id=alert_id, payload_json=_canonical({"camera_id": event["binding"]["camera_id"],
                        "task_id": event["binding"]["task_id"], "alert_id": alert_id, "event_id": event["event_id"]})))
            row = TaskEventRecord(event_id=event["event_id"], task_id=event["binding"]["task_id"],
                camera_id=event["binding"]["camera_id"], bundle_id=event["binding"]["bundle_id"],
                task_revision=event["binding"]["task_revision"], deployment_revision=event["binding"]["deployment_revision"],
                execution_id=event["binding"]["execution_id"], source_session_id=event["source"]["source_session_id"],
                source_epoch=event["source"]["source_epoch"], frame_index=event["source"]["frame_index"],
                occurred_at=datetime.fromisoformat(event["occurred_at"]), mode=payload["mode"], status="new", revision=1,
                payload_json=_canonical(payload), evidence_status=evidence_status, evidence_path=relative,
                evidence_sha256=digest, alert_id=alert_id)
            session.add(row)
            session.add(OutboxEvent(event_id=event["event_id"] + ":1", topic="task_events", event_type="task_event.created",
                resource_id=event["event_id"], payload_json=_canonical({"camera_id": row.camera_id,
                    "task_id": row.task_id, "event_id": row.event_id, "revision": 1, "mode": row.mode})))
            session.flush()
            result = self._serialize(row)
            session.commit()
            return result

    @staticmethod
    def _serialize(row):
        return {"event_id": row.event_id, "task_id": row.task_id, "camera_id": row.camera_id,
                "bundle_id": row.bundle_id, "task_revision": row.task_revision,
                "deployment_revision": row.deployment_revision, "execution_id": row.execution_id,
                "source_session_id": row.source_session_id, "source_epoch": row.source_epoch,
                "frame_index": row.frame_index, "occurred_at": _iso(row.occurred_at),
                "mode": row.mode, "status": row.status, "revision": row.revision,
                "payload": json.loads(row.payload_json), "evidence_status": row.evidence_status,
                "evidence_sha256": row.evidence_sha256, "alert_id": row.alert_id,
                "notes": row.notes, "reviewed_by": row.reviewed_by, "reviewed_at": _iso(row.reviewed_at),
                "created_at": _iso(row.created_at)}

    def list(self, task_id: str, limit: int = 100) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("event limit must be between 1 and 200")
        with self._db.get_session() as session:
            return [self._serialize(row) for row in session.scalars(select(TaskEventRecord).where(
                TaskEventRecord.task_id == task_id).order_by(TaskEventRecord.occurred_at.desc(),
                                                            TaskEventRecord.event_id.desc()).limit(limit))]

    def get(self, task_id: str, event_id: str) -> dict | None:
        with self._db.get_session() as session:
            row = session.get(TaskEventRecord, event_id)
            return self._serialize(row) if row is not None and row.task_id == task_id else None

    def authority_for(self, db) -> bool:
        """Providers are server-wired to this exact database, not request data."""
        return db is self._db

    def _collection_session(self, session) -> None:
        bind = session.get_bind()
        engine = getattr(bind, "engine", bind)
        if engine is not getattr(self._db, "_engine", None):
            raise TaskEventConflict("event collection provider belongs to another database")

    @staticmethod
    def _collection_binding(binding) -> dict:
        if (not isinstance(binding, dict) or any(key not in binding for key in _COLLECTION_BINDING_KEYS)
                or binding.get("template_id") != "object_detection.v1"
                or any(not isinstance(binding.get(key), str) or not binding[key]
                       for key in ("task_id", "camera_id"))
                or type(binding.get("task_revision")) is not int or binding["task_revision"] < 1):
            raise TaskEventConflict("event collection task binding is invalid")
        digest = binding.get("task_config_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise TaskEventConflict("event collection task digest is invalid")
        return {key: deepcopy(binding[key]) for key in _COLLECTION_BINDING_KEYS}

    @staticmethod
    def _collection_context_shape(context) -> dict:
        if (not isinstance(context, dict) or set(context) != set(_COLLECTION_CONTEXT_KEYS)
                or any(not isinstance(context.get(key), str) or not context[key]
                       for key in _COLLECTION_CONTEXT_KEYS if key not in {"deployment_revision", "source_epoch"})
                or any(type(context.get(key)) is not int or context[key] < 1
                       for key in ("deployment_revision", "source_epoch"))):
            raise TaskEventConflict("event collection execution context is invalid")
        digest = context["manifest_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise TaskEventConflict("event collection execution digest is invalid")
        return deepcopy(context)

    def collection_cursor(self, *, session=None) -> int:
        """Freeze a commit-order high-water mark without claiming WS delivery.

        Creation timestamps have only second precision on SQLite. The integer
        outbox primary key orders the event's already-atomic durable commits.
        """
        if session is None:
            with self._db.get_session() as owned:
                return self.collection_cursor(session=owned)
        self._collection_session(session)
        return session.scalar(select(func.max(OutboxEvent.id))) or 0

    def collection_context(self, binding, *, session=None) -> dict:
        """Read an exact applied/live execution while a collection is claimed."""
        binding = self._collection_binding(binding)
        if session is None:
            with self._db.get_session() as owned:
                return self.collection_context(binding, session=owned)
        self._collection_session(session)
        current = self._current(binding["task_id"]) if callable(self._current) else None
        if not isinstance(current, dict) or any(key not in current for key in _COLLECTION_CONTEXT_KEYS):
            raise TaskEventConflict("event collection execution is unavailable")
        context = self._collection_context_shape({key: current[key] for key in _COLLECTION_CONTEXT_KEYS})
        if context["task_id"] != binding["task_id"]:
            raise TaskEventConflict("event collection execution belongs to another task")
        bundle = session.get(TaskExecutionBundleRecord, context["bundle_id"])
        try:
            stored = json.loads(bundle.manifest_json) if bundle is not None else None
            models = {role: {"model_id": model["model_id"], "artifact_sha256": model["artifact_sha256"]}
                      for role, model in stored["models"].items()}
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise TaskEventConflict("event collection execution bundle is invalid") from exc
        event_binding = {**binding, **{key: context[key] for key in (
            "bundle_id", "manifest_sha256", "execution_id", "deployment_revision", "deployment_operation_id")},
            "models": models}
        self._verify_database_identity(session, {"binding": event_binding, "source": {
            "source_session_id": context["source_session_id"], "source_epoch": context["source_epoch"]}})
        return context

    def collection_context_current(self, binding, context) -> bool:
        try:
            return same_json_identity(self._collection_context_shape(context), self.collection_context(binding))
        except Exception:
            return False

    def _collection_document(self, row, binding, context=None, *, session) -> tuple[dict, str]:
        """Project immutable event identity; human review is a separate stream."""
        try:
            payload = json.loads(row.payload_json)
            immutable = {key: deepcopy(payload[key]) for key in _COLLECTION_PAYLOAD_KEYS}
            event_binding, source = immutable["binding"], immutable["source"]
            validate_event_source_identity(source)
            validate_event_source_identity(immutable["first_source"])
            if any(not same_json_identity(source[key], immutable["first_source"][key])
                   for key in ("camera_id", "source_session_id", "source_epoch", "source_runtime_hash")):
                raise ValueError("event first-frame source identity changed")
            if (immutable["first_source"]["frame_index"] > source["frame_index"]
                    or immutable["first_source"]["capture_monotonic_ts"] > source["capture_monotonic_ts"]):
                raise ValueError("event first-frame identity is from the future")
            if (immutable["schema_version"] != "argus.task-event.v1"
                    or immutable["event_type"] != "object_detected"
                    or immutable["event_id"] != row.event_id
                    or any(not same_json_identity(event_binding.get(key), value) for key, value in binding.items())):
                raise ValueError("event binding mismatch")
            for field, value in (
                ("task_id", row.task_id), ("camera_id", row.camera_id), ("bundle_id", row.bundle_id),
                ("task_revision", row.task_revision), ("deployment_revision", row.deployment_revision),
                ("execution_id", row.execution_id),
            ):
                if not same_json_identity(event_binding.get(field), value):
                    raise ValueError("event record binding mismatch")
            bundle = session.get(TaskExecutionBundleRecord, row.bundle_id)
            stored = json.loads(bundle.manifest_json) if bundle is not None else None
            expected_models = {role: {"model_id": model["model_id"], "artifact_sha256": model["artifact_sha256"]}
                               for role, model in stored["models"].items()}
            if (bundle is None or bundle.task_id != row.task_id or bundle.task_revision != row.task_revision
                    or bundle.manifest_sha256 != event_binding["manifest_sha256"]
                    or hashlib.sha256(bundle.manifest_json.encode()).hexdigest() != bundle.manifest_sha256
                    or any(not same_json_identity(stored.get(key), value) for key, value in binding.items())
                    or not same_json_identity(expected_models, event_binding.get("models"))):
                raise ValueError("event frozen model bundle identity changed")
            for field, value in (("camera_id", row.camera_id), ("source_session_id", row.source_session_id),
                                 ("source_epoch", row.source_epoch), ("frame_index", row.frame_index)):
                if not same_json_identity(source.get(field), value):
                    raise ValueError("event record source mismatch")
            if (immutable["mode"] != row.mode
                    or _iso(datetime.fromisoformat(immutable["occurred_at"])) != _iso(row.occurred_at)
                    or immutable["replay"].get("sha256") != row.evidence_sha256):
                raise ValueError("event record evidence mismatch")
            size = immutable["frame_size"]
            if (not isinstance(size, dict) or set(size) != {"width", "height"}
                    or any(type(size[key]) is not int or not 0 < size[key] <= 16384 for key in size)
                    or size["width"] * size["height"] > 64_000_000):
                raise ValueError("event frame dimensions invalid")
            if context is not None:
                actual = {**{key: event_binding.get(key) for key in _COLLECTION_CONTEXT_KEYS
                             if key not in {"source_session_id", "source_epoch"}},
                          "source_session_id": source.get("source_session_id"), "source_epoch": source.get("source_epoch")}
                if not same_json_identity(actual, context):
                    raise ValueError("event collection execution changed")
            digest = _digest({"payload": immutable, "evidence_sha256": row.evidence_sha256})
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise TaskEventConflict("event collection identity is invalid") from exc
        return immutable, digest

    def collection_events(self, binding, after_cursor, limit=16, *, context=None) -> list[dict]:
        binding = self._collection_binding(binding)
        if (type(after_cursor) is not int or not 0 <= after_cursor <= 2**63 - 1
                or type(limit) is not int or not 1 <= limit <= 16):
            raise ValueError("event collection cursor/limit is invalid")
        if context is not None:
            context = self._collection_context_shape(context)
            if context["task_id"] != binding["task_id"]:
                raise TaskEventConflict("event collection context belongs to another task")
        query = select(OutboxEvent.id, TaskEventRecord).select_from(OutboxEvent).join(
            TaskEventRecord, OutboxEvent.resource_id == TaskEventRecord.event_id).where(
            OutboxEvent.id > after_cursor, OutboxEvent.topic == "task_events",
            OutboxEvent.event_type == "task_event.created",
            OutboxEvent.event_id == TaskEventRecord.event_id + ":1",
            TaskEventRecord.task_id == binding["task_id"], TaskEventRecord.camera_id == binding["camera_id"],
            TaskEventRecord.task_revision == binding["task_revision"],
        )
        if context is not None:
            query = query.where(
                TaskEventRecord.bundle_id == context["bundle_id"],
                TaskEventRecord.execution_id == context["execution_id"],
                TaskEventRecord.deployment_revision == context["deployment_revision"],
                TaskEventRecord.source_session_id == context["source_session_id"],
                TaskEventRecord.source_epoch == context["source_epoch"],
            )
        with self._db.get_session() as session:
            return [{"cursor": cursor, "event_id": row.event_id,
                     "immutable_sha256": self._collection_document(row, binding, context, session=session)[1]}
                    for cursor, row in session.execute(query.order_by(OutboxEvent.id).limit(limit))]

    def collection_evidence(self, binding, event_id, expected_immutable_sha256, *, context=None) -> dict:
        binding = self._collection_binding(binding)
        if (not isinstance(event_id, str) or not event_id
                or not isinstance(expected_immutable_sha256, str) or len(expected_immutable_sha256) != 64
                or any(char not in "0123456789abcdef" for char in expected_immutable_sha256)):
            raise ValueError("event collection evidence identity is invalid")
        if context is not None:
            context = self._collection_context_shape(context)
        with self._db.get_session() as session:
            row = session.get(TaskEventRecord, event_id)
            if row is None or row.task_id != binding["task_id"]:
                raise LookupError("task event not found")
            creation = session.scalar(select(OutboxEvent.id).where(
                OutboxEvent.event_id == event_id + ":1", OutboxEvent.resource_id == event_id,
                OutboxEvent.topic == "task_events", OutboxEvent.event_type == "task_event.created"))
            if creation is None:
                raise TaskEventConflict("event creation receipt is unavailable")
            payload, digest = self._collection_document(row, binding, context, session=session)
            if digest != expected_immutable_sha256:
                raise TaskEventConflict("event collection evidence identity changed")
            if row.evidence_status != "ready" or not row.evidence_path or not row.evidence_sha256:
                raise TaskEventEvidenceUnavailable("task event keyframe is unavailable")
            if row.evidence_path != event_id + ".jpg":
                raise TaskEventEvidenceUnavailable("task event keyframe belongs to another event")
            path = self._safe_path(row.evidence_path)
            try:
                with path.open("rb") as stream:
                    data = stream.read(8 * 1024 * 1024 + 1)
            except OSError as exc:
                raise TaskEventEvidenceUnavailable("task event keyframe is missing or invalid") from exc
            if len(data) > 8 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != row.evidence_sha256:
                raise TaskEventEvidenceUnavailable("task event keyframe identity changed")
            return {"payload": payload, "source": deepcopy(payload["source"]),
                    "evidence_sha256": row.evidence_sha256,
                    "image_size": [payload["frame_size"]["width"], payload["frame_size"]["height"]],
                    "bytes": data, "mime": "image/jpeg"}

    def review(self, task_id, event_id, *, expected_revision, status, notes, actor):
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("an expected event revision is required")
        if not isinstance(status, str) or status not in {"acknowledged", "false_positive", "resolved"}:
            raise ValueError("unsupported event review status")
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 100:
            raise ValueError("server-authenticated review identity is required")
        if notes is not None and (not isinstance(notes, str) or len(notes) > 10000):
            raise ValueError("review notes are invalid")
        with self._db.get_session() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            row = session.get(TaskEventRecord, event_id)
            if row is None or row.task_id != task_id:
                raise LookupError("task event not found")
            if row.revision != expected_revision:
                if (row.revision == expected_revision + 1 and row.status == status
                        and row.reviewed_by == actor and (notes is None or row.notes == notes)):
                    return self._serialize(row)
                raise TaskEventConflict("task event review revision changed")
            payload = json.loads(row.payload_json)
            history = payload.setdefault("review_history", [])
            history.append({"revision": row.revision + 1, "previous_status": row.status,
                            "status": status, "notes": notes, "actor": actor,
                            "occurred_at": datetime.now(timezone.utc).isoformat()})
            if len(history) > 100:
                raise TaskEventConflict("event review history limit reached")
            row.payload_json = _canonical(payload)
            row.status, row.reviewed_by, row.reviewed_at = status, actor, datetime.now(timezone.utc)
            row.notes = notes if notes is not None else row.notes
            row.revision += 1
            session.add(OutboxEvent(event_id=row.event_id + ":" + str(row.revision), topic="task_events",
                event_type="task_event.updated", resource_id=row.event_id,
                payload_json=_canonical({"camera_id": row.camera_id, "task_id": task_id,
                    "event_id": event_id, "revision": row.revision, "status": row.status})))
            session.flush()
            result = self._serialize(row)
            session.commit()
            return result

    def load_evidence(self, task_id, event_id):
        with self._db.get_session() as session:
            row = session.get(TaskEventRecord, event_id)
            if row is None or row.task_id != task_id:
                raise LookupError("task event not found")
            if row.evidence_status != "ready" or not row.evidence_path or not row.evidence_sha256:
                raise TaskEventEvidenceUnavailable("task event keyframe is unavailable")
            if row.evidence_path != event_id + ".jpg":
                raise TaskEventEvidenceUnavailable("task event keyframe belongs to another event")
            path = self._safe_path(row.evidence_path)
            if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
                raise TaskEventEvidenceUnavailable("task event keyframe is missing or invalid")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != row.evidence_sha256:
                raise TaskEventEvidenceUnavailable("task event keyframe identity changed")
            return data, "image/jpeg"

    def replay_metadata(self, task_id, event_id):
        event = self.get(task_id, event_id)
        if event is None:
            raise LookupError("task event not found")
        try:
            self.load_evidence(task_id, event_id)
            ready, reason = True, None
        except TaskEventEvidenceUnavailable:
            ready, reason = False, "KEYFRAME_UNAVAILABLE"
        return {"schema_version": "argus.task-event-replay.v1", "event_id": event_id,
                "task_id": task_id, "kind": "keyframe_only", "video_available": False,
                "ready": ready, "unavailable_reason": reason, "source": event["payload"]["source"],
                "first_source": event["payload"]["first_source"], "object": event["payload"]["object"],
                "frame_size": event["payload"]["frame_size"], "image_sha256": event["evidence_sha256"]}
