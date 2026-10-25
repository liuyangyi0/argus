"""Task-bound collection/review/dataset/external-training workflows.

No legacy trainer, baseline mutation, remote SDK, model inference or publication
is invoked here. All mutation authority comes from a current task binding and
transactional workflow revision. HTTP adapters must authenticate human actors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
import weakref
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, text

from argus.capture.task_collector import TaskFrameCollector, controlled_path, png_image_size, safe_component
from argus.core.analysis_task_contracts import normalize_task_config
from argus.storage.models import (
    AnalysisTaskRecord, AnalysisTaskRevisionRecord, TaskDatasetRecord, TaskWorkflowRecord,
)

_LIVE_SERVICES = weakref.WeakValueDictionary()
_PROCESS_IDENTITY = uuid.uuid4().hex


def _process_alive(process_id: int) -> bool:
    """Only declare an owner lost when the OS positively reports its exit."""
    if type(process_id) is not int or process_id < 1:
        return True
    if process_id == os.getpid():
        return True
    if os.name == "nt":
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, process_id)
        if not handle:
            return ctypes.GetLastError() != 87  # ERROR_INVALID_PARAMETER: no such PID.
        try:
            code = ctypes.c_ulong()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TaskWorkflowConflict(RuntimeError):
    pass


class TaskWorkflowBlocked(RuntimeError):
    pass


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _actor(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 100:
        raise ValueError("a server-authenticated actor is required")
    return value


class TaskTrainingRecipeRegistry:
    """Explicit, immutable recipe capabilities; no implicit anomaly fallback."""

    _RECIPES = {
        "external_detection.v1": {
            "recipe_id": "external_detection.v1", "execution": "external",
            "templates": ["object_detection.v1", "ppe_compliance.v1"],
            "annotation_kind": "detection", "local_trainer_available": False,
        },
        "external_pose.v1": {
            "recipe_id": "external_pose.v1", "execution": "external",
            "templates": ["pose_estimation.v1"], "annotation_kind": "pose",
            "local_trainer_available": False,
        },
    }

    def list(self) -> list[dict]:
        return json.loads(_canonical(list(self._RECIPES.values())))

    def get(self, recipe_id: str) -> dict | None:
        recipe = self._RECIPES.get(recipe_id)
        return json.loads(_canonical(recipe)) if recipe else None


class TaskWorkflowService:
    def __init__(self, db, taskdata_root: str | Path, *, encoder=None, candidate_registrar=None):
        self._db = db
        self.root = Path(taskdata_root).resolve()
        self._encoder = encoder
        self._candidate_registrar = candidate_registrar
        self._owner_id = uuid.uuid4().hex
        self._collectors: dict[str, TaskFrameCollector] = {}
        self._collector_lock = threading.RLock()
        self._closed = False
        _LIVE_SERVICES[self._owner_id] = self
        self.recipes = TaskTrainingRecipeRegistry()

    def close(self) -> bool:
        """A successful close confirms worker, subscription and receipt cleanup."""
        with self._collector_lock:
            self._closed = True
            collectors = list(self._collectors.values())
        complete = True
        for collector in collectors:
            try:
                complete = collector.stop() and collector.retry_finalize() and complete
            except Exception:
                complete = False
        if complete:
            _LIVE_SERVICES.pop(self._owner_id, None)
        return complete

    @staticmethod
    def _begin(session) -> None:
        if session.get_bind().dialect.name != "sqlite":
            raise RuntimeError("workflow CAS currently requires SQLite")
        session.execute(text("BEGIN IMMEDIATE"))

    @staticmethod
    def _binding(session, task_id: str) -> dict:
        task = session.get(AnalysisTaskRecord, task_id)
        if task is None:
            raise KeyError("analysis task does not exist")
        if task.authority != "managed":
            raise TaskWorkflowBlocked("LEGACY_USE_EXISTING_SCENE_WORKFLOW")
        if getattr(task, "archived", False):
            raise TaskWorkflowBlocked("TASK_ARCHIVED")
        revision = session.scalar(select(AnalysisTaskRevisionRecord).where(
            AnalysisTaskRevisionRecord.task_id == task_id,
            AnalysisTaskRevisionRecord.revision == task.current_revision,
        ))
        if revision is None or _sha(revision.config_json) != revision.config_sha256:
            raise TaskWorkflowBlocked("TASK_CONFIG_INTEGRITY_FAILED")
        try:
            config = json.loads(revision.config_json)
            if (normalize_task_config(task.template_id, config) != config
                    or _canonical(config) != revision.config_json):
                raise ValueError("task configuration is not canonical")
        except (TypeError, ValueError) as exc:
            raise TaskWorkflowBlocked("TASK_CONFIG_INTEGRITY_FAILED") from exc
        return {
            "task_id": task.task_id, "camera_id": task.camera_id,
            "task_revision": task.current_revision, "task_config_sha256": revision.config_sha256,
            "template_id": task.template_id,
        }

    @staticmethod
    def _manifest(record) -> dict:
        if _sha(record.manifest_json) != record.manifest_sha256:
            raise TaskWorkflowBlocked("WORKFLOW_MANIFEST_INTEGRITY_FAILED")
        try:
            manifest = json.loads(record.manifest_json)
            binding = manifest["binding"]
            valid = (manifest.get("schema") == "argus.task-workflow.v1"
                     and binding["task_id"] == record.task_id
                     and binding["task_revision"] == record.task_revision
                     and binding["task_config_sha256"] == record.task_config_sha256
                     and binding["template_id"] == record.template_id
                     and isinstance(manifest["samples"], list)
                     and isinstance(manifest["collections"], list)
                     and isinstance(manifest["reviews"], dict)
                     and isinstance(manifest["review_history"], list))
            ids = [sample["sample_id"] for sample in manifest["samples"]]
            valid = valid and len(ids) == len(set(ids))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise TaskWorkflowBlocked("WORKFLOW_MANIFEST_SCHEMA_INVALID")
        return manifest

    @staticmethod
    def _task_config(session, binding: dict) -> dict:
        revision = session.scalar(select(AnalysisTaskRevisionRecord).where(
            AnalysisTaskRevisionRecord.task_id == binding["task_id"],
            AnalysisTaskRevisionRecord.revision == binding["task_revision"],
        ))
        if (revision is None or revision.config_sha256 != binding["task_config_sha256"]
                or _sha(revision.config_json) != revision.config_sha256):
            raise TaskWorkflowBlocked("TASK_CONFIG_INTEGRITY_FAILED")
        try:
            config = json.loads(revision.config_json)
            if normalize_task_config(binding["template_id"], config) != config or _canonical(config) != revision.config_json:
                raise ValueError("invalid bound task configuration")
            return config
        except (TypeError, ValueError) as exc:
            raise TaskWorkflowBlocked("TASK_CONFIG_INTEGRITY_FAILED") from exc

    def _record(self, session, workflow_id, expected_revision=None, *, check_binding=True):
        record = session.get(TaskWorkflowRecord, workflow_id)
        if record is None:
            raise KeyError("task workflow does not exist")
        if expected_revision is not None and record.revision != expected_revision:
            raise TaskWorkflowConflict("workflow revision changed")
        manifest = self._manifest(record)
        if check_binding and self._binding(session, record.task_id) != manifest["binding"]:
            raise TaskWorkflowConflict("task configuration changed; create a new bound workflow")
        return record, manifest

    @staticmethod
    def _save(session, record, manifest, state: str, *, operation=None):
        encoded = _canonical(manifest)
        record.manifest_json, record.manifest_sha256 = encoded, _sha(encoded)
        record.state = state
        record.revision += 1
        record.operation_json = _canonical(operation) if operation is not None else None
        record.updated_at = datetime.now(timezone.utc)
        session.flush()

    @classmethod
    def _view(cls, record) -> dict:
        return {
            "workflow_id": record.workflow_id, "task_id": record.task_id,
            "task_revision": record.task_revision, "task_config_sha256": record.task_config_sha256,
            "template_id": record.template_id, "state": record.state, "revision": record.revision,
            "manifest_sha256": record.manifest_sha256, "manifest": cls._manifest(record),
            "operation": json.loads(record.operation_json) if record.operation_json else None,
            "created_by": record.created_by, "created_at": record.created_at.isoformat(),
            "production_readiness": "NO-GO",
        }

    def create(self, task_id: str, *, actor: str, idempotency_key: str | None = None) -> dict:
        actor = _actor(actor)
        if idempotency_key is not None and (not idempotency_key or len(idempotency_key) > 200):
            raise ValueError("invalid idempotency key")
        workflow_id = (uuid.uuid5(uuid.NAMESPACE_URL, f"argus:task-workflow:{task_id}:{actor}:{idempotency_key}").hex
                       if idempotency_key else uuid.uuid4().hex)
        with self._db.get_session() as session:
            self._begin(session)
            existing = session.get(TaskWorkflowRecord, workflow_id)
            if existing is not None:
                return self._view(existing)
            binding = self._binding(session, task_id)
            manifest = {"schema": "argus.task-workflow.v1", "binding": binding,
                        "samples": [], "collections": [], "reviews": {}, "review_history": [],
                        "dataset_id": None, "training": None}
            encoded = _canonical(manifest)
            record = TaskWorkflowRecord(
                workflow_id=workflow_id, task_id=task_id, task_revision=binding["task_revision"],
                task_config_sha256=binding["task_config_sha256"], template_id=binding["template_id"],
                state="collectable", revision=1, manifest_json=encoded,
                manifest_sha256=_sha(encoded), created_by=actor,
            )
            session.add(record)
            session.flush()
            result = self._view(record)
            session.commit()
            return result

    def get(self, workflow_id: str) -> dict:
        with self._db.get_session() as session:
            record, _ = self._record(session, workflow_id, check_binding=False)
            result = self._view(record)
        with self._collector_lock:
            collector = self._collectors.get(workflow_id)
        result["collection_runtime"] = collector.snapshot() if collector else None
        return result

    def list(self, task_id: str | None = None) -> list[dict]:
        with self._db.get_session() as session:
            query = select(TaskWorkflowRecord).order_by(TaskWorkflowRecord.created_at)
            if task_id is not None:
                query = query.where(TaskWorkflowRecord.task_id == task_id)
            rows = [self._view(record) for record in session.scalars(query)]
        # Persisted operation state and the observed collector are separate
        # facts, especially when final manifest commit requires a retry.
        with self._collector_lock:
            collectors = dict(self._collectors)
        for row in rows:
            collector = collectors.get(row["workflow_id"])
            row["collection_runtime"] = collector.snapshot() if collector else None
        return rows

    def start_collection(self, workflow_id: str, *, expected_revision: int, manager, actor: str,
                         params: dict | None = None, collection_id: str | None = None) -> dict:
        _actor(actor)
        params = dict(params or {})
        if set(params) - {"max_samples", "max_duration", "interval", "queue_capacity", "max_storage_mb", "policy"}:
            raise ValueError("unknown collection parameters")
        collection_id = safe_component(collection_id) if collection_id is not None else uuid.uuid4().hex
        with self._collector_lock:
            if self._closed:
                raise TaskWorkflowBlocked("WORKFLOW_SERVICE_CLOSED")
            prior = self._collectors.get(workflow_id)
            if prior is not None and prior.snapshot()["state"] == "collecting":
                raise TaskWorkflowConflict("collection is already in progress")
            if prior is not None and (not prior.stop() or not prior.retry_finalize()):
                raise TaskWorkflowBlocked("COLLECTION_CLEANUP_NOT_CONFIRMED")
            with self._db.get_session() as session:
                self._begin(session)
                record, manifest = self._record(session, workflow_id, expected_revision)
                if record.state not in {"collectable", "review", "collection_failed"}:
                    raise TaskWorkflowConflict("workflow cannot collect in its current state")
                if any(receipt.get("collection_id") == collection_id for receipt in manifest["collections"]):
                    raise TaskWorkflowConflict("collection identity was already finalized")
                policy = self._task_config(session, manifest["binding"])["collection"]
                if policy["policy"] != "periodic" or params.pop("policy", "periodic") != "periodic":
                    raise TaskWorkflowBlocked("COLLECTION_POLICY_NOT_IMPLEMENTED")
                params.setdefault("max_samples", min(policy["max_samples"], 10000))
                params.setdefault("interval", policy["interval_seconds"])
                params.setdefault("max_storage_mb", policy["max_storage_mb"])
                if (type(params["max_samples"]) is not int or type(params["max_storage_mb"]) is not int
                        or type(params["interval"]) not in (int, float) or not math.isfinite(params["interval"])):
                    raise ValueError("invalid collection parameters")
                if (params["max_samples"] > policy["max_samples"]
                        or params["max_storage_mb"] > policy["max_storage_mb"]
                        or params["interval"] < policy["interval_seconds"]):
                    raise TaskWorkflowBlocked("COLLECTION_EXCEEDS_TASK_POLICY")
                collector = TaskFrameCollector(
                    manager, manifest["binding"], self.root, collection_id, **params,
                    encoder=self._encoder,
                    on_finish=lambda snapshot: self._finish_collection(workflow_id, collection_id, snapshot),
                )
                operation = {"kind": "collection", "collection_id": collection_id, "owner_id": self._owner_id,
                             "process_id": os.getpid(), "process_identity": _PROCESS_IDENTITY,
                             "requested_by": actor, "parameters": params}
                self._save(session, record, manifest, "collecting", operation=operation)
                session.commit()
            self._collectors[workflow_id] = collector
            try:
                collector.start()
            except Exception:
                collector.finish_start_failure()
        return self.get(workflow_id)

    def _finish_collection(self, workflow_id: str, collection_id: str, snapshot: dict) -> None:
        with self._db.get_session() as session:
            self._begin(session)
            # Captured samples stay bound to their original revision. A changed
            # task can retain evidence, but cannot freeze/train it as new data.
            record, manifest = self._record(session, workflow_id, check_binding=False)
            completed = next((entry for entry in manifest["collections"]
                              if entry["collection_id"] == collection_id), None)
            receipt = {key: value for key, value in snapshot.items() if key != "samples"}
            if completed is not None:
                samples = [sample for sample in manifest["samples"] if sample["collection_id"] == collection_id]
                def immutable(rows):
                    return [{key: value for key, value in row.items() if key != "auto_annotation"} for row in rows]
                if completed != receipt or immutable(samples) != immutable(snapshot["samples"]):
                    raise TaskWorkflowConflict("collection receipt changed after commit")
                return
            operation = json.loads(record.operation_json or "{}")
            if (record.state != "collecting" or operation.get("collection_id") != collection_id
                    or operation.get("owner_id") != self._owner_id):
                raise TaskWorkflowConflict("collection operation ownership changed")
            if snapshot["binding"] != manifest["binding"]:
                raise TaskWorkflowConflict("collection binding changed")
            for sample in snapshot["samples"]:
                self._verify_sample(sample, manifest["binding"])
            old_ids = {sample["sample_id"] for sample in manifest["samples"]}
            new_ids = [sample["sample_id"] for sample in snapshot["samples"]]
            if old_ids.intersection(new_ids) or len(new_ids) != len(set(new_ids)):
                raise TaskWorkflowConflict("duplicate collection samples")
            manifest["samples"].extend(snapshot["samples"])
            manifest["collections"].append(receipt)
            manifest["dataset_id"], manifest["training"] = None, None
            # Zero images are not success and cannot produce a trainable dataset.
            successful = snapshot["state"] in {"completed", "stopped"} and bool(snapshot["samples"])
            self._save(session, record, manifest, "review" if successful else "collection_failed")
            session.commit()

    def stop_collection(self, workflow_id: str, *, expected_revision: int, actor: str,
                        expected_collection_id: str | None = None) -> dict:
        _actor(actor)
        if expected_collection_id is not None:
            safe_component(expected_collection_id)
        with self._collector_lock:
            collector = self._collectors.get(workflow_id)
            with self._db.get_session() as session:
                self._begin(session)
                record, manifest = self._record(session, workflow_id, expected_revision, check_binding=False)
                operation = json.loads(record.operation_json or "{}")
                current_id = (operation.get("collection_id") if record.state == "collecting"
                              else collector.collection_id if collector is not None else None)
                if expected_collection_id is not None and current_id != expected_collection_id:
                    raise TaskWorkflowConflict("collection ownership changed")
                if (record.state == "collecting" and collector is not None
                        and operation.get("collection_id") != collector.collection_id):
                    prior = collector.snapshot()
                    if (prior.get("cleanup_confirmed") is not True or prior.get("persistence_pending") is True
                            or prior.get("state") == "collecting"):
                        raise TaskWorkflowBlocked("COLLECTION_CLEANUP_NOT_CONFIRMED")
                    # Start committed a new claim but lost its acknowledgement
                    # before installing the new worker. The retained terminal
                    # collector is not evidence that the new claim is alive.
                    collector = None
                if record.state != "collecting":
                    # A final receipt may already be committed while its
                    # subscription cleanup is still pending. Local ownership
                    # and the immutable receipt permit retry without changing
                    # dataset/review state or touching another collector.
                    if collector is None or not any(
                        receipt.get("collection_id") == collector.collection_id for receipt in manifest["collections"]
                    ):
                        raise TaskWorkflowConflict("collection is not in progress")
                    operation = None
                if collector is None:
                    owner = operation.get("owner_id")
                    same_process = (operation.get("process_id") == os.getpid()
                                    and operation.get("process_identity") == _PROCESS_IDENTITY)
                    known_lost = ((owner == self._owner_id)
                                  or (same_process and owner not in _LIVE_SERVICES)
                                  or not _process_alive(operation.get("process_id")))
                    if not known_lost:
                        raise TaskWorkflowBlocked("COLLECTION_OWNER_NOT_CONFIRMED_EXITED")
                    manifest["collections"].append({"collection_id": operation.get("collection_id"),
                                                     "state": "failed", "error_code": "COLLECTION_OWNER_LOST"})
                    self._save(session, record, manifest, "collection_failed")
                    session.commit()
                    return self.get(workflow_id)
                if operation is not None and (operation.get("owner_id") != self._owner_id
                                               or operation.get("collection_id") != collector.collection_id):
                    raise TaskWorkflowConflict("collection is owned by another process")
            # No DB lock is held while joining the worker/final receipt callback.
        if not collector.stop() or not collector.retry_finalize():
            raise TaskWorkflowBlocked("COLLECTION_CLEANUP_NOT_CONFIRMED")
        return self.get(workflow_id)

    def _sample_data(self, sample: dict, binding: dict) -> tuple[Path, bytes]:
        if sample.get("binding") != binding:
            raise TaskWorkflowConflict("sample task binding mismatch")
        relative = sample.get("relative_path", "")
        if not relative.startswith(f"{binding['task_id']}/{sample.get('collection_id')}/"):
            raise ValueError("sample is outside its task collection")
        path = controlled_path(self.root, relative, must_exist=True)
        with path.open("rb") as stream:
            data = stream.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise ValueError("task image is too large")
        if len(data) != sample.get("byte_size") or hashlib.sha256(data).hexdigest() != sample.get("image_sha256"):
            raise TaskWorkflowConflict("task image content changed")
        actual_size = png_image_size(data)
        image_size = sample.get("image_size")
        if (not isinstance(image_size, list) or len(image_size) != 2
                or any(type(value) is not int for value in image_size) or tuple(image_size) != actual_size):
            raise TaskWorkflowConflict("task image dimensions changed")
        identity = sample.get("identity", {})
        required = {"camera_id", "source_session_id", "source_epoch", "frame_index", "source_frame_index",
                    "pts", "time_base_num", "time_base_den", "capture_wall_ts", "capture_monotonic_ts",
                    "frame_index_provenance", "timestamp_provenance"}
        if (not isinstance(identity, dict) or not required <= identity.keys()
                or identity.get("camera_id") != binding["camera_id"]
                or not isinstance(identity.get("source_session_id"), str)
                or not identity.get("source_session_id")
                or type(identity.get("source_epoch")) is not int or identity["source_epoch"] < 1
                or type(identity.get("frame_index")) is not int or identity["frame_index"] < 0):
            raise ValueError("incomplete source identity")
        for field in ("capture_wall_ts", "capture_monotonic_ts"):
            value = identity[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid capture timestamp")
        for field in ("source_frame_index", "pts"):
            value = identity[field]
            if value is not None and (type(value) is not int or field == "source_frame_index" and value < 0):
                raise ValueError("invalid original frame identity")
        numerator, denominator = identity["time_base_num"], identity["time_base_den"]
        if ((numerator is None) != (denominator is None)
                or numerator is not None and (type(numerator) is not int or type(denominator) is not int
                                              or numerator < 1 or denominator < 1)
                or identity["pts"] is not None and numerator is None):
            raise ValueError("invalid original PTS time base")
        return path, data

    def _verify_sample(self, sample: dict, binding: dict) -> Path:
        return self._sample_data(sample, binding)[0]

    def read_sample_image(self, workflow_id: str, sample_id: str) -> bytes:
        with self._db.get_session() as session:
            _, manifest = self._record(session, workflow_id, check_binding=False)
            sample = next((row for row in manifest["samples"] if row["sample_id"] == sample_id), None)
            if sample is None:
                raise KeyError("sample does not exist")
            # Return precisely the bytes whose digest was checked. A file path
            # response would permit a second read after the verification.
            return self._sample_data(sample, manifest["binding"])[1]

    @staticmethod
    def _annotation(annotation: dict | None, template_id: str, allowed_labels: list[str] | None = None) -> dict:
        if not isinstance(annotation, dict) or annotation.get("schema") != "argus.task-annotation.v1":
            raise ValueError("approved samples require a versioned human annotation")
        expected = "pose" if template_id == "pose_estimation.v1" else "detection"
        if (set(annotation) != {"schema", "kind", "coordinate_space", "objects"}
                or annotation.get("coordinate_space") != "normalized"
                or annotation.get("kind") != expected or not isinstance(annotation.get("objects"), list)
                or len(annotation["objects"]) > 1000):
            raise ValueError("annotation is incompatible with its task template")
        labels = set(allowed_labels or [])
        if expected == "pose":
            labels = {"person"}
        elif template_id in {"ppe_compliance.v1", "ppe_compliance_dual.v1"}:
            labels = labels or {"person", "helmet", "bare_head"}

        def coordinate(value):
            return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1

        for item in annotation["objects"]:
            fields = {"label", "bbox_xyxy"} | ({"keypoint_schema", "keypoints"} if expected == "pose" else set())
            if not isinstance(item, dict) or set(item) != fields:
                raise ValueError("invalid annotation object fields")
            label = item["label"]
            if (not isinstance(label, str) or not label.strip() or len(label) > 100
                    or any(ord(char) < 32 for char in label) or labels and label not in labels
                    or template_id in {"ppe_compliance.v1", "ppe_compliance_dual.v1"}
                    and label not in {"person", "helmet", "bare_head"}):
                raise ValueError("annotation label is outside the task label schema")
            box = item["bbox_xyxy"]
            if (not isinstance(box, list) or len(box) != 4 or not all(coordinate(value) for value in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError("annotation box must be normalized non-degenerate xyxy")
            if expected == "pose":
                points = item["keypoints"]
                if item["keypoint_schema"] != "coco17.v1" or not isinstance(points, list) or len(points) != 17:
                    raise ValueError("pose annotation requires the complete COCO-17 schema")
                indexes = []
                for point in points:
                    if (not isinstance(point, dict) or set(point) != {"index", "x", "y", "visibility"}
                            or type(point["index"]) is not int or not 0 <= point["index"] < 17
                            or not coordinate(point["x"]) or not coordinate(point["y"])
                            or type(point["visibility"]) is not int or point["visibility"] not in {0, 1, 2}):
                        raise ValueError("invalid pose annotation keypoint")
                    indexes.append(point["index"])
                if len(set(indexes)) != 17:
                    raise ValueError("pose annotation keypoint indexes must be unique")
        return json.loads(_canonical(annotation))

    def _apply_review(self, manifest: dict, sample_id: str, *, expected_image_sha256: str,
                      decision: str, notes: str | None, annotation: dict | None, actor: str,
                      allowed_labels: list[str] | None = None) -> None:
        if decision not in {"approved", "rejected", "needs_review"}:
            raise ValueError("invalid review decision")
        sample = next((item for item in manifest["samples"] if item["sample_id"] == sample_id), None)
        if sample is None:
            raise KeyError("sample does not exist")
        if sample["image_sha256"] != expected_image_sha256:
            raise TaskWorkflowConflict("review image digest changed")
        self._verify_sample(sample, manifest["binding"])
        prior = manifest["reviews"].get(sample_id)
        if prior is not None:
            manifest["review_history"].append({"sample_id": sample_id, **prior})
        if notes is not None and (not isinstance(notes, str) or len(notes) > 10000):
            raise ValueError("invalid review notes")
        validated = self._annotation(annotation, manifest["binding"]["template_id"], allowed_labels) if decision == "approved" else None
        manifest["reviews"][sample_id] = {
            "review_revision": (prior["review_revision"] if prior else 0) + 1,
            "image_sha256": expected_image_sha256, "decision": decision,
            "notes": notes if notes is not None else prior.get("notes", "") if prior else "",
            "annotation": validated, "reviewed_by": actor, "origin": "human_review",
        }
        # Any change invalidates workflow-level approvals. Prior notes/reviews
        # remain in history; frozen datasets themselves are immutable.
        manifest["dataset_id"] = None
        manifest["training"] = None

    def review(self, workflow_id: str, sample_id: str, *, expected_revision: int,
               expected_image_sha256: str, decision: str, actor: str,
               notes: str | None = None, annotation: dict | None = None) -> dict:
        actor = _actor(actor)
        with self._db.get_session() as session:
            self._begin(session)
            record, manifest = self._record(session, workflow_id, expected_revision)
            if record.state in {"collecting", "candidate_importing"}:
                raise TaskWorkflowConflict("workflow operation is in progress")
            self._apply_review(manifest, sample_id, expected_image_sha256=expected_image_sha256,
                               decision=decision, notes=notes, annotation=annotation, actor=actor,
                               allowed_labels=self._task_config(session, manifest["binding"])["rules"]["labels"])
            self._save(session, record, manifest, "review")
            result = self._view(record)
            session.commit()
            return result

    def freeze(self, workflow_id: str, *, expected_revision: int, actor: str) -> dict:
        actor = _actor(actor)
        with self._db.get_session() as session:
            self._begin(session)
            record, manifest = self._record(session, workflow_id, expected_revision)
            if record.state != "review":
                raise TaskWorkflowConflict("workflow is not ready for dataset freezing")
            accepted = []
            config = self._task_config(session, manifest["binding"])
            for sample in manifest["samples"]:
                self._verify_sample(sample, manifest["binding"])
                review = manifest["reviews"].get(sample["sample_id"])
                if review is None or review["decision"] == "needs_review":
                    raise TaskWorkflowBlocked("HUMAN_REVIEW_INCOMPLETE")
                if review["image_sha256"] != sample["image_sha256"]:
                    raise TaskWorkflowConflict("review is stale")
                if review["decision"] == "approved":
                    self._annotation(review["annotation"], record.template_id, config["rules"]["labels"])
                    accepted.append({**sample, "review": review})
            sessions = sorted({sample["identity"]["source_session_id"] for sample in accepted})
            if len(sessions) < 2:
                raise TaskWorkflowBlocked("AT_LEAST_TWO_SOURCE_SESSIONS_REQUIRED")
            validation_session = sessions[-1]
            for sample in accepted:
                sample["split"] = "validation" if sample["identity"]["source_session_id"] == validation_session else "train"
            dataset_id = uuid.uuid4().hex
            dataset = {"schema": "argus.task-dataset.v1", "dataset_id": dataset_id,
                       "workflow_id": workflow_id, "workflow_revision": record.revision,
                       "workflow_manifest_sha256": record.manifest_sha256, "binding": manifest["binding"],
                       "split_policy": "source_session_disjoint.v1", "samples": accepted,
                       "label_schema": self._label_schema(record.template_id, config, accepted),
                       "frozen_by": actor, "production_readiness": "NO-GO"}
            encoded = _canonical(dataset)
            session.add(TaskDatasetRecord(dataset_id=dataset_id, task_id=record.task_id,
                                          task_revision=record.task_revision, manifest_json=encoded,
                                          manifest_sha256=_sha(encoded), created_by=actor))
            manifest["dataset_id"] = dataset_id
            self._save(session, record, manifest, "dataset_frozen")
            session.commit()
            return dataset

    def get_dataset(self, dataset_id: str) -> dict:
        with self._db.get_session() as session:
            record = session.get(TaskDatasetRecord, dataset_id)
            if record is None:
                raise KeyError("task dataset does not exist")
            return self._dataset(session, record)

    def _dataset(self, session, record, *, binding=None, workflow_id=None) -> dict:
        if (not record.manifest_sha256
                or _sha(record.manifest_json) != record.manifest_sha256):
            raise TaskWorkflowBlocked("DATASET_MANIFEST_INTEGRITY_FAILED")
        try:
            manifest = json.loads(record.manifest_json)
            actual = manifest["binding"]
            samples = manifest["samples"]
            valid = (manifest["schema"] == "argus.task-dataset.v1"
                     and manifest["dataset_id"] == record.dataset_id
                     and actual["task_id"] == record.task_id
                     and actual["task_revision"] == record.task_revision
                     and manifest["split_policy"] == "source_session_disjoint.v1"
                     and manifest["production_readiness"] == "NO-GO"
                     and isinstance(samples, list) and bool(samples)
                     and len({sample["sample_id"] for sample in samples}) == len(samples)
                     and (binding is None or binding == actual)
                     and (workflow_id is None or workflow_id == manifest["workflow_id"]))
            if not valid:
                raise ValueError("invalid frozen dataset")
            config = self._task_config(session, actual)
            sessions: dict[str, str] = {}
            splits = set()
            for sample in samples:
                self._verify_sample(sample, actual)
                review = sample["review"]
                if (review["decision"] != "approved" or review["origin"] != "human_review"
                        or review["image_sha256"] != sample["image_sha256"]
                        or not review["reviewed_by"] or sample["split"] not in {"train", "validation"}):
                    raise ValueError("invalid frozen review")
                self._annotation(review["annotation"], actual["template_id"], config["rules"]["labels"])
                source_session = sample["identity"]["source_session_id"]
                if source_session in sessions and sessions[source_session] != sample["split"]:
                    raise ValueError("source session leaks across dataset splits")
                sessions[source_session] = sample["split"]
                splits.add(sample["split"])
            if splits != {"train", "validation"}:
                raise ValueError("dataset needs session-disjoint train and validation data")
            if manifest["label_schema"] != self._label_schema(actual["template_id"], config, samples):
                raise ValueError("frozen dataset label schema changed")
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskWorkflowBlocked("DATASET_MANIFEST_SCHEMA_INVALID") from exc
        return manifest

    @staticmethod
    def _label_schema(template_id: str, config: dict, samples: list[dict]) -> dict:
        labels = config["rules"]["labels"]
        origin = "task_rules"
        if template_id == "pose_estimation.v1":
            labels, origin = ["person"], "template"
        elif template_id in {"ppe_compliance.v1", "ppe_compliance_dual.v1"} and not labels:
            labels, origin = ["person", "helmet", "bare_head"], "template"
        elif not labels:
            labels = sorted({item["label"] for sample in samples
                             for item in sample["review"]["annotation"]["objects"]})
            origin = "reviewed_annotations"
        return {"schema": "argus.task-label-schema.v1", "labels": labels, "origin": origin,
                "kind": "pose" if template_id == "pose_estimation.v1" else "detection"}

    def export_review(self, workflow_id: str, *, target: str) -> dict:
        if target not in {"cvat", "fiftyone"}:
            raise ValueError("unsupported review export target")
        current = self.get(workflow_id)
        for sample in current["manifest"]["samples"]:
            self._verify_sample(sample, current["manifest"]["binding"])
        # An outer envelope, not a mutation of existing ReviewBundle v1 or SDK
        # lock semantics. Calling this only returns JSON; no external API call.
        return {"schema": "argus.task-review-envelope.v1", "target": target,
                "workflow_id": workflow_id, "workflow_revision": current["revision"],
                "report_sha256": current["manifest_sha256"], "binding": current["manifest"]["binding"],
                "media_type": "images", "samples": current["manifest"]["samples"],
                "reviews": current["manifest"]["reviews"], "external_side_effects": False}

    def import_review(self, workflow_id: str, *, expected_revision: int, payload: dict, actor: str) -> dict:
        actor = _actor(actor)
        with self._db.get_session() as session:
            self._begin(session)
            record, manifest = self._record(session, workflow_id, expected_revision)
            if record.state in {"collecting", "candidate_importing"}:
                raise TaskWorkflowConflict("workflow operation is in progress")
            if (payload.get("schema") != "argus.task-review-envelope.v1"
                    or payload.get("target") not in {"cvat", "fiftyone"} or payload.get("workflow_id") != workflow_id
                    or payload.get("workflow_revision") != record.revision
                    or payload.get("report_sha256") != record.manifest_sha256
                    or payload.get("binding") != manifest["binding"]):
                raise TaskWorkflowConflict("review report or task binding changed")
            rows = payload.get("results")
            if not isinstance(rows, list) or not rows:
                raise ValueError("review import needs non-empty results")
            seen = set()
            allowed_labels = self._task_config(session, manifest["binding"])["rules"]["labels"]
            for row in rows:
                if not isinstance(row, dict) or set(row) - {"sample_id", "image_sha256", "origin", "decision", "notes", "annotation", "source"}:
                    raise ValueError("invalid review import row")
                sample_id = row["sample_id"]
                if sample_id in seen:
                    raise ValueError("duplicate sample review")
                seen.add(sample_id)
                if row.get("origin") == "auto":
                    sample = next((item for item in manifest["samples"] if item["sample_id"] == sample_id), None)
                    if sample is None or sample["image_sha256"] != row["image_sha256"]:
                        raise TaskWorkflowConflict("preannotation sample changed")
                    self._verify_sample(sample, manifest["binding"])
                    previous = manifest["reviews"].get(sample_id)
                    validated = self._annotation(row.get("annotation"), manifest["binding"]["template_id"], allowed_labels)
                    sample["auto_annotation"] = {"annotation": validated,
                                                 "source": row.get("source"), "imported_by": actor}
                    self._apply_review(manifest, sample_id, expected_image_sha256=row["image_sha256"],
                                       decision="needs_review", notes=previous.get("notes") if previous else None,
                                       annotation=None, actor=actor)
                elif row.get("origin") == "human":
                    self._apply_review(manifest, sample_id, expected_image_sha256=row["image_sha256"],
                                       decision=row["decision"], notes=row.get("notes"),
                                       annotation=row.get("annotation"), actor=actor, allowed_labels=allowed_labels)
                else:
                    raise ValueError("annotation origin must be explicit")
            self._save(session, record, manifest, "review")
            result = self._view(record)
            session.commit()
            return result

    def propose_training(self, workflow_id: str, *, expected_revision: int, recipe_id: str, actor: str) -> dict:
        actor = _actor(actor)
        with self._db.get_session() as session:
            self._begin(session)
            record, manifest = self._record(session, workflow_id, expected_revision)
            if record.state not in {"dataset_frozen", "training_blocked", "training_proposed"} or not manifest["dataset_id"]:
                raise TaskWorkflowConflict("freeze a reviewed dataset before proposing training")
            frozen_record = session.get(TaskDatasetRecord, manifest["dataset_id"])
            if frozen_record is None:
                raise TaskWorkflowBlocked("DATASET_MISSING")
            self._dataset(session, frozen_record, binding=manifest["binding"], workflow_id=workflow_id)
            recipe = self.recipes.get(recipe_id)
            blocked = ("RECIPE_NOT_REGISTERED" if recipe is None else
                       "RECIPE_TEMPLATE_MISMATCH" if record.template_id not in recipe["templates"] else None)
            manifest["training"] = {"recipe_id": recipe_id, "dataset_id": manifest["dataset_id"],
                                    "requested_by": actor, "approved_by": None,
                                    "execution": "external", "local_training_state": "blocked",
                                    "local_training_reason": "LOCAL_TRAINER_UNAVAILABLE",
                                    "blocked_reason": blocked, "candidate": None}
            self._save(session, record, manifest, "training_blocked" if blocked else "training_proposed")
            result = self._view(record)
            session.commit()
            return result

    def approve_training(self, workflow_id: str, *, expected_revision: int, actor: str) -> dict:
        actor = _actor(actor)
        with self._db.get_session() as session:
            self._begin(session)
            record, manifest = self._record(session, workflow_id, expected_revision)
            if record.state != "training_proposed" or not manifest["training"]:
                raise TaskWorkflowConflict("training proposal is not approvable")
            training = manifest["training"]
            recipe = self.recipes.get(training["recipe_id"])
            if (recipe is None or record.template_id not in recipe["templates"]
                    or training.get("blocked_reason") or training["dataset_id"] != manifest["dataset_id"]):
                raise TaskWorkflowBlocked("RECIPE_TEMPLATE_MISMATCH")
            frozen_record = session.get(TaskDatasetRecord, manifest["dataset_id"])
            if frozen_record is None:
                raise TaskWorkflowBlocked("DATASET_MISSING")
            frozen = self._dataset(session, frozen_record, binding=manifest["binding"], workflow_id=workflow_id)
            if not frozen["label_schema"]["labels"]:
                raise TaskWorkflowBlocked("LABEL_SCHEMA_EMPTY")
            manifest["training"]["approved_by"] = actor
            self._save(session, record, manifest, "training_export_ready")
            result = self._view(record)
            session.commit()
            return result

    def export_training_package(self, workflow_id: str) -> dict:
        with self._db.get_session() as session:
            record, manifest = self._record(session, workflow_id)
            if record.state not in {"training_export_ready", "candidate_importing", "candidate_registered"}:
                raise TaskWorkflowBlocked("APPROVED_EXTERNAL_TRAINING_REQUIRED")
            dataset = session.get(TaskDatasetRecord, manifest["dataset_id"])
            if dataset is None:
                raise TaskWorkflowBlocked("DATASET_MISSING")
            frozen = self._dataset(session, dataset, binding=manifest["binding"], workflow_id=workflow_id)
            if not frozen["label_schema"]["labels"]:
                raise TaskWorkflowBlocked("LABEL_SCHEMA_EMPTY")
            training = manifest["training"]
            recipe = self.recipes.get(training["recipe_id"])
            if (training["dataset_id"] != manifest["dataset_id"] or not training.get("approved_by")
                    or training.get("blocked_reason") or recipe is None
                    or record.template_id not in recipe["templates"]):
                raise TaskWorkflowBlocked("APPROVED_EXTERNAL_TRAINING_REQUIRED")
            package = {"schema": "argus.external-training-package.v1", "workflow_id": workflow_id,
                       "binding": manifest["binding"], "dataset": frozen, "recipe": manifest["training"]["recipe_id"],
                       "approved_by": manifest["training"]["approved_by"], "local_training_state": "blocked",
                       "production_readiness": "NO-GO"}
            return {**package, "package_sha256": _sha(_canonical(package))}

    def import_candidate(self, workflow_id: str, *, expected_revision: int, candidate: dict,
                         expected_package_sha256: str, actor: str) -> dict:
        actor = _actor(actor)
        if self._candidate_registrar is None:
            raise TaskWorkflowBlocked("P3_CANDIDATE_REGISTRAR_UNAVAILABLE")
        package = self.export_training_package(workflow_id)
        if package["package_sha256"] != expected_package_sha256:
            raise TaskWorkflowConflict("training package identity changed")
        operation_id = _sha(_canonical({"workflow_id": workflow_id, "package": expected_package_sha256,
                                       "candidate": candidate}))
        with self._db.get_session() as session:
            self._begin(session)
            record, manifest = self._record(session, workflow_id)
            completed = (manifest.get("training") or {}).get("candidate")
            if (record.state == "candidate_registered" and isinstance(completed, dict)
                    and completed.get("operation_id") == operation_id):
                return self._view(record)
            if record.revision != expected_revision:
                raise TaskWorkflowConflict("workflow revision changed")
            if record.state not in {"training_export_ready", "candidate_importing"}:
                raise TaskWorkflowConflict("candidate import is not available")
            existing = json.loads(record.operation_json or "{}")
            if existing and existing.get("operation_id") != operation_id:
                raise TaskWorkflowConflict("another candidate import requires reconciliation")
            if existing and not existing.get("requested_by"):
                # A legacy unknown write cannot be safely replayed under a
                # newly guessed actor: registry idempotency includes the actor.
                raise TaskWorkflowBlocked("CANDIDATE_IMPORT_ACTOR_UNRECORDED")
            registration_actor = _actor(existing.get("requested_by", actor))
            self._save(session, record, manifest, "candidate_importing", operation={
                "kind": "candidate_import", "operation_id": operation_id,
                "package_sha256": expected_package_sha256, "requested_by": registration_actor,
            })
            session.commit()
        # Only an injected P3 registrar may register a candidate. Its stable
        # idempotency key permits retry after an unknown commit; never publish.
        registered = self._candidate_registrar(
            task_id=package["binding"]["task_id"], candidate=candidate,
            training_manifest=package, actor=registration_actor, idempotency_key=operation_id,
        )
        if not isinstance(registered, dict) or not registered.get("artifact_id"):
            raise TaskWorkflowBlocked("CANDIDATE_REGISTRATION_NOT_CONFIRMED")
        with self._db.get_session() as session:
            self._begin(session)
            record, manifest = self._record(session, workflow_id)
            operation = json.loads(record.operation_json or "{}")
            if record.state != "candidate_importing" or operation.get("operation_id") != operation_id:
                raise TaskWorkflowConflict("candidate import state changed")
            manifest["training"]["candidate"] = {"artifact_id": registered["artifact_id"], "stage": "candidate",
                                                   "operation_id": operation_id}
            self._save(session, record, manifest, "candidate_registered")
            result = self._view(record)
            session.commit()
            return result
