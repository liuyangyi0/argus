"""Spawn-isolated shadow computation with one bounded shared-memory frame.

No process or SDK is started on import.  A worker is reusable only after its
previous process is confirmed dead and IPC/shared-memory cleanup succeeds.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import threading
import time
import uuid
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from argus.capture.camera import FrameData
from argus.core.analysis_task_contracts import (
    TaskModelDescriptor, model_role_incompatibilities, normalize_task_config, template_spec,
)


class TaskExecutionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_execution_manifest(manifest: dict) -> dict:
    """Detach the trusted registry response and reject authority ambiguity."""
    try:
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
        if (
            manifest["schema_version"] != "argus.task-execution-bundle.v1"
            or manifest["authority"] != "managed" or manifest["stage"] != "shadow"
            or manifest["template_id"] not in {
                "object_detection.v1", "pose_estimation.v1", "ppe_compliance.v1",
                "ppe_compliance_dual.v1", "fast_motion.v1",
            }
            or not all(isinstance(manifest[key], str) and manifest[key]
                       for key in ("task_id", "bundle_id", "camera_id"))
            or type(manifest["task_revision"]) is not int or manifest["task_revision"] < 1
        ):
            raise ValueError("manifest identity/authority is invalid")
        config = normalize_task_config(manifest["template_id"], manifest["config"])
        if config != manifest["config"]:
            raise ValueError("manifest must contain normalized configuration")
        digest = hashlib.sha256(json.dumps(
            config, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        if digest != manifest["task_config_sha256"]:
            raise ValueError("task configuration identity mismatch")
        roles = template_spec(manifest["template_id"])["model_roles"]
        if set(manifest["models"]) != set(roles):
            raise ValueError("manifest model roles mismatch")
        canonical = json.loads(json.dumps(manifest))
        stored_digest = canonical.pop("manifest_sha256")
        if set(canonical) != {
            "schema_version", "bundle_id", "task_id", "camera_id", "task_revision",
            "task_config_sha256", "template_id", "authority", "stage", "config", "models",
        }:
            raise ValueError("unexpected execution bundle fields")
        for model_role, model in manifest["models"].items():
            descriptor = {key: value for key, value in model.items()
                          if key not in {"model_id", "artifact_sha256", "artifact_path"}}
            parsed = TaskModelDescriptor.model_validate(descriptor).model_dump(mode="json")
            expected_adapter = roles[model_role]
            if (parsed != descriptor or descriptor["adapter_id"] != expected_adapter
                    or not isinstance(model.get("model_id"), str) or not model["model_id"]
                    or not isinstance(model.get("artifact_path"), str) or not model["artifact_path"]
                    or not _sha256(model.get("artifact_sha256"))):
                raise ValueError("model descriptor identity is invalid")
            if model_role_incompatibilities(
                manifest["template_id"], model_role, descriptor, config["rules"]["labels"],
            ):
                raise ValueError("model labels/adapter are incompatible with their task role")
            canonical["models"][model_role].pop("artifact_path")
        manifest_digest = hashlib.sha256(json.dumps(
            canonical, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        if not _sha256(stored_digest) or stored_digest != manifest_digest:
            raise ValueError("execution bundle digest mismatch")
        return manifest
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskExecutionError("INVALID_BUNDLE", str(exc)) from exc


def _sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def frame_provenance(frame: FrameData) -> dict:
    identity = frame.identity
    if (identity is None or not isinstance(identity.source_session_id, str) or not identity.source_session_id
            or type(identity.source_epoch) is not int or identity.source_epoch < 1):
        raise TaskExecutionError("INVALID_FRAME", "frame is missing source identity")
    if frame.camera_id != identity.camera_id:
        raise TaskExecutionError("INVALID_FRAME", "frame camera identity mismatch")
    if (type(identity.source_epoch) is not int or type(identity.frame_index) is not int
            or identity.frame_index < 0 or not math.isfinite(identity.capture_monotonic_ts)
            or not math.isfinite(identity.capture_wall_ts)
            or frame.timestamp != identity.capture_monotonic_ts
            or frame.frame_number != identity.frame_index):
        raise TaskExecutionError("INVALID_FRAME", "frame clock or sequence identity mismatch")
    # Do not expose camera URLs, credentials or local source paths in results.
    return {
        "camera_id": identity.camera_id,
        "source_session_id": identity.source_session_id,
        "source_epoch": identity.source_epoch,
        "source_runtime_hash": identity.runtime_source_hash,
        "frame_index": identity.frame_index,
        "source_frame_index": identity.source_frame_index,
        "pts": identity.pts,
        "time_base_num": identity.time_base_num,
        "time_base_den": identity.time_base_den,
        "frame_index_provenance": identity.frame_index_provenance,
        "timestamp_provenance": identity.timestamp_provenance,
        "capture_wall_ts": identity.capture_wall_ts,
        "capture_monotonic_ts": identity.capture_monotonic_ts,
    }


def _worker_main(connection, memory_name: str, capacity: int, manifest: dict, models_root: str) -> None:
    """Fixed spawn entry point, never an imported user-supplied callable."""
    memory = None
    try:
        from argus.core.task_adapters import build_task_adapter

        manifest = validate_execution_manifest(manifest)
        memory = shared_memory.SharedMemory(name=memory_name)
        adapter = build_task_adapter(manifest, models_root)
        connection.send({"kind": "ready", "provider": adapter.provider})
        while True:
            request = connection.recv()
            request_id = request["request_id"]
            if request["kind"] == "reset":
                adapter.reset(request["reason"])
                connection.send({"kind": "reset", "request_id": request_id})
                continue
            if request["kind"] != "infer":
                raise ValueError("unknown worker command")
            shape = request["shape"]
            if (len(shape) != 3 or shape[2] != 3 or min(shape) < 1
                    or int(np.prod(shape)) > capacity):
                raise ValueError("invalid shared frame shape")
            frame = np.ndarray(shape, dtype=np.uint8, buffer=memory.buf).copy()
            result = adapter.infer(frame, request["source"]["capture_monotonic_ts"])
            response = {"kind": "result", "request_id": request_id,
                        "source": request["source"], "data": result}
            # Result size is bounded independently of model output dimensions.
            if len(json.dumps(response, allow_nan=False).encode()) > 1024 * 1024:
                raise ValueError("worker result exceeds the bounded result contract")
            connection.send(response)
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send({"kind": "error", "code": getattr(exc, "code", "WORKER_FAILED"),
                             "message": str(exc)[:1000]})
        except (EOFError, OSError):
            pass
    finally:
        if memory is not None:
            memory.close()
        connection.close()


class IsolatedTaskWorker:
    """A single-flight process; timeout means kill/join, never a late reuse."""

    def __init__(
        self, manifest: dict, models_root: str | Path,
        *, context=None, memory_factory=None, max_frame_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        self.manifest = validate_execution_manifest(manifest)
        self.models_root = str(Path(models_root).resolve())
        self._context = context or multiprocessing.get_context("spawn")
        self._memory_factory = memory_factory or shared_memory.SharedMemory
        self._capacity = min(max_frame_bytes, self.manifest["config"]["resources"]["memory_limit_mb"]
                             * 1024 * 1024 // 4)
        self._lock = threading.RLock()
        # Only SDK request callers serialize here. stop() can always acquire
        # the short resource lock and kill an SDK blocked in load/run/reset.
        self._operation_lock = threading.Lock()
        self._cancelled = threading.Event()
        self._process = None
        self._connection = None
        self._child_connection = None
        self._memory = None
        self._process_started = False
        self._state = "stopped"
        self._reason: str | None = None
        self._provider: str | None = None
        self._request_id = 0
        self.execution_id = uuid.uuid4().hex

    def snapshot(self) -> dict:
        # Observation must not wait behind a timed-out SDK request. These
        # scalar fields are advisory; only start/stop under _lock own resources.
        return {"state": self._state, "reason": self._reason, "provider": self._provider,
                "execution_id": self.execution_id, "max_inflight": 1,
                "max_frame_bytes": self._capacity, "memory_limit_enforced": False,
                "production_allowed": False}

    def _receive(self, connection, timeout: float, timeout_code: str) -> dict:
        if not connection.poll(timeout):
            raise TaskExecutionError(timeout_code, "isolated worker did not respond before its deadline")
        response = connection.recv()
        if not isinstance(response, dict):
            raise TaskExecutionError("WORKER_FAILED", "invalid worker response")
        if response.get("kind") == "error":
            raise TaskExecutionError(str(response.get("code", "WORKER_FAILED")),
                                     str(response.get("message", "worker failed")))
        return response

    def _fail(self, exc: Exception) -> None:
        reason = str(exc)
        if not self.stop():
            raise TaskExecutionError("CLEANUP_PENDING", f"{reason}; worker ownership retained") from exc
        self._state = "failed"
        self._reason = reason
        if isinstance(exc, TaskExecutionError):
            raise exc
        raise TaskExecutionError("WORKER_FAILED", reason) from exc

    def start(self) -> None:
        with self._operation_lock:
            with self._lock:
                self._start_resources()
                if self._state == "running":
                    return
                connection = self._connection
            try:
                response = self._receive(
                    connection, self.manifest["config"]["resources"]["load_timeout_seconds"], "LOAD_TIMEOUT",
                )
                with self._lock:
                    if self._state != "starting" or self._connection is not connection:
                        raise TaskExecutionError("CANCELLED", "worker start was cancelled")
                    expected_provider = (
                        "classical-cpu" if self.manifest["template_id"] == "fast_motion.v1" else
                        {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[
                            self.manifest["config"]["resources"]["device"]
                        ]
                    )
                    if response.get("kind") != "ready" or response.get("provider") != expected_provider:
                        raise TaskExecutionError("WORKER_FAILED", "worker readiness was not confirmed")
                    self._provider = response["provider"]
                    self._state = "running"
            except Exception as exc:
                self._fail(exc)

    def _start_resources(self) -> None:
        try:
            if self._cancelled.is_set():
                raise TaskExecutionError("CANCELLED", "worker ownership was cancelled")
            if self._state == "running":
                return
            if self._process is not None or self._memory is not None or self._connection is not None:
                raise TaskExecutionError("CLEANUP_PENDING", "previous worker ownership has not been released")
            self._state, self._reason = "starting", None
            self._memory = self._memory_factory(create=True, size=self._capacity)
            self._connection, self._child_connection = self._context.Pipe(duplex=True)
            self._process = self._context.Process(
                target=_worker_main,
                args=(self._child_connection, self._memory.name, self._capacity,
                      self.manifest, self.models_root),
                daemon=True, name=f"task-worker-{self.manifest['task_id']}",
            )
            self._process.start()
            self._process_started = True
            self._child_connection.close()
            self._child_connection = None
        except Exception as exc:
            self._fail(exc)

    def reset(self, reason: str) -> None:
        with self._operation_lock:
            try:
                with self._lock:
                    if self._state != "running":
                        raise TaskExecutionError("WORKER_FAILED", "worker is not running")
                    self._request_id += 1
                    request_id, connection = self._request_id, self._connection
                    connection.send({"kind": "reset", "request_id": request_id, "reason": reason})
                response = self._receive(connection, self.manifest["config"]["resources"]["timeout_seconds"],
                                         "INFER_TIMEOUT")
                with self._lock:
                    if (self._state != "running" or self._connection is not connection
                            or response.get("kind") != "reset" or type(response.get("request_id")) is not int
                            or response.get("request_id") != request_id):
                        raise TaskExecutionError("WORKER_FAILED", "reset response identity mismatch or cancelled")
            except Exception as exc:
                self._fail(exc)

    def infer(self, frame: FrameData) -> dict:
        with self._operation_lock:
            try:
                source = frame_provenance(frame)
                if source["camera_id"] != self.manifest["camera_id"]:
                    raise TaskExecutionError("INVALID_FRAME", "frame is not from this task's camera")
                pixels = frame.frame
                if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3
                        or min(pixels.shape) < 1 or pixels.nbytes > self._capacity):
                    raise TaskExecutionError("INVALID_FRAME", "frame exceeds worker's BGR uint8 capacity")
                with self._lock:
                    if self._state != "running":
                        raise TaskExecutionError("WORKER_FAILED", "worker is not running")
                    shared = np.ndarray(pixels.shape, dtype=np.uint8, buffer=self._memory.buf)
                    shared[:] = pixels
                    del shared
                    self._request_id += 1
                    request_id, connection = self._request_id, self._connection
                    connection.send({"kind": "infer", "request_id": request_id,
                                     "shape": list(pixels.shape), "source": source})
                response = self._receive(connection, self.manifest["config"]["resources"]["timeout_seconds"],
                                         "INFER_TIMEOUT")
                if (self._state != "running" or self._connection is not connection
                        or response.get("kind") != "result" or type(response.get("request_id")) is not int
                        or response.get("request_id") != request_id
                        or response.get("source") != source):
                    raise TaskExecutionError("WORKER_FAILED", "inference result identity mismatch")
                data = response.get("data")
                states = ({"observed", "warming_up", "sampling_insufficient"}
                          if self.manifest["template_id"] == "fast_motion.v1" else {"observed"})
                if (not isinstance(data, dict) or not isinstance(data.get("state"), str)
                        or data["state"] not in states):
                    raise TaskExecutionError("INVALID_OUTPUT", "worker result does not match the task state contract")
                result = {
                    "schema_version": "argus.task-shadow-result.v1", "execution_id": self.execution_id,
                    "task_id": self.manifest["task_id"], "task_revision": self.manifest["task_revision"],
                    "task_config_sha256": self.manifest["task_config_sha256"],
                    "manifest_sha256": self.manifest["manifest_sha256"],
                    "bundle_id": self.manifest["bundle_id"], "template_id": self.manifest["template_id"],
                    "request_id": request_id, "source": source, "provider": self._provider,
                    "data": data, "production_allowed": False, "alerts_emitted": False,
                    "models": {role: {"model_id": value["model_id"], "artifact_sha256": value["artifact_sha256"]}
                               for role, value in self.manifest["models"].items()},
                }
                encoded = json.dumps(result, allow_nan=False)
                if len(encoded.encode()) > 1024 * 1024:
                    raise TaskExecutionError("WORKER_FAILED", "result exceeds memory store limit")
                return json.loads(encoded)
            except Exception as exc:
                self._fail(exc)

    def stop(self) -> bool:
        with self._lock:
            self._state = "stopping"
            timeout = self.manifest["config"]["resources"]["stop_timeout_seconds"]
            deadline = time.monotonic() + timeout
            try:
                started = self._process_started or (
                    self._process is not None and getattr(self._process, "pid", None) is not None)
                if self._process is not None and started and self._process.is_alive():
                    self._process.terminate()
                    self._process.join(max(0.0, (deadline - time.monotonic()) / 2))
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join(max(0.0, deadline - time.monotonic()))
                    if self._process.is_alive():
                        raise RuntimeError("worker process is still alive")
                if self._process is not None:
                    if started:
                        self._process.join(max(0.0, deadline - time.monotonic()))
                    self._process.close()
                    self._process = None
                    self._process_started = False
                for attribute in ("_connection", "_child_connection"):
                    connection = getattr(self, attribute)
                    if connection is not None:
                        connection.close()
                        setattr(self, attribute, None)
                if self._memory is not None:
                    self._memory.close()
                    try:
                        self._memory.unlink()
                    except FileNotFoundError:
                        pass
                    self._memory = None
                self._state, self._reason = "stopped", None
                self._provider = None
                return True
            except Exception as exc:
                self._state, self._reason = "cleanup_pending", str(exc)
                return False

    def cancel(self) -> bool:
        """Fence a control-plane cancellation even before spawn starts."""
        self._cancelled.set()
        return self.stop()
