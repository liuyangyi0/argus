"""Bounded collection of committed event keyframes, never future camera frames.

An event JPEG is a lossy derived image. Its PNG derivative keeps the original
frame identity and both digests; neither is presented as source video or as an
approved training label. All IO runs on the collector's dedicated thread.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import threading
import time
import uuid

from argus.capture.task_collector import TaskFrameCollector, controlled_path, png_image_size
from argus.core.task_events import same_json_identity, validate_event_sample_origin, validate_event_source_identity


def _decode_event_image(content: bytes, expected_size: list[int]):
    import cv2
    import numpy as np

    if (not isinstance(content, bytes) or not 0 < len(content) <= 8 * 1024 * 1024
            or not isinstance(expected_size, list) or len(expected_size) != 2
            or any(type(value) is not int or not 0 < value <= 16384 for value in expected_size)
            or math.prod(expected_size) > 16_000_000):
        raise RuntimeError("EVENT_IMAGE_INVALID")
    # Inspect the JPEG header before asking the decoder to allocate pixels.
    from PIL import Image
    from io import BytesIO

    with Image.open(BytesIO(content)) as image:
        if image.format != "JPEG" or image.size != tuple(expected_size):
            raise RuntimeError("EVENT_IMAGE_INVALID")
    pixels = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None or pixels.shape != (expected_size[1], expected_size[0], 3):
        raise RuntimeError("EVENT_IMAGE_INVALID")
    return pixels


class TaskEventCollector(TaskFrameCollector):
    """Poll an isolated creation cursor; reuse existing receipt/cleanup rules."""

    MAX_EXISTING_FRAMES = 20000

    def __init__(self, manager, task_binding, taskdata_root, collection_id, *,
                 event_provider, start_cursor: int, event_context: dict,
                 decoder=None, existing_sources: list[dict] | None = None, **kwargs):
        super().__init__(manager, task_binding, taskdata_root, collection_id, **kwargs)
        if (type(start_cursor) is not int or not 0 <= start_cursor <= 2**63 - 1
                or not isinstance(event_context, dict)):
            raise ValueError("invalid event collection identity")
        self._events = event_provider
        self._start_cursor = self._cursor = start_cursor
        self._event_context = deepcopy(event_context)
        self._decoder = decoder or _decode_event_image
        self._throttled = 0
        self._start_failed = False
        self._finalize_attempted = False
        self._accepted_frames: dict[tuple[str, str, int, int], str] = {}
        self._duplicate_frames = 0
        if existing_sources is not None and not isinstance(existing_sources, list):
            raise ValueError("existing event sources must be a list")
        if len(existing_sources or []) > self.MAX_EXISTING_FRAMES:
            raise RuntimeError("EVENT_COLLECTION_HISTORY_LIMIT")
        for source in existing_sources or []:
            try:
                validate_event_source_identity(source)
            except ValueError:
                raise RuntimeError("EVENT_COLLECTION_HISTORY_INVALID") from None
            if source["camera_id"] != self.binding["camera_id"]:
                raise RuntimeError("EVENT_COLLECTION_HISTORY_INVALID")
            key, identity = self._frame_identity(source)
            previous = self._accepted_frames.get(key)
            if previous is not None and previous != identity:
                raise RuntimeError("EVENT_FRAME_IDENTITY_CHANGED")
            self._accepted_frames[key] = identity
        self._existing_frame_count = len(self._accepted_frames)

    @staticmethod
    def _frame_identity(source: dict) -> tuple[tuple[str, str, int, int], str]:
        return ((source["camera_id"], source["source_session_id"], source["source_epoch"], source["frame_index"]),
                json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))

    @staticmethod
    def _thread_state(thread) -> str:
        if thread is None:
            return "absent"
        try:
            alive, started, ident = thread.is_alive(), thread._started.is_set(), thread.ident
        except Exception:
            return "unknown"
        if type(alive) is not bool or type(started) is not bool:
            return "unknown"
        if ident is None:
            return "not_started" if alive is False and started is False else "unknown"
        if type(ident) is not int or ident <= 0 or started is not True:
            return "unknown"
        return "running" if alive else "stopped"

    def _cleanup(self) -> bool:
        if (self._thread is not threading.current_thread()
                and self._thread_state(self._thread) not in {"absent", "not_started", "stopped"}):
            with self._lock:
                self._cleanup_confirmed = False
            return False
        return super()._cleanup()

    def stop(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        thread = self._thread
        state = self._thread_state(thread)
        if state == "unknown":
            with self._lock:
                self._error_code, self._cleanup_confirmed = "COLLECTION_WORKER_STATE_UNKNOWN", False
            return False
        if state == "not_started":
            with self._lock:
                if self._thread is thread:
                    self._thread = None
        elif state in {"running", "stopped"}:
            if thread is threading.current_thread():
                return False
            try:
                thread.join(timeout=max(0.0, timeout))
            except Exception:
                return False
            if self._thread_state(thread) != "stopped":
                return False
        complete = self._cleanup()
        if complete:
            with self._lock:
                # Unknown start failures can later be positively identified as
                # never started/exited. Preserve that recovery path for the
                # workflow's ordinary stop()/retry_finalize() protocol.
                if self._start_failed and self._finish_snapshot is None:
                    self._state, self._error_code = "failed", "COLLECTION_START_FAILED"
                    self._finish_snapshot = self.snapshot()
        return complete

    def retry_finalize(self) -> bool:
        # Only the worker's own final callback, or a positively quiescent
        # external observer, may replay the immutable receipt.
        if (self._thread is not threading.current_thread()
                and self._thread_state(self._thread) not in {"absent", "not_started", "stopped"}):
            return False
        with self._lock:
            if self._finish_snapshot is None:
                return False
            if self._finish_snapshot.get("cleanup_confirmed") is not True:
                # No callback is ever attempted with an unconfirmed cleanup.
                # Its first, unsent receipt may gain the later positive proof;
                # after any DB attempt the exact original receipt is frozen.
                if self._finalize_attempted or self._cleanup_confirmed is not True:
                    return False
                self._finish_snapshot = {**self._finish_snapshot, "cleanup_confirmed": True}
            self._finalize_attempted = True
        return super().retry_finalize()

    def snapshot(self) -> dict:
        with self._lock:
            result = {**super().snapshot(), "policy": "events", "kind": "event_keyframes",
                      "start_cursor": self._start_cursor, "last_cursor": self._cursor,
                      "event_context": deepcopy(self._event_context), "throttled_events": self._throttled,
                      "duplicate_frame_events": self._duplicate_frames,
                      "existing_frame_count": self._existing_frame_count}
            if self._thread is not threading.current_thread() and self._thread_state(self._thread) == "unknown":
                result["cleanup_confirmed"] = False
                result["state"], result["error_code"] = "cleanup_pending", "COLLECTION_WORKER_STATE_UNKNOWN"
            return result

    def start(self) -> None:
        with self._lock:
            if self._state != "created":
                raise RuntimeError("a collection can only start once")
            if self._events.collection_context_current(self.binding, self._event_context) is not True:
                raise RuntimeError("EVENT_EXECUTION_CHANGED")
            directory = controlled_path(self._root, f"{self._task_id}/{self.collection_id}")
            directory.mkdir(parents=True, exist_ok=True)
            controlled_path(self._root, f"{self._task_id}/{self.collection_id}")
            self._state = "collecting"
            self._thread = threading.Thread(target=self._run, name=f"event-collect-{self.collection_id}", daemon=True)
            try:
                self._thread.start()
            except Exception:
                self._start_failed = True
                self._state, self._error_code = "failed", "WORKER_START_FAILED"
                raise

    def finish_start_failure(self) -> bool:
        # Thread.start() can raise after creating its OS thread. Never publish
        # a terminal receipt/cleanup proof while that owner can still write.
        with self._lock:
            self._start_failed = True
        if not self.stop():
            return False
        return self.retry_finalize()

    def _validate_evidence(self, entry: dict, evidence: dict) -> dict:
        """Do not let a malformed provider write an uncommittable PNG first."""
        payload = evidence["payload"]
        source = deepcopy(evidence["source"])
        validate_event_source_identity(source)
        expected = json.dumps({"payload": payload, "evidence_sha256": evidence["evidence_sha256"]},
                              sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        size = evidence["image_size"]
        validate_event_sample_origin({"schema": "argus.task-event-sample.v1", "event_id": entry["event_id"],
            "immutable_sha256": entry["immutable_sha256"], "evidence_sha256": evidence["evidence_sha256"],
            "derivation": "verified_jpeg_to_png.v1", "payload": payload}, source, size, self.binding)
        if (not isinstance(entry["event_id"], str) or not entry["event_id"]
                or len(entry["event_id"]) > 64 or payload["event_id"] != entry["event_id"]
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value)
                       for value in (entry["immutable_sha256"], evidence["evidence_sha256"]))
                or hashlib.sha256(expected.encode()).hexdigest() != entry["immutable_sha256"]
                or not same_json_identity(payload["source"], source)
                or source["camera_id"] != self.binding["camera_id"]
                or any(not same_json_identity(payload["binding"].get(key), value) for key, value in self.binding.items())
                or any(not same_json_identity((source if key in {"source_session_id", "source_epoch"}
                                               else payload["binding"]).get(key), value)
                       for key, value in self._event_context.items())
                or not isinstance(size, list) or len(size) != 2
                or any(type(value) is not int or not 0 < value <= 16384 for value in size)
                or math.prod(size) > 16_000_000
                or not same_json_identity(size, [payload["frame_size"]["width"], payload["frame_size"]["height"]])):
            raise RuntimeError("EVENT_IMAGE_INVALID")
        return source

    def _run(self) -> None:
        deadline = time.monotonic() + self._max_duration
        last_accepted = float("-inf")
        try:
            while not self._stop.is_set() and time.monotonic() < deadline:
                if len(self._samples) >= self._max_samples:
                    break
                if self._events.collection_context_current(self.binding, self._event_context) is not True:
                    raise RuntimeError("EVENT_EXECUTION_CHANGED")
                batch = self._events.collection_events(self.binding, self._cursor,
                                                       limit=min(self._capacity, 16), context=self._event_context)
                if not batch:
                    self._stop.wait(min(0.1, max(0.0, deadline - time.monotonic())))
                    continue
                for entry in batch:
                    if (self._stop.is_set() or time.monotonic() >= deadline
                            or len(self._samples) >= self._max_samples):
                        break
                    cursor = entry["cursor"]
                    if type(cursor) is not int or cursor <= self._cursor:
                        raise RuntimeError("EVENT_CURSOR_INVALID")
                    evidence = self._events.collection_evidence(
                        self.binding, entry["event_id"], entry["immutable_sha256"], context=self._event_context)
                    source = self._validate_evidence(entry, evidence)
                    captured = source.get("capture_monotonic_ts")
                    if type(captured) not in (int, float) or not math.isfinite(captured):
                        raise RuntimeError("EVENT_IMAGE_INVALID")
                    frame_key, frame_identity = self._frame_identity(source)
                    prior_frame = self._accepted_frames.get(frame_key)
                    if prior_frame is not None:
                        if prior_frame != frame_identity:
                            raise RuntimeError("EVENT_FRAME_IDENTITY_CHANGED")
                        with self._lock:
                            self._duplicate_frames += 1
                            self._cursor = cursor
                        continue
                    if captured <= last_accepted:
                        raise RuntimeError("EVENT_FRAME_IDENTITY_CHANGED")
                    if captured - last_accepted < self._interval:
                        with self._lock:
                            self._throttled += 1
                            self._cursor = cursor
                        continue
                    content = evidence["bytes"]
                    if (evidence["mime"] != "image/jpeg" or not isinstance(content, bytes)
                            or hashlib.sha256(content).hexdigest() != evidence["evidence_sha256"]):
                        raise RuntimeError("EVENT_IMAGE_INVALID")
                    pixels = self._decoder(content, evidence["image_size"])
                    import numpy as np

                    if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.dtype("uint8")
                            or pixels.shape != (evidence["image_size"][1], evidence["image_size"][0], 3)):
                        raise RuntimeError("EVENT_IMAGE_INVALID")
                    encoded = self._encoder(pixels)
                    if (not isinstance(encoded, bytes) or not encoded
                            or list(png_image_size(encoded)) != evidence["image_size"]):
                        raise RuntimeError("IMAGE_ENCODING_FAILED")
                    if self._stop.is_set():
                        break
                    if self._events.collection_context_current(self.binding, self._event_context) is not True:
                        raise RuntimeError("EVENT_EXECUTION_CHANGED")
                    # Acceptance is a historical observation, not a distributed
                    # transaction with camera lifecycle. A later stop cannot
                    # rewrite this exact committed event into a newer one.
                    sample_id = uuid.uuid5(uuid.NAMESPACE_URL,
                                          f"argus:event-sample:{self.collection_id}:{entry['event_id']}").hex
                    relative = f"{self._task_id}/{self.collection_id}/{sample_id}.png"
                    self._write_image(relative, encoded)
                    for key in ("source_declared", "source_runtime"):
                        source.pop(key, None)
                    sample = {"sample_id": sample_id, "collection_id": self.collection_id,
                              "relative_path": relative, "image_sha256": hashlib.sha256(encoded).hexdigest(),
                              "byte_size": len(encoded), "identity": source,
                              "image_size": deepcopy(evidence["image_size"]), "binding": deepcopy(self.binding),
                              "auto_annotation": None,
                              "event_origin": {"schema": "argus.task-event-sample.v1",
                                               "event_id": entry["event_id"],
                                               "immutable_sha256": entry["immutable_sha256"],
                                               "evidence_sha256": evidence["evidence_sha256"],
                                               "derivation": "verified_jpeg_to_png.v1",
                                               "payload": deepcopy(evidence["payload"])}}
                    with self._lock:
                        self._samples.append(sample)
                        self._accepted_frames[frame_key] = frame_identity
                        self._cursor = cursor
                    last_accepted = captured
            with self._lock:
                self._state = "stopped" if self._stop.is_set() else "completed"
        except Exception as exc:
            allowed = {"EVENT_EXECUTION_CHANGED", "EVENT_CURSOR_INVALID", "EVENT_IMAGE_INVALID", "EVENT_FRAME_IDENTITY_CHANGED",
                       "IMAGE_ENCODING_FAILED", "IMAGE_WRITE_FAILED", "IMAGE_TOO_LARGE", "STORAGE_LIMIT_REACHED"}
            with self._lock:
                self._state, self._error_code = "failed", str(exc) if str(exc) in allowed else "EVENT_COLLECTION_FAILED"
        finally:
            self._cleanup()
            with self._lock:
                self._finish_snapshot = self.snapshot()
            self.retry_finalize()
