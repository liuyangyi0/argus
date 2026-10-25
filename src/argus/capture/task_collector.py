"""Bounded image-only collection from an already manager-owned source.

The worker never opens a camera, runs recognition, changes pipeline mode, or
rewrites video. Encoding and exclusive file writes run on this dedicated
thread, never under the source result-commit lock.
"""

from __future__ import annotations

import hashlib
import copy
from contextlib import contextmanager
import math
import os
import re
import threading
import time
import uuid
import struct
import zlib
from pathlib import Path
from typing import Callable

from argus.capture.source_runtime import CameraSourceRuntime


_STORAGE_LOCK = threading.Lock()


@contextmanager
def _storage_budget(root: Path, task_id: str):
    """Serialize byte-budget check/write across processes as well as threads."""
    with _STORAGE_LOCK:
        path = controlled_path(root, f"{task_id}/collection-budget.lock")
        with path.open("a+b") as stream:
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def safe_component(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("invalid task data identifier")
    if value in {".", ".."}:
        raise ValueError("invalid task data identifier")
    return value


def controlled_path(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    """Reject traversal and links, including Windows separators on POSIX."""
    relative = relative.replace("\\", "/")
    parts = relative.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("invalid task image path")
    def linked(path: Path) -> bool:
        try:
            return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)
        except FileNotFoundError:
            return False

    if linked(root):
        raise ValueError("task data links are not allowed")
    root = root.resolve()
    candidate = root
    for part in parts:
        safe_component(part)
        candidate = candidate / part
        if linked(candidate):
            raise ValueError("task data links are not allowed")
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("task image escaped its data root")
    if must_exist and not candidate.is_file():
        raise ValueError("task image is missing")
    return candidate


def _encode_png(pixels) -> bytes:
    import cv2

    success, encoded = cv2.imencode(".png", pixels)
    if not success:
        raise RuntimeError("image encoding failed")
    return encoded.tobytes()


def png_image_size(payload: bytes) -> tuple[int, int]:
    """Inspect bounded PNG structure and IHDR without invoking a media decoder."""
    if len(payload) < 45 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("task image must be PNG")
    offset, size, saw_data, ended = 8, None, False, False
    image_data = []
    while offset + 12 <= len(payload):
        length = struct.unpack_from(">I", payload, offset)[0]
        kind = payload[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(payload):
            raise ValueError("truncated task PNG")
        content = payload[offset + 8:end - 4]
        crc = struct.unpack_from(">I", payload, end - 4)[0]
        if zlib.crc32(kind + content) != crc:
            raise ValueError("task PNG checksum mismatch")
        if size is None:
            if kind != b"IHDR" or length != 13:
                raise ValueError("task PNG requires IHDR")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", content)
            if not 0 < width <= 16384 or not 0 < height <= 16384 or width * height > 64_000_000:
                raise ValueError("task PNG dimensions exceed their budget")
            if depth != 8 or color not in {0, 2, 4, 6} or (compression, filtering, interlace) != (0, 0, 0):
                raise ValueError("unsupported task PNG format")
            channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
            size = (width, height)
        elif kind == b"IHDR":
            raise ValueError("duplicate task PNG header")
        if kind == b"IDAT":
            saw_data = True
            image_data.append(content)
        if kind == b"IEND":
            ended = length == 0 and end == len(payload)
            break
        offset = end
    if not ended or not saw_data:
        raise ValueError("incomplete task PNG")
    # Bound both expansion and temporary allocations. No image/video SDK is
    # invoked, and invalid filters/truncated/concatenated zlib streams fail.
    stride, seen = size[0] * channels + 1, 0
    expected = size[1] * stride
    decoder = zlib.decompressobj()
    try:
        for compressed in image_data:
            remaining = compressed
            while remaining:
                block = decoder.decompress(remaining, 65536)
                remaining = decoder.unconsumed_tail
                for index in range((-seen) % stride, len(block), stride):
                    if block[index] > 4:
                        raise ValueError("invalid task PNG row filter")
                seen += len(block)
                if seen > expected or decoder.unused_data:
                    raise ValueError("task PNG expansion exceeds declared dimensions")
                if not block and remaining:
                    raise ValueError("invalid task PNG compressed stream")
        if not decoder.eof or seen != expected:
            raise ValueError("incomplete task PNG pixel data")
    except zlib.error as exc:
        raise ValueError("invalid task PNG compressed data") from exc
    return size


class TaskFrameCollector:
    """One bounded collection; results are images, not approved labels."""

    def __init__(
        self, manager, task_binding: dict, taskdata_root: str | Path, collection_id: str,
        *, max_samples: int = 100, max_duration: float = 60.0,
        interval: float = 1.0, queue_capacity: int = 8,
        max_storage_mb: int = 1024,
        encoder: Callable | None = None, on_finish: Callable | None = None,
    ) -> None:
        if type(max_samples) is not int or not 1 <= max_samples <= 10000:
            raise ValueError("max_samples must be between 1 and 10000")
        if type(max_duration) not in (int, float) or not math.isfinite(max_duration) or not 0 < max_duration <= 3600:
            raise ValueError("max_duration must be positive and at most 3600 seconds")
        if type(interval) not in (int, float) or not math.isfinite(interval) or not 0 <= interval <= 86400:
            raise ValueError("invalid sample interval")
        if type(queue_capacity) is not int or not 1 <= queue_capacity <= 64:
            raise ValueError("invalid collection queue capacity")
        if type(max_storage_mb) is not int or not 1 <= max_storage_mb <= 102400:
            raise ValueError("invalid collection storage limit")
        self.binding = dict(task_binding)
        for field in ("task_id", "camera_id", "template_id", "task_revision", "task_config_sha256"):
            if field not in self.binding:
                raise ValueError("incomplete task collection binding")
        self.collection_id = safe_component(collection_id)
        self._task_id = safe_component(self.binding["task_id"])
        self._manager = manager
        self._root = Path(taskdata_root).resolve()
        self._max_samples = max_samples
        self._max_duration = float(max_duration)
        self._interval = float(interval)
        self._capacity = queue_capacity
        self._storage_limit = max_storage_mb * 1024 * 1024
        self._encoder = encoder or _encode_png
        self._on_finish = on_finish
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._subscription = None
        self._source: CameraSourceRuntime | None = None
        self._state = "created"
        self._error_code: str | None = None
        self._samples: list[dict] = []
        self._discarded = 0
        self._dropped = 0
        self._subscription_name = f"collector:{self._task_id}:{self.collection_id}"
        self._finish_snapshot: dict | None = None
        self._finish_committed = False
        self._cleanup_confirmed = False

    def start(self) -> None:
        with self._lock:
            if self._state != "created":
                raise RuntimeError("a collection can only start once")
            source = self._manager.get_camera_source(self.binding["camera_id"])
            if not isinstance(source, CameraSourceRuntime):
                raise RuntimeError("manager-owned source is unavailable")
            directory = controlled_path(self._root, f"{self._task_id}/{self.collection_id}")
            directory.mkdir(parents=True, exist_ok=True)
            controlled_path(self._root, f"{self._task_id}/{self.collection_id}")
            self._source = source
            self._subscription = source.subscribe(
                self._subscription_name, mode="ordered_window", capacity=self._capacity,
            )
            self._state = "collecting"
            self._thread = threading.Thread(target=self._run, name=f"task-collect-{self.collection_id}", daemon=True)
            try:
                self._thread.start()
            except Exception:
                source.hub.unsubscribe(self._subscription_name, subscription=self._subscription)
                self._state, self._error_code = "failed", "WORKER_START_FAILED"
                raise

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "collection_id": self.collection_id, "state": self._state,
                "error_code": self._error_code, "binding": dict(self.binding),
                "samples": copy.deepcopy(self._samples),
                "discarded_frames": self._discarded,
                "dropped_frames": self._subscription.status.dropped_frames if self._subscription else self._dropped,
                "persistence_pending": self._finish_snapshot is not None and not self._finish_committed,
                "cleanup_confirmed": self._cleanup_confirmed,
            }

    def finish_start_failure(self) -> bool:
        with self._lock:
            self._state, self._error_code = "failed", "COLLECTION_START_FAILED"
        self._cleanup()
        with self._lock:
            self._finish_snapshot = self.snapshot()
        return self.retry_finalize()

    def _cleanup(self) -> bool:
        try:
            if self._source is not None and self._subscription is not None:
                self._source.hub.unsubscribe(self._subscription_name, subscription=self._subscription)
        except Exception:
            return False
        with self._lock:
            self._cleanup_confirmed = True
        return True

    def retry_finalize(self) -> bool:
        """Retry the identical receipt after an unknown database commit outcome."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._thread is not threading.current_thread():
                return False
            if self._finish_committed:
                return True
            receipt = copy.deepcopy(self._finish_snapshot)
        if receipt is None:
            return False
        try:
            if self._on_finish is not None:
                self._on_finish(receipt)
        except Exception:
            with self._lock:
                self._state, self._error_code = "failed", "MANIFEST_COMMIT_FAILED"
            return False
        with self._lock:
            self._finish_committed = True
            self._state, self._error_code = receipt["state"], receipt["error_code"]
        return True

    def _write_image(self, relative: str, encoded: bytes) -> None:
        if len(encoded) > 32 * 1024 * 1024:
            raise RuntimeError("IMAGE_TOO_LARGE")
        # All collector instances share this lock. Count retained and partial
        # task files too, so failed attempts cannot silently evade the budget.
        with _storage_budget(self._root, self._task_id):
            task_directory = controlled_path(self._root, self._task_id)
            used = 0
            for entry in task_directory.rglob("*"):
                checked = controlled_path(self._root, entry.relative_to(self._root).as_posix())
                if checked.is_file():
                    used += checked.stat().st_size
            if used + len(encoded) > self._storage_limit:
                raise RuntimeError("STORAGE_LIMIT_REACHED")
            path = controlled_path(self._root, relative)
            try:
                with path.open("xb") as stream:
                    stream.write(encoded)
                    stream.flush()
            except Exception:
                raise RuntimeError("IMAGE_WRITE_FAILED") from None

    def stop(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        if self._subscription is not None:
            try:
                self._subscription.close()
            except Exception:
                # Still join the worker and ask the source to release this
                # exact subscription. A failed first close must not prevent
                # cleanup retries or make stop skip ownership confirmation.
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        if thread is not None and thread.is_alive():
            return False
        return self._cleanup()

    def _run(self) -> None:
        deadline = time.monotonic() + self._max_duration
        last_accepted = float("-inf")
        assert self._subscription is not None and self._source is not None
        try:
            while not self._stop.is_set() and time.monotonic() < deadline:
                if len(self._samples) >= self._max_samples:
                    break
                frame = self._subscription.get(min(0.1, max(0.0, deadline - time.monotonic())))
                if frame is None:
                    continue
                now = time.monotonic()
                if now - last_accepted < self._interval:
                    continue
                identity = frame.identity
                if (identity is None or not identity.source_session_id
                        or identity.source_epoch < 1 or frame.camera_id != self.binding["camera_id"]):
                    raise RuntimeError("SOURCE_IDENTITY_MISSING")
                if self._manager.get_camera_source(self.binding["camera_id"]) is not self._source:
                    raise RuntimeError("SOURCE_REPLACED")
                token = (identity.source_session_id, identity.source_epoch)
                encoded = self._encoder(frame.frame)
                if not isinstance(encoded, bytes) or not encoded:
                    raise RuntimeError("IMAGE_ENCODING_FAILED")
                try:
                    if png_image_size(encoded) != (frame.frame.shape[1], frame.frame.shape[0]):
                        raise ValueError("image dimensions mismatch")
                except ValueError:
                    raise RuntimeError("IMAGE_ENCODING_FAILED") from None
                if self._stop.is_set():
                    break
                accepted: list[dict] = []

                def accept():
                    accepted.append(identity.to_dict())

                if (self._manager.get_camera_source(self.binding["camera_id"]) is not self._source
                        or not self._source.commit_if_current(token, accept)):
                    with self._lock:
                        self._discarded += 1
                    continue
                # The receipt is linearized above; a later reconnect does not
                # erase this historical frame. File I/O never holds source locks.
                sample_id = uuid.uuid4().hex
                relative = f"{self._task_id}/{self.collection_id}/{sample_id}.png"
                self._write_image(relative, encoded)
                source_identity = accepted[0]
                # URLs/source credentials are deliberately not exported.
                for key in ("source_declared", "source_runtime"):
                    source_identity.pop(key, None)
                sample = {
                    "sample_id": sample_id, "collection_id": self.collection_id,
                    "relative_path": relative, "image_sha256": hashlib.sha256(encoded).hexdigest(),
                    "byte_size": len(encoded), "identity": source_identity,
                    "image_size": [int(frame.frame.shape[1]), int(frame.frame.shape[0])],
                    "binding": dict(self.binding), "auto_annotation": None,
                }
                with self._lock:
                    self._samples.append(sample)
                last_accepted = now
            with self._lock:
                self._state = "stopped" if self._stop.is_set() else "completed"
        except Exception as exc:
            allowed = {"SOURCE_IDENTITY_MISSING", "SOURCE_REPLACED", "IMAGE_ENCODING_FAILED", "IMAGE_WRITE_FAILED",
                       "IMAGE_TOO_LARGE", "STORAGE_LIMIT_REACHED"}
            with self._lock:
                self._state = "failed"
                self._error_code = str(exc) if str(exc) in allowed else "COLLECTION_FAILED"
        finally:
            with self._lock:
                self._dropped = self._subscription.status.dropped_frames
            if not self._cleanup():
                with self._lock:
                    self._state, self._error_code = "failed", "SUBSCRIPTION_CLEANUP_FAILED"
            with self._lock:
                self._finish_snapshot = self.snapshot()
            self.retry_finalize()
