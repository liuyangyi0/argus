"""Model health status for the operator dashboard.

Each detector (anomaly, object, simplex, MOG2) holds a :class:`ModelStatus`
instance and updates it as load/inference succeeds or fails. The aggregation
endpoint ``/api/models/status`` walks all cameras and returns these records
so the frontend can show "this camera's anomaly model has been failing for
5 minutes" instead of silently falling back.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class ModelStatus:
    """Live health record for one detector on one camera.

    Attributes whose meaning varies per detector:

    ``backend``
        Anomaly: ``"torch-cuda"``, ``"torch-cpu"``, ``"openvino"``,
        ``"ssim-fallback"``, ``"disabled"``, ``"none"``.
        YOLO: ``"cuda"``, ``"cpu"``, ``"disabled"``.
        Fixed-algorithm detectors (simplex, MOG2): ``"enabled"`` or
        ``"disabled"``.

    ``last_error`` / ``last_error_ts``
        Populated on the most recent failure; cleared only by a subsequent
        success (so the UI can show "last failed 3s ago" next to a currently
        green status, signalling intermittent issues).
    """

    name: str
    camera_id: str
    loaded: bool = False
    backend: str = "none"
    model_path: str | None = None
    image_size: tuple[int, int] | None = None
    last_error: str | None = None
    last_error_ts: float | None = None
    consecutive_failures: int = 0
    total_inferences: int = 0
    total_failures: int = 0
    last_success_ts: float | None = None
    # Static per-detector metadata (not updated at runtime)
    extra: dict = field(default_factory=dict)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # ── State transitions ──

    def mark_loaded(self, backend: str, model_path: str | None = None,
                    image_size: tuple[int, int] | None = None) -> None:
        with self._lock:
            self.loaded = True
            self.backend = backend
            if model_path is not None:
                self.model_path = model_path
            if image_size is not None:
                self.image_size = image_size

    def mark_load_failed(self, error: str) -> None:
        with self._lock:
            self.loaded = False
            self.last_error = error
            self.last_error_ts = time.time()

    def mark_inference_success(self) -> None:
        with self._lock:
            self.total_inferences += 1
            self.consecutive_failures = 0
            self.last_success_ts = time.time()

    def mark_inference_failure(self, error: str) -> None:
        with self._lock:
            self.total_inferences += 1
            self.total_failures += 1
            self.consecutive_failures += 1
            self.last_error = error
            self.last_error_ts = time.time()

    def set_extra(self, **kwargs) -> None:
        with self._lock:
            self.extra.update(kwargs)

    # ── Serialization ──

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "camera_id": self.camera_id,
                "loaded": self.loaded,
                "backend": self.backend,
                "model_path": self.model_path,
                "image_size": list(self.image_size) if self.image_size else None,
                "last_error": self.last_error,
                "last_error_ts": self.last_error_ts,
                "consecutive_failures": self.consecutive_failures,
                "total_inferences": self.total_inferences,
                "total_failures": self.total_failures,
                "last_success_ts": self.last_success_ts,
                "extra": dict(self.extra),
            }
