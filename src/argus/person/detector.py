"""YOLO-based object detection for frame filtering.

Detects objects in frames using Ultralytics YOLO11n. Primary use case is
filtering out human operators (person masking), but also supports multi-class
COCO detection (YOLO-001) and object tracking (YOLO-002) for semantic
alert context.

YOLO-003: Refactored from YOLOPersonDetector to YOLOObjectDetector.
Backward-compatible aliases maintained for existing code.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()

# COCO class names for nuclear plant relevant classes
COCO_CLASS_NAMES: dict[int, str] = {
    0: "person",
    14: "bird",
    15: "cat",
    16: "dog",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    56: "chair",
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "cell_phone",
    73: "book",
    76: "scissors",
}

# Default: person only (backward compatible)
_DEFAULT_CLASSES = [0]

# DET-012: Shared YOLO model registry — one model instance across all pipelines
_shared_yolo_registry: dict[str, Any] = {}
_registry_lock = threading.Lock()


def get_shared_yolo(model_name: str) -> Any:
    """Get or create a shared YOLO model instance. Thread-safe."""
    with _registry_lock:
        if model_name not in _shared_yolo_registry:
            from ultralytics import YOLO

            model = YOLO(model_name)
            _shared_yolo_registry[model_name] = model
            logger.info("yolo.shared_loaded", model=model_name)
        return _shared_yolo_registry[model_name]


@dataclass
class ObjectDetection:
    """A detected object bounding box with semantic class (YOLO-001)."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = 0
    class_name: str = "person"
    track_id: int | None = None  # YOLO-002: from BoT-SORT tracking


@dataclass
class ObjectDetectionResult:
    """Result of the object detection stage (YOLO-003)."""

    persons: list[PersonDetection] = field(default_factory=list)
    has_persons: bool = False
    objects: list[ObjectDetection] = field(default_factory=list)
    has_objects: bool = False
    non_person_objects: list[ObjectDetection] = field(default_factory=list)
    masked_frame: np.ndarray | None = None  # frame with persons blacked out
    filter_available: bool = True  # False when YOLO model failed to load


# Backward-compatible aliases
PersonDetection = ObjectDetection
PersonFilterResult = ObjectDetectionResult


class YOLOObjectDetector:
    """Multi-class object detector with optional tracking (YOLO-001/002/003).

    Extends the original person-only detector to support:
    - Multi-class COCO detection (configurable class list)
    - BoT-SORT object tracking (persistent track IDs across frames)
    - Gaussian blur masking for person regions (DET-011)
    - Graceful degradation if YOLO unavailable
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.4,
        skip_frame_on_person: bool = False,
        mask_padding: int = 20,
        shared_model: Any | None = None,
        classes_to_detect: list[int] | None = None,
        enable_tracking: bool = False,
    ):
        self.confidence = confidence
        self.skip_frame_on_person = skip_frame_on_person
        self.mask_padding = mask_padding
        self.classes_to_detect = classes_to_detect or _DEFAULT_CLASSES
        self.enable_tracking = enable_tracking
        self._model = None
        self._model_name = model_name
        self._shared_model = shared_model
        self._available: bool | None = None  # None = not yet attempted

    def _ensure_model(self):
        """Lazy-load the YOLO model on first use. Degrades gracefully on failure."""
        if self._available is not None:
            return
        try:
            if self._shared_model is not None:
                self._model = self._shared_model
            else:
                self._model = get_shared_yolo(self._model_name)
            self._available = True
            logger.info(
                "yolo.loaded",
                model=self._model_name,
                classes=self.classes_to_detect,
                tracking=self.enable_tracking,
            )
        except Exception as e:
            self._available = False
            logger.error(
                "yolo.unavailable",
                error=str(e),
                msg="Person filtering OFFLINE — all frames will be analyzed unfiltered",
            )

    def detect(self, frame: np.ndarray) -> ObjectDetectionResult:
        """Detect objects in the frame.

        Returns empty result if YOLO is unavailable (graceful degradation).
        """
        self._ensure_model()

        if not self._available:
            return PersonFilterResult(persons=[], has_persons=False, filter_available=False)

        # YOLO-002: Use tracking if enabled, otherwise standard predict
        if self.enable_tracking:
            results = self._model.track(
                frame,
                classes=self.classes_to_detect,
                conf=self.confidence,
                persist=True,
                verbose=False,
            )
        else:
            results = self._model.predict(
                frame,
                classes=self.classes_to_detect,
                conf=self.confidence,
                verbose=False,
            )

        objects: list[ObjectDetection] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = COCO_CLASS_NAMES.get(class_id, f"class_{class_id}")

                # YOLO-002: Extract track ID if available
                track_id = None
                if self.enable_tracking and box.id is not None:
                    track_id = int(box.id[0])

                objects.append(ObjectDetection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                    track_id=track_id,
                ))

        # Split into persons vs non-person objects
        persons = [o for o in objects if o.class_id == 0]
        non_person = [o for o in objects if o.class_id != 0]
        has_persons = len(persons) > 0
        has_objects = len(objects) > 0

        # Mask person regions (blur, not non-person objects)
        masked_frame = None
        if has_persons and not self.skip_frame_on_person:
            masked_frame = frame.copy()
            h, w = frame.shape[:2]
            for p in persons:
                px1 = max(0, p.x1 - self.mask_padding)
                py1 = max(0, p.y1 - self.mask_padding)
                px2 = min(w, p.x2 + self.mask_padding)
                py2 = min(h, p.y2 + self.mask_padding)
                # DET-011: Gaussian blur instead of black fill
                region = masked_frame[py1:py2, px1:px2]
                if region.size > 0:
                    ksize = max(51, ((min(px2 - px1, py2 - py1) // 2) | 1))
                    masked_frame[py1:py2, px1:px2] = cv2.GaussianBlur(
                        region, (ksize, ksize), 0
                    )

        return ObjectDetectionResult(
            objects=objects,
            has_objects=has_objects,
            has_persons=has_persons,
            persons=persons,
            non_person_objects=non_person,
            masked_frame=masked_frame,
        )


# Backward-compatible alias
YOLOPersonDetector = YOLOObjectDetector
