"""YOLO-based person detection for frame filtering.

Detects people in frames so they can be masked out or the frame can be
skipped entirely, preventing human operators from triggering false anomaly
alerts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()

# COCO class index for "person"
_PERSON_CLASS = 0


@dataclass
class PersonDetection:
    """A detected person bounding box."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass
class PersonFilterResult:
    """Result of the person filtering stage."""

    persons: list[PersonDetection]
    has_persons: bool
    masked_frame: np.ndarray | None = None  # frame with persons blacked out
    filter_available: bool = True  # False when YOLO model failed to load


class YOLOPersonDetector:
    """Detects persons using YOLO and optionally masks them out.

    Uses Ultralytics YOLO for real-time person detection. In mask mode,
    detected person regions are blacked out so the anomaly detector
    doesn't flag them as anomalies.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.4,
        skip_frame_on_person: bool = False,
        mask_padding: int = 20,
    ):
        self.confidence = confidence
        self.skip_frame_on_person = skip_frame_on_person
        self.mask_padding = mask_padding
        self._model = None
        self._model_name = model_name
        self._available: bool | None = None  # None = not yet attempted

    def _ensure_model(self):
        """Lazy-load the YOLO model on first use. Degrades gracefully on failure."""
        if self._available is not None:
            return  # Already attempted (success or failure)
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_name)
            self._available = True
            logger.info("yolo.loaded", model=self._model_name)
        except Exception as e:
            self._available = False
            logger.error(
                "yolo.unavailable",
                error=str(e),
                msg="Person filtering OFFLINE — all frames will be analyzed unfiltered",
            )

    def detect(self, frame: np.ndarray) -> PersonFilterResult:
        """Detect persons in the frame.

        Returns empty result if YOLO is unavailable (graceful degradation).
        """
        self._ensure_model()

        if not self._available:
            return PersonFilterResult(persons=[], has_persons=False, filter_available=False)

        results = self._model.predict(
            frame,
            classes=[_PERSON_CLASS],
            conf=self.confidence,
            verbose=False,
        )

        persons = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                persons.append(PersonDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf))

        has_persons = len(persons) > 0
        masked_frame = None

        if has_persons and not self.skip_frame_on_person:
            # Create a copy and black out person regions
            masked_frame = frame.copy()
            h, w = frame.shape[:2]
            for p in persons:
                px1 = max(0, p.x1 - self.mask_padding)
                py1 = max(0, p.y1 - self.mask_padding)
                px2 = min(w, p.x2 + self.mask_padding)
                py2 = min(h, p.y2 + self.mask_padding)
                masked_frame[py1:py2, px1:px2] = 0

        return PersonFilterResult(
            persons=persons,
            has_persons=has_persons,
            masked_frame=masked_frame,
        )
