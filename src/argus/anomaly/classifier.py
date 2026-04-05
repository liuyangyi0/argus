"""Open vocabulary detection (OVD) for anomaly classification.

Uses YOLO-World to classify detected anomaly regions into semantic categories,
enabling risk-based severity adjustment (e.g., wrench=high, shadow=suppress).
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

# Default FOE vocabulary for nuclear plant environments
FOE_VOCAB = [
    "wrench", "bolt", "nut", "screwdriver", "hammer",
    "rag", "glove", "plastic bag", "tape", "wire",
    "insulation", "debris", "paint chip",
    "insect", "bird", "shadow", "reflection",
]


class OpenVocabClassifier:
    """Classifies anomaly regions using YOLO-World open vocabulary detection.

    Requires ultralytics>=8.3.0 with YOLO-World support.
    """

    def __init__(self, model_name: str = "yolov8s-worldv2.pt", vocabulary: list[str] | None = None):
        self._model = None
        self._model_name = model_name
        self._vocabulary = vocabulary or FOE_VOCAB
        self._loaded = False

    def load(self) -> None:
        """Load the YOLO-World model (lazy initialization)."""
        if self._loaded:
            return
        try:
            from ultralytics import YOLOWorld

            self._model = YOLOWorld(self._model_name)
            self._model.set_classes(self._vocabulary)
            self._loaded = True
            logger.info("classifier.loaded", model=self._model_name, vocab_size=len(self._vocabulary))
        except ImportError:
            logger.warning("classifier.yoloworld_not_available", msg="ultralytics YOLOWorld not found")
        except Exception as e:
            logger.error("classifier.load_failed", error=str(e))

    def classify(self, frame, bbox: tuple[int, int, int, int] | None = None) -> tuple[str, float] | None:
        """Classify anomaly region. Returns (label, confidence) or None.

        Args:
            frame: Full frame or cropped region (BGR numpy array).
            bbox: Optional (x, y, w, h) bounding box to crop before classification.

        Returns:
            Tuple of (label, confidence) if detection found, None otherwise.
        """
        if not self._loaded:
            self.load()
        if self._model is None:
            return None

        if bbox is not None:
            x, y, w, h = bbox
            crop = frame[y:y+h, x:x+w]
        else:
            crop = frame

        if crop.size == 0:
            return None

        try:
            results = self._model(crop, verbose=False)
            if results and results[0].boxes and len(results[0].boxes) > 0:
                cls_id = int(results[0].boxes[0].cls)
                conf = float(results[0].boxes[0].conf)
                label = results[0].names[cls_id]
                return label, conf
        except Exception as e:
            logger.error("classifier.predict_failed", error=str(e))

        return None
