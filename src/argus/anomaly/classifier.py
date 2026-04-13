"""Open vocabulary detection (OVD) for anomaly classification.

Uses YOLO-World to classify detected anomaly regions into semantic categories,
enabling risk-based severity adjustment (e.g., wrench=high, shadow=suppress).

Supports runtime vocabulary updates and vocabulary persistence for
nuclear-specific FOE (Foreign Object Exclusion) items.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Default FOE vocabulary for nuclear plant environments
FOE_VOCAB = [
    # Tools
    "wrench", "bolt", "nut", "screwdriver", "hammer",
    # Materials & personal items
    "rag", "glove", "plastic bag", "tape", "wire",
    "insulation", "debris", "paint chip",
    # Nuclear-specific FOE items
    "gasket", "o_ring", "cotter_pin", "safety_wire", "washer",
    "hose_clamp", "concrete_chip", "paint_flake", "metal_shaving",
    "fiber", "cloth_fragment", "graphite_gasket", "rubber_seal",
    "electrode_fragment", "zip_tie", "insulation_piece",
    "bearing_ball", "pen_cap", "lens_piece", "lubricant",
    "rivet", "screw", "rebar", "glass", "eyeglass_cord",
    # Noise / suppress classes
    "insect", "bird", "shadow", "reflection",
    # Interference objects (for suppression)
    "crane", "overhead_bridge", "scaffold",
]


class OpenVocabClassifier:
    """Classifies anomaly regions using YOLO-World open vocabulary detection.

    Requires ultralytics>=8.3.0 with YOLO-World support.

    Supports runtime vocabulary updates via :meth:`update_vocabulary` and
    vocabulary persistence to JSON via :meth:`save_vocabulary` /
    :meth:`load_vocabulary`.
    """

    def __init__(self, model_name: str = "yolov8s-worldv2.pt", vocabulary: list[str] | None = None):
        self._model = None
        self._model_name = model_name
        self._vocabulary = list(vocabulary or FOE_VOCAB)
        self._loaded = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    @property
    def vocabulary(self) -> list[str]:
        """Current vocabulary list."""
        return list(self._vocabulary)

    def update_vocabulary(self, new_vocab: list[str]) -> None:
        """Replace the vocabulary at runtime.

        YOLO-World supports hot-swapping the vocabulary via
        ``model.set_classes()`` without reloading the model weights.
        """
        self._vocabulary = list(new_vocab)
        if self._model is not None:
            try:
                self._model.set_classes(self._vocabulary)
                logger.info("classifier.vocabulary_updated", vocab_size=len(self._vocabulary))
            except Exception as e:
                logger.error("classifier.vocabulary_update_failed", error=str(e))

    def add_labels(self, labels: list[str]) -> None:
        """Append new labels to the vocabulary (deduplicated)."""
        existing = set(self._vocabulary)
        added = [l for l in labels if l not in existing]
        if added:
            self._vocabulary.extend(added)
            if self._model is not None:
                self._model.set_classes(self._vocabulary)
            logger.info("classifier.labels_added", added=added, total=len(self._vocabulary))

    def save_vocabulary(self, path: str | Path) -> None:
        """Persist current vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._vocabulary, ensure_ascii=False, indent=2))
        logger.info("classifier.vocabulary_saved", path=str(path), size=len(self._vocabulary))

    def load_vocabulary(self, path: str | Path) -> None:
        """Load vocabulary from a JSON file and apply it."""
        path = Path(path)
        if not path.exists():
            logger.warning("classifier.vocabulary_file_not_found", path=str(path))
            return
        vocab = json.loads(path.read_text())
        self.update_vocabulary(vocab)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

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
                cls_val = results[0].boxes[0].cls
                conf_val = results[0].boxes[0].conf
                cls_id = int(cls_val.item() if hasattr(cls_val, 'item') else cls_val)
                conf = float(conf_val.item() if hasattr(conf_val, 'item') else conf_val)
                label = results[0].names[cls_id]
                return label, conf
        except Exception as e:
            logger.error("classifier.predict_failed", error=str(e))

        return None
