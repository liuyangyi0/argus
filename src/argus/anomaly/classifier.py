"""Open vocabulary detection (OVD) for anomaly classification.

Uses YOLO-World to classify detected anomaly regions into semantic categories,
enabling risk-based severity adjustment (e.g., wrench=high, shadow=suppress).

Supports runtime vocabulary updates and vocabulary persistence for
industrial FOE (Foreign Object Exclusion) items.
"""

from __future__ import annotations

import json
import math
import threading
from numbers import Real
from pathlib import Path

import structlog

from argus.anomaly.auxiliary_runtime import (
    AuxiliaryBusyError,
    AuxiliaryUnavailableError,
    BoundedAuxiliaryRuntime,
)

logger = structlog.get_logger()

# Default FOE vocabulary for industrial facility environments
FOE_VOCAB = [
    # Tools
    "wrench", "bolt", "nut", "screwdriver", "hammer",
    # Materials & personal items
    "rag", "glove", "plastic bag", "tape", "wire",
    "insulation", "debris", "paint chip",
    # industrial FOE items
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

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        vocabulary: list[str] | None = None,
        timeout_seconds: float = 10.0,
        runtime_key: str | None = None,
    ):
        self._model = None
        self._model_name = model_name
        self._vocabulary = self._validate_vocabulary(
            FOE_VOCAB if vocabulary is None else vocabulary
        )
        self._loaded = False
        self._load_attempted = False
        self._loading = False
        self._mutating = False
        self._predicting = False
        self._closed = False
        self._device: str | None = None
        self._last_error: str | None = None
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self._timeout_seconds = float(timeout_seconds)
        self._lock = threading.RLock()
        self._runtime = BoundedAuxiliaryRuntime("yoloworld", runtime_key=runtime_key)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load explicitly, returning literal success only for a usable model.

        Both construction and prediction share the same bounded SDK slot.
        A failed attempt can be retried explicitly; lazy classification does
        not repeatedly download/rebuild a failed model on every input frame.
        """
        with self._lock:
            if self._closed:
                return False
            if self._loading or self._mutating or self._predicting or self._runtime.busy:
                return False
            if self._loaded and self._model is not None and self._last_error is None:
                return True
            self._load_attempted = True
            self._loading = True
            vocabulary = list(self._vocabulary)
            try:
                ticket = self._runtime.submit(lambda: self._create_model(vocabulary))
            except Exception as exc:
                self._loading = False
                self._last_error = str(exc)
                return False
        try:
            model, device = self._runtime.wait(ticket, self._timeout_seconds)
            if model is None:
                raise AuxiliaryUnavailableError("YOLO-World did not return a model")
            with self._lock:
                if self._closed:
                    return False
                self._model = model
                self._device = device
                self._loaded = True
                self._last_error = None
            logger.info("classifier.loaded", model=self._model_name, device=device)
            return True
        except Exception as exc:
            with self._lock:
                self._model = None
                self._loaded = False
                self._device = None
                self._last_error = str(exc)
            logger.warning("classifier.load_failed", error=str(exc))
            return False
        finally:
            with self._lock:
                self._loading = False

    def _create_model(self, vocabulary: list[str]):
        from ultralytics import YOLOWorld

        model = YOLOWorld(self._model_name)
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                model.to("cuda:0")
                device = "cuda:0"
        except ImportError:
            pass
        model.set_classes(vocabulary)
        return model, device

    def get_status(self) -> dict:
        """Expose target backend readiness rather than object attachment."""
        with self._lock:
            busy = self._runtime.busy or self._mutating or self._predicting
            cleanup_pending = self._runtime.cleanup_pending
            loaded = self._loaded and self._model is not None and not self._closed
            state = (
                "cleanup_failed" if self._closed and cleanup_pending else
                "stopped" if self._closed else
                "loading" if self._loading else
                "busy" if busy else
                "failed" if self._last_error else
                "ready" if loaded else "unloaded"
            )
            return {
                "state": state,
                "backend": "yoloworld" if loaded else None,
                "loaded": loaded,
                "ready": loaded and not self._loading and not busy and self._last_error is None,
                "degraded": self._last_error is not None or cleanup_pending,
                "reason": self._last_error,
                "busy": busy,
                "device": self._device,
                "cleanup_pending": cleanup_pending,
            }

    def shutdown(self) -> None:
        """Stop accepting calls and discard any late load/prediction result."""
        with self._lock:
            self._closed = True
            self._loaded = False
            self._model = None
            self._device = None
            try:
                self._runtime.shutdown()
            except Exception as exc:
                self._last_error = f"cleanup failed: {exc}"
                raise
            self._last_error = None

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    @property
    def vocabulary(self) -> list[str]:
        """Current vocabulary list."""
        with self._lock:
            return list(self._vocabulary)

    @staticmethod
    def _validate_vocabulary(vocabulary: list[str]) -> list[str]:
        if not isinstance(vocabulary, list) or not vocabulary:
            raise ValueError("vocabulary must be a nonempty list")
        if any(not isinstance(label, str) or not label.strip() for label in vocabulary):
            raise ValueError("vocabulary labels must be nonempty strings")
        return list(dict.fromkeys(label.strip() for label in vocabulary))

    def update_vocabulary(self, new_vocab: list[str]) -> None:
        """Replace the vocabulary at runtime.

        YOLO-World supports hot-swapping the vocabulary via
        ``model.set_classes()`` without reloading the model weights.
        """
        vocabulary = self._validate_vocabulary(new_vocab)
        with self._lock:
            if self._closed:
                raise AuxiliaryUnavailableError("classifier is stopped")
            if self._loading or self._mutating or self._predicting or self._runtime.busy:
                raise AuxiliaryBusyError("classifier has an unfinished SDK call")
            model = self._model
            if model is None:
                self._vocabulary = vocabulary
                return
            ticket = self._runtime.submit(lambda: model.set_classes(vocabulary))
            self._mutating = True
        try:
            self._runtime.wait(ticket, self._timeout_seconds)
            with self._lock:
                if self._closed:
                    raise AuxiliaryUnavailableError("classifier is stopped")
                self._vocabulary = vocabulary
                self._last_error = None
        except Exception as exc:
            # set_classes may have partially mutated SDK state.  Do not use it
            # with an uncommitted vocabulary; a fresh explicit load is required.
            with self._lock:
                self._model = None
                self._loaded = False
                self._last_error = str(exc)
            raise
        finally:
            with self._lock:
                self._mutating = False
        logger.info("classifier.vocabulary_updated", vocab_size=len(vocabulary))

    def add_labels(self, labels: list[str]) -> None:
        """Append new labels to the vocabulary (deduplicated)."""
        self.update_vocabulary(list(dict.fromkeys([*self.vocabulary, *labels])))

    def save_vocabulary(self, path: str | Path) -> None:
        """Persist current vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vocabulary = self.vocabulary
        path.write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("classifier.vocabulary_saved", path=str(path), size=len(vocabulary))

    def load_vocabulary(self, path: str | Path) -> None:
        """Load vocabulary from a JSON file and apply it."""
        path = Path(path)
        if not path.exists():
            logger.warning("classifier.vocabulary_file_not_found", path=str(path))
            raise FileNotFoundError(f"Configured classifier vocabulary does not exist: {path}")
        vocab = json.loads(path.read_text(encoding="utf-8"))
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
        return self._classify(frame, bbox, return_evidence=False)

    def classify_with_evidence(
        self, frame, bbox: tuple[int, int, int, int] | None = None,
    ) -> dict | None:
        """Return semantic evidence with a validated full-frame XYXY box.

        Missing/invalid detector localization is represented by ``bbox=None``;
        a label alone cannot prove that it describes the anomaly being graded.
        """
        return self._classify(frame, bbox, return_evidence=True)

    def _classify(self, frame, bbox, *, return_evidence: bool):
        with self._lock:
            if self._closed:
                raise AuxiliaryUnavailableError("classifier is stopped")
        if frame is None or frame.size == 0:
            return None
        if bbox is not None:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                return None
            height, width = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x+w), min(height, y+h)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame
            x1 = y1 = 0

        if crop.size == 0:
            return None
        with self._lock:
            lazy_load = not self._loaded and not self._load_attempted
        if lazy_load:
            self.load()
        with self._lock:
            if self._loading or self._mutating or self._predicting:
                raise AuxiliaryBusyError("classifier has an unfinished SDK call")
            if self._closed or not self._loaded or self._model is None:
                raise AuxiliaryUnavailableError(self._last_error or "classifier is not loaded")
            model = self._model
            # The caller may reuse/mutate its frame after a timeout.
            crop = crop.copy()
            ticket = self._runtime.submit(lambda: self._predict(
                model, crop, return_evidence=return_evidence, offset=(x1, y1),
            ))
            self._predicting = True
        try:
            result = self._runtime.wait(ticket, self._timeout_seconds)
            with self._lock:
                if self._closed:
                    raise AuxiliaryUnavailableError("classifier is stopped")
                self._last_error = None
            return result
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            logger.warning("classifier.predict_failed", error=str(exc))
            raise
        finally:
            with self._lock:
                self._predicting = False

    @staticmethod
    def _predict(model, crop, *, return_evidence=False, offset=(0, 0)):
        results = model(crop, verbose=False)
        if results and results[0].boxes and len(results[0].boxes) > 0:
            cls_val = results[0].boxes[0].cls
            conf_val = results[0].boxes[0].conf
            cls_value = cls_val.item() if hasattr(cls_val, "item") else cls_val
            confidence_value = conf_val.item() if hasattr(conf_val, "item") else conf_val
            if (
                isinstance(cls_value, bool) or not isinstance(cls_value, Real)
                or not math.isfinite(cls_value) or cls_value < 0
                or int(cls_value) != cls_value
            ):
                raise ValueError("classifier returned an invalid class ID")
            if isinstance(confidence_value, bool) or not isinstance(confidence_value, Real):
                raise ValueError("classifier returned an invalid confidence")
            cls_id = int(cls_value)
            confidence = float(confidence_value)
            label = results[0].names[cls_id]
            if not isinstance(label, str) or not label.strip():
                raise ValueError("classifier returned an invalid label")
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("classifier returned an invalid confidence")
            if return_evidence:
                location = None
                raw_box = getattr(results[0].boxes[0], "xyxy", None)
                if raw_box is not None:
                    try:
                        values = raw_box.tolist() if hasattr(raw_box, "tolist") else raw_box
                        if isinstance(values, (list, tuple)) and len(values) == 1:
                            values = values[0]
                        if (
                            isinstance(values, (list, tuple)) and len(values) == 4
                            and all(isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v) for v in values)
                        ):
                            bx1, by1, bx2, by2 = map(float, values)
                            height, width = crop.shape[:2]
                            if 0 <= bx1 < bx2 <= width and 0 <= by1 < by2 <= height:
                                location = [bx1 + offset[0], by1 + offset[1], bx2 + offset[0], by2 + offset[1]]
                    except (TypeError, ValueError, OverflowError):
                        pass
                return {"label": label, "confidence": confidence, "bbox": location}
            return label, confidence
        return None
