"""SAM 2 instance segmentation for anomaly regions (D2).

Given anomaly peak points from heatmap analysis, segments individual objects
to provide precise boundaries for anomaly classification and reporting.

Missing/failed SDKs are reported as unavailable rather than normal empty
results.  Contour segmentation is an explicit, visibly degraded opt-in.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np
import structlog

from argus.anomaly.auxiliary_runtime import (
    AuxiliaryBusyError,
    AuxiliaryUnavailableError,
    BoundedAuxiliaryRuntime,
)

logger = structlog.get_logger()


@dataclass
class SegmentedObject:
    """A single segmented object instance."""

    mask: np.ndarray  # binary mask (H, W), dtype=uint8, values 0/255
    area_px: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[int, int]
    confidence: float


@dataclass
class SegmentationResult:
    """Aggregated result from instance segmentation."""

    objects: list[SegmentedObject] = field(default_factory=list)
    num_objects: int = 0
    total_area_px: int = 0
    backend: str = "sam2"
    degraded: bool = False


# ---------------------------------------------------------------------------
# Peak extraction from anomaly heatmap
# ---------------------------------------------------------------------------

def extract_peak_points(
    anomaly_map: np.ndarray,
    max_points: int = 5,
    min_score: float = 0.7,
    min_distance_px: int = 30,
) -> list[tuple[int, int]]:
    """Extract top-N local maxima from an anomaly heatmap.

    Args:
        anomaly_map: 2-D float32 array (H, W), values in [0, 1].
        max_points: Maximum number of peak points to return.
        min_score: Minimum anomaly value for a point to qualify.
        min_distance_px: Minimum pixel distance between two peaks
            (non-maximum suppression radius).

    Returns:
        List of ``(x, y)`` pixel coordinates sorted by descending score.
    """
    if anomaly_map is None or anomaly_map.size == 0:
        return []

    # Ensure 2-D
    amap = anomaly_map.squeeze()
    if amap.ndim != 2:
        return []

    # Dilate to find local maxima (each pixel equals its neighbourhood max)
    kernel_size = max(3, min_distance_px // 2 * 2 + 1)  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(amap.astype(np.float32), kernel)

    # A pixel is a local maximum if it equals its dilated value
    local_max_mask = (amap >= dilated) & (amap >= min_score)
    ys, xs = np.nonzero(local_max_mask)

    if len(ys) == 0:
        return []

    scores = amap[ys, xs]
    order = np.argsort(-scores)  # descending

    # Greedy NMS by distance
    selected: list[tuple[int, int]] = []
    selected_coords: list[tuple[int, int]] = []
    for idx in order:
        if len(selected) >= max_points:
            break
        cx, cy = int(xs[idx]), int(ys[idx])
        too_close = False
        for sx, sy in selected_coords:
            if (cx - sx) ** 2 + (cy - sy) ** 2 < min_distance_px ** 2:
                too_close = True
                break
        if not too_close:
            selected.append((cx, cy))
            selected_coords.append((cx, cy))

    return selected


# ---------------------------------------------------------------------------
# Instance segmenter
# ---------------------------------------------------------------------------

class InstanceSegmenter:
    """SAM 2 based instance segmentation for anomaly regions.

    Requires the ``sam2`` package (``pip install sam2``).
    An explicit ``allow_contour_fallback`` may enable degraded segmentation;
    its result/status must never be represented as loaded SAM 2.
    """

    def __init__(
        self,
        model_size: str = "small",
        min_mask_area_px: int = 100,
        timeout_seconds: float = 10.0,
        allow_contour_fallback: bool = False,
        runtime_key: str | None = None,
    ):
        self._model_size = model_size
        self._min_mask_area_px = min_mask_area_px
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self._timeout_seconds = float(timeout_seconds)
        self._predictor = None
        self._loaded = False
        self._sam2_available = False
        self._allow_contour_fallback = bool(allow_contour_fallback)
        self._fallback_ready = False
        self._load_attempted = False
        self._loading = False
        self._predicting = False
        self._closed = False
        self._last_error: str | None = None
        self._device: str | None = None
        self._lock = threading.RLock()
        self._runtime = BoundedAuxiliaryRuntime("sam2", runtime_key=runtime_key)

    # ------------------------------------------------------------------
    # Runtime parameter updates (D2 hot-reload, stage 2.6)
    # ------------------------------------------------------------------

    def update_runtime_params(
        self,
        *,
        min_mask_area_px: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """Patch runtime-tunable parameters in place.

        Only ``min_mask_area_px`` and ``timeout_seconds`` are reload-safe
        here — ``model_size`` would require re-downloading weights and
        resetting the predictor, which we intentionally do NOT support at
        runtime (operators should restart the pipeline for that).

        Note that ``max_points`` and ``min_anomaly_score`` belong to the
        **pipeline's** peak-extraction step, not to this class, so they
        are pushed onto the pipeline separately (see
        ``CameraManager.update_segmenter_params``).
        """
        if min_mask_area_px is not None and min_mask_area_px <= 0:
            raise ValueError("min_mask_area_px must be positive")
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        with self._lock:
            if self._closed:
                raise AuxiliaryUnavailableError("segmenter is stopped")
            if self._loading or self._predicting or self._runtime.busy:
                raise AuxiliaryBusyError("segmenter has an unfinished SDK call")
            if min_mask_area_px is not None:
                self._min_mask_area_px = int(min_mask_area_px)
            if timeout_seconds is not None:
                self._timeout_seconds = float(timeout_seconds)
        logger.info(
            "segmenter.runtime_params_updated",
            min_mask_area_px=self._min_mask_area_px,
            timeout_seconds=self._timeout_seconds,
        )

    def load(self) -> bool:
        """Load SAM 2 within the same one-call budget used by inference.

        Only the actual SAM 2 predictor returns True.  A timeout/failure cannot
        attach a late predictor, and an explicit subsequent call may retry.
        """
        with self._lock:
            if self._closed:
                return False
            if self._loading or self._predicting or self._runtime.busy:
                return False
            if (
                self._loaded and self._sam2_available and self._predictor is not None
                and self._last_error is None
            ):
                return True
            self._load_attempted = True
            self._loading = True
            try:
                ticket = self._runtime.submit(self._create_predictor)
            except Exception as exc:
                self._loading = False
                self._last_error = str(exc)
                return False
        try:
            predictor, device = self._runtime.wait(ticket, self._timeout_seconds)
            if predictor is None:
                raise AuxiliaryUnavailableError("SAM 2 did not return a predictor")
            with self._lock:
                if self._closed:
                    return False
                self._predictor = predictor
                self._sam2_available = True
                self._loaded = True
                self._fallback_ready = False
                self._device = device
                self._last_error = None
            logger.info(
                "segmenter.sam2_loaded",
                model_size=self._model_size,
                device=device,
            )
            return True
        except Exception as exc:
            with self._lock:
                self._predictor = None
                self._sam2_available = False
                self._loaded = False
                self._device = None
                self._last_error = str(exc)
                self._fallback_ready = self._allow_contour_fallback and not self._closed
            logger.warning(
                "segmenter.sam2_load_failed",
                error=str(exc),
                contour_fallback=self._fallback_ready,
            )
            return False
        finally:
            with self._lock:
                self._loading = False

    def _create_predictor(self):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        size_map = {
            "tiny": "facebook/sam2-hiera-tiny",
            "small": "facebook/sam2-hiera-small",
            "base_plus": "facebook/sam2-hiera-base-plus",
            "large": "facebook/sam2-hiera-large",
        }
        model_id = size_map.get(self._model_size, size_map["small"])
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        return predictor, device

    def get_status(self) -> dict:
        with self._lock:
            busy = self._runtime.busy or self._predicting
            cleanup_pending = self._runtime.cleanup_pending
            loaded = (
                self._loaded and self._sam2_available
                and self._predictor is not None and not self._closed
            )
            fallback = self._fallback_ready and not self._closed
            state = (
                "cleanup_failed" if self._closed and cleanup_pending else
                "stopped" if self._closed else
                "loading" if self._loading else
                "busy" if busy else
                "degraded" if fallback else
                "failed" if self._last_error else
                "ready" if loaded else "unloaded"
            )
            return {
                "state": state,
                "backend": "sam2" if loaded else "contour" if fallback else None,
                "loaded": loaded,
                "ready": loaded and not self._loading and not busy and self._last_error is None,
                "degraded": fallback or self._last_error is not None or cleanup_pending,
                "reason": self._last_error,
                "busy": busy,
                "device": self._device,
                "cleanup_pending": cleanup_pending,
            }

    def segment(
        self,
        frame: np.ndarray,
        prompt_points: list[tuple[int, int]],
    ) -> SegmentationResult:
        """Segment objects at the given anomaly peak points.

        Args:
            frame: BGR image (H, W, 3).
            prompt_points: ``(x, y)`` pixel coordinates of anomaly peaks.

        Returns:
            :class:`SegmentationResult` with per-object masks, bboxes, etc.
        """
        with self._lock:
            if self._closed:
                raise AuxiliaryUnavailableError("segmenter is stopped")
        if frame is None or frame.size == 0 or not prompt_points:
            return SegmentationResult()
        with self._lock:
            lazy_load = not self._loaded and not self._load_attempted
        if lazy_load:
            self.load()
        with self._lock:
            if self._closed:
                raise AuxiliaryUnavailableError("segmenter is stopped")
            if self._loading or self._predicting or self._runtime.busy:
                raise AuxiliaryBusyError("segmenter has an unfinished SDK call")
            if self._sam2_available and self._loaded and self._predictor is not None:
                backend = "sam2"
                predictor = self._predictor
                owned_frame = frame.copy()
                points = list(prompt_points)
                ticket = self._runtime.submit(
                    lambda: self._segment_sam2(owned_frame, points, predictor=predictor)
                )
            elif self._fallback_ready:
                backend = "contour"
                owned_frame = frame.copy()
                points = list(prompt_points)
                ticket = self._runtime.submit(
                    lambda: self._segment_contour_fallback(owned_frame, points)
                )
            else:
                raise AuxiliaryUnavailableError(self._last_error or "SAM 2 is not loaded")
            self._predicting = True
        try:
            result = self._runtime.wait(ticket, self._timeout_seconds)
            with self._lock:
                if self._closed:
                    raise AuxiliaryUnavailableError("segmenter is stopped")
                if backend == "sam2":
                    self._last_error = None
            result.backend = backend
            result.degraded = backend != "sam2"
            return result
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            logger.warning(
                "segmenter.segment_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise
        finally:
            with self._lock:
                self._predicting = False

    def shutdown(self) -> None:
        """Reject new calls and late results without waiting on a stuck SDK."""
        with self._lock:
            self._closed = True
            self._loaded = False
            self._sam2_available = False
            self._fallback_ready = False
            self._predictor = None
            self._device = None
            try:
                self._runtime.shutdown()
            except Exception as exc:
                self._last_error = f"cleanup failed: {exc}"
                raise
            self._last_error = None

    # ------------------------------------------------------------------
    # SAM 2 path
    # ------------------------------------------------------------------

    def _segment_sam2(
        self,
        frame: np.ndarray,
        prompt_points: list[tuple[int, int]],
        *,
        predictor=None,
    ) -> SegmentationResult:
        """Run SAM 2 image predictor on each prompt point."""
        predictor = self._predictor if predictor is None else predictor
        if predictor is None:
            raise AuxiliaryUnavailableError("SAM 2 predictor is not loaded")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = frame.shape[:2]
        objects: list[SegmentedObject] = []

        try:
            predictor.set_image(rgb)
            for px, py in prompt_points:
                point_coords = np.array([[px, py]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                if masks is not None and len(masks) > 0:
                    best_idx = int(np.argmax(scores))
                    mask = masks[best_idx]
                    confidence = float(scores[best_idx])
                    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                        raise ValueError("SAM 2 returned an invalid confidence")
                    obj = self._mask_to_object(mask, h, w, confidence)
                    if obj is not None:
                        objects.append(obj)
        finally:
            predictor.reset_image()

        total_area = sum(o.area_px for o in objects)
        return SegmentationResult(
            objects=objects,
            num_objects=len(objects),
            total_area_px=total_area,
        )

    # ------------------------------------------------------------------
    # Contour fallback
    # ------------------------------------------------------------------

    def _segment_contour_fallback(
        self,
        frame: np.ndarray,
        prompt_points: list[tuple[int, int]],
    ) -> SegmentationResult:
        """Contour-based fallback when SAM 2 is unavailable."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape[:2]
        objects: list[SegmentedObject] = []

        for px, py in prompt_points:
            r = 50  # search radius
            y1, y2 = max(0, py - r), min(h, py + r)
            x1, x2 = max(0, px - r), min(w, px + r)
            region = gray[y1:y2, x1:x2]

            if region.size == 0:
                continue

            _, thresh = cv2.threshold(
                region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            area = int(cv2.contourArea(largest))
            if area < self._min_mask_area_px:
                continue

            bx, by, bw, bh = cv2.boundingRect(largest)
            full_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(full_mask[y1:y2, x1:x2], [largest], -1, 255, -1)

            # Centroid via moments on the local contour
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
            else:
                cx, cy = x1 + bx + bw // 2, y1 + by + bh // 2

            objects.append(SegmentedObject(
                mask=full_mask,
                area_px=area,
                bbox=(x1 + bx, y1 + by, bw, bh),
                centroid=(cx, cy),
                confidence=1.0,
            ))

        total_area = sum(o.area_px for o in objects)
        return SegmentationResult(
            objects=objects,
            num_objects=len(objects),
            total_area_px=total_area,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mask_to_object(
        self,
        mask: np.ndarray,
        frame_h: int,
        frame_w: int,
        confidence: float,
    ) -> SegmentedObject | None:
        """Convert a boolean/float mask from SAM 2 to a :class:`SegmentedObject`."""
        binary = (mask > 0.5).astype(np.uint8) * 255

        # Ensure mask is 2-D
        if binary.ndim == 3:
            binary = binary.squeeze()
        if binary.shape != (frame_h, frame_w):
            binary = cv2.resize(binary, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

        area = int(np.count_nonzero(binary))
        if area < self._min_mask_area_px:
            return None

        # Bounding box
        ys, xs = np.nonzero(binary)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

        # Centroid
        cx = int(xs.mean())
        cy = int(ys.mean())

        return SegmentedObject(
            mask=binary,
            area_px=area,
            bbox=bbox,
            centroid=(cx, cy),
            confidence=confidence,
        )
