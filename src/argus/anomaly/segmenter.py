"""SAM 2 instance segmentation for anomaly regions (D2).

Given anomaly peak points from heatmap analysis, segments individual objects
to provide precise boundaries for anomaly classification and reporting.

If the ``sam2`` package is not installed the class still initialises but
returns empty results — the pipeline must never crash because of a missing
optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import structlog

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
    Falls back to contour-based segmentation if SAM 2 is not available.
    """

    def __init__(
        self,
        model_size: str = "small",
        min_mask_area_px: int = 100,
    ):
        self._model_size = model_size
        self._min_mask_area_px = min_mask_area_px
        self._predictor = None
        self._loaded = False
        self._sam2_available = False

    def load(self) -> None:
        """Load SAM 2 model (lazy initialisation)."""
        if self._loaded:
            return

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Map model_size to SAM 2 checkpoint names
            size_map = {
                "tiny": "facebook/sam2-hiera-tiny",
                "small": "facebook/sam2-hiera-small",
                "base_plus": "facebook/sam2-hiera-base-plus",
                "large": "facebook/sam2-hiera-large",
            }
            model_id = size_map.get(self._model_size, size_map["small"])

            self._predictor = SAM2ImagePredictor.from_pretrained(model_id)
            self._sam2_available = True
            self._loaded = True
            logger.info(
                "segmenter.sam2_loaded",
                model_size=self._model_size,
                model_id=model_id,
            )
        except ImportError:
            self._sam2_available = False
            self._loaded = True
            logger.info(
                "segmenter.sam2_not_installed",
                msg="sam2 package not found — using contour-based fallback",
            )
        except Exception as exc:
            self._sam2_available = False
            self._loaded = True
            logger.warning(
                "segmenter.sam2_load_failed",
                error=str(exc),
                msg="Failed to load SAM 2 model — using contour-based fallback",
            )

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
        if not self._loaded:
            self.load()

        if frame is None or frame.size == 0 or not prompt_points:
            return SegmentationResult()

        try:
            if self._sam2_available and self._predictor is not None:
                return self._segment_sam2(frame, prompt_points)
            return self._segment_contour_fallback(frame, prompt_points)
        except Exception as exc:
            logger.warning(
                "segmenter.segment_error",
                error=str(exc),
                error_type=type(exc).__name__,
                msg="Segmentation failed — returning empty result",
            )
            return SegmentationResult()

    # ------------------------------------------------------------------
    # SAM 2 path
    # ------------------------------------------------------------------

    def _segment_sam2(
        self,
        frame: np.ndarray,
        prompt_points: list[tuple[int, int]],
    ) -> SegmentationResult:
        """Run SAM 2 image predictor on each prompt point."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)

        h, w = frame.shape[:2]
        objects: list[SegmentedObject] = []

        for px, py in prompt_points:
            point_coords = np.array([[px, py]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)  # foreground

            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

            # Pick best mask (highest score)
            if masks is not None and len(masks) > 0:
                best_idx = int(np.argmax(scores))
                mask = masks[best_idx]
                confidence = float(scores[best_idx])
                obj = self._mask_to_object(mask, h, w, confidence)
                if obj is not None:
                    objects.append(obj)

        self._predictor.reset_image()

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
