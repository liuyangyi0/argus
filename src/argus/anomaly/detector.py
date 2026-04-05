"""Anomalib-based anomaly detection.

Wraps Anomalib models (PatchCore, EfficientAD, AnomalyDINO) for inference.
Only requires "normal" images for training — any deviation from the learned
normal appearance is flagged as anomalous with a score and heatmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class AnomalyResult:
    """Result of anomaly detection on a single frame."""

    anomaly_score: float  # 0.0 = normal, 1.0 = highly anomalous
    anomaly_map: np.ndarray | None  # per-pixel anomaly heatmap (H, W), 0-1
    is_anomalous: bool
    threshold: float
    detection_failed: bool = False  # True when prediction errored out


class AnomalibDetector:
    """Anomaly detection using Anomalib models.

    Supports loading pre-trained models for inference. Training is handled
    separately via scripts/train_model.py.

    For cold-start (no trained model available), falls back to a simple
    SSIM-based comparison against baseline images.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = 0.7,
        image_size: tuple[int, int] = (256, 256),
        ssim_baseline_frames: int = 15,
        ssim_sensitivity: float = 50.0,
        ssim_midpoint: float = 0.015,
    ):
        self.threshold = threshold
        self.image_size = image_size
        self._model_path = Path(model_path) if model_path else None
        self._engine = None
        self._loaded = False
        self._ssim_baseline_frames = ssim_baseline_frames
        self._ssim_sensitivity = ssim_sensitivity
        self._ssim_midpoint = ssim_midpoint

    def load(self) -> bool:
        """Load the Anomalib model for inference.

        Returns True if model loaded successfully, False if no model available.
        """
        if self._model_path is None or not self._model_path.exists():
            # SSIM fallback uses frame difference — needs a much lower threshold
            # than Anomalib models (frame diff 95th percentile is typically 0.0-0.3)
            self._ssim_threshold = min(self.threshold, 0.15)
            logger.warning(
                "anomaly.no_model",
                path=str(self._model_path),
                msg="No trained model found, using SSIM fallback",
                ssim_threshold=self._ssim_threshold,
            )
            return False

        # Try OpenVINO first (faster inference), then fall back to Torch
        suffix = self._model_path.suffix.lower()
        if suffix in (".xml", ".onnx", ".bin"):
            try:
                from anomalib.deploy import OpenVINOInferencer
                self._engine = OpenVINOInferencer(path=self._model_path)
                self._loaded = True
                logger.info("anomaly.model_loaded_openvino", path=str(self._model_path))
                return True
            except Exception as e:
                logger.warning("anomaly.openvino_failed", error=str(e))

        # Torch inferencer for .ckpt / .pt files (or OpenVINO fallback)
        try:
            import os
            os.environ.setdefault("TRUST_REMOTE_CODE", "1")
            from anomalib.deploy import TorchInferencer
            self._engine = TorchInferencer(path=self._model_path)
            self._loaded = True
            logger.info("anomaly.model_loaded_torch", path=str(self._model_path))
            return True
        except Exception as e:
            logger.error("anomaly.load_failed", error=str(e))
            return False

    def predict(self, frame: np.ndarray) -> AnomalyResult:
        """Run anomaly detection on a frame.

        Args:
            frame: BGR image from camera (after person masking).

        Returns:
            AnomalyResult with anomaly score and heatmap.
        """
        if self._loaded and self._engine is not None:
            return self._predict_anomalib(frame)
        return self._predict_ssim_fallback(frame)

    def _predict_anomalib(self, frame: np.ndarray) -> AnomalyResult:
        """Predict using loaded Anomalib model."""
        # Validate frame
        if frame is None or frame.size == 0:
            logger.warning("anomaly.invalid_frame", msg="Empty or None frame")
            return self._safe_result()

        try:
            # Resize to model's expected input size
            resized = cv2.resize(frame, self.image_size)

            # Convert BGR to RGB for anomalib
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            prediction = self._engine.predict(rgb)

            # Extract score — Anomalib 2.x returns Tensors, not numpy
            raw_score = prediction.pred_score
            if hasattr(raw_score, "item"):
                anomaly_score = float(raw_score.squeeze().item())
            else:
                anomaly_score = float(raw_score)

            if not np.isfinite(anomaly_score):
                logger.error(
                    "anomaly.nan_score",
                    raw_score=str(prediction.pred_score),
                    msg="Model returned NaN/Inf score, treating as normal",
                )
                return self._safe_result()

            # Clamp to valid range
            anomaly_score = max(0.0, min(anomaly_score, 1.0))

            anomaly_map = None
            if prediction.anomaly_map is not None:
                amap = prediction.anomaly_map.squeeze()
                # Convert Tensor to numpy if needed
                if hasattr(amap, "detach"):
                    amap = amap.detach().cpu().numpy()
                anomaly_map = np.asarray(amap, dtype=np.float32)
                # Replace NaN/Inf in heatmap
                if not np.all(np.isfinite(anomaly_map)):
                    logger.warning("anomaly.nan_heatmap", msg="Heatmap contains NaN/Inf")
                    anomaly_map = np.nan_to_num(anomaly_map, nan=0.0, posinf=1.0, neginf=0.0)
                # Normalize to 0-1 range
                map_min, map_max = anomaly_map.min(), anomaly_map.max()
                if map_max > map_min:
                    anomaly_map = (anomaly_map - map_min) / (map_max - map_min)
                else:
                    anomaly_map = np.zeros_like(anomaly_map, dtype=np.float32)

            return AnomalyResult(
                anomaly_score=anomaly_score,
                anomaly_map=anomaly_map,
                is_anomalous=anomaly_score >= self.threshold,
                threshold=self.threshold,
            )
        except Exception as e:
            logger.error("anomaly.predict_failed", error=str(e), error_type=type(e).__name__)
            return self._safe_result()

    def _safe_result(self) -> AnomalyResult:
        """Return a safe default result when prediction fails.

        Sets detection_failed=True so the pipeline can distinguish between
        'no anomaly detected' and 'detection itself failed'.
        """
        return AnomalyResult(
            anomaly_score=0.0,
            anomaly_map=None,
            is_anomalous=False,
            threshold=self.threshold,
            detection_failed=True,
        )

    def _predict_ssim_fallback(self, frame: np.ndarray) -> AnomalyResult:
        """SSIM-based fallback when no Anomalib model is available.

        Uses a running average baseline (not just the first frame) and learns
        the noise floor from the first N frames. Only changes significantly
        above the noise floor are flagged as anomalies.

        This handles AI-generated video artifacts and camera sensor noise
        that would otherwise cause false positives.
        """
        import math

        if frame is None or frame.size == 0:
            return self._safe_result()

        # Resize to standard size for consistent comparison
        resized = cv2.resize(frame, self.image_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Phase 1: Collect baseline frames (first N frames → build avg + noise model)
        baseline_count = getattr(self, "_ssim_baseline_count", 0)
        BASELINE_FRAMES = self._ssim_baseline_frames

        if baseline_count == 0:
            self._ssim_baseline_acc = gray.copy()
            self._ssim_baseline_count = 1
            self._ssim_noise_floor = 0.0
            self._ssim_frame_diffs = []
            logger.info("anomaly.ssim_calibrating", msg="Collecting baseline frames...")
            return AnomalyResult(
                anomaly_score=0.0, anomaly_map=None, is_anomalous=False,
                threshold=self.threshold, detection_failed=False,
            )

        if baseline_count < BASELINE_FRAMES:
            # Track frame-to-frame differences to learn noise level
            prev = self._ssim_baseline_acc / baseline_count
            diff_to_prev = cv2.absdiff(gray, prev)
            diff_blurred = cv2.GaussianBlur(diff_to_prev, (11, 11), 0) / 255.0
            noise_sample = float(cv2.blur(diff_blurred, (16, 16)).max())
            self._ssim_frame_diffs.append(noise_sample)

            # Accumulate for average baseline
            self._ssim_baseline_acc += gray
            self._ssim_baseline_count += 1

            if baseline_count == BASELINE_FRAMES - 1:
                # Finalize baseline and noise floor
                self._ssim_baseline = (self._ssim_baseline_acc / self._ssim_baseline_count).astype(np.float32)
                # Noise floor = IQR-based estimation (robust to outliers from early motion)
                diffs = np.array(self._ssim_frame_diffs)
                q25, q75 = float(np.percentile(diffs, 25)), float(np.percentile(diffs, 75))
                iqr = q75 - q25
                self._ssim_noise_floor = q75 + 1.5 * iqr if iqr > 0 else float(np.median(diffs)) * 1.5
                logger.info(
                    "anomaly.ssim_calibrated",
                    baseline_frames=self._ssim_baseline_count,
                    noise_floor=round(self._ssim_noise_floor, 4),
                    diffs=str([round(d, 4) for d in self._ssim_frame_diffs]),
                )
            return AnomalyResult(
                anomaly_score=0.0, anomaly_map=None, is_anomalous=False,
                threshold=self.threshold, detection_failed=False,
            )

        # Phase 2: Detection — compare against learned baseline
        diff = cv2.absdiff(gray, self._ssim_baseline)

        # Strong blur to suppress pixel-level noise
        diff_blur = cv2.GaussianBlur(diff, (15, 15), 0)

        # Threshold: zero out anything below noise floor (suppress background noise)
        diff_float = diff_blur / 255.0
        noise_mask = diff_float > (self._ssim_noise_floor * 0.5)
        diff_clean = diff_float * noise_mask.astype(np.float32)

        # Score: peak local density after noise removal
        pooled = cv2.blur(diff_clean, (16, 16))
        raw_score = float(pooled.max())

        # Subtract noise floor so only real changes produce signal
        signal = max(0.0, raw_score - self._ssim_noise_floor)

        # Sigmoid normalization (signal → 0-1 anomaly score)
        # After noise subtraction, signal is small (0.01-0.10 range)
        # Map so that signal ~0.01 → ~0.5 (info), ~0.03 → ~0.9 (high)
        sensitivity = self._ssim_sensitivity
        midpoint = self._ssim_midpoint
        if signal <= 0:
            anomaly_score = 0.0
        else:
            anomaly_score = 1.0 / (1.0 + math.exp(-sensitivity * (signal - midpoint)))
        anomaly_score = max(0.0, min(anomaly_score, 1.0))

        # Build anomaly map — use noise-cleaned version so heatmap highlights
        # only real changes, not background noise/artifacts
        anomaly_map = diff_clean

        return AnomalyResult(
            anomaly_score=anomaly_score,
            anomaly_map=anomaly_map,
            is_anomalous=anomaly_score >= self.threshold,
            threshold=self.threshold,
        )

    def hot_reload(self, new_model_path: Path) -> bool:
        """Hot-reload the anomaly model without stopping inference.

        Loads the new model into a temporary variable first. If loading
        succeeds, atomically swaps it in. If it fails, keeps the old model.
        Thread-safe via a lock around the engine swap.
        """
        import threading

        if not hasattr(self, "_reload_lock"):
            self._reload_lock = threading.Lock()

        logger.info("anomaly.hot_reload_start", path=str(new_model_path))

        try:
            # Try OpenVINO first
            try:
                from anomalib.deploy import OpenVINOInferencer
                new_engine = OpenVINOInferencer(path=new_model_path)
            except (ImportError, Exception):
                from anomalib.deploy import TorchInferencer
                new_engine = TorchInferencer(path=new_model_path)

            # Atomic swap
            with self._reload_lock:
                self._engine = new_engine
                self._model_path = new_model_path
                self._loaded = True

            logger.info("anomaly.hot_reload_success", path=str(new_model_path))
            return True

        except Exception as e:
            logger.error("anomaly.hot_reload_failed", error=str(e), msg="Keeping old model")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


@dataclass
class _TileInfo:
    """Coordinates of a tile within the original frame."""

    x: int
    y: int
    w: int
    h: int


class MultiScaleDetector:
    """Sliding window multi-scale anomaly detector.

    Wraps an AnomalibDetector and runs it on multiple overlapping tiles
    of the input frame, plus once on the full frame. This dramatically
    improves detection of small objects that get lost when a 1920×1080
    frame is squeezed down to 256×256.

    The final score is the maximum across all tiles and the full frame,
    so a small anomaly in one tile won't be diluted by normal regions.

    The heatmap is assembled by mapping each tile's anomaly_map back
    to its position in the original frame, taking the max at overlaps.
    """

    def __init__(
        self,
        base_detector: AnomalibDetector,
        tile_size: int = 512,
        tile_overlap: float = 0.25,
    ):
        self._base = base_detector
        self._tile_size = tile_size
        self._stride = max(1, int(tile_size * (1.0 - tile_overlap)))
        self.threshold = base_detector.threshold

    @property
    def is_loaded(self) -> bool:
        return self._base.is_loaded

    def load(self) -> bool:
        return self._base.load()

    def hot_reload(self, new_model_path: Path) -> bool:
        return self._base.hot_reload(new_model_path)

    def predict(self, frame: np.ndarray) -> AnomalyResult:
        """Run multi-scale detection: tiles + full frame, return best score.

        Falls back to single-scale for SSIM mode (no trained model) since
        SSIM frame-diff works on the full frame and each tile would need
        its own independent calibration.
        """
        if frame is None or frame.size == 0:
            return self._base.predict(frame)

        # SSIM fallback mode: skip tiling, run full frame only
        if not self._base.is_loaded:
            return self._base.predict(frame)

        h, w = frame.shape[:2]

        # If frame is smaller than tile size, just run single-scale
        if h <= self._tile_size and w <= self._tile_size:
            return self._base.predict(frame)

        # Generate tile coordinates
        tiles = self._generate_tiles(h, w)

        # Run detection on each tile
        best_score = 0.0
        best_result: AnomalyResult | None = None
        tile_results: list[tuple[_TileInfo, AnomalyResult]] = []

        for tile_info in tiles:
            crop = frame[
                tile_info.y : tile_info.y + tile_info.h,
                tile_info.x : tile_info.x + tile_info.w,
            ]
            result = self._base.predict(crop)
            tile_results.append((tile_info, result))

            if result.anomaly_score > best_score:
                best_score = result.anomaly_score
                best_result = result

        # Also run full-frame detection (catches large anomalies)
        global_result = self._base.predict(frame)
        if global_result.anomaly_score > best_score:
            best_score = global_result.anomaly_score
            best_result = global_result

        if best_result is None:
            best_result = global_result

        # Merge heatmaps from all tiles into full-frame heatmap
        merged_map = self._merge_heatmaps(h, w, tile_results, global_result)

        logger.debug(
            "multiscale.result",
            tiles=len(tiles),
            best_score=round(best_score, 3),
            global_score=round(global_result.anomaly_score, 3),
        )

        return AnomalyResult(
            anomaly_score=best_score,
            anomaly_map=merged_map,
            is_anomalous=best_score >= self.threshold,
            threshold=self.threshold,
        )

    def _generate_tiles(self, frame_h: int, frame_w: int) -> list[_TileInfo]:
        """Generate sliding window tile coordinates covering the frame.

        Tiles that would extend beyond the frame edge are clamped so they
        stay within bounds (the last tile in each row/column may overlap
        more with its neighbor).
        """
        tiles = []
        y = 0
        while y < frame_h:
            tile_h = min(self._tile_size, frame_h - y)
            # If remaining height is too small, extend tile upward
            if tile_h < self._tile_size and y > 0:
                y = max(0, frame_h - self._tile_size)
                tile_h = frame_h - y

            x = 0
            while x < frame_w:
                tile_w = min(self._tile_size, frame_w - x)
                if tile_w < self._tile_size and x > 0:
                    x = max(0, frame_w - self._tile_size)
                    tile_w = frame_w - x

                tiles.append(_TileInfo(x=x, y=y, w=tile_w, h=tile_h))

                if x + tile_w >= frame_w:
                    break
                x += self._stride

            if y + tile_h >= frame_h:
                break
            y += self._stride

        return tiles

    def _merge_heatmaps(
        self,
        frame_h: int,
        frame_w: int,
        tile_results: list[tuple[_TileInfo, AnomalyResult]],
        global_result: AnomalyResult,
    ) -> np.ndarray | None:
        """Merge tile heatmaps into a single full-frame heatmap.

        Each tile's anomaly_map is resized to its tile dimensions and
        placed at its position in the output. Overlapping regions use
        the maximum value. The global full-frame heatmap is also blended in.
        """
        merged = np.zeros((frame_h, frame_w), dtype=np.float32)
        has_any_map = False

        for tile_info, result in tile_results:
            if result.anomaly_map is None:
                continue
            has_any_map = True
            # Resize tile heatmap from model size to tile dimensions
            tile_map = cv2.resize(
                result.anomaly_map.astype(np.float32),
                (tile_info.w, tile_info.h),
                interpolation=cv2.INTER_LINEAR,
            )
            # Place into merged map, taking max at overlaps
            region = merged[
                tile_info.y : tile_info.y + tile_info.h,
                tile_info.x : tile_info.x + tile_info.w,
            ]
            np.maximum(region, tile_map, out=region)

        # Blend global heatmap (resized to full frame)
        if global_result.anomaly_map is not None:
            has_any_map = True
            global_map = cv2.resize(
                global_result.anomaly_map.astype(np.float32),
                (frame_w, frame_h),
                interpolation=cv2.INTER_LINEAR,
            )
            np.maximum(merged, global_map, out=merged)

        return merged if has_any_map else None
