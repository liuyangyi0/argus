"""Anomalib-based anomaly detection.

Wraps Anomalib models (PatchCore, EfficientAD, AnomalyDINO) for inference.
Only requires "normal" images for training — any deviation from the learned
normal appearance is flagged as anomalous with a score and heatmap.
"""

from __future__ import annotations

import threading
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
    raw_score: float | None = None  # Pre-calibration score (None if no calibration)


@dataclass
class DetectorStatus:
    """Current operational status of the anomaly detector (DET-004)."""

    mode: str  # "anomalib" or "ssim_fallback"
    model_path: str | None
    model_loaded: bool
    threshold: float
    ssim_calibration_progress: float  # 0.0 to 1.0, relevant in SSIM mode
    ssim_calibrated: bool
    ssim_noise_floor: float | None
    is_quantized: bool = False  # True when the loaded model contains INT8 ops
    minmax_broken: bool = False  # True when PostProcessor MinMax is not fit (sigmoid fallback)


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
        enable_calibration: bool = True,
    ):
        self.threshold = threshold
        self.image_size = image_size
        self._model_path = Path(model_path) if model_path else None
        self._engine = None
        self._loaded = False
        self._enable_calibration = enable_calibration
        self._calibration_scores: np.ndarray | None = None  # sorted scores for p-value
        self._calibration_n: int = 0
        self._ssim_baseline_frames = ssim_baseline_frames
        self._ssim_sensitivity = ssim_sensitivity
        self._ssim_midpoint = ssim_midpoint
        self._ssim_baseline_count = 0
        self._ssim_noise_floor: float | None = None
        self._reload_lock = threading.Lock()
        self._ssim_lock = threading.Lock()  # protects SSIM baseline calibration state
        self._minmax_broken = False  # True when PostProcessor MinMax is not fit

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
        if suffix == ".ckpt":
            logger.error(
                "anomaly.unsupported_checkpoint",
                path=str(self._model_path),
                msg="Lightning checkpoint is not deployable; use model.pt or model.xml",
            )
            return False
        if suffix in (".xml", ".onnx", ".bin"):
            try:
                from anomalib.deploy import OpenVINOInferencer
                self._engine = OpenVINOInferencer(path=self._model_path)
                self._loaded = True
                self._check_minmax_normalization()
                logger.info("anomaly.model_loaded_openvino", path=str(self._model_path))
                self._load_calibration()
                return True
            except Exception as e:
                logger.warning("anomaly.openvino_failed", error=str(e))

        # Torch inferencer for .ckpt / .pt files (or OpenVINO fallback)
        try:
            import os
            # Required for Anomalib Torch model loading (TorchInferencer).
            # Only affects the Torch fallback path when OpenVINO is unavailable.
            # Security: enables arbitrary code execution from model files — only load trusted models.
            os.environ.setdefault("TRUST_REMOTE_CODE", "1")
            from anomalib.deploy import TorchInferencer
            self._engine = TorchInferencer(path=self._model_path)
            self._loaded = True
            self._check_minmax_normalization()
            logger.info("anomaly.model_loaded_torch", path=str(self._model_path))
            self._load_calibration()
            return True
        except Exception as e:
            logger.error("anomaly.load_failed", error=str(e))
            return False

    def _check_minmax_normalization(self) -> None:
        """Detect broken PostProcessor MinMax normalization.

        When a model is exported without fitting the PostProcessor (e.g.
        min=inf, max=-inf), all scores are normalized to 1.0 regardless
        of input. We detect this at load time and flag it so predict()
        can fall back to raw scores with sigmoid normalization.
        """
        try:
            model = getattr(self._engine, "model", None)
            if model is None:
                return
            pp = getattr(model, "post_processor", None)
            if pp is None:
                return
            img_mm = getattr(pp, "_image_min_max_metric", None)
            if img_mm is None:
                return
            mm_min = getattr(img_mm, "min", None)
            mm_max = getattr(img_mm, "max", None)
            if mm_min is None or mm_max is None:
                return
            import math
            min_val = float(mm_min)
            max_val = float(mm_max)
            if math.isinf(min_val) or math.isinf(max_val) or min_val >= max_val:
                self._minmax_broken = True
                logger.warning(
                    "anomaly.minmax_not_fit",
                    min=min_val,
                    max=max_val,
                    msg="PostProcessor MinMax not fit — using raw score with sigmoid normalization",
                )
        except Exception:
            pass

    def calibrate_raw_scores(self, baseline_dir: Path | None = None) -> None:
        """Calibrate raw score normalization using baseline images.

        When MinMax is broken, this method runs baseline images through the
        inner model to determine the normal score range, then sets a sigmoid
        midpoint at mean + 3*std. Call this after load() when _minmax_broken.
        """
        if not self._minmax_broken or not self._loaded:
            return
        if baseline_dir is None or not baseline_dir.is_dir():
            return

        import torch
        model = self._engine.model
        inner = getattr(model, "model", model)
        scores = []
        for img_path in sorted(baseline_dir.iterdir())[:30]:
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            resized = cv2.resize(frame, self.image_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            inp = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                out = inner(inp)
            raw = float(out.pred_score.squeeze().item())
            if np.isfinite(raw):
                scores.append(raw)

        if len(scores) >= 3:
            mean = float(np.mean(scores))
            std = max(float(np.std(scores)), 1.0)
            self._raw_score_midpoint = mean + 3 * std
            self._raw_score_scale = std * 3
            logger.info(
                "anomaly.raw_score_calibrated",
                n_baselines=len(scores),
                mean=round(mean, 2),
                std=round(std, 2),
                midpoint=round(self._raw_score_midpoint, 2),
                scale=round(self._raw_score_scale, 2),
            )
        else:
            logger.warning(
                "anomaly.raw_score_calibration_insufficient",
                n_baselines=len(scores),
                msg="Not enough baselines for calibration, using default sigmoid",
            )

    def _normalize_raw_score(self, raw_val: float) -> float:
        """Normalize a raw PatchCore score to 0-1 using sigmoid.

        Uses calibrated midpoint/scale if available, otherwise defaults.
        """
        import math
        midpoint = getattr(self, "_raw_score_midpoint", 55.0)
        scale = getattr(self, "_raw_score_scale", 8.0)
        return 1.0 / (1.0 + math.exp(-(raw_val - midpoint) / scale))

    def _load_calibration(self) -> None:
        """Load conformal calibration data from calibration.json near the model.

        Searches for calibration.json in the model file's directory and up to
        3 parent directories (covers model files nested in subdirectories).
        """
        if not self._enable_calibration or self._model_path is None:
            return

        try:
            from argus.alerts.calibration import ConformalCalibrator

            # Search model dir and parents for calibration.json
            search_dir = self._model_path.parent
            for _ in range(4):  # current dir + 3 parents
                cal_path = search_dir / "calibration.json"
                if cal_path.exists():
                    scores = ConformalCalibrator.load_sorted_scores(cal_path)
                    if scores is not None and len(scores) > 0:
                        self._calibration_scores = scores
                        self._calibration_n = len(scores)
                        logger.info(
                            "anomaly.calibration_loaded",
                            path=str(cal_path),
                            n_samples=self._calibration_n,
                        )
                    else:
                        logger.info(
                            "anomaly.calibration_no_scores",
                            path=str(cal_path),
                            msg="calibration.json found but no sorted_scores, skipping p-value calibration",
                        )
                    return
                search_dir = search_dir.parent

            logger.debug("anomaly.no_calibration", msg="No calibration.json found near model")
        except Exception as e:
            logger.warning("anomaly.calibration_load_failed", error=str(e))

    def _apply_calibration(self, raw_score: float) -> float:
        """Convert a raw anomaly score to a calibrated p-value.

        p_value = (number of calibration scores >= raw_score) / n_samples

        Higher p-value means more calibration samples had scores at or above
        this level, so the observation is more "normal". We invert it so that
        a higher calibrated score still means more anomalous:
        calibrated_score = 1 - p_value
        """
        if self._calibration_scores is None:
            return raw_score

        # searchsorted with side='left' gives the index of the first score >= raw_score
        # So n - index = number of scores >= raw_score
        idx = np.searchsorted(self._calibration_scores, raw_score, side="left")
        n_ge = self._calibration_n - idx
        p_value = n_ge / self._calibration_n
        calibrated = 1.0 - p_value
        return max(0.0, min(calibrated, 1.0))

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

    def predict_batch(self, frames: list[np.ndarray]) -> list[AnomalyResult]:
        """Run anomaly detection on a batch of frames.

        Processes each frame independently (Anomalib models don't natively
        support batch inference), but avoids per-call overhead by reusing
        the loaded model reference.

        Args:
            frames: List of BGR images.

        Returns:
            List of AnomalyResult, one per frame.
        """
        return [self.predict(frame) for frame in frames]

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

            # When PostProcessor MinMax is not fit, the normalized score is
            # always 1.0 (useless). Use the raw inner model score directly
            # to avoid running inference twice.
            if self._minmax_broken:
                import torch
                model = self._engine.model
                inner = getattr(model, "model", model)
                inp = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                with torch.no_grad():
                    raw_out = inner(inp)
                raw_val = float(raw_out.pred_score.squeeze().item())
                if not np.isfinite(raw_val):
                    return self._safe_result()
                anomaly_score = self._normalize_raw_score(raw_val)
                # Use anomaly_map from inner output if available
                prediction = raw_out
            else:
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

            # Apply conformal calibration if available
            raw_score_out = None
            if self._calibration_scores is not None:
                raw_score_out = anomaly_score
                anomaly_score = self._apply_calibration(anomaly_score)

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
                raw_score=raw_score_out,
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
        # Lock protects calibration state (_ssim_baseline_acc, _ssim_frame_diffs, etc.)
        # to prevent race conditions during concurrent SSIM predictions.
        with self._ssim_lock:
            baseline_count = self._ssim_baseline_count
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

    def get_status(self) -> DetectorStatus:
        """Return current detector operational status (DET-004)."""
        calibrated = self._ssim_baseline_count >= self._ssim_baseline_frames
        if self._loaded:
            calibration_progress = 1.0
        elif self._ssim_baseline_count > 0:
            calibration_progress = min(1.0, self._ssim_baseline_count / self._ssim_baseline_frames)
        else:
            calibration_progress = 0.0

        # Detect INT8 quantization by inspecting OpenVINO model ops
        is_quantized = False
        if self._loaded and self._engine is not None:
            is_quantized = self._detect_quantization()

        return DetectorStatus(
            mode="anomalib" if self._loaded else "ssim_fallback",
            model_path=str(self._model_path) if self._model_path else None,
            model_loaded=self._loaded,
            threshold=self.threshold,
            ssim_calibration_progress=calibration_progress,
            ssim_calibrated=calibrated,
            ssim_noise_floor=self._ssim_noise_floor,
            is_quantized=is_quantized,
            minmax_broken=self._minmax_broken,
        )

    def _detect_quantization(self) -> bool:
        """Check if the loaded OpenVINO model contains INT8 quantized ops."""
        try:
            import openvino as ov

            if self._model_path is None:
                return False

            model_xml = self._model_path
            if model_xml.suffix.lower() != ".xml":
                return False

            core = ov.Core()
            model = core.read_model(str(model_xml))

            # Check for FakeQuantize or i8 element types indicating INT8
            for op in model.get_ordered_ops():
                op_type = op.get_type_name()
                if op_type == "FakeQuantize":
                    return True
                # Check for INT8 element types in outputs
                for output in op.outputs():
                    et = output.get_element_type()
                    if "i8" in str(et) or "u8" in str(et):
                        return True
            return False
        except Exception:
            return False

    def hot_reload(self, new_model_path: Path) -> bool:
        """Hot-reload the anomaly model without stopping inference.

        Loads the new model into a temporary variable first. If loading
        succeeds, atomically swaps it in. If it fails, keeps the old model.
        Thread-safe via a lock around the engine swap.
        """
        logger.info("anomaly.hot_reload_start", path=str(new_model_path))

        try:
            if new_model_path.suffix.lower() == ".ckpt":
                raise ValueError("Lightning checkpoint is not deployable; use model.pt or model.xml")

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

            # Reload calibration data for the new model
            self._calibration_scores = None
            self._calibration_n = 0
            self._load_calibration()

            logger.info("anomaly.hot_reload_success", path=str(new_model_path))
            return True

        except Exception as e:
            logger.error("anomaly.hot_reload_failed", error=str(e), msg="Keeping old model")
            return False

    def hot_reload_with_warmup(
        self,
        new_model_path: Path,
        expected_hash: str | None = None,
        warmup_frames: list[np.ndarray] | None = None,
        warmup_count: int = 10,
        max_latency_ms: float = 500.0,
        baseline_dir: Path | None = None,
        callback: "threading.Event | None" = None,
    ) -> dict:
        """Hot-reload with SHA256 verification, warmup, and latency check.

        Runs in the calling thread (caller should spawn a thread if async needed).
        Returns a dict with status and details.
        """
        import time

        from argus.storage.model_registry import ModelRegistry

        result = {
            "success": False,
            "sha256_verified": False,
            "warmup_latency_ms": None,
            "error": None,
        }

        logger.info("anomaly.warmup_reload_start", path=str(new_model_path))

        try:
            if expected_hash:
                actual_hash = ModelRegistry._compute_dir_hash(new_model_path)
                if actual_hash != expected_hash:
                    result["error"] = (
                        f"SHA256 mismatch: expected {expected_hash}, got {actual_hash}"
                    )
                    logger.error("anomaly.warmup_sha256_mismatch", **result)
                    if callback:
                        callback.set()
                    return result
                result["sha256_verified"] = True

            # Step 2: Load new model
            try:
                from anomalib.deploy import OpenVINOInferencer
                new_engine = OpenVINOInferencer(path=new_model_path)
            except Exception:
                from anomalib.deploy import TorchInferencer
                new_engine = TorchInferencer(path=new_model_path)

            # Warmup with baseline frames
            if warmup_frames is None and baseline_dir and baseline_dir.is_dir():
                warmup_frames = []
                for img_path in sorted(baseline_dir.iterdir())[:warmup_count]:
                    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        frame = cv2.imread(str(img_path))
                        if frame is not None:
                            warmup_frames.append(frame)

            if warmup_frames:
                latencies = []
                for frame in warmup_frames[:warmup_count]:
                    t0 = time.perf_counter()
                    new_engine.predict(frame)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000)

                avg_latency = sum(latencies) / len(latencies) if latencies else 0
                result["warmup_latency_ms"] = round(avg_latency, 1)

                if avg_latency > max_latency_ms:
                    result["error"] = (
                        f"Warmup latency {avg_latency:.1f}ms exceeds limit {max_latency_ms}ms"
                    )
                    logger.warning("anomaly.warmup_latency_exceeded", **result)
                    if callback:
                        callback.set()
                    return result

                logger.info(
                    "anomaly.warmup_complete",
                    avg_latency_ms=result["warmup_latency_ms"],
                    frames=len(latencies),
                )

            # Step 4: Atomic swap
            with self._reload_lock:
                self._engine = new_engine
                self._model_path = new_model_path
                self._loaded = True

            self._calibration_scores = None
            self._calibration_n = 0
            self._load_calibration()

            result["success"] = True
            logger.info("anomaly.warmup_reload_success", path=str(new_model_path))

        except Exception as e:
            result["error"] = str(e)
            logger.error("anomaly.warmup_reload_failed", error=str(e))

        if callback:
            callback.set()

        return result

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_calibrated(self) -> bool:
        """Whether conformal calibration data is loaded."""
        return self._calibration_scores is not None


@dataclass
class _TileInfo:
    """Coordinates of a tile within the original frame."""

    x: int
    y: int
    w: int
    h: int


class MultiScaleDetector:
    """Sliding window multi-scale anomaly detector with pyramid support.

    Wraps an AnomalibDetector and runs it on multiple overlapping tiles
    of the input frame, plus once on the full frame. This dramatically
    improves detection of small objects that get lost when a 1920×1080
    frame is squeezed down to 256×256.

    Pyramid mode (default): runs 3 levels (512/768/1024) to capture
    anomalies at different scales — small foreign objects, medium debris,
    and large structural changes. NMS merges results across levels.

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
        pyramid_mode: bool = False,
        pyramid_sizes: list[int] | None = None,
    ):
        self._base = base_detector
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._stride = max(1, int(tile_size * (1.0 - tile_overlap)))
        self.threshold = base_detector.threshold
        self._pyramid_mode = pyramid_mode
        self._pyramid_sizes = pyramid_sizes or [512, 768, 1024]
        # Pre-compute strides for each pyramid level
        self._pyramid_strides = [
            max(1, int(sz * (1.0 - tile_overlap))) for sz in self._pyramid_sizes
        ]

    @property
    def is_loaded(self) -> bool:
        return self._base.is_loaded

    @property
    def is_calibrated(self) -> bool:
        return self._base.is_calibrated

    def load(self) -> bool:
        return self._base.load()

    def get_status(self) -> DetectorStatus:
        return self._base.get_status()

    def hot_reload(self, new_model_path: Path) -> bool:
        return self._base.hot_reload(new_model_path)

    def calibrate_raw_scores(self, baseline_dir: Path | None = None) -> None:
        return self._base.calibrate_raw_scores(baseline_dir)

    def predict(self, frame: np.ndarray) -> AnomalyResult:
        """Run multi-scale detection: tiles + full frame, return best score.

        Falls back to single-scale for SSIM mode (no trained model) since
        SSIM frame-diff works on the full frame and each tile would need
        its own independent calibration.

        Pyramid mode: runs 3 levels (512/768/1024) and merges results,
        capturing anomalies at different spatial scales.
        """
        if frame is None or frame.size == 0:
            return self._base.predict(frame)

        # SSIM fallback mode: skip tiling, run full frame only
        if not self._base.is_loaded:
            return self._base.predict(frame)

        h, w = frame.shape[:2]

        # If frame is smaller than smallest tile, just run single-scale
        min_tile = self._pyramid_sizes[0] if self._pyramid_mode else self._tile_size
        if h <= min_tile and w <= min_tile:
            return self._base.predict(frame)

        best_score = 0.0
        best_result: AnomalyResult | None = None
        all_tile_results: list[tuple[_TileInfo, AnomalyResult]] = []

        if self._pyramid_mode:
            # Pyramid: run at each scale level (512/768/1024)
            for level_idx, (pyr_size, pyr_stride) in enumerate(
                zip(self._pyramid_sizes, self._pyramid_strides)
            ):
                if h <= pyr_size and w <= pyr_size:
                    continue  # frame too small for this level
                tiles = self._generate_tiles(h, w, tile_size=pyr_size, stride=pyr_stride)
                for tile_info in tiles:
                    crop = frame[
                        tile_info.y : tile_info.y + tile_info.h,
                        tile_info.x : tile_info.x + tile_info.w,
                    ]
                    result = self._base.predict(crop)
                    all_tile_results.append((tile_info, result))
                    if result.anomaly_score > best_score:
                        best_score = result.anomaly_score
                        best_result = result

            logger.debug(
                "multiscale.pyramid_result",
                levels=len(self._pyramid_sizes),
                total_tiles=len(all_tile_results),
                best_score=round(best_score, 3),
            )
        else:
            # Single tile_size mode (legacy)
            tiles = self._generate_tiles(h, w)
            for tile_info in tiles:
                crop = frame[
                    tile_info.y : tile_info.y + tile_info.h,
                    tile_info.x : tile_info.x + tile_info.w,
                ]
                result = self._base.predict(crop)
                all_tile_results.append((tile_info, result))
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
        merged_map = self._merge_heatmaps(h, w, all_tile_results, global_result)

        logger.debug(
            "multiscale.result",
            tiles=len(all_tile_results),
            best_score=round(best_score, 3),
            global_score=round(global_result.anomaly_score, 3),
            pyramid=self._pyramid_mode,
        )

        return AnomalyResult(
            anomaly_score=best_score,
            anomaly_map=merged_map,
            is_anomalous=best_score >= self.threshold,
            threshold=self.threshold,
        )

    def _generate_tiles(
        self, frame_h: int, frame_w: int,
        tile_size: int | None = None, stride: int | None = None,
    ) -> list[_TileInfo]:
        """Generate sliding window tile coordinates covering the frame.

        Tiles that would extend beyond the frame edge are clamped so they
        stay within bounds (the last tile in each row/column may overlap
        more with its neighbor).
        """
        ts = tile_size or self._tile_size
        st = stride or self._stride
        tiles = []
        y = 0
        while y < frame_h:
            tile_h = min(ts, frame_h - y)
            # If remaining height is too small, extend tile upward
            if tile_h < ts and y > 0:
                y = max(0, frame_h - ts)
                tile_h = frame_h - y

            x = 0
            while x < frame_w:
                tile_w = min(ts, frame_w - x)
                if tile_w < ts and x > 0:
                    x = max(0, frame_w - ts)
                    tile_w = frame_w - x

                tiles.append(_TileInfo(x=x, y=y, w=tile_w, h=tile_h))

                if x + tile_w >= frame_w:
                    break
                x += st

            if y + tile_h >= frame_h:
                break
            y += st

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

        if not has_any_map:
            return None

        # NMS: suppress overlapping anomaly peaks across pyramid levels.
        # Extract connected regions above threshold, suppress lower-scoring
        # regions that overlap significantly with higher-scoring ones.
        if self._pyramid_mode:
            merged = self._nms_suppress(merged)

        return merged

    def _nms_suppress(
        self, heatmap: np.ndarray, iou_threshold: float = 0.5
    ) -> np.ndarray:
        """Non-Maximum Suppression on heatmap anomaly regions.

        Extracts connected components above the detection threshold,
        computes bounding boxes, and suppresses lower-scoring overlapping
        regions — keeping only the strongest detection per spatial area.
        """
        binary = (heatmap >= self.threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        if num_labels <= 2:
            return heatmap  # 0 or 1 region — nothing to suppress

        # Collect regions with their max scores
        regions: list[tuple[float, int, int, int, int, int]] = []  # (score, label, x, y, w, h)
        for i in range(1, num_labels):  # skip background (0)
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            mask = labels[y:y+h, x:x+w] == i
            region_score = float(heatmap[y:y+h, x:x+w][mask].max())
            regions.append((region_score, i, x, y, w, h))

        # Sort by score descending
        regions.sort(key=lambda r: r[0], reverse=True)

        suppressed = set()
        for idx, (score_a, label_a, x_a, y_a, w_a, h_a) in enumerate(regions):
            if label_a in suppressed:
                continue
            for jdx in range(idx + 1, len(regions)):
                score_b, label_b, x_b, y_b, w_b, h_b = regions[jdx]
                if label_b in suppressed:
                    continue
                # Compute IoU of bounding boxes
                ix = max(0, min(x_a + w_a, x_b + w_b) - max(x_a, x_b))
                iy = max(0, min(y_a + h_a, y_b + h_b) - max(y_a, y_b))
                intersection = ix * iy
                union = w_a * h_a + w_b * h_b - intersection
                iou = intersection / max(union, 1)
                if iou >= iou_threshold:
                    suppressed.add(label_b)

        # Zero out suppressed regions
        if suppressed:
            for label_id in suppressed:
                heatmap[labels == label_id] = 0.0

        return heatmap
