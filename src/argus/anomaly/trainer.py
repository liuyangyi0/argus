"""Model training orchestration.

Coordinates the full training pipeline: validate baselines, split train/val,
train model, validate output, evaluate on validation set, generate quality
report, recommend threshold, and record results.
"""

from __future__ import annotations

import random
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import structlog

from argus.anomaly.baseline import BaselineManager
from argus.storage.models import BaselineState

logger = structlog.get_logger()

MIN_BASELINE_IMAGES: int = 30

_EMPTY_VAL_STATS: dict = {
    "scores": [],
    "mean": 0.0,
    "std": 0.0,
    "min": 0.0,
    "max": 0.0,
    "p5": 0.0,
    "p25": 0.0,
    "p50": 0.0,
    "p75": 0.0,
    "p95": 0.0,
}

MODEL_INFO = {
    "patchcore": {
        "name": "PatchCore",
        "description": "高精度异常检测，适合静态场景",
        "speed": "中等",
        "epochs": 1,
        "memory": "高",
    },
    "efficient_ad": {
        "name": "EfficientAD",
        "description": "高效异常检测，适合边缘部署",
        "speed": "快",
        "epochs": 70,
        "memory": "低",
    },
    "fastflow": {
        "name": "FastFlow",
        "description": "基于正则化流的实时异常检测",
        "speed": "很快",
        "epochs": 70,
        "memory": "中等",
    },
    "padim": {
        "name": "PaDiM",
        "description": "轻量级异常检测，适合嵌入式设备",
        "speed": "很快",
        "epochs": 1,
        "memory": "低",
    },
    "dinomaly2": {
        "name": "Dinomaly",
        "description": "DINOv2 重建异常检测，支持少样本 (8-shot)",
        "speed": "中等",
        "epochs": 1,
        "memory": "高",
    },
}

# Mapping from user-facing backbone names to anomalib Dinomaly encoder_name values
_DINOMALY_BACKBONE_MAP = {
    "dinov2_vits14": "dinov2reg_vit_small_14",
    "dinov2_vitb14": "dinov2reg_vit_base_14",
    "dinov2_vitl14": "dinov2reg_vit_large_14",
}


def _resolve_dinomaly_backbone(backbone: str) -> str:
    """Resolve user-facing backbone name to anomalib Dinomaly encoder_name.

    If the name is already in anomalib format (e.g. 'dinov2reg_vit_base_14'),
    it is returned as-is.
    """
    return _DINOMALY_BACKBONE_MAP.get(backbone, backbone)


class TrainingStatus(str, Enum):
    IDLE = "idle"
    VALIDATING = "validating"
    SPLITTING = "splitting"
    TRAINING = "training"
    EXPORTING = "exporting"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class QualityReport:
    """Quality assessment of a trained model."""

    grade: str  # A / B / C / F
    score_stats: dict = field(default_factory=dict)
    threshold_recommended: float = 0.0
    suggestions: list[str] = field(default_factory=list)


@dataclass
class TrainingResult:
    status: TrainingStatus
    model_path: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    image_count: int = 0
    train_count: int = 0
    val_count: int = 0
    pre_validation: dict | None = None
    val_stats: dict | None = None
    quality_report: QualityReport | None = None
    threshold_recommended: float | None = None
    output_validation: dict | None = None
    model_version_id: str | None = None
    validation_report: object | None = None  # ValidationReport from training_validator


def _list_images(directory: Path) -> list[Path]:
    """Return sorted list of .png and .jpg files in directory."""
    return sorted(list(directory.glob("*.png")) + list(directory.glob("*.jpg")))


def _build_calibration_dataset(
    ov_model,
    val_dir: Path | None,
    max_images: int = 100,
) -> list[np.ndarray]:
    """Build a calibration dataset for INT8 quantization from validation images.

    Returns a list of preprocessed numpy arrays matching the model's input shape.
    Each element is a single-batch NCHW float32 tensor ready for NNCF.
    """
    if val_dir is None or not val_dir.exists():
        return []

    val_normal = val_dir / "normal" if (val_dir / "normal").exists() else val_dir
    images = _list_images(val_normal)
    if not images:
        return []

    # Determine model input shape from the OpenVINO model
    input_layer = ov_model.input(0)
    input_shape = input_layer.shape  # e.g., [1, 3, 256, 256]

    # Handle dynamic shapes — fall back to 256x256
    try:
        _, channels, height, width = [
            int(d) if not isinstance(d, int) and hasattr(d, 'get_length') and d.get_length() != -1
            else int(d) if isinstance(d, int)
            else 256
            for d in input_shape
        ]
    except (ValueError, TypeError):
        channels, height, width = 3, 256, 256

    calibration_data = []
    for img_path in images[:max_images]:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # Resize and convert BGR -> RGB
        resized = cv2.resize(frame, (width, height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # HWC -> NCHW float32, normalized to [0, 1]
        blob = rgb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # CHW -> NCHW

        calibration_data.append(blob)

    logger.info(
        "training.calibration_dataset_built",
        images=len(calibration_data),
        input_shape=f"{1}x{channels}x{height}x{width}",
    )
    return calibration_data


class ModelTrainer:
    """Orchestrates anomaly model training and export.

    Wraps Anomalib's training API into a simple interface that can be
    called from the scheduler or the dashboard.
    """

    def __init__(
        self,
        baseline_manager: BaselineManager,
        models_dir: str | Path = "data/models",
        exports_dir: str | Path = "data/exports",
        model_registry=None,
        training_validator=None,
        validation_config: dict | None = None,
    ):
        self._baseline_manager = baseline_manager
        self._models_dir = Path(models_dir)
        self._exports_dir = Path(exports_dir)
        self._model_registry = model_registry
        self._training_validator = training_validator
        self._validation_config = validation_config or {}
        self._status = TrainingStatus.IDLE
        self._last_result: TrainingResult | None = None

    @property
    def exports_dir(self) -> Path:
        return self._exports_dir

    @property
    def status(self) -> TrainingStatus:
        return self._status

    @property
    def last_result(self) -> TrainingResult | None:
        return self._last_result

    def _fail(self, start: float, **kwargs) -> TrainingResult:
        """Build a FAILED TrainingResult and update internal state."""
        kwargs.setdefault("duration_seconds", time.monotonic() - start)
        result = TrainingResult(status=TrainingStatus.FAILED, **kwargs)
        self._status = TrainingStatus.FAILED
        self._last_result = result
        return result

    def train(
        self,
        camera_id: str,
        zone_id: str = "default",
        model_type: str = "patchcore",
        image_size: int = 256,
        export_format: str | None = "openvino",
        quantization: str = "fp16",
        calibration_images: int = 100,
        progress_callback: callable | None = None,
        anomaly_config: object | None = None,
        resume_from: str | None = None,
        group_id: str | None = None,
        backbone_checkpoint: str | None = None,
        skip_baseline_validation: bool = False,
    ) -> TrainingResult:
        """Train an anomaly detection model for a specific camera/zone.

        Full pipeline: validate -> split -> train -> export -> evaluate -> report.

        Args:
            anomaly_config: Optional AnomalyConfig with model-specific parameters
                (e.g. dinomaly_backbone, dinomaly_encoder_layers for dinomaly2).
        """
        start = time.monotonic()

        def _progress(pct: int, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        if group_id:
            baseline_dir = self._baseline_manager.get_group_baseline_dir(group_id, zone_id)
        else:
            baseline_dir = self._baseline_manager.get_baseline_dir(camera_id, zone_id)

        # Lifecycle gate: only Verified or Active baselines can be used for training
        lifecycle = getattr(self._baseline_manager, "_lifecycle", None)
        if lifecycle and baseline_dir.exists():
            version_name = baseline_dir.name
            lifecycle_camera = f"group:{group_id}" if group_id else camera_id
            ver_rec = lifecycle.get_version(lifecycle_camera, zone_id, version_name)
            if ver_rec and ver_rec.state == BaselineState.DRAFT:
                return self._fail(
                    start,
                    error=f"基���版本 {version_name} 处于 Draft 状态，未通过审核，不能用于训练",
                )

        image_count = self._baseline_manager.count_images(camera_id, zone_id)

        # Dinomaly2 supports few-shot mode with fewer images
        min_images = MIN_BASELINE_IMAGES
        if model_type == "dinomaly2" and anomaly_config is not None:
            few_shot_min = getattr(anomaly_config, "dinomaly_few_shot_images", 8)
            if image_count < MIN_BASELINE_IMAGES and image_count >= few_shot_min:
                min_images = few_shot_min
                logger.info(
                    "trainer.few_shot_mode",
                    camera_id=camera_id,
                    image_count=image_count,
                    few_shot_min=few_shot_min,
                )

        if image_count < min_images:
            return self._fail(
                start,
                error=f"基线图片不足: {image_count} 张 (需要 >= {min_images})",
                image_count=image_count,
            )

        # Pre-training validation (TRN-001)
        self._status = TrainingStatus.VALIDATING
        _progress(5, "正在验证基线质量...")

        pre_validation = self._validate_baseline_quality(baseline_dir, min_images=min_images)
        if not pre_validation["passed"]:
            if skip_baseline_validation:
                logger.warning(
                    "training.validation_skipped",
                    camera_id=camera_id,
                    errors=pre_validation["errors"],
                    msg="Baseline validation failed but skip_baseline_validation=True",
                )
            else:
                error_msg = "; ".join(pre_validation["errors"])
                return self._fail(
                    start,
                    error=f"基线质量验证失败: {error_msg}",
                    image_count=image_count,
                    pre_validation=pre_validation,
                )

        logger.info(
            "training.started",
            camera_id=camera_id,
            zone_id=zone_id,
            model_type=model_type,
            images=image_count,
            corruption_rate=pre_validation["corruption_rate"],
            near_duplicate_rate=pre_validation["near_duplicate_rate"],
            brightness_std=round(pre_validation["brightness_std"], 2),
            validation_skipped=skip_baseline_validation,
        )

        output_dir = self._models_dir / camera_id / zone_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train/val split (TRN-002)
        self._status = TrainingStatus.SPLITTING
        _progress(10, "正在切分训练/验证集...")

        train_dir, val_dir, train_count, val_count = self._split_train_val(
            baseline_dir, output_dir
        )

        # Anomalib training
        self._status = TrainingStatus.TRAINING
        _progress(15, f"正在训练 {model_type} 模型 ({train_count} 张训练图片)...")

        common_fail_kwargs = dict(
            image_count=image_count,
            train_count=train_count,
            val_count=val_count,
            pre_validation=pre_validation,
        )

        try:
            engine, model = self._train_anomalib(
                data_dir=train_dir,
                output_dir=output_dir,
                model_type=model_type,
                image_size=image_size,
                anomaly_config=anomaly_config,
                resume_from=resume_from,
                backbone_checkpoint=backbone_checkpoint,
            )
        except ImportError:
            return self._fail(start, error="anomalib 未安装", **common_fail_kwargs)
        except Exception as e:
            logger.error("training.failed", error=str(e))
            return self._fail(start, error=str(e), **common_fail_kwargs)

        # Export
        export_path_str = None
        if export_format:
            self._status = TrainingStatus.EXPORTING
            quant_label = f" + {quantization}" if quantization != "fp32" else ""
            _progress(70, f"正在导出 {export_format}{quant_label} 格式...")
            try:
                export_path_str = str(self._exports_dir / camera_id / zone_id)
                self._export_model(
                    engine=engine,
                    model=model,
                    export_format=export_format,
                    export_path=export_path_str,
                    quantization=quantization,
                    val_dir=val_dir,
                    calibration_images=calibration_images,
                )
            except Exception as e:
                logger.error("training.export_failed", error=str(e))

        # Output validation + smoke test (TRN-006)
        # Returns (validation_dict, loaded_detector_or_None)
        _progress(80, "正在验证模型输出...")
        output_validation, detector = self._validate_output(
            output_dir,
            Path(export_path_str) if export_path_str else None,
            val_dir,
        )

        if not output_validation["checkpoint_valid"] and not output_validation["smoke_test_passed"]:
            return self._fail(
                start,
                error="训练未产生有效模型文件",
                output_validation=output_validation,
                **common_fail_kwargs,
            )

        # Post-training validation on val set (TRN-003)
        # Reuses detector from smoke test to avoid redundant model load
        self._status = TrainingStatus.EVALUATING
        _progress(85, f"正在验证集上评估模型 ({val_count} 张)...")

        val_stats = self._validate_on_val_set(val_dir, detector=detector)

        # Threshold recommendation (TRN-005)
        threshold_recommended = self._recommend_threshold(val_stats)

        # Quality report (TRN-004)
        quality_report = self._generate_quality_report(val_stats, threshold_recommended)

        # Three-step validation pipeline (if configured)
        validation_report = None
        if self._training_validator is not None and detector is not None:
            _progress(90, "正在执行三步验证（AUROC/合成Recall/历史回放）...")
            try:
                validation_report = self._training_validator.validate(
                    detector=detector,
                    camera_id=camera_id,
                    zone_id=zone_id,
                    val_dir=val_dir,
                    baseline_dir=baseline_dir,
                    threshold=threshold_recommended,
                    auroc_threshold=self._validation_config.get(
                        "validation_auroc_threshold", 0.99
                    ),
                    recall_threshold=self._validation_config.get(
                        "validation_recall_threshold", 0.95
                    ),
                    replay_days=self._validation_config.get(
                        "historical_replay_days", 30
                    ),
                )
                if not validation_report.all_passed:
                    quality_report.grade = "F"
                    quality_report.suggestions.append(
                        "三步验证未通过 — 模型不得部署"
                    )
                    logger.warning(
                        "training.validation_failed",
                        camera_id=camera_id,
                        auroc=validation_report.auroc,
                        recall=validation_report.recall,
                    )
            except Exception as e:
                logger.error("training.validation_error", error=str(e))

        _progress(95, f"训练完成 — 质量等级: {quality_report.grade}")

        # C4: Register model version in registry
        model_version_id = None
        if self._model_registry is not None:
            try:
                model_version_id = self._model_registry.register(
                    model_path=output_dir,
                    baseline_dir=baseline_dir,
                    camera_id=camera_id,
                    model_type=model_type,
                    training_params={
                        "image_size": image_size,
                        "export_format": export_format,
                        "quantization": quantization,
                        "train_count": train_count,
                        "val_count": val_count,
                        "quality_grade": quality_report.grade,
                    },
                )
                self._model_registry.activate(model_version_id)
                logger.info(
                    "training.model_registered",
                    model_version_id=model_version_id,
                    camera_id=camera_id,
                )
            except Exception as e:
                logger.error("training.model_registry_failed", error=str(e))

        duration = time.monotonic() - start
        result = TrainingResult(
            status=TrainingStatus.COMPLETE,
            model_path=str(output_dir),
            duration_seconds=duration,
            image_count=image_count,
            train_count=train_count,
            val_count=val_count,
            pre_validation=pre_validation,
            val_stats=val_stats,
            quality_report=quality_report,
            threshold_recommended=threshold_recommended,
            output_validation=output_validation,
            model_version_id=model_version_id,
            validation_report=validation_report,
        )
        self._status = TrainingStatus.COMPLETE
        self._last_result = result

        logger.info(
            "training.complete",
            camera_id=camera_id,
            duration=f"{duration:.1f}s",
            model_path=str(output_dir),
            quality_grade=quality_report.grade,
            threshold=round(threshold_recommended, 3),
        )

        # Conformal calibration (A2-3): run on val set scores if detector available
        if detector is not None and val_stats.get("scores"):
            try:
                from argus.alerts.calibration import ConformalCalibrator

                cal_scores = np.array(val_stats["scores"])
                if len(cal_scores) >= 50:
                    calibrator = ConformalCalibrator()
                    cal_result = calibrator.calibrate(cal_scores)
                    cal_path = output_dir / "calibration.json"
                    calibrator.save(
                        cal_result, cal_path, sorted_scores=np.sort(cal_scores)
                    )
                    logger.info("trainer.calibration_saved", path=str(cal_path))
                else:
                    logger.warning(
                        "trainer.calibration_skipped",
                        reason=f"Not enough scores for calibration ({len(cal_scores)} < 50)",
                    )
            except Exception as e:
                logger.warning("trainer.calibration_failed", error=str(e))

        # Cleanup split directory
        split_dir = output_dir / "_split"
        if split_dir.exists():
            shutil.rmtree(split_dir, ignore_errors=True)

        return result

    # ── TRN-001: Pre-training validation ──

    def _validate_baseline_quality(
        self, baseline_dir: Path, min_images: int = MIN_BASELINE_IMAGES,
    ) -> dict:
        """Validate baseline image quality before training.

        Thresholds: corruption < 10%, near-duplicate < 80%, brightness std > 2.0.
        """
        images = _list_images(baseline_dir)
        total = len(images)
        errors: list[str] = []

        if total < min_images:
            return {
                "passed": False,
                "corruption_rate": 0.0,
                "near_duplicate_rate": 0.0,
                "brightness_std": 0.0,
                "errors": [f"图片数量不足: {total} (需要 >= {min_images})"],
            }

        corrupted = 0
        brightness_values: list[float] = []
        hashes: list[np.ndarray] = []

        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None or frame.size == 0:
                corrupted += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(float(gray.mean()))

            # Perceptual hash: 8x8 grayscale, threshold at mean -> 64-bit hash
            small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            phash = (small > small.mean()).flatten().astype(np.uint8)
            hashes.append(phash)

        corruption_rate = corrupted / total if total > 0 else 0.0

        if corruption_rate >= 0.1:
            errors.append(f"损坏率过高: {corruption_rate:.1%} (上限 10%)")

        brightness_std = float(np.std(brightness_values)) if brightness_values else 0.0
        if brightness_std <= 2.0:
            errors.append(f"亮度多样性不足: std={brightness_std:.1f} (需要 > 2.0)")

        # Vectorized near-duplicate detection via hamming distance
        near_duplicate_count = 0
        n = len(hashes)
        if n > 1:
            hash_matrix = np.array(hashes)  # (n, 64)
            duplicate_flags = np.zeros(n, dtype=bool)
            for i in range(n):
                if duplicate_flags[i]:
                    continue
                # Hamming distances from i to all j > i
                dists = np.sum(hash_matrix[i] != hash_matrix[i + 1:], axis=1)
                dup_mask = dists <= 5  # very similar (out of 64 bits)
                new_dups = dup_mask & ~duplicate_flags[i + 1:]
                near_duplicate_count += int(new_dups.sum())
                duplicate_flags[i + 1:] |= dup_mask

        near_duplicate_rate = near_duplicate_count / total if total > 0 else 0.0

        if near_duplicate_rate >= 0.8:
            errors.append(f"近似重复率过高: {near_duplicate_rate:.1%} (上限 80%)")

        return {
            "passed": len(errors) == 0,
            "corruption_rate": corruption_rate,
            "near_duplicate_rate": near_duplicate_rate,
            "brightness_std": brightness_std,
            "errors": errors,
        }

    # ── TRN-002: Auto train/val split ──

    @staticmethod
    def _split_train_val(
        baseline_dir: Path, output_dir: Path, seed: int = 42
    ) -> tuple[Path, Path, int, int]:
        """Split baseline images into train (80%) and val (20%) sets.

        Files are physically copied to ensure portability.
        Returns (train_dir, val_dir, train_count, val_count).
        val_dir contains a 'normal' subdirectory with the actual images.
        """
        images = _list_images(baseline_dir)

        rng = random.Random(seed)
        shuffled = list(images)
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * 0.8)
        train_images = shuffled[:split_idx]
        val_images = shuffled[split_idx:]

        split_dir = output_dir / "_split"
        train_dir = split_dir / "train" / "normal"
        val_dir = split_dir / "val" / "normal"

        if split_dir.exists():
            shutil.rmtree(split_dir)

        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        for img in train_images:
            shutil.copy2(str(img), str(train_dir / img.name))

        for img in val_images:
            shutil.copy2(str(img), str(val_dir / img.name))

        logger.info(
            "training.split_complete",
            train=len(train_images),
            val=len(val_images),
            total=len(images),
        )
        return train_dir.parent, val_dir.parent, len(train_images), len(val_images)

    # ── TRN-003: Post-training validation ──

    def _validate_on_val_set(
        self, val_dir: Path, *, detector=None
    ) -> dict:
        """Run inference on validation set and collect score distribution.

        If detector is provided, reuses it instead of loading the model again.
        """
        val_normal = val_dir / "normal" if (val_dir / "normal").exists() else val_dir
        val_images = _list_images(val_normal)

        if not val_images:
            logger.warning("training.no_val_images", val_dir=str(val_dir))
            return dict(_EMPTY_VAL_STATS)

        if detector is None:
            logger.warning("training.no_detector_for_validation")
            return dict(_EMPTY_VAL_STATS)

        scores: list[float] = []
        try:
            for img_path in val_images:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                result = detector.predict(frame)
                scores.append(result.anomaly_score)
        except Exception as e:
            logger.error("training.val_inference_failed", error=str(e))
            return dict(_EMPTY_VAL_STATS)

        if not scores:
            return dict(_EMPTY_VAL_STATS)

        arr = np.array(scores)
        p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95]).tolist()
        return {
            "scores": scores,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95,
        }

    # ── TRN-004: Quality report ──

    @staticmethod
    def _generate_quality_report(
        val_stats: dict, threshold_recommended: float
    ) -> QualityReport:
        """Generate a quality report based on validation score distribution."""
        mean = val_stats.get("mean", 0.0)
        std = val_stats.get("std", 0.0)
        max_score = val_stats.get("max", 0.0)
        suggestions: list[str] = []

        if mean < 0.3 and std < 0.1 and max_score < threshold_recommended * 0.8:
            grade = "A"
        elif mean < 0.4 and std < 0.15 and max_score < threshold_recommended:
            grade = "B"
        elif mean < 0.5 and max_score < threshold_recommended:
            grade = "C"
        else:
            grade = "F"

        if not val_stats.get("scores"):
            suggestions.append("验证集为空，无法评估模型质量")
            return QualityReport(
                grade="F",
                score_stats=val_stats,
                threshold_recommended=threshold_recommended,
                suggestions=suggestions,
            )

        if grade in ("F", "C"):
            suggestions.append("建议重新采集更多样化的基线图片（不同光照、角度）")
        if std > 0.15:
            suggestions.append("基线图片质量不一致，检查是否有异常值或场景变化")
        if max_score > 0 and max_score > threshold_recommended * 0.7:
            suggestions.append("验证集最高分接近阈值，存在误报风险")
        if mean > 0.4:
            suggestions.append("模型在正常图片上的平均分偏高，可能欠拟合")
        if grade == "A" and not suggestions:
            suggestions.append("模型质量优秀，可直接部署使用")

        return QualityReport(
            grade=grade,
            score_stats=val_stats,
            threshold_recommended=threshold_recommended,
            suggestions=suggestions,
        )

    # ── TRN-005: Threshold recommendation ──

    @staticmethod
    def _recommend_threshold(val_stats: dict) -> float:
        """Recommend an anomaly threshold based on validation scores.

        Formula: max(mean + 2.5*std, max_score * 1.05), capped at [0.1, 0.95].
        """
        if not val_stats.get("scores"):
            return 0.7

        mean = val_stats.get("mean", 0.0)
        std = val_stats.get("std", 0.0)
        max_score = val_stats.get("max", 0.0)

        threshold = max(mean + 2.5 * std, max_score * 1.05)
        threshold = min(threshold, 0.95)
        threshold = max(threshold, 0.1)
        return round(threshold, 4)

    # ── TRN-006: Output validation ──

    def _validate_output(
        self, output_dir: Path, export_path: Path | None, val_dir: Path
    ) -> tuple[dict, object | None]:
        """Validate training output: checkpoint integrity, export files, smoke test.

        Returns (validation_dict, loaded_detector_or_None) so the caller
        can reuse the loaded detector for val-set evaluation.
        """
        result = {
            "checkpoint_valid": False,
            "export_valid": False,
            "smoke_test_passed": False,
            "inference_latency_ms": None,
            "errors": [],
        }
        loaded_detector = None

        model_file = self._find_best_model_file(output_dir)
        if model_file is None:
            result["errors"].append("未找到模型文件")
        elif model_file.stat().st_size == 0:
            result["errors"].append("模型文件大小为 0")
        else:
            result["checkpoint_valid"] = True

        if export_path and export_path.exists():
            export_files = (
                list(export_path.rglob("*.xml"))
                + list(export_path.rglob("*.onnx"))
            )
            if export_files and all(f.stat().st_size > 0 for f in export_files):
                result["export_valid"] = True
            else:
                result["errors"].append("导出文件缺失或为空")
        elif export_path:
            result["errors"].append(f"导出目录不存在: {export_path}")

        # Smoke inference test — keep detector loaded for reuse
        if model_file:
            val_normal = val_dir / "normal" if (val_dir / "normal").exists() else val_dir
            test_images = _list_images(val_normal)
            if test_images:
                try:
                    from argus.anomaly.detector import AnomalibDetector

                    detector = AnomalibDetector(model_path=model_file, threshold=0.5)
                    if detector.load():
                        frame = cv2.imread(str(test_images[0]))
                        if frame is not None:
                            t0 = time.monotonic()
                            pred = detector.predict(frame)
                            latency = (time.monotonic() - t0) * 1000
                            if np.isfinite(pred.anomaly_score) and 0 <= pred.anomaly_score <= 1:
                                result["smoke_test_passed"] = True
                                result["inference_latency_ms"] = round(latency, 1)
                                loaded_detector = detector
                            else:
                                result["errors"].append(f"推理结果异常: score={pred.anomaly_score}")
                except Exception as e:
                    result["errors"].append(f"冒烟测试失败: {e}")

        return result, loaded_detector

    # ── TRN-008: Model A/B comparison ──

    def compare_models(
        self, old_model_path: str | Path, new_model_path: str | Path, val_dir: str | Path
    ) -> dict:
        """Compare two models on the same validation set.

        Returns score distributions, latency comparison, and recommendation.
        """
        from argus.anomaly.detector import AnomalibDetector

        val_path = Path(val_dir)
        val_normal = val_path / "normal" if (val_path / "normal").exists() else val_path
        val_images = _list_images(val_normal)

        if not val_images:
            return {"error": "验证集为空", "recommendation": "inconclusive"}

        results = {}
        for label, model_path in [("old", Path(old_model_path)), ("new", Path(new_model_path))]:
            detector = AnomalibDetector(model_path=model_path, threshold=0.5)
            if not detector.load():
                return {"error": f"无法加载{label}模型: {model_path}", "recommendation": "inconclusive"}

            scores = []
            total_latency = 0.0
            for img_path in val_images:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                t0 = time.monotonic()
                pred = detector.predict(frame)
                total_latency += time.monotonic() - t0
                scores.append(pred.anomaly_score)

            arr = np.array(scores) if scores else np.array([0.0])
            avg_latency = (total_latency / len(scores) * 1000) if scores else 0.0

            results[label] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "max": float(arr.max()),
                "p95": float(np.percentile(arr, 95)),
                "latency_ms": round(avg_latency, 1),
            }

        old_s = results["old"]
        new_s = results["new"]
        if new_s["mean"] <= old_s["mean"] and new_s["max"] <= old_s["max"]:
            recommendation = "deploy"
            reason = "新模型在验证集上的平均分和最高分均不高于旧模型"
        elif new_s["mean"] > old_s["mean"] * 1.2:
            recommendation = "keep_old"
            reason = "新模型平均分显著高于旧模型，建议保留旧模型"
        else:
            recommendation = "review"
            reason = "新旧模型表现接近，建议人工评估"

        return {
            "old": old_s,
            "new": new_s,
            "delta_mean": round(new_s["mean"] - old_s["mean"], 4),
            "delta_max": round(new_s["max"] - old_s["max"], 4),
            "delta_latency_ms": round(new_s["latency_ms"] - old_s["latency_ms"], 1),
            "recommendation": recommendation,
            "reason": reason,
        }

    # ── Internal helpers ──

    @staticmethod
    def _find_best_model_file(model_dir: Path) -> Path | None:
        """Find the best model file in a training output directory.

        Preference: .xml (OpenVINO) > .onnx > .ckpt > .pt
        """
        for pattern in ("**/*.xml", "**/*.onnx", "**/*.ckpt", "**/*.pt"):
            files = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                return files[0]
        return None

    def _train_anomalib(
        self,
        data_dir: Path,
        output_dir: Path,
        model_type: str,
        image_size: int,
        anomaly_config: object | None = None,
        resume_from: str | None = None,
        backbone_checkpoint: str | None = None,
    ) -> tuple:
        """Execute Anomalib training. Raises ImportError if anomalib is not installed.

        Returns (engine, model) so the caller can use them for export.
        data_dir should be the train split directory (containing a 'normal' subdirectory).

        Args:
            anomaly_config: Optional AnomalyConfig for model-specific parameters.
        """
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage
        import torch

        resize_transform = Compose([
            ToImage(),
            Resize((image_size, image_size)),
            ToDtype(torch.float32, scale=True),
        ])

        datamodule = Folder(
            name="baseline",
            root=str(data_dir),
            normal_dir="normal",
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,
            augmentations=resize_transform,
        )

        if model_type == "efficient_ad":
            from anomalib.models import EfficientAd
            model = EfficientAd()
        elif model_type == "fastflow":
            from anomalib.models import Fastflow
            model = Fastflow(
                backbone="resnet18",  # Lightweight for real-time
                flow_steps=8,
            )
        elif model_type == "padim":
            from anomalib.models import Padim
            model = Padim(
                backbone="resnet18",
                layers=["layer1", "layer2", "layer3"],
            )
        elif model_type == "dinomaly2":
            # Dinomaly2 uses DINOv2 backbone with reconstruction-based anomaly detection
            try:
                from anomalib.models import Dinomaly

                # Build constructor kwargs from anomaly_config
                dinomaly_kwargs: dict = {}
                if anomaly_config is not None:
                    backbone = getattr(anomaly_config, "dinomaly_backbone", None)
                    if backbone:
                        dinomaly_kwargs["encoder_name"] = _resolve_dinomaly_backbone(backbone)
                    encoder_layers = getattr(anomaly_config, "dinomaly_encoder_layers", None)
                    if encoder_layers:
                        dinomaly_kwargs["target_layers"] = encoder_layers

                model = Dinomaly(**dinomaly_kwargs)
                logger.info(
                    "trainer.dinomaly_created",
                    **{k: str(v) for k, v in dinomaly_kwargs.items()},
                )

                if backbone_checkpoint:
                    try:
                        import torch
                        state_dict = torch.load(
                            backbone_checkpoint, map_location="cpu", weights_only=True,
                        )
                        model.model.encoder.load_state_dict(state_dict, strict=False)
                        logger.info(
                            "trainer.custom_backbone_loaded",
                            path=backbone_checkpoint,
                        )
                    except Exception as e:
                        logger.warning(
                            "trainer.custom_backbone_failed",
                            path=backbone_checkpoint,
                            error=str(e),
                        )
            except ImportError:
                # Fallback: if Dinomaly not in current anomalib version, use PatchCore
                logger.warning(
                    "trainer.dinomaly_not_available",
                    msg="Dinomaly not found in anomalib, falling back to PatchCore",
                )
                from anomalib.models import Patchcore
                model = Patchcore(
                    backbone="wide_resnet50_2",
                    layers=["layer2", "layer3"],
                    coreset_sampling_ratio=0.1,
                )
        else:  # patchcore (default)
            from anomalib.models import Patchcore
            model = Patchcore(
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                coreset_sampling_ratio=0.1,
            )

        # FastFlow and EfficientAD need multiple epochs; PatchCore/Padim/Dinomaly need 1
        if model_type in ("efficient_ad", "fastflow"):
            max_epochs = 70
        else:
            max_epochs = 1  # PatchCore, Padim, and Dinomaly only need feature extraction

        # Early stopping + checkpointing for multi-epoch models
        callbacks = []
        if max_epochs > 1:
            try:
                from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

                callbacks.append(EarlyStopping(
                    monitor="image_AUROC",
                    patience=10,
                    mode="max",
                    min_delta=0.001,
                ))
                callbacks.append(ModelCheckpoint(
                    monitor="image_AUROC",
                    mode="max",
                    save_top_k=1,
                    filename="best-{epoch}-{image_AUROC:.4f}",
                ))
                logger.info("trainer.callbacks_enabled", model_type=model_type, max_epochs=max_epochs)
            except ImportError:
                logger.warning("trainer.callbacks_unavailable", msg="EarlyStopping/ModelCheckpoint not available")

        engine = Engine(
            default_root_dir=str(output_dir),
            max_epochs=max_epochs,
            callbacks=callbacks if callbacks else None,
        )

        # Checkpoint resumption
        fit_kwargs = {"model": model, "datamodule": datamodule}
        if resume_from and Path(resume_from).exists():
            fit_kwargs["ckpt_path"] = resume_from
            logger.info("trainer.resuming_from_checkpoint", path=resume_from)

        engine.fit(**fit_kwargs)

        return engine, model

    @staticmethod
    def _export_model(
        engine,
        model,
        export_format: str,
        export_path: str,
        quantization: str = "fp16",
        val_dir: Path | None = None,
        calibration_images: int = 100,
    ) -> None:
        """Export trained model to an optimized inference format.

        CRIT-02/HIGH-12: Use correct Anomalib 2.x export API (export_mode parameter).

        If quantization == "int8" and export_format == "openvino", runs NNCF
        post-training quantization on the exported FP model using validation
        images as calibration data.
        """
        # Anomalib 2.x uses export_mode (str), not export_type (enum)
        engine.export(
            model=model,
            export_mode=export_format,  # "openvino" or "onnx"
        )
        logger.info("training.exported", format=export_format, path=export_path)

        # INT8 post-training quantization (B2)
        if quantization == "int8" and export_format == "openvino":
            ModelTrainer._quantize_int8(
                export_path=Path(export_path),
                val_dir=val_dir,
                calibration_images=calibration_images,
            )

    @staticmethod
    def _quantize_int8(
        export_path: Path,
        val_dir: Path | None = None,
        calibration_images: int = 100,
    ) -> None:
        """Apply INT8 post-training quantization to an exported OpenVINO model.

        Uses NNCF's quantize() with calibration data from validation images.
        Falls back gracefully if nncf is not installed.
        """
        try:
            import nncf
        except ImportError:
            logger.warning(
                "training.int8_skipped",
                reason="nncf not installed — run: pip install argus[quantize]",
            )
            return

        try:
            import openvino as ov
        except ImportError:
            logger.warning(
                "training.int8_skipped",
                reason="openvino not installed",
            )
            return

        # Find the exported .xml model file
        xml_files = sorted(export_path.rglob("*.xml"))
        if not xml_files:
            logger.warning("training.int8_skipped", reason="No .xml model found in export path")
            return

        model_xml = xml_files[0]
        logger.info("training.int8_quantizing", model=str(model_xml))

        try:
            # Load FP model
            core = ov.Core()
            ov_model = core.read_model(model_xml)

            # Build calibration dataset from validation images
            cal_images = _build_calibration_dataset(
                ov_model=ov_model,
                val_dir=val_dir,
                max_images=calibration_images,
            )

            if not cal_images:
                logger.warning(
                    "training.int8_skipped",
                    reason="No calibration images available",
                )
                return

            # Run NNCF post-training quantization
            quantized_model = nncf.quantize(
                ov_model,
                nncf.Dataset(cal_images),
                subset_size=min(len(cal_images), calibration_images),
                model_type=nncf.ModelType.TRANSFORMER,
                preset=nncf.QuantizationPreset.MIXED,
            )

            # Save INT8 model — overwrite the FP model files
            model_bin = model_xml.with_suffix(".bin")
            ov.save_model(quantized_model, str(model_xml))

            logger.info(
                "training.int8_complete",
                model=str(model_xml),
                original_size_mb=round(model_bin.stat().st_size / 1024 / 1024, 1)
                if model_bin.exists() else None,
            )

        except Exception as e:
            logger.error("training.int8_failed", error=str(e))
            logger.warning(
                "training.int8_fallback",
                msg="Keeping FP model — INT8 quantization failed",
            )
