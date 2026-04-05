"""Training validation and quality assessment.

Provides pre-training validation (image count, corruption, duplicates),
dataset splitting, post-training model quality checks, threshold
recommendation, and output validation for anomaly detection models.

TRN-001 through TRN-006 implementation.
"""

from __future__ import annotations

import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def _list_images(directory: Path) -> list[Path]:
    """List image files in a directory (non-recursive)."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# TRN-001: Pre-training validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of pre-training baseline validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    image_count: int = 0
    corrupt_count: int = 0
    corrupt_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    brightness_std: float = 0.0


class TrainingValidator:
    """Validate baseline images before training (TRN-001)."""

    MIN_IMAGES = 30
    MAX_CORRUPT_RATIO = 0.10
    MAX_DUPLICATE_RATIO = 0.80
    MIN_BRIGHTNESS_STD = 2.0
    DUPLICATE_SAMPLE_SIZE = 50  # max pairwise comparisons grow O(n^2)

    def validate_baseline(self, image_dir: Path) -> ValidationResult:
        """Validate baseline images before training.

        Checks:
        - image_count >= 30 (error if fewer)
        - corrupt_ratio < 10% (error if higher)
        - duplicate_ratio < 80% (warning)
        - brightness_std > 2.0 (warning if lower)
        """
        errors: list[str] = []
        warnings: list[str] = []

        images = _list_images(image_dir)
        image_count = len(images)

        if image_count < self.MIN_IMAGES:
            errors.append(
                f"Insufficient images: {image_count} (need >= {self.MIN_IMAGES})"
            )

        # Corruption check
        corrupt_count, valid_frames, brightnesses = self._check_corruption(images)
        corrupt_ratio = corrupt_count / max(image_count, 1)

        if corrupt_ratio >= self.MAX_CORRUPT_RATIO:
            errors.append(
                f"Too many corrupt images: {corrupt_count}/{image_count} "
                f"({corrupt_ratio:.1%} >= {self.MAX_CORRUPT_RATIO:.0%})"
            )

        # Brightness diversity
        brightness_std = 0.0
        if len(brightnesses) >= 2:
            brightness_std = float(np.std(brightnesses))
            if brightness_std < self.MIN_BRIGHTNESS_STD:
                warnings.append(
                    f"Low brightness diversity (std={brightness_std:.2f}): "
                    f"images may be too similar"
                )

        # Duplicate detection
        duplicate_ratio = self._check_duplicates(valid_frames)
        if duplicate_ratio >= self.MAX_DUPLICATE_RATIO:
            warnings.append(
                f"High duplicate ratio ({duplicate_ratio:.1%}): "
                f"consider adding more diverse images"
            )

        valid = len(errors) == 0

        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            image_count=image_count,
            corrupt_count=corrupt_count,
            corrupt_ratio=corrupt_ratio,
            duplicate_ratio=duplicate_ratio,
            brightness_std=brightness_std,
        )

    def _check_corruption(
        self, images: list[Path]
    ) -> tuple[int, list[np.ndarray], list[float]]:
        """Read images, count failures, collect valid frames and brightness."""
        corrupt = 0
        valid_frames: list[np.ndarray] = []
        brightnesses: list[float] = []

        for path in images:
            img = cv2.imread(str(path))
            if img is None:
                corrupt += 1
                continue
            valid_frames.append(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightnesses.append(float(np.mean(gray)))

        return corrupt, valid_frames, brightnesses

    def _check_duplicates(self, frames: list[np.ndarray]) -> float:
        """Estimate duplicate ratio using pixel correlation on a subset.

        Uses normalised cross-correlation via cv2.matchTemplate (TM_CCOEFF_NORMED).
        A pair with correlation >= 0.98 is considered a duplicate.
        """
        if len(frames) < 2:
            return 0.0

        # Subsample to keep computation tractable
        sample_size = min(len(frames), self.DUPLICATE_SAMPLE_SIZE)
        rng = random.Random(42)
        sample = rng.sample(frames, sample_size)

        # Resize all to a small standard size for fast comparison
        small = []
        for f in sample:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            small.append(resized)

        duplicate_pairs = 0
        total_pairs = 0

        for i in range(len(small)):
            for j in range(i + 1, len(small)):
                total_pairs += 1
                result = cv2.matchTemplate(
                    small[i], small[j], cv2.TM_CCOEFF_NORMED
                )
                correlation = float(result[0, 0])
                if correlation >= 0.98:
                    duplicate_pairs += 1

        if total_pairs == 0:
            return 0.0

        return duplicate_pairs / total_pairs


# ---------------------------------------------------------------------------
# TRN-002: Auto train/val split
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Result of dataset splitting."""

    train_count: int
    val_count: int
    split_ratio: float
    train_dir: Path
    val_dir: Path


class DatasetSplitter:
    """Split baseline images into train/val directories (TRN-002)."""

    def split(
        self,
        image_dir: Path,
        output_dir: Path,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> SplitResult:
        """Split baseline images into train/val directories.

        Physical copy to output_dir/train/ and output_dir/val/.
        Fixed seed for reproducibility.
        """
        images = _list_images(image_dir)

        rng = random.Random(seed)
        shuffled = list(images)
        rng.shuffle(shuffled)

        val_count = max(1, int(len(shuffled) * val_ratio))
        val_images = shuffled[:val_count]
        train_images = shuffled[val_count:]

        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for img_path in train_images:
            shutil.copy2(str(img_path), str(train_dir / img_path.name))

        for img_path in val_images:
            shutil.copy2(str(img_path), str(val_dir / img_path.name))

        actual_ratio = val_count / max(len(shuffled), 1)

        return SplitResult(
            train_count=len(train_images),
            val_count=len(val_images),
            split_ratio=actual_ratio,
            train_dir=train_dir,
            val_dir=val_dir,
        )


# ---------------------------------------------------------------------------
# TRN-005: Threshold recommendation
# ---------------------------------------------------------------------------

def recommend_threshold(scores: list[float]) -> float:
    """Recommend detection threshold based on normal image score distribution.

    threshold = max(mean + 2.5*std, max_score * 1.05)
    Clamped to [0.1, 0.95].
    """
    if not scores:
        return 0.5  # sensible default when no data

    arr = np.array(scores, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    max_score = float(np.max(arr))

    threshold = max(mean + 2.5 * std, max_score * 1.05)
    return float(np.clip(threshold, 0.1, 0.95))


# ---------------------------------------------------------------------------
# TRN-004: Quality report / grading
# ---------------------------------------------------------------------------

@dataclass
class ModelQualityReport:
    """Post-training model quality assessment."""

    grade: str  # "A", "B", "C", "F"
    score_mean: float
    score_std: float
    score_max: float
    score_p95: float
    threshold: float
    recommended_threshold: float
    train_count: int
    val_count: int
    inference_time_ms: float
    suggestions: list[str] = field(default_factory=list)


def _compute_grade(
    score_max: float,
    score_std: float,
    threshold: float,
) -> str:
    """Determine quality grade based on score distribution.

    A: max_score < threshold * 0.6 and std < 0.05
    B: max_score < threshold * 0.8 and std < 0.1
    C: max_score < threshold
    F: max_score >= threshold
    """
    if score_max >= threshold:
        return "F"
    if score_max < threshold * 0.6 and score_std < 0.05:
        return "A"
    if score_max < threshold * 0.8 and score_std < 0.1:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# TRN-003: Post-training validation
# ---------------------------------------------------------------------------

class PostTrainingValidator:
    """Run inference on validation set and assess model quality (TRN-003)."""

    def validate_model(
        self,
        model_path: Path,
        val_dir: Path,
        threshold: float = 0.7,
        train_count: int = 0,
    ) -> ModelQualityReport:
        """Run inference on validation set and collect score distribution.

        Loads model via AnomalibDetector, runs predict() on each validation
        image, and builds a ModelQualityReport with grading.
        """
        from argus.anomaly.detector import AnomalibDetector

        detector = AnomalibDetector(model_path=model_path, threshold=threshold)
        detector.load()

        val_images = _list_images(val_dir)
        val_count = len(val_images)
        scores: list[float] = []
        total_ms = 0.0

        for img_path in val_images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            t0 = time.perf_counter()
            result = detector.predict(frame)
            elapsed = (time.perf_counter() - t0) * 1000.0
            total_ms += elapsed
            scores.append(result.anomaly_score)

        if not scores:
            return ModelQualityReport(
                grade="F",
                score_mean=0.0,
                score_std=0.0,
                score_max=0.0,
                score_p95=0.0,
                threshold=threshold,
                recommended_threshold=0.5,
                train_count=train_count,
                val_count=val_count,
                inference_time_ms=0.0,
                suggestions=["No valid validation images could be processed"],
            )

        arr = np.array(scores, dtype=np.float64)
        score_mean = float(np.mean(arr))
        score_std = float(np.std(arr))
        score_max = float(np.max(arr))
        score_p95 = float(np.percentile(arr, 95))
        avg_ms = total_ms / len(scores)
        rec_threshold = recommend_threshold(scores)
        grade = _compute_grade(score_max, score_std, threshold)

        suggestions: list[str] = []
        if grade == "F":
            suggestions.append(
                "Model scores normal images above threshold - retrain with more/better data"
            )
        if score_std > 0.15:
            suggestions.append(
                "High score variance - baseline images may be inconsistent"
            )
        if avg_ms > 200:
            suggestions.append(
                "Slow inference - consider using OpenVINO export for faster speed"
            )

        return ModelQualityReport(
            grade=grade,
            score_mean=score_mean,
            score_std=score_std,
            score_max=score_max,
            score_p95=score_p95,
            threshold=threshold,
            recommended_threshold=rec_threshold,
            train_count=train_count,
            val_count=val_count,
            inference_time_ms=avg_ms,
            suggestions=suggestions,
        )


# ---------------------------------------------------------------------------
# TRN-006: Output validation
# ---------------------------------------------------------------------------

@dataclass
class OutputValidationResult:
    """Result of training output validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checkpoint_files: list[str] = field(default_factory=list)
    export_files: list[str] = field(default_factory=list)


class OutputValidator:
    """Validate training output files (TRN-006)."""

    CHECKPOINT_EXTENSIONS = {".ckpt", ".pt", ".xml"}
    EXPORT_EXTENSIONS = {".xml", ".onnx", ".bin"}

    def validate_output(self, output_dir: Path) -> OutputValidationResult:
        """Validate training output files.

        Checks:
        - At least one checkpoint file exists (.ckpt, .pt, .xml)
        - File size > 0
        - Try loading with AnomalibDetector (smoke test)
        - If export directory exists, validate export files too
        """
        errors: list[str] = []
        warnings: list[str] = []
        checkpoint_files: list[str] = []
        export_files: list[str] = []

        if not output_dir.is_dir():
            errors.append(f"Output directory does not exist: {output_dir}")
            return OutputValidationResult(
                valid=False,
                errors=errors,
            )

        # Find checkpoint files (recursive)
        for ext in self.CHECKPOINT_EXTENSIONS:
            for f in output_dir.rglob(f"*{ext}"):
                if f.is_file():
                    checkpoint_files.append(str(f))

        if not checkpoint_files:
            errors.append("No checkpoint files found (.ckpt, .pt, .xml)")
        else:
            # Check file sizes
            for ckpt in checkpoint_files:
                size = Path(ckpt).stat().st_size
                if size == 0:
                    errors.append(f"Empty checkpoint file: {ckpt}")

            # Smoke test: try loading the first checkpoint
            first_ckpt = Path(checkpoint_files[0])
            try:
                from argus.anomaly.detector import AnomalibDetector

                detector = AnomalibDetector(
                    model_path=first_ckpt, threshold=0.5
                )
                loaded = detector.load()
                if not loaded:
                    warnings.append(
                        f"Model load returned False for {first_ckpt.name} "
                        f"(may need anomalib installed)"
                    )
            except Exception as e:
                warnings.append(f"Smoke test failed for {first_ckpt.name}: {e}")

        # Check for export files
        export_dir = output_dir / "exports"
        if export_dir.is_dir():
            for ext in self.EXPORT_EXTENSIONS:
                for f in export_dir.rglob(f"*{ext}"):
                    if f.is_file():
                        export_files.append(str(f))
            if not export_files:
                warnings.append("Export directory exists but contains no export files")

        valid = len(errors) == 0

        return OutputValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            checkpoint_files=checkpoint_files,
            export_files=export_files,
        )
