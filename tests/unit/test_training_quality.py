"""Tests for training validation and quality assessment (TRN-001~006)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from argus.anomaly.quality import (
    DatasetSplitter,
    ModelQualityReport,
    OutputValidator,
    PostTrainingValidator,
    TrainingValidator,
    ValidationResult,
    _compute_grade,
    recommend_threshold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(path: Path, width: int = 64, height: int = 64, brightness: int = 128) -> None:
    """Create a small dummy image file."""
    img = np.full((height, width, 3), brightness, dtype=np.uint8)
    # Add some noise so images are not perfect duplicates
    rng = np.random.RandomState(hash(str(path)) % 2**31)
    noise = rng.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    cv2.imwrite(str(path), img)


def _make_images(directory: Path, count: int, brightness: int = 128) -> list[Path]:
    """Create multiple dummy images in a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        p = directory / f"img_{i:04d}.png"
        _make_image(p, brightness=brightness + i * 2)  # vary brightness slightly
        paths.append(p)
    return paths


def _make_corrupt_image(path: Path) -> None:
    """Create a file that cv2.imread will fail to read."""
    path.write_bytes(b"not a valid image file at all")


# ---------------------------------------------------------------------------
# TRN-001: Pre-training validation
# ---------------------------------------------------------------------------

class TestTrainingValidator:
    def test_too_few_images_error(self, tmp_path: Path) -> None:
        """< 30 images should produce a blocking error."""
        img_dir = tmp_path / "baseline"
        _make_images(img_dir, 10)

        validator = TrainingValidator()
        result = validator.validate_baseline(img_dir)

        assert not result.valid
        assert result.image_count == 10
        assert any("Insufficient" in e for e in result.errors)

    def test_enough_good_images_valid(self, tmp_path: Path) -> None:
        """50 good images should pass validation."""
        img_dir = tmp_path / "baseline"
        _make_images(img_dir, 50)

        validator = TrainingValidator()
        result = validator.validate_baseline(img_dir)

        assert result.valid
        assert result.image_count == 50
        assert result.corrupt_count == 0
        assert result.errors == []

    def test_corrupt_images_above_threshold(self, tmp_path: Path) -> None:
        """> 10% corrupt images should produce a blocking error."""
        img_dir = tmp_path / "baseline"
        img_dir.mkdir(parents=True)

        # 27 good + 8 corrupt = 35 total, corrupt ratio ~22.8%
        for i in range(27):
            _make_image(img_dir / f"good_{i:04d}.png")
        for i in range(8):
            _make_corrupt_image(img_dir / f"bad_{i:04d}.png")

        validator = TrainingValidator()
        result = validator.validate_baseline(img_dir)

        assert not result.valid
        assert result.corrupt_count == 8
        assert result.corrupt_ratio >= 0.10
        assert any("corrupt" in e.lower() for e in result.errors)

    def test_corrupt_images_below_threshold(self, tmp_path: Path) -> None:
        """< 10% corrupt images should not block (just inform)."""
        img_dir = tmp_path / "baseline"
        img_dir.mkdir(parents=True)

        # 32 good + 2 corrupt = 34, corrupt ratio ~5.9%
        for i in range(32):
            _make_image(img_dir / f"good_{i:04d}.png")
        for i in range(2):
            _make_corrupt_image(img_dir / f"bad_{i:04d}.png")

        validator = TrainingValidator()
        result = validator.validate_baseline(img_dir)

        assert result.valid
        assert result.corrupt_count == 2

    def test_low_brightness_diversity_warning(self, tmp_path: Path) -> None:
        """All images with identical brightness should warn."""
        img_dir = tmp_path / "baseline"
        img_dir.mkdir(parents=True)

        # Create 35 images with SAME brightness (no variation)
        for i in range(35):
            img = np.full((64, 64, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i:04d}.png"), img)

        validator = TrainingValidator()
        result = validator.validate_baseline(img_dir)

        assert result.brightness_std < 2.0
        assert any("brightness" in w.lower() or "similar" in w.lower() for w in result.warnings)

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should fail with insufficient images."""
        img_dir = tmp_path / "empty"
        img_dir.mkdir()

        validator = TrainingValidator()
        result = validator.validate_baseline(img_dir)

        assert not result.valid
        assert result.image_count == 0


# ---------------------------------------------------------------------------
# TRN-002: Dataset splitting
# ---------------------------------------------------------------------------

class TestDatasetSplitter:
    def test_creates_train_val_directories(self, tmp_path: Path) -> None:
        """Split should create train/ and val/ with correct ratios."""
        src = tmp_path / "src"
        _make_images(src, 50)
        out = tmp_path / "output"

        splitter = DatasetSplitter()
        result = splitter.split(src, out, val_ratio=0.2)

        assert result.train_dir.is_dir()
        assert result.val_dir.is_dir()
        assert result.val_count == 10  # 20% of 50
        assert result.train_count == 40
        assert result.train_count + result.val_count == 50

        # Verify files actually exist
        train_files = list(result.train_dir.iterdir())
        val_files = list(result.val_dir.iterdir())
        assert len(train_files) == 40
        assert len(val_files) == 10

    def test_deterministic_with_same_seed(self, tmp_path: Path) -> None:
        """Same seed should produce identical splits."""
        src = tmp_path / "src"
        _make_images(src, 30)

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        splitter = DatasetSplitter()
        r1 = splitter.split(src, out1, val_ratio=0.2, seed=42)
        r2 = splitter.split(src, out2, val_ratio=0.2, seed=42)

        train1 = sorted(p.name for p in r1.train_dir.iterdir())
        train2 = sorted(p.name for p in r2.train_dir.iterdir())
        val1 = sorted(p.name for p in r1.val_dir.iterdir())
        val2 = sorted(p.name for p in r2.val_dir.iterdir())

        assert train1 == train2
        assert val1 == val2

    def test_different_seed_different_split(self, tmp_path: Path) -> None:
        """Different seeds should (very likely) produce different splits."""
        src = tmp_path / "src"
        _make_images(src, 30)

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        splitter = DatasetSplitter()
        r1 = splitter.split(src, out1, val_ratio=0.2, seed=42)
        r2 = splitter.split(src, out2, val_ratio=0.2, seed=99)

        val1 = sorted(p.name for p in r1.val_dir.iterdir())
        val2 = sorted(p.name for p in r2.val_dir.iterdir())

        # With 30 images and 6 val, different seeds should almost certainly differ
        assert val1 != val2

    def test_small_dataset_at_least_one_val(self, tmp_path: Path) -> None:
        """Even with few images, at least one goes to val."""
        src = tmp_path / "src"
        _make_images(src, 5)
        out = tmp_path / "output"

        splitter = DatasetSplitter()
        result = splitter.split(src, out, val_ratio=0.2)

        assert result.val_count >= 1
        assert result.train_count >= 1


# ---------------------------------------------------------------------------
# TRN-005: Threshold recommendation
# ---------------------------------------------------------------------------

class TestRecommendThreshold:
    def test_known_scores(self) -> None:
        """Known score distribution should return predictable threshold."""
        scores = [0.1, 0.12, 0.15, 0.11, 0.13]
        threshold = recommend_threshold(scores)

        arr = np.array(scores)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        max_s = float(np.max(arr))
        expected = max(mean + 2.5 * std, max_s * 1.05)
        expected = float(np.clip(expected, 0.1, 0.95))

        assert abs(threshold - expected) < 1e-9

    def test_empty_scores_returns_default(self) -> None:
        """Empty scores should return a sensible default."""
        assert recommend_threshold([]) == 0.5

    def test_clamped_to_lower_bound(self) -> None:
        """Very small scores should clamp threshold to 0.1."""
        scores = [0.001, 0.002, 0.001]
        threshold = recommend_threshold(scores)
        assert threshold >= 0.1

    def test_clamped_to_upper_bound(self) -> None:
        """Very high scores should clamp threshold to 0.95."""
        scores = [0.9, 0.91, 0.92, 0.93, 0.95]
        threshold = recommend_threshold(scores)
        assert threshold <= 0.95

    def test_single_score(self) -> None:
        """Single score should still work."""
        threshold = recommend_threshold([0.3])
        # max(0.3 + 0, 0.3*1.05) = 0.315
        assert 0.1 <= threshold <= 0.95


# ---------------------------------------------------------------------------
# TRN-004: Quality grading
# ---------------------------------------------------------------------------

class TestModelGrading:
    def test_grade_a(self) -> None:
        """Grade A: max < threshold*0.6, std < 0.05."""
        grade = _compute_grade(score_max=0.3, score_std=0.02, threshold=0.7)
        assert grade == "A"

    def test_grade_b(self) -> None:
        """Grade B: max < threshold*0.8, std < 0.1."""
        grade = _compute_grade(score_max=0.5, score_std=0.08, threshold=0.7)
        assert grade == "B"

    def test_grade_c(self) -> None:
        """Grade C: max < threshold but doesn't meet A or B criteria."""
        grade = _compute_grade(score_max=0.65, score_std=0.12, threshold=0.7)
        assert grade == "C"

    def test_grade_f(self) -> None:
        """Grade F: max >= threshold."""
        grade = _compute_grade(score_max=0.75, score_std=0.05, threshold=0.7)
        assert grade == "F"

    def test_grade_a_boundary(self) -> None:
        """At the boundary of A: max == threshold*0.6 should be B or C, not A."""
        grade = _compute_grade(score_max=0.42, score_std=0.04, threshold=0.7)
        # 0.42 == 0.7 * 0.6, so NOT less than, should not be A
        assert grade != "A"

    def test_grade_f_exact_threshold(self) -> None:
        """Exactly at threshold should be F."""
        grade = _compute_grade(score_max=0.7, score_std=0.01, threshold=0.7)
        assert grade == "F"

    def test_model_quality_report_fields(self) -> None:
        """ModelQualityReport should hold all required fields."""
        report = ModelQualityReport(
            grade="B",
            score_mean=0.2,
            score_std=0.05,
            score_max=0.4,
            score_p95=0.35,
            threshold=0.7,
            recommended_threshold=0.55,
            train_count=80,
            val_count=20,
            inference_time_ms=15.0,
            suggestions=["Consider more training data"],
        )
        assert report.grade == "B"
        assert report.train_count == 80
        assert len(report.suggestions) == 1


# ---------------------------------------------------------------------------
# TRN-003: Post-training validation (mocked)
# ---------------------------------------------------------------------------

class TestPostTrainingValidator:
    def test_validate_model_grade_a(self, tmp_path: Path) -> None:
        """Mocked model scoring low on normal images should get grade A."""
        val_dir = tmp_path / "val"
        _make_images(val_dir, 10)

        mock_result = MagicMock()
        mock_result.anomaly_score = 0.15

        with patch("argus.anomaly.detector.AnomalibDetector") as MockDetector:
            instance = MockDetector.return_value
            instance.predict.return_value = mock_result

            validator = PostTrainingValidator()
            # Pass a dummy model path
            model_path = tmp_path / "model.ckpt"
            model_path.write_bytes(b"fake")
            report = validator.validate_model(model_path, val_dir, threshold=0.7)

        assert report.grade == "A"
        assert report.val_count == 10
        assert report.score_max < 0.7 * 0.6

    def test_validate_model_grade_f(self, tmp_path: Path) -> None:
        """Model scoring above threshold should get grade F."""
        val_dir = tmp_path / "val"
        _make_images(val_dir, 5)

        mock_result = MagicMock()
        mock_result.anomaly_score = 0.85

        with patch("argus.anomaly.detector.AnomalibDetector") as MockDetector:
            instance = MockDetector.return_value
            instance.predict.return_value = mock_result

            validator = PostTrainingValidator()
            model_path = tmp_path / "model.ckpt"
            model_path.write_bytes(b"fake")
            report = validator.validate_model(model_path, val_dir, threshold=0.7)

        assert report.grade == "F"
        assert any("retrain" in s.lower() for s in report.suggestions)

    def test_validate_model_no_images(self, tmp_path: Path) -> None:
        """Empty val dir should return grade F with suggestion."""
        val_dir = tmp_path / "val"
        val_dir.mkdir()

        with patch("argus.anomaly.detector.AnomalibDetector") as MockDetector:
            validator = PostTrainingValidator()
            model_path = tmp_path / "model.ckpt"
            model_path.write_bytes(b"fake")
            report = validator.validate_model(model_path, val_dir)

        assert report.grade == "F"
        assert report.val_count == 0


# ---------------------------------------------------------------------------
# TRN-006: Output validation
# ---------------------------------------------------------------------------

class TestOutputValidator:
    def test_missing_checkpoint_error(self, tmp_path: Path) -> None:
        """No checkpoint files should produce an error."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        validator = OutputValidator()
        result = validator.validate_output(output_dir)

        assert not result.valid
        assert any("No checkpoint" in e for e in result.errors)

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Non-existent output dir should produce an error."""
        validator = OutputValidator()
        result = validator.validate_output(tmp_path / "does_not_exist")

        assert not result.valid
        assert any("does not exist" in e for e in result.errors)

    def test_valid_checkpoint(self, tmp_path: Path) -> None:
        """Valid checkpoint file should pass (with mocked smoke test)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        ckpt = output_dir / "model.ckpt"
        ckpt.write_bytes(b"\x00" * 1024)  # non-empty file

        with patch("argus.anomaly.detector.AnomalibDetector") as MockDetector:
            instance = MockDetector.return_value
            instance.load.return_value = True

            validator = OutputValidator()
            result = validator.validate_output(output_dir)

        assert result.valid
        assert len(result.checkpoint_files) == 1
        assert result.errors == []

    def test_empty_checkpoint_file(self, tmp_path: Path) -> None:
        """Zero-byte checkpoint should produce an error."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        ckpt = output_dir / "model.pt"
        ckpt.write_bytes(b"")  # empty file

        validator = OutputValidator()
        result = validator.validate_output(output_dir)

        assert not result.valid
        assert any("Empty checkpoint" in e for e in result.errors)

    def test_export_directory_with_files(self, tmp_path: Path) -> None:
        """Export files should be detected when present."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        ckpt = output_dir / "model.ckpt"
        ckpt.write_bytes(b"\x00" * 1024)

        export_dir = output_dir / "exports"
        export_dir.mkdir()
        (export_dir / "model.xml").write_bytes(b"\x00" * 512)
        (export_dir / "model.bin").write_bytes(b"\x00" * 512)

        with patch("argus.anomaly.detector.AnomalibDetector") as MockDetector:
            instance = MockDetector.return_value
            instance.load.return_value = True

            validator = OutputValidator()
            result = validator.validate_output(output_dir)

        assert result.valid
        assert len(result.export_files) == 2

    def test_export_directory_empty_warns(self, tmp_path: Path) -> None:
        """Empty export directory should produce a warning."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        ckpt = output_dir / "model.ckpt"
        ckpt.write_bytes(b"\x00" * 1024)

        export_dir = output_dir / "exports"
        export_dir.mkdir()  # empty

        with patch("argus.anomaly.detector.AnomalibDetector") as MockDetector:
            instance = MockDetector.return_value
            instance.load.return_value = True

            validator = OutputValidator()
            result = validator.validate_output(output_dir)

        assert result.valid  # still valid, just a warning
        assert any("no export files" in w.lower() for w in result.warnings)

    def test_multiple_checkpoint_types(self, tmp_path: Path) -> None:
        """Should find .ckpt, .pt, and .xml checkpoint files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        for name in ["model.ckpt", "model.pt", "model.xml"]:
            (output_dir / name).write_bytes(b"\x00" * 100)

        with patch("argus.anomaly.detector.AnomalibDetector") as MockDetector:
            instance = MockDetector.return_value
            instance.load.return_value = True

            validator = OutputValidator()
            result = validator.validate_output(output_dir)

        assert result.valid
        assert len(result.checkpoint_files) == 3
