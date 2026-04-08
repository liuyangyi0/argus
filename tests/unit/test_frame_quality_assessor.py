"""Tests for frame quality assessment (ANO-001)."""

from __future__ import annotations

import numpy as np
import pytest

from argus.core.frame_quality import FrameQualityAssessor, QualityScore


@pytest.fixture
def assessor() -> FrameQualityAssessor:
    return FrameQualityAssessor()


class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_dataclass_fields(self):
        score = QualityScore(
            acceptable=True,
            overall_score=0.8,
            blur_score=0.9,
            exposure_score=0.7,
            noise_score=0.85,
            entropy=6.5,
            issues=[],
        )
        assert score.acceptable is True
        assert score.overall_score == 0.8
        assert score.issues == []

    def test_default_issues_list(self):
        score = QualityScore(
            acceptable=True,
            overall_score=0.5,
            blur_score=0.5,
            exposure_score=0.5,
            noise_score=0.5,
            entropy=4.0,
        )
        assert score.issues == []


class TestBlackFrame:
    """Black frames should have low exposure and entropy scores."""

    def test_black_frame_low_exposure(self, assessor: FrameQualityAssessor):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = assessor.assess(frame)
        assert result.exposure_score < 0.2
        assert "underexposed" in " ".join(result.issues).lower() or "dark" in " ".join(result.issues).lower()

    def test_black_frame_low_entropy(self, assessor: FrameQualityAssessor):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = assessor.assess(frame)
        assert result.entropy < 1.0

    def test_black_frame_not_acceptable(self, assessor: FrameQualityAssessor):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = assessor.assess(frame)
        assert result.acceptable is False


class TestWhiteFrame:
    """White frames should have low exposure score (overexposed)."""

    def test_white_frame_low_exposure(self, assessor: FrameQualityAssessor):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = assessor.assess(frame)
        assert result.exposure_score < 0.2
        assert "overexposed" in " ".join(result.issues).lower() or "bright" in " ".join(result.issues).lower()

    def test_white_frame_low_entropy(self, assessor: FrameQualityAssessor):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = assessor.assess(frame)
        assert result.entropy < 1.0


class TestRandomNoiseFrame:
    """Random noise should have reasonable quality (high entropy, varied brightness)."""

    def test_noise_frame_acceptable(self, assessor: FrameQualityAssessor):
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        result = assessor.assess(frame)
        # Random noise has high entropy and mid-range brightness
        assert result.entropy > 5.0
        assert result.exposure_score > 0.5

    def test_noise_frame_high_entropy(self, assessor: FrameQualityAssessor):
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        result = assessor.assess(frame)
        assert result.entropy > 7.0


class TestBlurredFrame:
    """Heavily blurred frames should have low blur_score."""

    def test_blurred_low_score(self, assessor: FrameQualityAssessor):
        # Create a frame with sharp edges then blur heavily
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        import cv2
        blurred = cv2.GaussianBlur(frame, (51, 51), 20)
        result = assessor.assess(blurred)
        assert result.blur_score < 0.4

    def test_sharp_vs_blurred(self, assessor: FrameQualityAssessor):
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        import cv2
        blurred = cv2.GaussianBlur(frame, (51, 51), 20)
        sharp_result = assessor.assess(frame)
        blur_result = assessor.assess(blurred)
        assert sharp_result.blur_score > blur_result.blur_score


class TestUniformGray:
    """Uniform gray frame should have very low entropy."""

    def test_uniform_gray_low_entropy(self, assessor: FrameQualityAssessor):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = assessor.assess(frame)
        assert result.entropy < 1.0
        assert any("entropy" in issue.lower() for issue in result.issues)


class TestConfidenceMultiplier:
    """confidence_multiplier should return correct ranges."""

    def test_good_quality_returns_one(self, assessor: FrameQualityAssessor):
        quality = QualityScore(
            acceptable=True, overall_score=0.9,
            blur_score=0.9, exposure_score=0.9,
            noise_score=0.9, entropy=7.0, issues=[],
        )
        assert assessor.confidence_multiplier(quality) == 1.0

    def test_medium_quality_range(self, assessor: FrameQualityAssessor):
        quality = QualityScore(
            acceptable=True, overall_score=0.65,
            blur_score=0.6, exposure_score=0.6,
            noise_score=0.6, entropy=5.0, issues=[],
        )
        mult = assessor.confidence_multiplier(quality)
        assert 0.7 <= mult <= 1.0

    def test_poor_quality_range(self, assessor: FrameQualityAssessor):
        quality = QualityScore(
            acceptable=False, overall_score=0.3,
            blur_score=0.2, exposure_score=0.3,
            noise_score=0.3, entropy=2.0, issues=["blurry"],
        )
        mult = assessor.confidence_multiplier(quality)
        assert 0.3 <= mult <= 0.7

    def test_zero_quality_returns_minimum(self, assessor: FrameQualityAssessor):
        quality = QualityScore(
            acceptable=False, overall_score=0.0,
            blur_score=0.0, exposure_score=0.0,
            noise_score=0.0, entropy=0.0, issues=["empty"],
        )
        mult = assessor.confidence_multiplier(quality)
        assert mult == pytest.approx(0.3)

    def test_boundary_08(self, assessor: FrameQualityAssessor):
        quality = QualityScore(
            acceptable=True, overall_score=0.8,
            blur_score=0.8, exposure_score=0.8,
            noise_score=0.8, entropy=6.0, issues=[],
        )
        assert assessor.confidence_multiplier(quality) == 1.0

    def test_boundary_05(self, assessor: FrameQualityAssessor):
        quality = QualityScore(
            acceptable=True, overall_score=0.5,
            blur_score=0.5, exposure_score=0.5,
            noise_score=0.5, entropy=4.0, issues=[],
        )
        assert assessor.confidence_multiplier(quality) == pytest.approx(0.7)


class TestEmptyFrame:
    """Empty/None frames should be handled gracefully."""

    def test_none_frame(self, assessor: FrameQualityAssessor):
        result = assessor.assess(None)
        assert result.acceptable is False
        assert result.overall_score == 0.0

    def test_empty_array(self, assessor: FrameQualityAssessor):
        frame = np.array([], dtype=np.uint8)
        result = assessor.assess(frame)
        assert result.acceptable is False


class TestGrayscaleInput:
    """Grayscale frames (2D arrays) should work correctly."""

    def test_grayscale_frame(self, assessor: FrameQualityAssessor):
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (100, 100), dtype=np.uint8)
        result = assessor.assess(frame)
        assert 0.0 <= result.overall_score <= 1.0
        assert result.entropy > 0


class TestCustomThresholds:
    """Custom threshold values should affect assessment."""

    def test_strict_blur_threshold(self):
        strict = FrameQualityAssessor(blur_threshold=500.0)
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        result = strict.assess(frame)
        # With a very high threshold, blur score should be lower
        default = FrameQualityAssessor()
        default_result = default.assess(frame)
        assert result.blur_score <= default_result.blur_score
