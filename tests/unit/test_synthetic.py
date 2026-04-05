"""Tests for synthetic data generation and recall evaluation (D3)."""

import cv2
import numpy as np
import pytest

from argus.validation.recall_test import load_synthetic_pairs, evaluate_recall
from argus.validation.synthetic import (
    generate_synthetic,
    _match_brightness,
    _blur_mask_edges,
    _extract_mask,
    _random_perspective_transform,
)


def _make_baseline_image(w: int = 200, h: int = 150, color=(80, 120, 90)):
    """Create a simple baseline image with some texture."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    noise = np.random.RandomState(0).randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def _make_foe_object_alpha(w: int = 50, h: int = 50):
    """Create an FOE object with alpha channel (BGRA)."""
    obj = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.circle(obj, (w // 2, h // 2), min(w, h) // 3, (0, 0, 255, 255), -1)
    return obj


def _make_foe_object_white_bg(w: int = 50, h: int = 50):
    """Create an FOE object with white background (BGR)."""
    obj = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(obj, (10, 10), (w - 10, h - 10), (30, 30, 200), -1)
    return obj


def _setup_dirs(tmp_path, n_baselines=3, n_objects=2, use_alpha=True):
    """Set up baseline and object directories for testing."""
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    output_dir = tmp_path / "output"

    for i in range(n_baselines):
        img = _make_baseline_image()
        cv2.imwrite(str(baseline_dir / f"baseline_{i:03d}.png"), img)

    for i in range(n_objects):
        if use_alpha:
            obj = _make_foe_object_alpha()
        else:
            obj = _make_foe_object_white_bg()
        cv2.imwrite(str(objects_dir / f"foe_{i:03d}.png"), obj)

    return baseline_dir, objects_dir, output_dir


class TestSyntheticGeneration:

    def test_synthetic_generation_produces_valid_images(self, tmp_path):
        """Generated images should be valid PNGs with correct dimensions and matching masks."""
        baseline_dir, objects_dir, output_dir = _setup_dirs(tmp_path)
        count = 5
        generated = generate_synthetic(
            baseline_dir=baseline_dir,
            objects_dir=objects_dir,
            output_dir=output_dir,
            count=count,
            seed=42,
            min_scale=0.5,
            max_scale=1.0,
        )

        assert generated > 0
        assert generated <= count

        # Check that output images exist and are loadable
        composites = sorted(output_dir.glob("synthetic_*.png"))
        masks = sorted((output_dir / "masks").glob("synthetic_*_mask.png"))
        assert len(composites) == generated
        assert len(masks) == generated

        # Validate each image
        for comp_path in composites:
            img = cv2.imread(str(comp_path))
            assert img is not None, f"Failed to load {comp_path}"
            assert img.shape[0] > 0 and img.shape[1] > 0

        # Validate masks are same spatial dims as composites
        for mask_path in masks:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            assert mask is not None
            bg = _make_baseline_image()
            assert mask.shape[:2] == (bg.shape[0], bg.shape[1])

    def test_synthetic_with_white_bg_objects(self, tmp_path):
        """Generation works with white-background FOE objects (no alpha)."""
        baseline_dir, objects_dir, output_dir = _setup_dirs(
            tmp_path, use_alpha=False
        )
        generated = generate_synthetic(
            baseline_dir=baseline_dir,
            objects_dir=objects_dir,
            output_dir=output_dir,
            count=3,
            seed=123,
        )
        assert generated > 0

    def test_no_baselines_returns_zero(self, tmp_path):
        """Empty baseline directory returns 0 generated."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        objects_dir = tmp_path / "objects"
        objects_dir.mkdir()
        obj = _make_foe_object_alpha()
        cv2.imwrite(str(objects_dir / "foe.png"), obj)

        result = generate_synthetic(
            baseline_dir=empty_dir,
            objects_dir=objects_dir,
            output_dir=tmp_path / "out",
            count=5,
        )
        assert result == 0

    def test_no_objects_returns_zero(self, tmp_path):
        """Empty objects directory returns 0 generated."""
        baseline_dir = tmp_path / "baselines"
        baseline_dir.mkdir()
        bg = _make_baseline_image()
        cv2.imwrite(str(baseline_dir / "bg.png"), bg)

        result = generate_synthetic(
            baseline_dir=baseline_dir,
            objects_dir=tmp_path / "empty_objects",
            output_dir=tmp_path / "out",
            count=5,
        )
        assert result == 0

    def test_seed_reproducibility(self, tmp_path):
        """Same seed produces identical outputs."""
        baseline_dir, objects_dir, _ = _setup_dirs(tmp_path)
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        generate_synthetic(
            baseline_dir=baseline_dir, objects_dir=objects_dir,
            output_dir=out1, count=3, seed=99,
        )
        generate_synthetic(
            baseline_dir=baseline_dir, objects_dir=objects_dir,
            output_dir=out2, count=3, seed=99,
        )

        for f1 in sorted(out1.glob("synthetic_*.png")):
            f2 = out2 / f1.name
            img1 = cv2.imread(str(f1))
            img2 = cv2.imread(str(f2))
            assert img1 is not None and img2 is not None
            np.testing.assert_array_equal(img1, img2)


class TestMaskGeneration:

    def test_mask_is_binary(self, tmp_path):
        """Ground truth masks should contain both 0 and non-zero values."""
        baseline_dir, objects_dir, output_dir = _setup_dirs(tmp_path)
        generate_synthetic(
            baseline_dir=baseline_dir,
            objects_dir=objects_dir,
            output_dir=output_dir,
            count=3,
            seed=42,
        )

        masks_dir = output_dir / "masks"
        for mask_path in masks_dir.glob("*.png"):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            unique = set(np.unique(mask))
            assert 0 in unique, "Mask should have background (0) pixels"
            assert len(unique - {0}) > 0, "Mask should have foreground pixels"

    def test_mask_has_nonzero_region(self, tmp_path):
        """Each mask should have a non-empty foreground region."""
        baseline_dir, objects_dir, output_dir = _setup_dirs(tmp_path)
        generate_synthetic(
            baseline_dir=baseline_dir,
            objects_dir=objects_dir,
            output_dir=output_dir,
            count=3,
            seed=42,
        )

        masks_dir = output_dir / "masks"
        for mask_path in masks_dir.glob("*.png"):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            assert mask.sum() > 0, f"Mask {mask_path.name} has no foreground"

    def test_mask_dimensions_match_image(self, tmp_path):
        """Mask and composite image should have the same spatial dimensions."""
        baseline_dir, objects_dir, output_dir = _setup_dirs(tmp_path)
        generate_synthetic(
            baseline_dir=baseline_dir,
            objects_dir=objects_dir,
            output_dir=output_dir,
            count=2,
            seed=42,
        )

        for img_path in sorted(output_dir.glob("synthetic_*.png")):
            img = cv2.imread(str(img_path))
            mask_path = output_dir / "masks" / f"{img_path.stem}_mask.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            assert img.shape[:2] == mask.shape[:2]


class TestBrightnessMatching:

    def test_brightness_matching_adjusts_foreground(self):
        """Brightness matching should shift foreground luminance toward background."""
        # Dark background region
        bg = np.full((50, 50, 3), 40, dtype=np.uint8)
        # Bright foreground
        fg = np.full((50, 50, 3), 200, dtype=np.uint8)
        mask = np.full((50, 50), 255, dtype=np.uint8)

        adjusted = _match_brightness(fg, bg, mask)
        assert adjusted.mean() < fg.mean(), (
            f"Adjusted mean {adjusted.mean():.1f} should be less than original {fg.mean():.1f}"
        )

    def test_brightness_matching_brightens_dark_foreground(self):
        """Brightness matching should brighten a dark foreground on a light background."""
        bg = np.full((50, 50, 3), 200, dtype=np.uint8)
        fg = np.full((50, 50, 3), 40, dtype=np.uint8)
        mask = np.full((50, 50), 255, dtype=np.uint8)

        adjusted = _match_brightness(fg, bg, mask)
        assert adjusted.mean() > fg.mean(), (
            f"Adjusted mean {adjusted.mean():.1f} should be greater than original {fg.mean():.1f}"
        )

    def test_brightness_matching_empty_mask(self):
        """Empty mask should return foreground unchanged."""
        bg = np.full((50, 50, 3), 100, dtype=np.uint8)
        fg = np.full((50, 50, 3), 200, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)

        adjusted = _match_brightness(fg, bg, mask)
        np.testing.assert_array_equal(adjusted, fg)


class TestEdgeBlending:

    def test_blur_mask_edges_produces_soft_edges(self):
        """Blurred mask should have intermediate values at edges."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)

        soft = _blur_mask_edges(mask, sigma=3.0)
        assert soft.dtype == np.float32
        assert soft.max() <= 1.0
        assert soft.min() >= 0.0

        # Should have intermediate values (not just 0 and 1)
        unique_count = len(np.unique(np.round(soft, 2)))
        assert unique_count > 2, "Blurred mask should have gradient values"

    def test_blur_sigma_zero_returns_binary(self):
        """Sigma=0 should return a binary float mask."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        result = _blur_mask_edges(mask, sigma=0.0)
        unique = set(np.unique(result))
        assert unique <= {0.0, 1.0}


class TestPerspectiveTransform:

    def test_perspective_preserves_shape(self):
        """Perspective transform should keep the same image dimensions."""
        rng = np.random.default_rng(42)
        obj = np.zeros((60, 80, 4), dtype=np.uint8)
        cv2.circle(obj, (40, 30), 20, (255, 0, 0, 255), -1)

        transformed = _random_perspective_transform(obj, rng)
        assert transformed.shape == obj.shape

    def test_perspective_modifies_content(self):
        """Perspective transform should actually change the pixel content."""
        rng = np.random.default_rng(42)
        obj = np.zeros((60, 80, 3), dtype=np.uint8)
        cv2.rectangle(obj, (10, 10), (70, 50), (128, 128, 128), -1)

        transformed = _random_perspective_transform(obj, rng)
        assert not np.array_equal(obj, transformed)


class TestMaskExtraction:

    def test_extract_mask_alpha_channel(self):
        """Alpha channel extraction should use the 4th channel as mask."""
        obj = np.zeros((50, 50, 4), dtype=np.uint8)
        obj[10:40, 10:40, 3] = 255

        mask = _extract_mask(obj)
        assert mask.shape == (50, 50)
        assert mask[20, 20] == 255
        assert mask[0, 0] == 0

    def test_extract_mask_white_background(self):
        """White background detection should invert: white=bg, dark=fg."""
        obj = np.full((50, 50, 3), 255, dtype=np.uint8)
        obj[15:35, 15:35] = (30, 30, 30)

        mask = _extract_mask(obj)
        assert mask.shape == (50, 50)
        assert mask[25, 25] == 255, "Dark region should be foreground"
        assert mask[0, 0] == 0, "White region should be background"


class TestSyntheticPairs:

    def test_load_synthetic_pairs(self, tmp_path):
        """Load paired synthetic images and masks."""
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        for i in range(5):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"synthetic_{i:05d}.png"), img)
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.imwrite(str(masks_dir / f"synthetic_{i:05d}_mask.png"), mask)

        pairs = load_synthetic_pairs(tmp_path)
        assert len(pairs) == 5

    def test_load_synthetic_pairs_empty_dir(self, tmp_path):
        """Empty directory returns no pairs."""
        pairs = load_synthetic_pairs(tmp_path)
        assert len(pairs) == 0

    def test_load_synthetic_pairs_missing_mask(self, tmp_path):
        """Images without matching masks are skipped."""
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        for i in range(3):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"synthetic_{i:05d}.png"), img)

        for i in range(2):
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.imwrite(str(masks_dir / f"synthetic_{i:05d}_mask.png"), mask)

        pairs = load_synthetic_pairs(tmp_path)
        assert len(pairs) == 2


class TestRecallEvaluation:

    def test_recall_evaluation_runs(self, tmp_path):
        """Recall evaluation completes without error."""
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        for i in range(3):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"synthetic_{i:05d}.png"), img)
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.imwrite(str(masks_dir / f"synthetic_{i:05d}_mask.png"), mask)

        class MockDetector:
            def predict(self, frame):
                class Result:
                    anomaly_score = 0.8
                return Result()

        result = evaluate_recall(MockDetector(), tmp_path, threshold=0.5)
        assert result["recall"] == 1.0
        assert result["tp"] == 3
        assert result["fn"] == 0
        assert result["total"] == 3
        assert len(result["scores"]) == 3

    def test_recall_all_below_threshold(self, tmp_path):
        """All scores below threshold gives recall=0."""
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        for i in range(3):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"synthetic_{i:05d}.png"), img)
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.imwrite(str(masks_dir / f"synthetic_{i:05d}_mask.png"), mask)

        class LowScoreDetector:
            def predict(self, frame):
                class Result:
                    anomaly_score = 0.1
                return Result()

        result = evaluate_recall(LowScoreDetector(), tmp_path, threshold=0.5)
        assert result["recall"] == 0.0
        assert result["fn"] == 3
        assert result["tp"] == 0

    def test_recall_empty_dir(self, tmp_path):
        """Empty synthetic dir returns zero recall with no errors."""
        class MockDetector:
            def predict(self, frame):
                class Result:
                    anomaly_score = 0.8
                return Result()

        result = evaluate_recall(MockDetector(), tmp_path, threshold=0.5)
        assert result["recall"] == 0.0
        assert result["total"] == 0

    def test_recall_partial_detection(self, tmp_path):
        """Mixed detection results produce correct partial recall."""
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        for i in range(4):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"synthetic_{i:05d}.png"), img)
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.imwrite(str(masks_dir / f"synthetic_{i:05d}_mask.png"), mask)

        class AlternatingDetector:
            def __init__(self):
                self._call_count = 0

            def predict(self, frame):
                self._call_count += 1
                class Result:
                    pass
                r = Result()
                r.anomaly_score = 0.9 if self._call_count % 2 == 1 else 0.1
                return r

        result = evaluate_recall(AlternatingDetector(), tmp_path, threshold=0.5)
        assert result["recall"] == 0.5
        assert result["tp"] == 2
        assert result["fn"] == 2
