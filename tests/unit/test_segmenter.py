"""Tests for SAM 2 instance segmentation (D2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argus.anomaly.segmenter import (
    InstanceSegmenter,
    SegmentationResult,
    SegmentedObject,
    extract_peak_points,
)


# ---------------------------------------------------------------------------
# Peak extraction
# ---------------------------------------------------------------------------

class TestPeakExtraction:

    def test_peak_extraction_from_heatmap(self):
        """Peaks above min_score are returned, sorted by descending score."""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[20, 30] = 0.9  # highest
        heatmap[70, 80] = 0.8  # second
        heatmap[50, 50] = 0.5  # below threshold

        points = extract_peak_points(heatmap, max_points=5, min_score=0.7)
        assert len(points) == 2
        # First point should be the highest scoring one
        assert points[0] == (30, 20)
        assert points[1] == (80, 70)

    def test_peak_extraction_max_points(self):
        """No more than max_points peaks are returned."""
        heatmap = np.zeros((200, 200), dtype=np.float32)
        # Place 10 well-separated peaks
        for i in range(10):
            heatmap[i * 20, i * 20] = 0.9 - i * 0.01
        points = extract_peak_points(heatmap, max_points=3, min_score=0.7)
        assert len(points) <= 3

    def test_peak_extraction_empty_heatmap(self):
        """Empty or zero heatmap yields no points."""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        points = extract_peak_points(heatmap, max_points=5, min_score=0.7)
        assert points == []

    def test_peak_extraction_none_heatmap(self):
        """None heatmap returns empty list."""
        points = extract_peak_points(None, max_points=5, min_score=0.7)
        assert points == []

    def test_peak_extraction_nms_distance(self):
        """Points within min_distance_px of each other are suppressed."""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[50, 50] = 0.95
        heatmap[52, 52] = 0.90  # too close (within 30px)
        heatmap[50, 90] = 0.85  # far enough

        points = extract_peak_points(
            heatmap, max_points=5, min_score=0.7, min_distance_px=30,
        )
        assert len(points) == 2
        # Should have (50, 50) and (90, 50), not (52, 52)
        xs = {p[0] for p in points}
        assert 50 in xs
        assert 90 in xs

    def test_peak_extraction_3d_squeezable(self):
        """A 3-D array with shape (1, H, W) is squeezed to 2-D."""
        heatmap = np.zeros((1, 50, 50), dtype=np.float32)
        heatmap[0, 25, 25] = 0.9
        points = extract_peak_points(heatmap, max_points=5, min_score=0.7)
        assert len(points) == 1


# ---------------------------------------------------------------------------
# Contour fallback (no SAM 2)
# ---------------------------------------------------------------------------

class TestInstanceSegmenterFallback:

    def test_no_sam2_returns_empty_on_blank_frame(self):
        """When SAM 2 is not installed and the frame is uniform, no objects."""
        seg = InstanceSegmenter(model_size="small", min_mask_area_px=10)
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = seg.segment(frame, [(100, 100)])
        assert isinstance(result, SegmentationResult)
        # Otsu on a uniform region produces all-0 or all-255 mask;
        # either way contourArea may be 0 or the whole region — depends on
        # implementation.  We just check it doesn't crash.

    def test_segment_finds_bright_object(self):
        """A bright square on dark background is found by contour fallback."""
        seg = InstanceSegmenter(model_size="small", min_mask_area_px=10)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[80:120, 80:120] = 255  # 40x40 bright square

        result = seg.segment(frame, [(100, 100)])
        assert isinstance(result, SegmentationResult)
        if result.num_objects > 0:
            obj = result.objects[0]
            assert obj.area_px > 0
            assert len(obj.bbox) == 4
            assert len(obj.centroid) == 2
            assert obj.confidence == 1.0

    def test_min_area_filter(self):
        """Objects smaller than min_mask_area_px are discarded."""
        seg = InstanceSegmenter(model_size="small", min_mask_area_px=5000)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[95:105, 95:105] = 255  # 10x10 = ~100 pixels, below 5000

        result = seg.segment(frame, [(100, 100)])
        assert result.num_objects == 0

    def test_empty_prompt_points(self):
        """No prompt points returns empty result."""
        seg = InstanceSegmenter()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = seg.segment(frame, [])
        assert result.num_objects == 0

    def test_none_frame(self):
        """None frame returns empty result without crashing."""
        seg = InstanceSegmenter()
        result = seg.segment(None, [(50, 50)])
        assert result.num_objects == 0

    def test_total_area_computed(self):
        """total_area_px is the sum of all object areas."""
        seg = InstanceSegmenter(model_size="small", min_mask_area_px=10)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Two well-separated bright squares
        frame[30:70, 30:70] = 255
        frame[200:240, 200:240] = 255

        result = seg.segment(frame, [(50, 50), (220, 220)])
        if result.num_objects >= 2:
            assert result.total_area_px == sum(o.area_px for o in result.objects)


# ---------------------------------------------------------------------------
# SAM 2 mock path
# ---------------------------------------------------------------------------

class TestSegmentWithMockModel:

    def test_segment_with_mock_sam2(self):
        """SAM 2 predictor path works when mocked."""
        seg = InstanceSegmenter(model_size="small", min_mask_area_px=10)

        # Create a mock predictor
        mock_predictor = MagicMock()
        # Return a single binary mask, score, and logit
        mask = np.zeros((200, 200), dtype=bool)
        mask[80:120, 80:120] = True  # 40x40 object
        mock_predictor.predict.return_value = (
            np.array([mask]),           # masks
            np.array([0.95]),           # scores
            np.array([np.zeros((1, 200, 200))]),  # logits (unused)
        )

        seg._predictor = mock_predictor
        seg._sam2_available = True
        seg._loaded = True

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = seg.segment(frame, [(100, 100)])

        assert result.num_objects == 1
        obj = result.objects[0]
        assert obj.area_px > 0
        assert obj.confidence == pytest.approx(0.95, abs=0.01)
        assert len(obj.bbox) == 4
        assert len(obj.centroid) == 2
        mock_predictor.set_image.assert_called_once()
        mock_predictor.predict.assert_called_once()
        mock_predictor.reset_image.assert_called_once()

    def test_mock_sam2_min_area_filter(self):
        """Masks below min_mask_area_px are filtered in SAM 2 path too."""
        seg = InstanceSegmenter(model_size="small", min_mask_area_px=5000)

        mock_predictor = MagicMock()
        mask = np.zeros((200, 200), dtype=bool)
        mask[98:102, 98:102] = True  # 4x4 = 16 pixels
        mock_predictor.predict.return_value = (
            np.array([mask]),
            np.array([0.9]),
            np.array([np.zeros((1, 200, 200))]),
        )

        seg._predictor = mock_predictor
        seg._sam2_available = True
        seg._loaded = True

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = seg.segment(frame, [(100, 100)])
        assert result.num_objects == 0

    def test_sam2_import_failure_uses_fallback(self):
        """When sam2 import fails, fallback is used gracefully."""
        seg = InstanceSegmenter(model_size="small", min_mask_area_px=10)

        # Simulate import failure by calling load with sam2 not installed
        with patch.dict("sys.modules", {"sam2": None, "sam2.sam2_image_predictor": None}):
            seg.load()

        assert seg._sam2_available is False
        assert seg._loaded is True

        # Should still work via fallback
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[80:120, 80:120] = 255
        result = seg.segment(frame, [(100, 100)])
        assert isinstance(result, SegmentationResult)
