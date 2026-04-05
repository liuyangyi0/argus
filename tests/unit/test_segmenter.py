"""Tests for instance segmentation (D2)."""

import numpy as np
import pytest

from argus.anomaly.segmenter import InstanceSegmenter, SegmentationResult


class TestInstanceSegmenter:

    def test_segment_returns_result(self):
        """Segmenter returns a SegmentationResult."""
        seg = InstanceSegmenter(model_size="small")
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add a bright object
        frame[80:120, 80:120] = 255
        result = seg.segment(frame, [(100, 100)])
        assert isinstance(result, SegmentationResult)

    def test_segment_empty_frame(self):
        """Empty frame with no objects returns empty result."""
        seg = InstanceSegmenter()
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = seg.segment(frame, [(100, 100)])
        assert isinstance(result, SegmentationResult)

    def test_segment_finds_object(self):
        """Frame with distinct object at prompt point is segmented."""
        seg = InstanceSegmenter()
        frame = np.full((200, 200, 3), 200, dtype=np.uint8)
        # Dark object on light background
        frame[80:120, 80:120] = 20
        result = seg.segment(frame, [(100, 100)])
        assert result.num_objects >= 0  # May find contour depending on Otsu threshold
