"""Tests for the Simplex safety channel detector."""

import time

import cv2
import numpy as np
import pytest

from argus.prefilter.simple_detector import SimplexDetector, SimplexResult


class TestSimplexDetector:

    def _make_gray(self, value=128, shape=(480, 640)):
        return np.full(shape, value, dtype=np.uint8)

    def test_detect_static_object(self):
        """参考帧纯灰 → 当前帧有黑色矩形 → 等待 min_static_seconds 后检测到。"""
        detector = SimplexDetector(
            diff_threshold=20, min_area_px=100, min_static_seconds=0.01,
            match_radius_px=100,
        )
        ref = self._make_gray(128)
        detector.set_reference(ref)

        # Frame with a large dark rectangle
        frame = self._make_gray(128)
        frame[100:250, 100:350] = 0  # 150x250=37500 px

        # First detection: starts tracking
        result = detector.detect(frame)
        assert result.max_static_seconds >= 0

        # Small delay then re-detect to exceed min_static_seconds
        time.sleep(0.02)
        result = detector.detect(frame)
        assert result.has_detection is True
        assert len(result.static_regions) > 0

    def test_no_detection_on_reference(self):
        """参考帧 == 当前帧 → 无检测。"""
        detector = SimplexDetector(min_area_px=100, min_static_seconds=0.0)
        ref = self._make_gray(128)
        detector.set_reference(ref)

        result = detector.detect(ref.copy())
        assert result.has_detection is False
        assert len(result.static_regions) == 0

    def test_small_object_filtered(self):
        """面积 < min_area_px → 不检测。"""
        detector = SimplexDetector(
            diff_threshold=20, min_area_px=5000, min_static_seconds=0.0,
        )
        ref = self._make_gray(128)
        detector.set_reference(ref)

        # Tiny 10x10 white patch (100 px < 5000 min)
        frame = self._make_gray(128)
        frame[200:210, 200:210] = 255

        time.sleep(0.01)
        result = detector.detect(frame)
        # Should not detect (too small)
        assert result.has_detection is False

    def test_moving_object_not_static(self):
        """物体每帧移动 → centroid 不匹配 → 不触发静止检测。"""
        detector = SimplexDetector(
            diff_threshold=20, min_area_px=100, min_static_seconds=0.5,
            match_radius_px=10,  # tight matching radius
        )
        ref = self._make_gray(128)
        detector.set_reference(ref)

        for i in range(10):
            frame = self._make_gray(128)
            # Move rectangle far each frame (beyond match_radius)
            y = 50 + i * 30
            frame[y:y+50, 100:200] = 255
            result = detector.detect(frame)

        # Object is moving, first_seen keeps resetting → not static
        assert result.has_detection is False

    def test_reset_clears_tracking(self):
        """reset() 后 tracked regions 清空。"""
        detector = SimplexDetector(min_area_px=100, min_static_seconds=0.0)
        ref = self._make_gray(128)
        detector.set_reference(ref)

        frame = self._make_gray(128)
        frame[100:250, 100:350] = 0
        detector.detect(frame)

        assert len(detector._tracked) > 0
        detector.reset()
        assert len(detector._tracked) == 0

    def test_reference_frame_required(self):
        """未设置 reference → 返回空结果。"""
        detector = SimplexDetector()
        frame = self._make_gray(128)
        result = detector.detect(frame)
        assert result.has_detection is False
        assert len(result.static_regions) == 0
