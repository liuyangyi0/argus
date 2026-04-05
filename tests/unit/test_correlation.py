"""Tests for cross-camera correlation (C3)."""

import time

import numpy as np
import pytest

from argus.core.correlation import (
    CameraOverlapPair,
    CorrelationResult,
    CrossCameraCorrelator,
)


class TestCrossCameraCorrelator:

    def _identity_pair(self):
        """Create a pair with identity homography (same view)."""
        return CameraOverlapPair(
            camera_a="cam_a",
            camera_b="cam_b",
            homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

    def test_corroborated_when_partner_agrees(self):
        """两台摄像头同一位置都有高分 → corroborated=True。"""
        pair = self._identity_pair()
        correlator = CrossCameraCorrelator([pair])

        # cam_b has high anomaly at (100, 100)
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[80:120, 80:120] = 0.8
        now = time.time()
        correlator.update("cam_b", anomaly_map, now)

        result = correlator.check("cam_a", (100, 100), now)
        assert result.corroborated is True
        assert result.partner_camera == "cam_b"
        assert result.partner_score_at_location > 0.3

    def test_uncorroborated_when_partner_clean(self):
        """cam_a 异常但 cam_b 对应位置正常 → corroborated=False。"""
        pair = self._identity_pair()
        correlator = CrossCameraCorrelator([pair])

        # cam_b has no anomaly
        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        now = time.time()
        correlator.update("cam_b", anomaly_map, now)

        result = correlator.check("cam_a", (100, 100), now)
        assert result.corroborated is False

    def test_no_partners_defaults_to_corroborated(self):
        """无 overlap pair → corroborated=True。"""
        correlator = CrossCameraCorrelator([])
        result = correlator.check("cam_solo", (50, 50), time.time())
        assert result.corroborated is True

    def test_stale_partner_data_ignored(self):
        """partner 数据超过 5s → 不使用。"""
        pair = self._identity_pair()
        correlator = CrossCameraCorrelator([pair])

        anomaly_map = np.ones((200, 200), dtype=np.float32) * 0.9
        old_time = time.time() - 10  # 10 seconds ago
        correlator.update("cam_b", anomaly_map, old_time)

        result = correlator.check("cam_a", (100, 100), time.time())
        assert result.corroborated is False

    def test_homography_projection_correct(self):
        """已知 H 矩阵 → 投影坐标正确。"""
        # Shift homography: maps (x,y) -> (x+50, y+30)
        pair = CameraOverlapPair(
            camera_a="cam_a",
            camera_b="cam_b",
            homography=[[1, 0, 50], [0, 1, 30], [0, 0, 1]],
        )
        correlator = CrossCameraCorrelator([pair])

        # cam_b has anomaly at (150, 130) = (100+50, 100+30)
        anomaly_map = np.zeros((300, 300), dtype=np.float32)
        anomaly_map[110:150, 130:170] = 0.9
        now = time.time()
        correlator.update("cam_b", anomaly_map, now)

        # Check from cam_a perspective at (100, 100) → should project to (150, 130)
        result = correlator.check("cam_a", (100, 100), now)
        assert result.corroborated is True

    def test_configurable_threshold(self):
        """自定义 threshold → 只有超过该阈值才算 corroborated。"""
        pair = self._identity_pair()
        # High threshold: 0.5 score should NOT corroborate
        correlator = CrossCameraCorrelator([pair], corroboration_threshold=0.8)

        anomaly_map = np.zeros((200, 200), dtype=np.float32)
        anomaly_map[80:120, 80:120] = 0.5  # Below threshold of 0.8
        now = time.time()
        correlator.update("cam_b", anomaly_map, now)

        result = correlator.check("cam_a", (100, 100), now)
        assert result.corroborated is False

    def test_reverse_direction_uses_inverse_homography(self):
        """cam_b 查询 cam_a 的数据时使用逆矩阵。"""
        pair = CameraOverlapPair(
            camera_a="cam_a",
            camera_b="cam_b",
            homography=[[1, 0, 50], [0, 1, 30], [0, 0, 1]],
        )
        correlator = CrossCameraCorrelator([pair])

        # cam_a has anomaly at (100, 100)
        anomaly_map = np.zeros((300, 300), dtype=np.float32)
        anomaly_map[80:120, 80:120] = 0.8
        now = time.time()
        correlator.update("cam_a", anomaly_map, now)

        # From cam_b at (150, 130) → inverse projects to (100, 100) in cam_a
        result = correlator.check("cam_b", (150, 130), now)
        assert result.corroborated is True
        assert result.partner_camera == "cam_a"

    def test_prune_stale_removes_old_maps(self):
        """prune_stale 清除超时数据。"""
        pair = self._identity_pair()
        correlator = CrossCameraCorrelator([pair])

        old_time = time.time() - 60  # 60 seconds ago
        anomaly_map = np.ones((200, 200), dtype=np.float32) * 0.9
        correlator.update("cam_b", anomaly_map, old_time)
        assert "cam_b" in correlator._recent_maps

        correlator.prune_stale(max_age=30.0)
        assert "cam_b" not in correlator._recent_maps
