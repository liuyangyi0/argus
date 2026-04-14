"""Tests for the camera calibration module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from argus.physics.calibration import CalibrationData, CalibrationTool, CameraCalibration


def _make_calibration_data(camera_id: str = "test-cam") -> CalibrationData:
    """Create a synthetic calibration with a simple pinhole model."""
    # Simple pinhole: focal length 1000px, principal point at center of 1920x1080
    K = np.array([
        [1000.0, 0.0, 960.0],
        [0.0, 1000.0, 540.0],
        [0.0, 0.0, 1.0],
    ])
    # Camera at (0, 0, 5000mm) looking down at the pool surface (Z=0)
    R = np.eye(3)
    t = np.array([0.0, 0.0, 5000.0])
    dist = np.zeros(5)

    return CalibrationData(
        camera_id=camera_id,
        camera_matrix=K,
        dist_coeffs=dist,
        rotation_matrix=R,
        translation_vector=t,
        reprojection_error=0.5,
        image_size=(1920, 1080),
    )


class TestCalibrationData:
    def test_to_dict_from_dict_roundtrip(self):
        """Serialization roundtrip should preserve all fields."""
        original = _make_calibration_data()
        d = original.to_dict()
        restored = CalibrationData.from_dict(d)

        assert restored.camera_id == original.camera_id
        np.testing.assert_array_almost_equal(restored.camera_matrix, original.camera_matrix)
        np.testing.assert_array_almost_equal(restored.dist_coeffs, original.dist_coeffs)
        np.testing.assert_array_almost_equal(restored.rotation_matrix, original.rotation_matrix)
        assert restored.reprojection_error == pytest.approx(0.5)
        assert restored.image_size == (1920, 1080)

    def test_json_roundtrip(self):
        """Save to JSON and reload should produce identical data."""
        original = _make_calibration_data()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(original.to_dict(), f)
            path = f.name

        restored = CalibrationData.from_dict(json.loads(Path(path).read_text()))
        np.testing.assert_array_almost_equal(restored.camera_matrix, original.camera_matrix)


class TestCameraCalibration:
    def test_world_to_pixel_to_world_roundtrip(self):
        """pixel_to_world → world_to_pixel should be approximately inverse."""
        data = _make_calibration_data()
        cal = CameraCalibration(data, pool_surface_z_mm=0.0)

        # Project a world point to pixel
        px, py = cal.world_to_pixel(500.0, 300.0, 0.0)

        # Project back to world
        wp = cal.pixel_to_world(px, py, z_mm=0.0)

        assert wp.x_mm == pytest.approx(500.0, abs=5.0)
        assert wp.y_mm == pytest.approx(300.0, abs=5.0)

    def test_parallel_ray_no_crash(self):
        """Ray parallel to pool surface should not crash."""
        data = _make_calibration_data()
        # Camera looking exactly sideways (R rotated 90 degrees)
        data.rotation_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        cal = CameraCalibration(data, pool_surface_z_mm=0.0)
        # Should not raise, even if result is approximate
        wp = cal.pixel_to_world(960.0, 540.0)
        assert isinstance(wp.x_mm, float)

    def test_pixel_distance_to_mm(self):
        """Known pixel distance should map to expected real-world distance."""
        data = _make_calibration_data()
        cal = CameraCalibration(data, pool_surface_z_mm=0.0)

        dist = cal.pixel_distance_to_mm(960.0, 540.0, 1060.0, 540.0)
        # 100 pixels at f=1000, distance=5000mm → ~500mm real distance
        assert dist > 0
        assert dist == pytest.approx(500.0, rel=0.2)


class TestCalibrationTool:
    def test_insufficient_frames_raises(self):
        """Should raise ValueError with fewer than 5 frames."""
        tool = CalibrationTool(camera_id="test")
        with pytest.raises(ValueError, match="5"):
            tool.calibrate()
