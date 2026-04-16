"""Shared fixtures for integration tests.

Provides synthetic frame generators, minimal pipeline configs, and
mock camera/detector objects for testing multi-component interactions
without real hardware or trained models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argus.config.schema import (
    AlertConfig,
    AnomalyConfig,
    CameraConfig,
    MOG2Config,
    PersonFilterConfig,
    SeverityThresholds,
    SimplexConfig,
    SuppressionConfig,
    TemporalConfirmation,
)


# ---------------------------------------------------------------------------
# Synthetic frame generators
# ---------------------------------------------------------------------------

def make_gray_frame(
    width: int = 640, height: int = 480, value: int = 128
) -> np.ndarray:
    """Create a solid gray BGR frame."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def make_anomaly_frame(
    width: int = 640,
    height: int = 480,
    base_value: int = 128,
    patch_value: int = 0,
    patch_rect: tuple[int, int, int, int] = (100, 150, 200, 300),
) -> np.ndarray:
    """Create a frame with an anomalous rectangular patch.

    patch_rect is (y1, x1, y2, x2).
    """
    frame = make_gray_frame(width, height, base_value)
    y1, x1, y2, x2 = patch_rect
    frame[y1:y2, x1:x2] = patch_value
    return frame


def frame_sequence(frames: list[np.ndarray], camera_id: str = "test_cam"):
    """Yield FrameData objects from a list of frames."""
    from argus.capture.camera import FrameData
    import time

    for i, f in enumerate(frames):
        yield FrameData(
            frame=f.copy(),
            camera_id=camera_id,
            timestamp=time.monotonic(),
            frame_number=i,
            resolution=(f.shape[1], f.shape[0]),
        )


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@pytest.fixture
def integration_camera_config():
    """Minimal camera config for integration tests (no real camera)."""
    return CameraConfig(
        camera_id="int_cam",
        name="Integration Camera",
        source="synthetic",
        protocol="file",
        fps_target=5,
        resolution=(640, 480),
        mog2=MOG2Config(
            history=50,
            heartbeat_frames=10,
            var_threshold=25.0,
            change_pct_threshold=0.005,
            enable_stabilization=False,
        ),
        person_filter=PersonFilterConfig(model_name="yolo11n.pt"),
        anomaly=AnomalyConfig(
            threshold=0.5,
            ssim_baseline_frames=5,
        ),
        simplex=SimplexConfig(enabled=False),
    )


@pytest.fixture
def fast_alert_config():
    """Alert config with low thresholds and fast evidence accumulation."""
    return AlertConfig(
        severity_thresholds=SeverityThresholds(
            info=0.3, low=0.5, medium=0.7, high=0.9,
        ),
        temporal=TemporalConfirmation(

            max_gap_seconds=10.0,
            evidence_lambda=0.8,
            evidence_threshold=1.0,
            min_spatial_overlap=0.0,
        ),
        suppression=SuppressionConfig(
            same_zone_window_seconds=10.0,
            same_camera_window_seconds=10.0,
        ),
    )


@pytest.fixture
def integration_db(tmp_path):
    """Temporary SQLite database for integration tests."""
    from argus.storage.database import Database

    db = Database(database_url=f"sqlite:///{tmp_path / 'integration_test.db'}")
    db.initialize()
    yield db
    db.close()
