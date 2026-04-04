"""Shared test fixtures."""

import numpy as np
import pytest

from argus.config.schema import (
    AlertConfig,
    AnomalyConfig,
    CameraConfig,
    MOG2Config,
    PersonFilterConfig,
)


@pytest.fixture
def blank_frame():
    """A 480x640 black frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def noisy_frame():
    """A 480x640 frame with random noise (simulates change)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def frame_with_white_patch(blank_frame):
    """A black frame with a white rectangle (simulates anomaly)."""
    frame = blank_frame.copy()
    frame[100:200, 150:300] = 255  # white patch
    return frame


@pytest.fixture
def camera_config():
    """A test camera config using file protocol."""
    return CameraConfig(
        camera_id="test_cam",
        name="Test Camera",
        source="test_video.mp4",
        protocol="file",
        fps_target=30,
        mog2=MOG2Config(
            history=50,
            var_threshold=25.0,
            detect_shadows=True,
            change_pct_threshold=0.005,
        ),
        person_filter=PersonFilterConfig(
            confidence=0.4,
            skip_frame_on_person=False,
            model_name="yolo11n.pt",
        ),
        anomaly=AnomalyConfig(
            model_type="patchcore",
            threshold=0.7,
            image_size=(256, 256),
        ),
    )


@pytest.fixture
def alert_config():
    """A test alert config."""
    return AlertConfig()
