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
from argus.core.event_bus import EventBus


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
def frame_with_anomaly_patch(blank_frame):
    """A black frame with a saturated red rectangle (simulates anomaly).

    Pure white (BGR=255,255,255) is HSV saturation=0 and gets filtered out by
    the prefilter's chromaticity-based shadow suppression. Use a saturated
    color so the patch survives shadow filtering.
    """
    frame = blank_frame.copy()
    frame[100:200, 150:300] = (0, 0, 255)
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


@pytest.fixture
def event_bus():
    """A fresh EventBus instance, cleared after test."""
    bus = EventBus()
    yield bus
    bus.clear()


@pytest.fixture
def frame_sequence(blank_frame, noisy_frame):
    """A sequence of 10 frames alternating blank and noisy."""
    return [blank_frame if i % 2 == 0 else noisy_frame for i in range(10)]
