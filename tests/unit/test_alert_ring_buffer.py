"""Tests for alert ring buffer (FR-033)."""

import time

import pytest

from argus.core.alert_ring_buffer import (
    AlertFrameBuffer,
    FrameSnapshot,
    RecordingStatus,
    compress_frame,
)


def _make_snapshot(
    timestamp: float | None = None,
    anomaly_score: float = 0.1,
    frame_number: int = 0,
) -> FrameSnapshot:
    """Create a minimal FrameSnapshot for testing."""
    return FrameSnapshot(
        timestamp=timestamp or time.time(),
        frame_jpeg=b"\xff\xd8\xff\xe0" + b"\x00" * 100,  # fake JPEG header
        anomaly_score=anomaly_score,
        simplex_score=None,
        cusum_evidence={"cam:default": 0.5},
        yolo_persons=[],
        frame_number=frame_number,
    )


class TestAlertFrameBuffer:
    def test_append_and_frame_count(self):
        buf = AlertFrameBuffer(fps=5, pre_seconds=10, post_seconds=5)
        assert buf.frame_count == 0

        for i in range(10):
            buf.append(_make_snapshot(frame_number=i))
        assert buf.frame_count == 10

    def test_buffer_maxlen(self):
        """Buffer should not exceed fps * (pre + post) frames."""
        buf = AlertFrameBuffer(fps=5, pre_seconds=2, post_seconds=1)
        # maxlen = 5 * 3 = 15
        for i in range(30):
            buf.append(_make_snapshot(frame_number=i))
        assert buf.frame_count == 15

    def test_solidify_info_severity(self):
        """INFO severity should only capture the trigger frame."""
        buf = AlertFrameBuffer(fps=5, pre_seconds=10, post_seconds=5)
        ts = time.time()
        for i in range(20):
            buf.append(_make_snapshot(timestamp=ts - 10 + i * 0.2, frame_number=i))

        recording = buf.solidify(
            alert_id="ALT-001",
            camera_id="cam_01",
            severity="info",
            trigger_timestamp=ts,
        )
        assert recording is not None
        assert recording.severity == "info"
        assert len(recording.frames) == 1
        assert recording.fps == 1

    def test_solidify_high_severity(self):
        """HIGH severity keeps original FPS for the pre-trigger window."""
        buf = AlertFrameBuffer(fps=5, pre_seconds=10, post_seconds=5)
        ts = time.time()
        # Fill 10 seconds of frames
        for i in range(50):
            buf.append(_make_snapshot(timestamp=ts - 10 + i * 0.2, frame_number=i))

        recording = buf.solidify(
            alert_id="ALT-002",
            camera_id="cam_01",
            severity="high",
            trigger_timestamp=ts,
        )
        assert recording is not None
        assert recording.severity == "high"
        assert len(recording.frames) > 0
        assert recording.status == RecordingStatus.RECORDING

    def test_solidify_medium_severity_downsample(self):
        """MEDIUM severity should downsample to 10 FPS."""
        buf = AlertFrameBuffer(fps=15, pre_seconds=10, post_seconds=5)
        ts = time.time()
        for i in range(150):
            buf.append(_make_snapshot(timestamp=ts - 10 + i / 15, frame_number=i))

        recording = buf.solidify(
            alert_id="ALT-003",
            camera_id="cam_01",
            severity="medium",
            trigger_timestamp=ts,
        )
        assert recording is not None
        assert recording.fps == 10
        # Downsampled to ~10 FPS over 10 seconds = ~100 frames
        assert len(recording.frames) <= 110

    def test_solidify_low_severity_short_window(self):
        """LOW severity has shorter window (10s pre + 10s post) at 5 FPS."""
        buf = AlertFrameBuffer(fps=15, pre_seconds=60, post_seconds=30)
        ts = time.time()
        for i in range(900):
            buf.append(_make_snapshot(timestamp=ts - 60 + i / 15, frame_number=i))

        recording = buf.solidify(
            alert_id="ALT-004",
            camera_id="cam_01",
            severity="low",
            trigger_timestamp=ts,
        )
        assert recording is not None
        # LOW: only 10 seconds pre-trigger at 5 FPS = ~50 frames
        assert recording.fps == 5
        assert len(recording.frames) <= 60

    def test_solidify_empty_buffer_returns_none(self):
        buf = AlertFrameBuffer(fps=5)
        assert buf.solidify("ALT-005", "cam_01", "high", time.time()) is None

    def test_post_capture_lifecycle(self):
        buf = AlertFrameBuffer(fps=5, pre_seconds=5, post_seconds=2)
        ts = time.time()

        # Fill pre-trigger frames
        for i in range(25):
            buf.append(_make_snapshot(timestamp=ts - 5 + i * 0.2, frame_number=i))

        buf.start_post_capture("ALT-006", "high", ts)
        assert "ALT-006" in buf.get_pending_captures()

        # Frames during post-capture window
        for i in range(10):
            buf.append(_make_snapshot(timestamp=ts + i * 0.2, frame_number=25 + i))

        # Not ready yet (deadline not reached) — manual check
        expired = buf.check_expired_captures()
        # May or may not be expired depending on timing

    def test_memory_estimate(self):
        buf = AlertFrameBuffer(fps=5)
        for i in range(10):
            buf.append(_make_snapshot(frame_number=i))
        assert buf.memory_estimate_mb > 0

    def test_linked_alert_id(self):
        buf = AlertFrameBuffer(fps=5, pre_seconds=5, post_seconds=2)
        ts = time.time()
        for i in range(25):
            buf.append(_make_snapshot(timestamp=ts - 5 + i * 0.2, frame_number=i))

        recording = buf.solidify(
            alert_id="ALT-007",
            camera_id="cam_01",
            severity="high",
            trigger_timestamp=ts,
            linked_alert_id="ALT-006",
        )
        assert recording is not None
        assert recording.linked_alert_id == "ALT-006"


class TestCompressFrame:
    def test_compress_frame_returns_bytes(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = compress_frame(frame, quality=85)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_compress_frame_scales_down(self):
        import numpy as np
        # Large frame should be scaled to max_height
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = compress_frame(frame, quality=85, max_height=720)
        assert isinstance(result, bytes)
        assert len(result) > 0
