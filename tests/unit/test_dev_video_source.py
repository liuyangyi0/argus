"""Tests for the local development video source helper."""

from __future__ import annotations

import cv2
import numpy as np

from argus.capture.camera import CameraCapture
from argus.runtime.dev_video import create_dev_video


def test_create_dev_video_can_be_read_by_file_camera(tmp_path):
    video_path = tmp_path / "dev_camera.avi"

    meta = create_dev_video(
        video_path,
        width=160,
        height=120,
        fps=5,
        seconds=2,
        anomaly_start_s=1.0,
    )

    assert video_path.exists()
    assert meta["frames"] == 10
    assert meta["anomaly_start_frame"] == 5
    assert meta["motion"] == "settle"

    camera = CameraCapture(
        camera_id="dev_file",
        source=str(video_path),
        protocol="file",
        fps_target=0,
        resolution=(160, 120),
    )
    try:
        assert camera.connect() is True
        frames = [camera.read() for _ in range(12)]
    finally:
        camera.release()

    assert all(frame is not None for frame in frames)
    assert frames[0].resolution == (160, 120)
    # File sources loop after EOF, which keeps a dev camera stream stable.
    assert frames[-1].frame_number == 12


def test_create_dev_video_default_anomaly_settles(tmp_path):
    video_path = tmp_path / "settled_dev_camera.avi"

    create_dev_video(
        video_path,
        width=160,
        height=120,
        fps=5,
        seconds=4,
        anomaly_start_s=1.0,
    )

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        for _ in range(16):
            ok, frame = capture.read()
            assert ok
            frames.append(frame)
    finally:
        capture.release()

    # After one second of entry motion, the object should hold still so the
    # default spatial-continuity gate can accumulate alert evidence.
    diff = cv2.absdiff(frames[12], frames[13])
    assert float(np.max(diff)) <= 1.0
