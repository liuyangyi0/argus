"""Tests for the local development video source helper."""

from __future__ import annotations

import cv2
import numpy as np

from argus.capture.camera import CameraCapture
from argus.prefilter.fast_motion import FastMotionDetector
from argus.runtime.dev_video import DEV_VIDEO_MOTIONS, create_dev_video


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


def test_create_dev_video_stable_scene_never_introduces_anomaly(tmp_path):
    video_path = tmp_path / "stable_dev_camera.avi"

    meta = create_dev_video(
        video_path,
        width=160,
        height=120,
        fps=5,
        seconds=4,
        anomaly_start_s=1.0,
        motion="stable",
    )

    assert "stable" in DEV_VIDEO_MOTIONS
    assert meta["motion"] == "stable"

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        for _ in range(16):
            ok, frame = capture.read()
            assert ok
            frames.append(frame)
    finally:
        capture.release()

    before_to_after = cv2.absdiff(frames[4], frames[12])
    settled = cv2.absdiff(frames[12], frames[13])
    assert float(np.max(before_to_after)) <= 1.0
    assert float(np.max(settled)) <= 1.0


def test_create_dev_video_book_scene_places_static_book(tmp_path):
    video_path = tmp_path / "book_dev_camera.avi"

    meta = create_dev_video(
        video_path,
        width=160,
        height=120,
        fps=5,
        seconds=4,
        anomaly_start_s=1.0,
        motion="book",
    )

    assert meta["motion"] == "book"

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        for _ in range(16):
            ok, frame = capture.read()
            assert ok
            frames.append(frame)
    finally:
        capture.release()

    changed = cv2.absdiff(frames[4], frames[12])
    settled = cv2.absdiff(frames[12], frames[13])
    assert float(np.mean(changed)) > 3.0
    assert float(np.max(settled)) <= 1.0


def test_create_dev_video_projectile_triggers_fast_motion_detector(tmp_path):
    video_path = tmp_path / "projectile_dev_camera.avi"

    meta = create_dev_video(
        video_path,
        width=320,
        height=180,
        fps=60,
        seconds=5,
        anomaly_start_s=0.5,
        motion="projectile",
    )

    assert "projectile" in DEV_VIDEO_MOTIONS
    assert meta["motion"] == "projectile"

    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        min_area_px=2,
        max_area_px=1500,
        min_streak_length_px=4,
        min_confidence=0.55,
        fps_hint=60,
    )
    capture = cv2.VideoCapture(str(video_path))
    detections = []
    detection_frames = []
    try:
        for idx in range(meta["frames"]):
            ok, frame = capture.read()
            assert ok
            result = detector.process(frame, timestamp=idx / 60)
            if result.has_detection:
                detections.extend(result.candidates)
                detection_frames.append(idx)
    finally:
        capture.release()

    assert detections
    assert any(candidate.to_detected_object()["class_name"] == "fast_projectile" for candidate in detections)
    assert any(idx > meta["anomaly_start_frame"] + 120 for idx in detection_frames)
