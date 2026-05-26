"""Deterministic local video source for Argus development."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _base_frame(width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), 118, dtype=np.uint8)

    cv2.rectangle(frame, (30, 30), (width - 30, height - 30), (145, 145, 145), 2)
    cv2.line(frame, (0, height // 2), (width, height // 2), (95, 95, 95), 1)
    cv2.line(frame, (width // 2, 0), (width // 2, height), (95, 95, 95), 1)
    cv2.putText(
        frame,
        "ARGUS DEV SOURCE",
        (24, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    return frame


def create_dev_video(
    output: Path,
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 10,
    seconds: int = 20,
    anomaly_start_s: float = 6.0,
    motion: str = "settle",
) -> dict[str, int | str]:
    """Generate a loopable development video and return summary metadata."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if fps <= 0 or seconds <= 0:
        raise ValueError("fps and seconds must be positive")
    if motion not in {"settle", "moving"}:
        raise ValueError("motion must be 'settle' or 'moving'")

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output}")

    frame_count = int(fps * seconds)
    anomaly_start_frame = int(fps * anomaly_start_s)
    base = _base_frame(width, height)

    try:
        for idx in range(frame_count):
            frame = base.copy()
            if idx >= anomaly_start_frame:
                t = idx - anomaly_start_frame
                if motion == "moving":
                    span = max(width - 170, 1)
                    x = 60 + (t * 9) % span
                    y = height // 2 - 36 + int(18 * np.sin(t / 8.0))
                else:
                    settle_frames = max(fps, 1)
                    start_x = 60
                    target_x = max(20, min(width - 100, width // 2 - 20))
                    ease = min(1.0, t / settle_frames)
                    x = int(start_x + (target_x - start_x) * ease)
                    y = height // 2 - 36
                cv2.rectangle(frame, (x, y), (x + 70, y + 52), (0, 0, 255), -1)
                cv2.rectangle(frame, (x, y), (x + 70, y + 52), (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    "FOE",
                    (x + 16, y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        writer.release()

    return {
        "output": str(output),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frame_count,
        "anomaly_start_frame": anomaly_start_frame,
        "motion": motion,
    }
