"""Deterministic local video source for Argus development."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEV_VIDEO_MOTIONS = ("stable", "settle", "moving", "book", "projectile")


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


def _draw_foreign_object(frame: np.ndarray, x: int, y: int) -> None:
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


def _draw_book(frame: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    cover = (82, 52, 26)
    edge = (28, 24, 20)
    pages = (220, 214, 190)
    accent = (35, 85, 165)

    cv2.rectangle(frame, (x, y), (x + width, y + height), cover, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), edge, 2)
    cv2.rectangle(
        frame,
        (x + max(3, width // 16), y + max(3, height // 12)),
        (x + width - max(5, width // 10), y + height - max(4, height // 10)),
        pages,
        -1,
    )
    spine_w = max(7, width // 7)
    cv2.rectangle(frame, (x, y), (x + spine_w, y + height), accent, -1)
    cv2.line(
        frame,
        (x + width // 2, y + max(5, height // 9)),
        (x + width // 2, y + height - max(5, height // 9)),
        (170, 160, 135),
        1,
    )
    cv2.putText(
        frame,
        "BOOK",
        (x + spine_w + max(5, width // 16), y + height // 2 + max(4, height // 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.35, min(0.7, width / 150.0)),
        edge,
        1,
        cv2.LINE_AA,
    )


def _draw_projectile(frame: np.ndarray, x: int, y: int, length: int, thickness: int) -> None:
    x1 = max(0, x - length)
    x2 = min(frame.shape[1] - 1, x)
    y1 = max(0, y - thickness // 2)
    y2 = min(frame.shape[0] - 1, y + max(1, thickness // 2))
    if x2 <= 0 or x1 >= frame.shape[1] or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), -1)
    cv2.circle(frame, (x2, y), max(2, thickness), (255, 255, 255), -1)


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
    if motion not in DEV_VIDEO_MOTIONS:
        allowed = "', '".join(DEV_VIDEO_MOTIONS)
        raise ValueError(f"motion must be one of '{allowed}'")

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
            if idx >= anomaly_start_frame and motion != "stable":
                t = idx - anomaly_start_frame
                if motion == "moving":
                    span = max(width - 170, 1)
                    x = 60 + (t * 9) % span
                    y = height // 2 - 36 + int(18 * np.sin(t / 8.0))
                    _draw_foreign_object(frame, x, y)
                elif motion == "book":
                    book_w = max(42, min(128, width // 4))
                    book_h = max(30, int(book_w * 0.68))
                    settle_frames = max(fps, 1)
                    target_x = max(12, min(width - book_w - 12, width // 2 - book_w // 2))
                    target_y = max(12, min(height - book_h - 18, height // 2 - book_h // 2))
                    start_y = max(0, target_y - max(18, height // 5))
                    ease = min(1.0, t / settle_frames)
                    y = int(start_y + (target_y - start_y) * ease)
                    _draw_book(frame, target_x, y, book_w, book_h)
                elif motion == "projectile":
                    travel_frames = max(6, min(frame_count - anomaly_start_frame, fps))
                    cycle_frames = max(travel_frames + max(fps // 2, 1), fps * 2)
                    t = t % cycle_frames
                    if t >= travel_frames:
                        writer.write(frame)
                        continue
                    speed = max(8, int(round(width / max(travel_frames, 1))))
                    x = -max(12, width // 28) + t * speed
                    y = int(height * 0.42 + height * 0.08 * np.sin(t / max(fps / 7.0, 1.0)))
                    length = max(8, width // 38)
                    thickness = max(2, min(5, height // 120))
                    _draw_projectile(frame, x, y, length, thickness)
                else:
                    settle_frames = max(fps, 1)
                    start_x = 60
                    target_x = max(20, min(width - 100, width // 2 - 20))
                    ease = min(1.0, t / settle_frames)
                    x = int(start_x + (target_x - start_x) * ease)
                    y = height // 2 - 36
                    _draw_foreign_object(frame, x, y)
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
