"""Fast-motion detector for tiny fly-through objects.

This is a lightweight classical-CV channel for objects that may only appear
for one or two frames. It complements SimplexDetector: Simplex is deliberately
static-object focused, while this detector is transient-motion focused.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FastMotionCandidate:
    """One tiny high-speed motion candidate in full-resolution coordinates."""

    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area_px: int
    streak_length_px: float
    confidence: float
    track_id: int | None = None
    speed_px_per_sec: float | None = None
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)

    def to_detected_object(self) -> dict:
        return {
            "class_name": "fast_projectile",
            "confidence": round(self.confidence, 3),
            "track_id": self.track_id,
            "bbox": [int(v) for v in self.bbox],
            "centroid": [round(self.centroid[0], 1), round(self.centroid[1], 1)],
            "area_px": int(self.area_px),
            "streak_length_px": round(float(self.streak_length_px), 1),
            "speed_px_per_sec": (
                round(float(self.speed_px_per_sec), 1)
                if self.speed_px_per_sec is not None
                else None
            ),
            "trajectory_points": [
                {"t": round(float(t), 4), "x": round(float(x), 1), "y": round(float(y), 1)}
                for t, x, y in self.trajectory
            ],
        }


@dataclass
class FastMotionResult:
    """Detector output for one frame."""

    has_detection: bool = False
    candidates: list[FastMotionCandidate] = field(default_factory=list)
    anomaly_map: np.ndarray | None = None
    max_confidence: float = 0.0


class FastMotionDetector:
    """Detect tiny, transient motion without running heavyweight ML models."""

    def __init__(
        self,
        *,
        process_width: int = 960,
        diff_threshold: int = 18,
        background_alpha: float = 0.03,
        min_area_px: int = 2,
        max_area_px: int = 1500,
        min_streak_length_px: int = 4,
        min_confidence: float = 0.6,
        max_motion_fraction: float = 0.015,
        max_streak_frame_fraction: float = 0.25,
        max_candidates_per_frame: int = 5,
        fps_hint: float | None = None,
        trajectory_history_length: int = 16,
    ) -> None:
        self._process_width = process_width
        self._diff_threshold = diff_threshold
        self._background_alpha = background_alpha
        self._min_area_px = min_area_px
        self._max_area_px = max_area_px
        self._min_streak_length_px = min_streak_length_px
        self._min_confidence = min_confidence
        self._max_motion_fraction = max_motion_fraction
        self._max_streak_frame_fraction = max_streak_frame_fraction
        self._max_candidates = max_candidates_per_frame
        self._fps_hint = fps_hint
        self._trajectory_history_length = trajectory_history_length
        self._prev_gray: np.ndarray | None = None
        self._background: np.ndarray | None = None
        self._prev_candidates: list[FastMotionCandidate] = []
        self._prev_timestamp: float | None = None
        self._next_track_id = 1
        self._track_histories: dict[int, list[tuple[float, float, float]]] = {}

    def process(self, frame: np.ndarray, *, timestamp: float | None = None) -> FastMotionResult:
        """Run detection on one BGR frame."""
        if frame is None or frame.size == 0:
            return FastMotionResult()

        ts = timestamp if timestamp is not None else time.time()
        frame_h, frame_w = frame.shape[:2]
        scale = min(1.0, self._process_width / max(frame_w, 1))
        proc_w = int(round(frame_w * scale))
        proc_h = int(round(frame_h * scale))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if scale < 1.0:
            small = cv2.resize(gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            small = gray.copy()

        if self._prev_gray is None or self._background is None:
            self._prev_gray = small
            self._background = small.astype(np.float32)
            self._prev_timestamp = ts
            return FastMotionResult()

        frame_diff = cv2.absdiff(small, self._prev_gray)
        bg_u8 = cv2.convertScaleAbs(self._background)
        bg_diff = cv2.absdiff(small, bg_u8)
        motion = cv2.max(frame_diff, bg_diff)
        _, mask = cv2.threshold(motion, self._diff_threshold, 255, cv2.THRESH_BINARY)

        # Close one-pixel gaps in motion streaks without erasing tiny objects.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        motion_fraction = float(np.count_nonzero(mask)) / float(mask.size)
        if self._max_motion_fraction > 0 and motion_fraction > self._max_motion_fraction:
            cv2.accumulateWeighted(small, self._background, self._background_alpha)
            self._prev_gray = small
            self._prev_candidates = []
            self._prev_timestamp = ts
            return FastMotionResult()

        candidates, accepted_mask = self._extract_candidates(
            motion=motion,
            mask=mask,
            frame_shape=(frame_h, frame_w),
            scale=scale,
            timestamp=ts,
        )

        cv2.accumulateWeighted(small, self._background, self._background_alpha)
        self._prev_gray = small
        self._prev_candidates = candidates
        self._prev_timestamp = ts

        if not candidates:
            return FastMotionResult()

        anomaly_map = cv2.resize(
            accepted_mask.astype(np.float32) / 255.0,
            (frame_w, frame_h),
            interpolation=cv2.INTER_NEAREST,
        )
        max_conf = max(candidate.confidence for candidate in candidates)
        return FastMotionResult(
            has_detection=True,
            candidates=candidates,
            anomaly_map=anomaly_map,
            max_confidence=max_conf,
        )

    def reset(self) -> None:
        self._prev_gray = None
        self._background = None
        self._prev_candidates.clear()
        self._prev_timestamp = None
        self._track_histories.clear()

    def _extract_candidates(
        self,
        *,
        motion: np.ndarray,
        mask: np.ndarray,
        frame_shape: tuple[int, int],
        scale: float,
        timestamp: float,
    ) -> tuple[list[FastMotionCandidate], np.ndarray]:
        frame_h, frame_w = frame_shape
        inv_scale = 1.0 / max(scale, 1e-6)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        accepted: list[FastMotionCandidate] = []
        accepted_mask = np.zeros_like(mask)

        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < self._min_area_px or area > self._max_area_px:
                continue

            streak = float(max(w, h) * inv_scale)
            full_area = int(round(area * inv_scale * inv_scale))
            max_scene_streak = self._max_streak_frame_fraction * max(frame_w, frame_h)
            if self._max_streak_frame_fraction > 0 and streak > max_scene_streak:
                return [], np.zeros_like(mask)
            if streak < self._min_streak_length_px and full_area < self._min_area_px * 4:
                continue

            label_mask = labels == label
            mean_motion = float(motion[label_mask].mean()) / 255.0
            streak_score = min(1.0, streak / max(self._min_streak_length_px, 1))
            area_score = min(1.0, full_area / max(self._min_area_px * 16, 1))
            confidence = min(1.0, 0.25 + mean_motion * 0.45 + streak_score * 0.2 + area_score * 0.1)
            if confidence < self._min_confidence:
                continue

            cx_small, cy_small = centroids[label]
            x1 = max(0, int(math.floor(x * inv_scale)))
            y1 = max(0, int(math.floor(y * inv_scale)))
            x2 = min(frame_w, int(math.ceil((x + w) * inv_scale)))
            y2 = min(frame_h, int(math.ceil((y + h) * inv_scale)))
            centroid = (float(cx_small * inv_scale), float(cy_small * inv_scale))

            candidate = FastMotionCandidate(
                bbox=(x1, y1, x2, y2),
                centroid=centroid,
                area_px=full_area,
                streak_length_px=streak,
                confidence=confidence,
            )
            self._assign_track(candidate, timestamp)
            accepted.append(candidate)
            accepted_mask[label_mask] = 255

        accepted.sort(key=lambda item: item.confidence, reverse=True)
        if len(accepted) > self._max_candidates:
            keep = accepted[: self._max_candidates]
            keep_ids = {id(item) for item in keep}
            accepted_mask[:] = 0
            for label in range(1, num_labels):
                for item in keep:
                    x1, y1, x2, y2 = item.bbox
                    sx1 = int(x1 * scale)
                    sy1 = int(y1 * scale)
                    sx2 = max(sx1 + 1, int(x2 * scale))
                    sy2 = max(sy1 + 1, int(y2 * scale))
                    accepted_mask[sy1:sy2, sx1:sx2] = mask[sy1:sy2, sx1:sx2]
                    keep_ids.discard(id(item))
            accepted = keep
        return accepted, accepted_mask

    def _assign_track(self, candidate: FastMotionCandidate, timestamp: float) -> None:
        best: FastMotionCandidate | None = None
        best_dist = 120.0
        for previous in self._prev_candidates:
            dx = candidate.centroid[0] - previous.centroid[0]
            dy = candidate.centroid[1] - previous.centroid[1]
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best = previous
                best_dist = dist

        if best is not None and best.track_id is not None and self._prev_timestamp is not None:
            candidate.track_id = best.track_id
            dt = max(timestamp - self._prev_timestamp, 1e-6)
            candidate.speed_px_per_sec = best_dist / dt
            self._append_history(candidate, timestamp)
            return

        candidate.track_id = self._next_track_id
        self._next_track_id += 1
        if self._fps_hint and self._fps_hint > 0:
            candidate.speed_px_per_sec = candidate.streak_length_px * float(self._fps_hint)
        self._append_history(candidate, timestamp)

    def _append_history(self, candidate: FastMotionCandidate, timestamp: float) -> None:
        if candidate.track_id is None:
            return
        history = self._track_histories.setdefault(candidate.track_id, [])
        history.append((
            float(timestamp),
            float(candidate.centroid[0]),
            float(candidate.centroid[1]),
        ))
        if len(history) > self._trajectory_history_length:
            del history[:-self._trajectory_history_length]
        candidate.trajectory = list(history)
