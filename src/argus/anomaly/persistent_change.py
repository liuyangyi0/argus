"""Persistent fixed-scene change scorer.

Detects objects that appear, disappear, or are left behind in a fixed camera
view by comparing the current frame with a slowly adapting reference image.
The scorer is intentionally classical CV only so it can run before heavyweight
model inference and prevent MOG2 no-change early exits from hiding persistent
scene changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class PersistentChangeCandidate:
    """A persistent localized scene-change candidate in original-frame coordinates."""

    track_id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[float, float]
    area_px: int
    mean_diff: float
    max_diff: float
    first_seen_ts: float
    last_seen_ts: float
    seen_frames: int
    persistence_seconds: float
    score: float
    # Tracks are retained briefly across missed frames.  Downstream temporal
    # confirmation must not count such a stale bbox as a fresh observation.
    observed_this_frame: bool = True
    # Exact changed-pixel support, cropped to ``bbox`` at original-frame
    # resolution.  This is deliberately runtime-only (``to_dict`` does not
    # serialize the array): downstream anomaly attribution must never treat the
    # whole bounding rectangle as explained support for a thin/concave object.
    support_mask: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def to_detected_object(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "class_name": "static_foreign",
            "confidence": round(self.score, 3),
            "track_id": self.track_id,
            "bbox": [x, y, x + w, y + h],
            "area_px": self.area_px,
            "persistence_seconds": round(self.persistence_seconds, 3),
        }

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "centroid": [round(self.centroid[0], 3), round(self.centroid[1], 3)],
            "area_px": self.area_px,
            "mean_diff": round(self.mean_diff, 3),
            "max_diff": round(self.max_diff, 3),
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
            "seen_frames": self.seen_frames,
            "persistence_seconds": round(self.persistence_seconds, 3),
            "score": round(self.score, 4),
            "observed_this_frame": self.observed_this_frame,
        }


@dataclass
class PersistentChangeResult:
    """Per-frame persistent-change scorer output."""

    ready: bool
    has_detection: bool = False
    score: float = 0.0
    anomaly_map: np.ndarray | None = None
    candidates: list[PersistentChangeCandidate] = field(default_factory=list)
    active_tracks: list[PersistentChangeCandidate] = field(default_factory=list)
    changed_fraction: float = 0.0
    warmup_remaining: int = 0
    suppression_reason: str | None = None

    def evidence(self) -> dict:
        return {
            "ready": self.ready,
            "has_detection": self.has_detection,
            "score": round(self.score, 4),
            "changed_fraction": round(self.changed_fraction, 6),
            "warmup_remaining": self.warmup_remaining,
            "suppression_reason": self.suppression_reason,
            "candidate_count": len(self.candidates),
            "active_track_count": len(self.active_tracks),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "active_tracks": [track.to_dict() for track in self.active_tracks],
        }


@dataclass
class _TrackState:
    track_id: int
    centroid_x: float
    centroid_y: float
    bbox: tuple[int, int, int, int]
    area_px: int
    mean_diff: float
    max_diff: float
    first_seen_ts: float
    last_seen_ts: float
    support_mask: np.ndarray | None = None
    seen_frames: int = 1
    missed_frames: int = 0


class PersistentChangeScorer:
    """Score localized changes that persist against a fixed-scene reference."""

    def __init__(
        self,
        *,
        process_width: int = 640,
        reference_frames: int = 15,
        diff_threshold: int = 28,
        min_area_px: int = 250,
        max_area_fraction: float = 0.20,
        max_change_fraction: float = 0.12,
        min_persistence_seconds: float = 2.0,
        min_persistence_frames: int = 3,
        learning_alpha: float = 0.002,
        morph_kernel_size: int = 5,
        match_radius_px: int = 40,
        max_missed_frames: int = 3,
        alert_score: float = 0.82,
        max_candidates_per_frame: int = 5,
    ) -> None:
        self._process_width = int(process_width)
        self._reference_frames = int(reference_frames)
        self._diff_threshold = int(diff_threshold)
        self._min_area_px = int(min_area_px)
        self._max_area_fraction = float(max_area_fraction)
        self._max_change_fraction = float(max_change_fraction)
        self._min_persistence_seconds = float(min_persistence_seconds)
        self._min_persistence_frames = int(min_persistence_frames)
        self._learning_alpha = float(learning_alpha)
        self._match_radius_px = int(match_radius_px)
        self._max_missed_frames = int(max_missed_frames)
        self._alert_score = float(alert_score)
        self._max_candidates_per_frame = int(max_candidates_per_frame)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(morph_kernel_size), int(morph_kernel_size)),
        )
        self._reference: np.ndarray | None = None
        self._reference_samples = 0
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    @property
    def is_ready(self) -> bool:
        return self._reference is not None and self._reference_samples >= self._reference_frames

    def reset_tracks(self) -> None:
        self._tracks.clear()

    def reset(self, *, clear_reference: bool = False) -> None:
        self.reset_tracks()
        if clear_reference:
            self._reference = None
            self._reference_samples = 0

    def set_reference(self, frame: np.ndarray) -> None:
        gray, _scale_x, _scale_y = self._prepare_gray(frame)
        self._reference = gray.astype(np.float32)
        self._reference_samples = self._reference_frames
        self.reset_tracks()

    def process(self, frame: np.ndarray, *, timestamp: float | None = None) -> PersistentChangeResult:
        ts = time.time() if timestamp is None else float(timestamp)
        gray, scale_x, scale_y = self._prepare_gray(frame)
        if self._reference is None:
            self._reference = gray.astype(np.float32)
            self._reference_samples = 1
            return PersistentChangeResult(
                ready=False,
                warmup_remaining=max(0, self._reference_frames - self._reference_samples),
                suppression_reason="reference_warmup",
            )

        if self._reference_samples < self._reference_frames:
            alpha = 1.0 / float(self._reference_samples + 1)
            cv2.accumulateWeighted(gray, self._reference, alpha)
            self._reference_samples += 1
            return PersistentChangeResult(
                ready=False,
                warmup_remaining=max(0, self._reference_frames - self._reference_samples),
                suppression_reason="reference_warmup",
            )

        reference_u8 = np.clip(self._reference, 0, 255).astype(np.uint8)
        diff = cv2.absdiff(gray, reference_u8)
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        _, mask = cv2.threshold(blurred, self._diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        changed_fraction = float(np.count_nonzero(mask)) / float(mask.size)
        if changed_fraction > self._max_change_fraction:
            self.reset_tracks()
            self._update_reference(gray, mask=None)
            return PersistentChangeResult(
                ready=True,
                changed_fraction=changed_fraction,
                suppression_reason="global_change",
            )

        regions = self._extract_regions(mask, diff, frame.shape, scale_x, scale_y)
        active_tracks = self._update_tracks(regions, ts)
        candidates = [
            track for track in active_tracks
            if track.persistence_seconds >= self._min_persistence_seconds
            and track.seen_frames >= self._min_persistence_frames
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[: self._max_candidates_per_frame]

        self._update_reference(gray, mask=mask)
        score = max((candidate.score for candidate in candidates), default=0.0)
        anomaly_map = self._build_anomaly_map(mask, score) if candidates else None
        suppression_reason = None
        if regions and not candidates:
            suppression_reason = "persistence_not_met"

        return PersistentChangeResult(
            ready=True,
            has_detection=bool(candidates),
            score=score,
            anomaly_map=anomaly_map,
            candidates=candidates,
            active_tracks=active_tracks,
            changed_fraction=changed_fraction,
            suppression_reason=suppression_reason,
        )

    def _prepare_gray(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        height, width = gray.shape[:2]
        if width > self._process_width:
            scale = self._process_width / float(width)
            new_size = (self._process_width, max(1, int(round(height * scale))))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        scale_x = gray.shape[1] / float(width)
        scale_y = gray.shape[0] / float(height)
        return gray, scale_x, scale_y

    def _extract_regions(
        self,
        mask: np.ndarray,
        diff: np.ndarray,
        frame_shape: tuple[int, ...],
        scale_x: float,
        scale_y: float,
    ) -> list[dict]:
        frame_h, frame_w = frame_shape[:2]
        max_area_px = max(self._min_area_px, int(frame_h * frame_w * self._max_area_fraction))
        binary_mask = (mask > 0).astype(np.uint8)
        _component_count, component_labels = cv2.connectedComponents(
            binary_mask,
            connectivity=8,
        )
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        regions: list[dict] = []
        for contour in contours:
            area_scaled = float(cv2.contourArea(contour))
            if area_scaled <= 0:
                continue
            area_px = int(round(area_scaled / max(scale_x * scale_y, 1e-6)))
            if area_px < self._min_area_px or area_px > max_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            x0 = max(0, min(frame_w - 1, int(round(x / scale_x))))
            y0 = max(0, min(frame_h - 1, int(round(y / scale_y))))
            x1 = max(x0 + 1, min(frame_w, int(round((x + w) / scale_x))))
            y1 = max(y0 + 1, min(frame_h, int(round((y + h) / scale_y))))
            contour_fill = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_fill, [contour], -1, 1, thickness=-1)
            labels_in_contour = component_labels[contour_fill > 0]
            labels_in_contour = labels_in_contour[labels_in_contour > 0]
            if labels_in_contour.size == 0:
                continue
            label_counts = np.bincount(labels_in_contour)
            component_label = int(np.argmax(label_counts))
            component_mask = component_labels == component_label
            values = diff[component_mask > 0]
            mean_diff = float(values.mean()) if values.size else 0.0
            max_diff = float(values.max()) if values.size else 0.0
            cx = (x + w / 2.0) / scale_x
            cy = (y + h / 2.0) / scale_y
            support_crop = component_mask[y : y + h, x : x + w].astype(np.uint8)
            support_mask = cv2.resize(
                support_crop,
                (x1 - x0, y1 - y0),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            regions.append({
                "centroid": (cx, cy),
                "bbox": (x0, y0, x1 - x0, y1 - y0),
                "area_px": area_px,
                "mean_diff": mean_diff,
                "max_diff": max_diff,
                "support_mask": support_mask,
            })
        return regions

    def _update_tracks(
        self,
        regions: list[dict],
        timestamp: float,
    ) -> list[PersistentChangeCandidate]:
        unmatched = set(self._tracks.keys())
        for region in regions:
            cx, cy = region["centroid"]
            best_id = None
            best_dist = float(self._match_radius_px)
            for track_id in list(unmatched):
                track = self._tracks[track_id]
                dist = ((cx - track.centroid_x) ** 2 + (cy - track.centroid_y) ** 2) ** 0.5
                if dist < best_dist:
                    best_id = track_id
                    best_dist = dist
            if best_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _TrackState(
                    track_id=track_id,
                    centroid_x=float(cx),
                    centroid_y=float(cy),
                    bbox=region["bbox"],
                    area_px=int(region["area_px"]),
                    mean_diff=float(region["mean_diff"]),
                    max_diff=float(region["max_diff"]),
                    first_seen_ts=timestamp,
                    last_seen_ts=timestamp,
                    support_mask=region["support_mask"],
                )
            else:
                track = self._tracks[best_id]
                track.centroid_x = float(cx)
                track.centroid_y = float(cy)
                track.bbox = region["bbox"]
                track.area_px = int(region["area_px"])
                track.mean_diff = float(region["mean_diff"])
                track.max_diff = float(region["max_diff"])
                track.support_mask = region["support_mask"]
                track.last_seen_ts = timestamp
                track.seen_frames += 1
                track.missed_frames = 0
                unmatched.discard(best_id)

        for track_id in list(unmatched):
            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self._max_missed_frames:
                del self._tracks[track_id]

        return [self._candidate_from_track(track) for track in self._tracks.values()]

    def _candidate_from_track(self, track: _TrackState) -> PersistentChangeCandidate:
        persistence = max(0.0, track.last_seen_ts - track.first_seen_ts)
        persistence_ratio = min(
            1.0,
            persistence / max(self._min_persistence_seconds, 1e-6),
        )
        diff_ratio = min(1.0, track.mean_diff / 96.0)
        score = min(0.99, self._alert_score + 0.10 * persistence_ratio + 0.07 * diff_ratio)
        return PersistentChangeCandidate(
            track_id=track.track_id,
            bbox=track.bbox,
            centroid=(track.centroid_x, track.centroid_y),
            area_px=track.area_px,
            mean_diff=track.mean_diff,
            max_diff=track.max_diff,
            first_seen_ts=track.first_seen_ts,
            last_seen_ts=track.last_seen_ts,
            seen_frames=track.seen_frames,
            persistence_seconds=persistence,
            score=score,
            observed_this_frame=track.missed_frames == 0,
            support_mask=(
                track.support_mask.copy()
                if track.support_mask is not None
                else None
            ),
        )

    def _update_reference(self, gray: np.ndarray, mask: np.ndarray | None) -> None:
        if self._reference is None or self._learning_alpha <= 0:
            return
        if mask is None or not np.any(mask):
            cv2.accumulateWeighted(gray, self._reference, self._learning_alpha)
            return
        stable = mask == 0
        if not np.any(stable):
            return
        ref = self._reference
        ref[stable] = (
            (1.0 - self._learning_alpha) * ref[stable]
            + self._learning_alpha * gray.astype(np.float32)[stable]
        )

    @staticmethod
    def _build_anomaly_map(mask: np.ndarray, score: float) -> np.ndarray:
        anomaly_map = (mask.astype(np.float32) / 255.0) * float(score)
        return np.clip(anomaly_map, 0.0, 1.0)
