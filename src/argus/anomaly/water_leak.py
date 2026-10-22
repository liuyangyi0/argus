"""Visible water leak / drip / pooling scorer for fixed cameras.

This head is intentionally scoped to RGB-visible leak signs: localized
darkening, bright/specular wet patches, vertical drip streaks, and growing
pool-like regions. It does not claim to detect invisible gas or clear fluid
without visible image evidence.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class WaterLeakCandidate:
    """A localized visible water leak sign in original-frame coordinates."""

    track_id: int
    leak_type: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[float, float]
    area_px: int
    growth_area_px: int
    mean_absdiff: float
    mean_signed_delta: float
    first_seen_ts: float
    last_seen_ts: float
    seen_frames: int
    persistence_seconds: float
    vertical_extent_px: float
    vertical_aspect: float
    score: float

    def to_detected_object(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "class_name": "visible_water_leak",
            "confidence": round(self.score, 3),
            "track_id": self.track_id,
            "bbox": [x, y, x + w, y + h],
            "leak_type": self.leak_type,
            "area_px": self.area_px,
            "growth_area_px": self.growth_area_px,
            "persistence_seconds": round(self.persistence_seconds, 3),
            "vertical_extent_px": round(float(self.vertical_extent_px), 1),
            "vertical_aspect": round(float(self.vertical_aspect), 2),
        }

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "leak_type": self.leak_type,
            "bbox": list(self.bbox),
            "centroid": [round(self.centroid[0], 3), round(self.centroid[1], 3)],
            "area_px": self.area_px,
            "growth_area_px": self.growth_area_px,
            "mean_absdiff": round(self.mean_absdiff, 3),
            "mean_signed_delta": round(self.mean_signed_delta, 3),
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
            "seen_frames": self.seen_frames,
            "persistence_seconds": round(self.persistence_seconds, 3),
            "vertical_extent_px": round(float(self.vertical_extent_px), 3),
            "vertical_aspect": round(float(self.vertical_aspect), 3),
            "score": round(self.score, 4),
        }


@dataclass
class WaterLeakResult:
    """Per-frame visible water leak scorer output."""

    ready: bool
    has_detection: bool = False
    score: float = 0.0
    anomaly_map: np.ndarray | None = None
    candidates: list[WaterLeakCandidate] = field(default_factory=list)
    active_tracks: list[WaterLeakCandidate] = field(default_factory=list)
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
class _Region:
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area_px: int
    mean_absdiff: float
    mean_signed_delta: float
    vertical_extent_px: float
    vertical_aspect: float
    leak_type: str


@dataclass
class _TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid_x: float
    centroid_y: float
    area_px: int
    first_area_px: int
    max_area_px: int
    mean_absdiff: float
    mean_signed_delta: float
    vertical_extent_px: float
    vertical_aspect: float
    leak_type: str
    first_seen_ts: float
    last_seen_ts: float
    seen_frames: int = 1
    missed_frames: int = 0


class WaterLeakHead:
    """Detect visible drip, leak, and pooling signs against a fixed reference."""

    def __init__(
        self,
        *,
        process_width: int = 640,
        reference_frames: int = 15,
        diff_threshold: int = 22,
        darkening_threshold: int = 14,
        brightening_threshold: int = 18,
        specular_value_threshold: int = 180,
        specular_saturation_max: int = 80,
        min_area_px: int = 120,
        max_area_fraction: float = 0.12,
        max_change_fraction: float = 0.08,
        min_growth_area_px: int = 80,
        min_persistence_seconds: float = 1.0,
        min_persistence_frames: int = 3,
        min_drip_length_px: int = 18,
        min_vertical_aspect: float = 2.0,
        learning_alpha: float = 0.001,
        morph_kernel_size: int = 5,
        match_radius_px: int = 50,
        max_missed_frames: int = 5,
        alert_score: float = 0.84,
        max_candidates_per_frame: int = 5,
    ) -> None:
        self._process_width = int(process_width)
        self._reference_frames = int(reference_frames)
        self._diff_threshold = int(diff_threshold)
        self._darkening_threshold = int(darkening_threshold)
        self._brightening_threshold = int(brightening_threshold)
        self._specular_value_threshold = int(specular_value_threshold)
        self._specular_saturation_max = int(specular_saturation_max)
        self._min_area_px = int(min_area_px)
        self._max_area_fraction = float(max_area_fraction)
        self._max_change_fraction = float(max_change_fraction)
        self._min_growth_area_px = int(min_growth_area_px)
        self._min_persistence_seconds = float(min_persistence_seconds)
        self._min_persistence_frames = int(min_persistence_frames)
        self._min_drip_length_px = int(min_drip_length_px)
        self._min_vertical_aspect = float(min_vertical_aspect)
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
        gray, _bgr, _scale_x, _scale_y = self._prepare(frame)
        self._reference = gray.astype(np.float32)
        self._reference_samples = self._reference_frames
        self.reset_tracks()

    def process(self, frame: np.ndarray, *, timestamp: float | None = None) -> WaterLeakResult:
        ts = time.time() if timestamp is None else float(timestamp)
        gray, bgr_small, scale_x, scale_y = self._prepare(frame)
        if self._reference is None:
            self._reference = gray.astype(np.float32)
            self._reference_samples = 1
            return WaterLeakResult(
                ready=False,
                warmup_remaining=max(0, self._reference_frames - self._reference_samples),
                suppression_reason="reference_warmup",
            )

        if self._reference_samples < self._reference_frames:
            alpha = 1.0 / float(self._reference_samples + 1)
            cv2.accumulateWeighted(gray, self._reference, alpha)
            self._reference_samples += 1
            ready = self._reference_samples >= self._reference_frames
            return WaterLeakResult(
                ready=ready,
                warmup_remaining=max(0, self._reference_frames - self._reference_samples),
                suppression_reason="reference_warmup",
            )

        reference_u8 = np.clip(self._reference, 0, 255).astype(np.uint8)
        signed = gray.astype(np.int16) - reference_u8.astype(np.int16)
        absdiff = np.abs(signed).astype(np.uint8)
        _, diff_mask = cv2.threshold(absdiff, self._diff_threshold, 255, cv2.THRESH_BINARY)

        dark_mask = signed <= -self._darkening_threshold
        bright_mask = signed >= self._brightening_threshold
        hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
        specular_mask = (
            (hsv[:, :, 2] >= self._specular_value_threshold)
            & (hsv[:, :, 1] <= self._specular_saturation_max)
            & (diff_mask > 0)
        )
        cue_mask = dark_mask | bright_mask | specular_mask
        mask = np.where((diff_mask > 0) & cue_mask, 255, 0).astype(np.uint8)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        changed_fraction = float(np.count_nonzero(mask)) / float(mask.size)
        if changed_fraction > self._max_change_fraction:
            self.reset_tracks()
            self._update_reference(gray, mask=None)
            return WaterLeakResult(
                ready=True,
                changed_fraction=changed_fraction,
                suppression_reason="global_change",
            )

        regions = self._extract_regions(mask, absdiff, signed, frame.shape, scale_x, scale_y)
        active_tracks = self._update_tracks(regions, ts)
        candidates = [
            track for track in active_tracks
            if track.persistence_seconds >= self._min_persistence_seconds
            and track.seen_frames >= self._min_persistence_frames
            and (
                track.growth_area_px >= self._min_growth_area_px
                or track.leak_type == "drip"
            )
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[: self._max_candidates_per_frame]

        self._update_reference(gray, mask=mask)
        score = max((candidate.score for candidate in candidates), default=0.0)
        anomaly_map = self._build_anomaly_map(mask, frame.shape, score) if candidates else None
        suppression_reason = None
        if not candidates:
            if not active_tracks:
                suppression_reason = "no_visible_water_change"
            elif any(track.leak_type == "pooling" for track in active_tracks):
                suppression_reason = "growth_not_met"
            else:
                suppression_reason = "persistence_not_met"

        return WaterLeakResult(
            ready=True,
            has_detection=bool(candidates),
            score=score,
            anomaly_map=anomaly_map,
            candidates=candidates,
            active_tracks=active_tracks,
            changed_fraction=changed_fraction,
            suppression_reason=suppression_reason,
        )

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        h, w = frame.shape[:2]
        scale = min(1.0, self._process_width / max(w, 1))
        if scale < 1.0:
            resized = cv2.resize(
                frame,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = frame.copy()
        bgr = resized if resized.ndim == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        scale_x = bgr.shape[1] / max(w, 1)
        scale_y = bgr.shape[0] / max(h, 1)
        return gray, bgr, scale_x, scale_y

    def _extract_regions(
        self,
        mask: np.ndarray,
        absdiff: np.ndarray,
        signed: np.ndarray,
        frame_shape: tuple[int, ...],
        scale_x: float,
        scale_y: float,
    ) -> list[_Region]:
        frame_h, frame_w = frame_shape[:2]
        inv_x = 1.0 / max(scale_x, 1e-6)
        inv_y = 1.0 / max(scale_y, 1e-6)
        max_area_px = self._max_area_fraction * frame_w * frame_h
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        regions: list[_Region] = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            full_area = int(round(area * inv_x * inv_y))
            if full_area < self._min_area_px or full_area > max_area_px:
                continue
            label_mask = labels == label
            mean_absdiff = float(absdiff[label_mask].mean())
            mean_signed_delta = float(signed[label_mask].mean())
            full_w = max(1.0, float(w) * inv_x)
            full_h = max(1.0, float(h) * inv_y)
            vertical_aspect = full_h / full_w
            leak_type = (
                "drip"
                if full_h >= self._min_drip_length_px
                and vertical_aspect >= self._min_vertical_aspect
                else "pooling"
            )
            cx_small, cy_small = centroids[label]
            x1 = max(0, int(math.floor(x * inv_x)))
            y1 = max(0, int(math.floor(y * inv_y)))
            x2 = min(frame_w, int(math.ceil((x + w) * inv_x)))
            y2 = min(frame_h, int(math.ceil((y + h) * inv_y)))
            regions.append(_Region(
                bbox=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                centroid=(float(cx_small * inv_x), float(cy_small * inv_y)),
                area_px=full_area,
                mean_absdiff=mean_absdiff,
                mean_signed_delta=mean_signed_delta,
                vertical_extent_px=full_h,
                vertical_aspect=vertical_aspect,
                leak_type=leak_type,
            ))
        return regions

    def _update_tracks(self, regions: list[_Region], timestamp: float) -> list[WaterLeakCandidate]:
        matched_track_ids: set[int] = set()
        for region in regions:
            best_id = None
            best_dist = float(self._match_radius_px)
            for track_id, track in self._tracks.items():
                if track_id in matched_track_ids:
                    continue
                dist = math.hypot(
                    region.centroid[0] - track.centroid_x,
                    region.centroid[1] - track.centroid_y,
                )
                if dist <= best_dist:
                    best_id = track_id
                    best_dist = dist

            if best_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _TrackState(
                    track_id=track_id,
                    bbox=region.bbox,
                    centroid_x=region.centroid[0],
                    centroid_y=region.centroid[1],
                    area_px=region.area_px,
                    first_area_px=region.area_px,
                    max_area_px=region.area_px,
                    mean_absdiff=region.mean_absdiff,
                    mean_signed_delta=region.mean_signed_delta,
                    vertical_extent_px=region.vertical_extent_px,
                    vertical_aspect=region.vertical_aspect,
                    leak_type=region.leak_type,
                    first_seen_ts=timestamp,
                    last_seen_ts=timestamp,
                )
                matched_track_ids.add(track_id)
                continue

            track = self._tracks[best_id]
            track.bbox = region.bbox
            track.centroid_x = region.centroid[0]
            track.centroid_y = region.centroid[1]
            track.area_px = region.area_px
            track.max_area_px = max(track.max_area_px, region.area_px)
            track.mean_absdiff = region.mean_absdiff
            track.mean_signed_delta = region.mean_signed_delta
            track.vertical_extent_px = region.vertical_extent_px
            track.vertical_aspect = region.vertical_aspect
            track.leak_type = "drip" if region.leak_type == "drip" else track.leak_type
            track.last_seen_ts = timestamp
            track.seen_frames += 1
            track.missed_frames = 0
            matched_track_ids.add(best_id)

        for track_id in list(self._tracks):
            if track_id in matched_track_ids:
                continue
            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self._max_missed_frames:
                del self._tracks[track_id]

        active = [
            self._candidate_from_track(track, timestamp)
            for track in self._tracks.values()
            if track.missed_frames == 0
        ]
        active.sort(key=lambda item: item.score, reverse=True)
        return active

    def _candidate_from_track(
        self,
        track: _TrackState,
        timestamp: float,
    ) -> WaterLeakCandidate:
        persistence = max(0.0, timestamp - track.first_seen_ts)
        growth_area = max(0, track.max_area_px - track.first_area_px)
        growth_score = min(1.0, growth_area / max(self._min_growth_area_px, 1))
        persistence_score = min(1.0, persistence / max(self._min_persistence_seconds, 1e-6))
        contrast_score = min(1.0, track.mean_absdiff / max(self._diff_threshold * 4, 1))
        score = min(
            0.99,
            self._alert_score
            + 0.06 * growth_score
            + 0.04 * persistence_score
            + 0.03 * contrast_score,
        )
        return WaterLeakCandidate(
            track_id=track.track_id,
            leak_type=track.leak_type,
            bbox=track.bbox,
            centroid=(track.centroid_x, track.centroid_y),
            area_px=track.area_px,
            growth_area_px=growth_area,
            mean_absdiff=track.mean_absdiff,
            mean_signed_delta=track.mean_signed_delta,
            first_seen_ts=track.first_seen_ts,
            last_seen_ts=track.last_seen_ts,
            seen_frames=track.seen_frames,
            persistence_seconds=persistence,
            vertical_extent_px=track.vertical_extent_px,
            vertical_aspect=track.vertical_aspect,
            score=score,
        )

    def _update_reference(self, gray: np.ndarray, mask: np.ndarray | None) -> None:
        if self._reference is None or self._learning_alpha <= 0:
            return
        if mask is None:
            cv2.accumulateWeighted(gray, self._reference, self._learning_alpha)
            return
        stable_mask = cv2.bitwise_not(mask)
        if np.count_nonzero(stable_mask):
            cv2.accumulateWeighted(
                gray,
                self._reference,
                self._learning_alpha,
                mask=stable_mask,
            )

    def _build_anomaly_map(
        self,
        mask: np.ndarray,
        frame_shape: tuple[int, ...],
        score: float,
    ) -> np.ndarray:
        frame_h, frame_w = frame_shape[:2]
        small = mask.astype(np.float32) / 255.0 * float(score)
        return cv2.resize(small, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
