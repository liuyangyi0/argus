"""Visible smoke / steam / dust / mist plume scorer for fixed RGB cameras.

This head is scoped to RGB-visible plume signs only. It looks for localized,
diffuse, low-saturation changes that persist against a fixed reference. It
does not claim to detect invisible or colorless gas.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class PlumeCandidate:
    """A localized visible plume sign in original-frame coordinates."""

    track_id: int
    plume_type: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[float, float]
    area_px: int
    growth_area_px: int
    mean_absdiff: float
    mean_signed_delta: float
    mean_saturation: float
    mean_value: float
    edge_density: float
    fill_fraction: float
    first_seen_ts: float
    last_seen_ts: float
    seen_frames: int
    persistence_seconds: float
    score: float

    def to_detected_object(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "class_name": "visible_plume",
            "confidence": round(self.score, 3),
            "track_id": self.track_id,
            "bbox": [x, y, x + w, y + h],
            "plume_type": self.plume_type,
            "area_px": self.area_px,
            "growth_area_px": self.growth_area_px,
            "persistence_seconds": round(self.persistence_seconds, 3),
            "edge_density": round(self.edge_density, 4),
            "fill_fraction": round(self.fill_fraction, 4),
        }

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "plume_type": self.plume_type,
            "bbox": list(self.bbox),
            "centroid": [round(self.centroid[0], 3), round(self.centroid[1], 3)],
            "area_px": self.area_px,
            "growth_area_px": self.growth_area_px,
            "mean_absdiff": round(self.mean_absdiff, 3),
            "mean_signed_delta": round(self.mean_signed_delta, 3),
            "mean_saturation": round(self.mean_saturation, 3),
            "mean_value": round(self.mean_value, 3),
            "edge_density": round(self.edge_density, 6),
            "fill_fraction": round(self.fill_fraction, 6),
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
            "seen_frames": self.seen_frames,
            "persistence_seconds": round(self.persistence_seconds, 3),
            "score": round(self.score, 4),
        }


@dataclass
class PlumeResult:
    """Per-frame visible plume scorer output."""

    ready: bool
    has_detection: bool = False
    score: float = 0.0
    anomaly_map: np.ndarray | None = None
    candidates: list[PlumeCandidate] = field(default_factory=list)
    active_tracks: list[PlumeCandidate] = field(default_factory=list)
    changed_fraction: float = 0.0
    warmup_remaining: int = 0
    suppression_reason: str | None = None
    suppressed_region_count: int = 0

    def evidence(self) -> dict:
        return {
            "ready": self.ready,
            "has_detection": self.has_detection,
            "score": round(self.score, 4),
            "changed_fraction": round(self.changed_fraction, 6),
            "warmup_remaining": self.warmup_remaining,
            "suppression_reason": self.suppression_reason,
            "suppressed_region_count": self.suppressed_region_count,
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
    mean_saturation: float
    mean_value: float
    edge_density: float
    fill_fraction: float
    plume_type: str


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
    mean_saturation: float
    mean_value: float
    edge_density: float
    fill_fraction: float
    plume_type: str
    first_seen_ts: float
    last_seen_ts: float
    seen_frames: int = 1
    missed_frames: int = 0


class PlumeHead:
    """Detect visible smoke, steam, dust, and mist signs against a fixed reference."""

    def __init__(
        self,
        *,
        process_width: int = 640,
        reference_frames: int = 15,
        diff_threshold: int = 16,
        brightening_threshold: int = 12,
        darkening_threshold: int = 12,
        max_saturation: int = 150,
        min_area_px: int = 180,
        max_area_fraction: float = 0.25,
        max_change_fraction: float = 0.18,
        min_growth_area_px: int = 100,
        min_persistence_seconds: float = 1.0,
        min_persistence_frames: int = 3,
        max_edge_density: float = 0.08,
        max_bbox_fill_fraction: float = 0.90,
        learning_alpha: float = 0.001,
        morph_kernel_size: int = 7,
        match_radius_px: int = 70,
        max_missed_frames: int = 6,
        alert_score: float = 0.83,
        max_candidates_per_frame: int = 5,
    ) -> None:
        self._process_width = int(process_width)
        self._reference_frames = int(reference_frames)
        self._diff_threshold = int(diff_threshold)
        self._brightening_threshold = int(brightening_threshold)
        self._darkening_threshold = int(darkening_threshold)
        self._max_saturation = int(max_saturation)
        self._min_area_px = int(min_area_px)
        self._max_area_fraction = float(max_area_fraction)
        self._max_change_fraction = float(max_change_fraction)
        self._min_growth_area_px = int(min_growth_area_px)
        self._min_persistence_seconds = float(min_persistence_seconds)
        self._min_persistence_frames = int(min_persistence_frames)
        self._max_edge_density = float(max_edge_density)
        self._max_bbox_fill_fraction = float(max_bbox_fill_fraction)
        self._learning_alpha = float(learning_alpha)
        self._match_radius_px = int(match_radius_px)
        self._max_missed_frames = int(max_missed_frames)
        self._alert_score = float(alert_score)
        self._max_candidates_per_frame = int(max_candidates_per_frame)
        k = max(3, int(morph_kernel_size))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
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
        gray, _bgr, _hsv, _scale_x, _scale_y = self._prepare(frame)
        self._reference = gray.astype(np.float32)
        self._reference_samples = self._reference_frames
        self.reset_tracks()

    def process(self, frame: np.ndarray, *, timestamp: float | None = None) -> PlumeResult:
        ts = time.time() if timestamp is None else float(timestamp)
        gray, _bgr_small, hsv, scale_x, scale_y = self._prepare(frame)
        if self._reference is None:
            self._reference = gray.astype(np.float32)
            self._reference_samples = 1
            return PlumeResult(
                ready=False,
                warmup_remaining=max(0, self._reference_frames - self._reference_samples),
                suppression_reason="reference_warmup",
            )

        if self._reference_samples < self._reference_frames:
            alpha = 1.0 / float(self._reference_samples + 1)
            cv2.accumulateWeighted(gray, self._reference, alpha)
            self._reference_samples += 1
            ready = self._reference_samples >= self._reference_frames
            return PlumeResult(
                ready=ready,
                warmup_remaining=max(0, self._reference_frames - self._reference_samples),
                suppression_reason="reference_warmup",
            )

        reference_u8 = np.clip(self._reference, 0, 255).astype(np.uint8)
        signed = gray.astype(np.int16) - reference_u8.astype(np.int16)
        absdiff = np.abs(signed).astype(np.uint8)
        blurred_diff = cv2.GaussianBlur(absdiff, (5, 5), 0)
        _, diff_mask = cv2.threshold(
            blurred_diff,
            self._diff_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        low_saturation = hsv[:, :, 1] <= self._max_saturation
        bright_or_dark = (
            signed >= self._brightening_threshold
        ) | (signed <= -self._darkening_threshold)
        mask = np.where((diff_mask > 0) & low_saturation & bright_or_dark, 255, 0).astype(
            np.uint8,
        )
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        changed_fraction = float(np.count_nonzero(mask)) / float(mask.size)
        if changed_fraction > self._max_change_fraction:
            self.reset_tracks()
            self._update_reference(gray, mask=None)
            return PlumeResult(
                ready=True,
                changed_fraction=changed_fraction,
                suppression_reason="global_change",
            )

        regions, suppressed_count = self._extract_regions(
            mask,
            absdiff,
            signed,
            gray,
            hsv,
            frame.shape,
            scale_x,
            scale_y,
        )
        active_tracks = self._update_tracks(regions, ts)
        candidates = [
            track for track in active_tracks
            if track.persistence_seconds >= self._min_persistence_seconds
            and track.seen_frames >= self._min_persistence_frames
            and (
                track.growth_area_px >= self._min_growth_area_px
                or track.area_px >= self._min_area_px * 4
            )
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[: self._max_candidates_per_frame]

        self._update_reference(gray, mask=mask)
        score = max((candidate.score for candidate in candidates), default=0.0)
        anomaly_map = self._build_anomaly_map(mask, frame.shape, score) if candidates else None
        suppression_reason = None
        if not candidates:
            if suppressed_count and not active_tracks:
                suppression_reason = "solid_edge_suppressed"
            elif not active_tracks:
                suppression_reason = "no_visible_plume_change"
            elif any(track.growth_area_px < self._min_growth_area_px for track in active_tracks):
                suppression_reason = "growth_not_met"
            else:
                suppression_reason = "persistence_not_met"

        return PlumeResult(
            ready=True,
            has_detection=bool(candidates),
            score=score,
            anomaly_map=anomaly_map,
            candidates=candidates,
            active_tracks=active_tracks,
            changed_fraction=changed_fraction,
            suppression_reason=suppression_reason,
            suppressed_region_count=suppressed_count,
        )

    def _prepare(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        scale_x = bgr.shape[1] / max(w, 1)
        scale_y = bgr.shape[0] / max(h, 1)
        return gray, bgr, hsv, scale_x, scale_y

    def _extract_regions(
        self,
        mask: np.ndarray,
        absdiff: np.ndarray,
        signed: np.ndarray,
        gray: np.ndarray,
        hsv: np.ndarray,
        frame_shape: tuple[int, ...],
        scale_x: float,
        scale_y: float,
    ) -> tuple[list[_Region], int]:
        frame_h, frame_w = frame_shape[:2]
        inv_x = 1.0 / max(scale_x, 1e-6)
        inv_y = 1.0 / max(scale_y, 1e-6)
        max_area_px = self._max_area_fraction * frame_w * frame_h
        edges = cv2.Canny(gray, 50, 150)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        regions: list[_Region] = []
        suppressed_count = 0
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            full_area = int(round(area * inv_x * inv_y))
            if full_area < self._min_area_px or full_area > max_area_px:
                continue
            label_mask = labels == label
            bbox_area = max(1, int(w) * int(h))
            edge_density = float(np.count_nonzero(edges[label_mask])) / float(max(area, 1))
            fill_fraction = float(area) / float(bbox_area)
            if (
                edge_density > self._max_edge_density
                or fill_fraction > self._max_bbox_fill_fraction
            ):
                suppressed_count += 1
                continue
            mean_absdiff = float(absdiff[label_mask].mean())
            mean_signed_delta = float(signed[label_mask].mean())
            mean_saturation = float(hsv[:, :, 1][label_mask].mean())
            mean_value = float(hsv[:, :, 2][label_mask].mean())
            plume_type = self._classify_plume(mean_signed_delta, mean_value)
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
                mean_saturation=mean_saturation,
                mean_value=mean_value,
                edge_density=edge_density,
                fill_fraction=fill_fraction,
                plume_type=plume_type,
            ))
        return regions, suppressed_count

    def _update_tracks(self, regions: list[_Region], timestamp: float) -> list[PlumeCandidate]:
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
                    mean_saturation=region.mean_saturation,
                    mean_value=region.mean_value,
                    edge_density=region.edge_density,
                    fill_fraction=region.fill_fraction,
                    plume_type=region.plume_type,
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
            track.mean_saturation = region.mean_saturation
            track.mean_value = region.mean_value
            track.edge_density = region.edge_density
            track.fill_fraction = region.fill_fraction
            track.plume_type = region.plume_type
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
    ) -> PlumeCandidate:
        persistence = max(0.0, timestamp - track.first_seen_ts)
        growth_area = max(0, track.max_area_px - track.first_area_px)
        growth_score = min(1.0, growth_area / max(self._min_growth_area_px, 1))
        persistence_score = min(1.0, persistence / max(self._min_persistence_seconds, 1e-6))
        contrast_score = min(1.0, track.mean_absdiff / max(self._diff_threshold * 4, 1))
        diffuse_score = max(0.0, 1.0 - track.edge_density / max(self._max_edge_density, 1e-6))
        score = min(
            0.99,
            self._alert_score
            + 0.05 * growth_score
            + 0.04 * persistence_score
            + 0.03 * contrast_score
            + 0.02 * diffuse_score,
        )
        return PlumeCandidate(
            track_id=track.track_id,
            plume_type=track.plume_type,
            bbox=track.bbox,
            centroid=(track.centroid_x, track.centroid_y),
            area_px=track.area_px,
            growth_area_px=growth_area,
            mean_absdiff=track.mean_absdiff,
            mean_signed_delta=track.mean_signed_delta,
            mean_saturation=track.mean_saturation,
            mean_value=track.mean_value,
            edge_density=track.edge_density,
            fill_fraction=track.fill_fraction,
            first_seen_ts=track.first_seen_ts,
            last_seen_ts=track.last_seen_ts,
            seen_frames=track.seen_frames,
            persistence_seconds=persistence,
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

    @staticmethod
    def _build_anomaly_map(
        mask: np.ndarray,
        frame_shape: tuple[int, ...],
        score: float,
    ) -> np.ndarray:
        frame_h, frame_w = frame_shape[:2]
        small = mask.astype(np.float32) / 255.0 * float(score)
        return cv2.resize(small, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

    def _classify_plume(self, mean_signed_delta: float, mean_value: float) -> str:
        if mean_signed_delta >= self._brightening_threshold:
            return "steam_or_mist"
        if mean_signed_delta <= -self._darkening_threshold:
            return "smoke_or_dust"
        if mean_value < 90:
            return "smoke_or_dust"
        return "visible_plume"
