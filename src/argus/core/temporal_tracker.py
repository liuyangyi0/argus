"""Temporal anomaly tracking across frames.

Tracks anomaly regions across consecutive frames to determine persistence
and movement patterns. Stationary anomalies that persist for many frames
receive a severity boost, enabling the alert grader to escalate persistent
foreign objects more aggressively.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class TrackedAnomaly:
    """An anomaly region tracked across multiple frames."""

    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    consecutive_frames: int  # continuous presence count
    centroid_x: float
    centroid_y: float
    max_score: float
    avg_score: float
    velocity: tuple[float, float]  # (dx, dy) pixels per frame
    is_stationary: bool  # velocity magnitude < threshold
    persistence_seconds: float
    trajectory_history: list[tuple[float, float, float]] = field(default_factory=list)
    # Each entry: (timestamp_seconds, centroid_x, centroid_y)
    area_px: int = 0  # Detected region area in pixels


@dataclass
class TemporalAnalysis:
    """Result of temporal analysis for a single frame."""

    active_tracks: list[TrackedAnomaly]
    new_tracks: int  # tracks started this frame
    lost_tracks: int  # tracks that disappeared
    severity_boost: float  # 0.0-0.3 boost based on persistence
    # Stationary anomaly for 5+ frames -> +0.1 boost
    # Stationary anomaly for 15+ frames -> +0.2 boost
    # Stationary anomaly for 30+ frames -> +0.3 boost (max)


@dataclass
class _TrackState:
    """Internal mutable state for a tracked anomaly."""

    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    consecutive_frames: int
    centroid_x: float
    centroid_y: float
    prev_centroid_x: float
    prev_centroid_y: float
    max_score: float
    score_sum: float
    score_count: int
    matched_this_frame: bool = False
    trajectory_history: list[tuple[float, float, float]] = field(default_factory=list)
    area_px: int = 0
    max_history_length: int = 300


class TemporalAnomalyTracker:
    """Track anomaly regions across frames for persistence analysis.

    Uses centroid-based matching: if a region's centroid is within
    match_distance pixels of a previous frame's region, they're the same track.
    """

    def __init__(
        self,
        match_distance: float = 50.0,
        max_gap_frames: int = 5,
        stationary_threshold: float = 10.0,
        fps: float = 5.0,
        trajectory_history_length: int = 300,
    ):
        self._match_distance = match_distance
        self._max_gap_frames = max_gap_frames
        self._stationary_threshold = stationary_threshold
        self._fps = fps
        self._trajectory_history_length = trajectory_history_length
        self._tracks: dict[int, _TrackState] = {}
        self._next_id = 1

    def update(self, regions: list, frame_number: int) -> TemporalAnalysis:
        """Update tracks with new frame's anomaly regions.

        regions: list of objects with centroid_x, centroid_y, max_score attributes

        Algorithm:
        1. For each region, find closest existing track within match_distance
        2. If match found: update track (centroid, score, consecutive_frames)
        3. If no match: create new track
        4. Tracks not matched for max_gap_frames: remove
        5. Compute velocity from centroid history
        6. Return TemporalAnalysis with severity boost
        """
        # Mark all tracks as unmatched
        for track in self._tracks.values():
            track.matched_this_frame = False

        new_tracks_count = 0

        for region in regions:
            cx = region.centroid_x
            cy = region.centroid_y
            score = region.max_score

            # Find closest unmatched track within match_distance
            best_track_id = None
            best_dist = self._match_distance

            for tid, track in self._tracks.items():
                if track.matched_this_frame:
                    continue
                dx = cx - track.centroid_x
                dy = cy - track.centroid_y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < best_dist:
                    best_dist = dist
                    best_track_id = tid

            now = time.time()
            region_area = getattr(region, "area_px", 0)

            if best_track_id is not None:
                # Update existing track
                track = self._tracks[best_track_id]
                track.prev_centroid_x = track.centroid_x
                track.prev_centroid_y = track.centroid_y
                track.centroid_x = cx
                track.centroid_y = cy
                track.last_seen_frame = frame_number
                track.consecutive_frames += 1
                track.max_score = max(track.max_score, score)
                track.score_sum += score
                track.score_count += 1
                track.matched_this_frame = True
                track.area_px = region_area
                # Append to trajectory history
                track.trajectory_history.append((now, cx, cy))
                if len(track.trajectory_history) > track.max_history_length:
                    track.trajectory_history = track.trajectory_history[
                        -track.max_history_length :
                    ]
            else:
                # Create new track
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _TrackState(
                    track_id=tid,
                    first_seen_frame=frame_number,
                    last_seen_frame=frame_number,
                    consecutive_frames=1,
                    centroid_x=cx,
                    centroid_y=cy,
                    prev_centroid_x=cx,
                    prev_centroid_y=cy,
                    max_score=score,
                    score_sum=score,
                    score_count=1,
                    matched_this_frame=True,
                    trajectory_history=[(now, cx, cy)],
                    area_px=region_area,
                    max_history_length=self._trajectory_history_length,
                )
                new_tracks_count += 1

        # Remove tracks not seen for max_gap_frames
        lost_ids = [
            tid
            for tid, track in self._tracks.items()
            if not track.matched_this_frame
            and (frame_number - track.last_seen_frame) > self._max_gap_frames
        ]
        lost_tracks_count = len(lost_ids)
        for tid in lost_ids:
            del self._tracks[tid]

        # Build active track list and compute severity boost
        active: list[TrackedAnomaly] = []
        max_boost = 0.0

        for track in self._tracks.values():
            dx = track.centroid_x - track.prev_centroid_x
            dy = track.centroid_y - track.prev_centroid_y
            velocity = (dx, dy)
            speed = math.sqrt(dx * dx + dy * dy)
            is_stationary = speed < self._stationary_threshold

            duration_frames = track.last_seen_frame - track.first_seen_frame + 1
            persistence_seconds = duration_frames / self._fps if self._fps > 0 else 0.0

            avg_score = track.score_sum / track.score_count if track.score_count > 0 else 0.0

            tracked = TrackedAnomaly(
                track_id=track.track_id,
                first_seen_frame=track.first_seen_frame,
                last_seen_frame=track.last_seen_frame,
                consecutive_frames=track.consecutive_frames,
                centroid_x=track.centroid_x,
                centroid_y=track.centroid_y,
                max_score=track.max_score,
                avg_score=avg_score,
                velocity=velocity,
                is_stationary=is_stationary,
                persistence_seconds=persistence_seconds,
                trajectory_history=list(track.trajectory_history),
                area_px=track.area_px,
            )
            active.append(tracked)

            boost = self.get_severity_boost(tracked)
            if boost > max_boost:
                max_boost = boost

        return TemporalAnalysis(
            active_tracks=active,
            new_tracks=new_tracks_count,
            lost_tracks=lost_tracks_count,
            severity_boost=max_boost,
        )

    def get_severity_boost(self, track: TrackedAnomaly) -> float:
        """Calculate severity boost based on track persistence.

        Only stationary anomalies receive a boost:
        - 5+ consecutive frames:  +0.1
        - 15+ consecutive frames: +0.2
        - 30+ consecutive frames: +0.3 (max)
        """
        if not track.is_stationary:
            return 0.0

        frames = track.consecutive_frames
        if frames >= 30:
            return 0.3
        if frames >= 15:
            return 0.2
        if frames >= 5:
            return 0.1
        return 0.0

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 1
