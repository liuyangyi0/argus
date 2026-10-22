"""Unit tests for the fast-motion projectile detector."""

from __future__ import annotations

import numpy as np

from argus.prefilter.fast_motion import FastMotionCandidate, FastMotionDetector


def _blank_frame() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _gray_frame(value: int) -> np.ndarray:
    return np.full((1080, 1920, 3), value, dtype=np.uint8)


def _candidate_at(x: float, y: float = 300.0) -> FastMotionCandidate:
    return FastMotionCandidate(
        bbox=(int(x) - 2, int(y) - 2, int(x) + 2, int(y) + 2),
        centroid=(x, y),
        area_px=16,
        streak_length_px=4.0,
        confidence=0.9,
    )


def test_tiny_1080p_dot_generates_fast_projectile_candidate():
    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        min_area_px=2,
        max_area_px=1500,
        min_streak_length_px=4,
        min_confidence=0.55,
    )
    assert detector.process(_blank_frame(), timestamp=1.0).has_detection is False

    frame = _blank_frame()
    frame[300:305, 600:605] = 255

    result = detector.process(frame, timestamp=1.0 + 1 / 60)

    assert result.has_detection is True
    assert result.max_confidence >= 0.55
    assert result.anomaly_map is not None
    candidate = result.candidates[0]
    assert candidate.to_detected_object()["class_name"] == "fast_projectile"
    x1, y1, x2, y2 = candidate.bbox
    assert x1 <= 605 and x2 >= 600
    assert y1 <= 305 and y2 >= 300
    assert candidate.trajectory


def test_low_amplitude_noise_does_not_trigger():
    rng = np.random.default_rng(7)
    detector = FastMotionDetector(process_width=960, diff_threshold=18)
    detector.process(_blank_frame(), timestamp=1.0)

    noise = rng.integers(0, 8, size=(1080, 1920, 3), dtype=np.uint8)
    result = detector.process(noise, timestamp=1.0 + 1 / 60)

    assert result.has_detection is False
    assert result.candidates == []


def test_global_brightness_shift_below_threshold_does_not_trigger():
    detector = FastMotionDetector(process_width=960, diff_threshold=18)
    detector.process(_gray_frame(70), timestamp=1.0)

    result = detector.process(_gray_frame(84), timestamp=1.0 + 1 / 60)

    assert result.has_detection is False
    assert result.candidates == []


def test_smooth_background_micro_jitter_does_not_trigger():
    detector = FastMotionDetector(process_width=960, diff_threshold=18)
    gradient = np.linspace(40, 160, 1920, dtype=np.uint8)
    base = np.repeat(gradient[None, :], 1080, axis=0)
    frame = np.dstack([base, base, base])
    detector.process(frame, timestamp=1.0)

    shifted = frame.copy()
    shifted[:, 1:] = frame[:, :-1]
    shifted[:, 0] = frame[:, 0]
    result = detector.process(shifted, timestamp=1.0 + 1 / 60)

    assert result.has_detection is False
    assert result.candidates == []


def test_jpeg_like_low_amplitude_block_noise_does_not_trigger():
    detector = FastMotionDetector(process_width=960, diff_threshold=18)
    detector.process(_gray_frame(96), timestamp=1.0)

    noisy = _gray_frame(96)
    noisy[::16, :] = 104
    noisy[:, ::16] = 88
    result = detector.process(noisy, timestamp=1.0 + 1 / 60)

    assert result.has_detection is False
    assert result.candidates == []


def test_subtle_localized_reflection_flicker_does_not_trigger():
    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        min_area_px=2,
        min_streak_length_px=4,
        min_confidence=0.60,
    )
    detector.process(_gray_frame(130), timestamp=1.0)

    frame = _gray_frame(130)
    frame[612:624, 1590:1612] = 158
    result = detector.process(frame, timestamp=1.0 + 1 / 60)

    assert result.has_detection is False
    assert result.candidates == []


def test_short_bright_streak_triggers_fast_projectile():
    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        min_area_px=2,
        min_streak_length_px=4,
        min_confidence=0.55,
        fps_hint=60,
    )
    detector.process(_blank_frame(), timestamp=1.0)

    frame = _blank_frame()
    frame[500:503, 700:730] = 255
    result = detector.process(frame, timestamp=1.0 + 1 / 60)

    assert result.has_detection is True
    assert result.candidates[0].to_detected_object()["class_name"] == "fast_projectile"
    assert result.candidates[0].streak_length_px >= 4


def test_fragmented_streak_is_grouped_as_one_projectile_candidate():
    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        min_area_px=8,
        min_streak_length_px=8,
        min_confidence=0.55,
        streak_gap_px=5,
    )
    detector.process(_blank_frame(), timestamp=1.0)

    frame = _blank_frame()
    frame[500:504, 700:704] = 255
    frame[500:504, 710:714] = 255
    frame[500:504, 720:724] = 255
    result = detector.process(frame, timestamp=1.0 + 1 / 60)

    assert result.has_detection is True
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    detected = candidate.to_detected_object()
    assert detected["class_name"] == "fast_projectile"
    assert detected["fragment_count"] == 3
    assert candidate.streak_length_px >= 20
    assert candidate.mean_motion > 0.9


def test_fragmented_streak_respects_gap_limit():
    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        min_area_px=8,
        min_streak_length_px=8,
        min_confidence=0.55,
        streak_gap_px=2,
    )
    detector.process(_blank_frame(), timestamp=1.0)

    frame = _blank_frame()
    frame[500:504, 700:704] = 255
    frame[500:504, 710:714] = 255
    frame[500:504, 720:724] = 255
    result = detector.process(frame, timestamp=1.0 + 1 / 60)

    assert result.has_detection is False
    assert result.candidates == []


def test_broad_scene_change_is_left_to_anomaly_path():
    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        min_area_px=2,
        min_streak_length_px=4,
        min_confidence=0.55,
        max_motion_fraction=0.015,
    )
    detector.process(_gray_frame(96), timestamp=1.0)

    frame = _gray_frame(96)
    frame[420:620, 760:1080] = 170
    result = detector.process(frame, timestamp=1.0 + 1 / 60)

    assert result.has_detection is False
    assert result.candidates == []
    assert result.suppression_reason == "global_motion"
    assert result.motion_fraction > 0.015


def test_long_structural_edge_is_left_to_anomaly_path():
    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        min_area_px=2,
        min_streak_length_px=4,
        min_confidence=0.55,
        max_streak_frame_fraction=0.25,
    )
    detector.process(_blank_frame(), timestamp=1.0)

    frame = _blank_frame()
    frame[420:423, 100:720] = 255
    result = detector.process(frame, timestamp=1.0 + 1 / 60)

    assert result.has_detection is False
    assert result.candidates == []


def test_dark_small_object_against_bright_background_triggers():
    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        min_area_px=2,
        min_streak_length_px=4,
        min_confidence=0.55,
    )
    detector.process(_gray_frame(180), timestamp=1.0)

    frame = _gray_frame(180)
    frame[440:446, 900:906] = 0
    result = detector.process(frame, timestamp=1.0 + 1 / 60)

    assert result.has_detection is True
    candidate = result.candidates[0]
    assert candidate.to_detected_object()["class_name"] == "fast_projectile"
    x1, y1, x2, y2 = candidate.bbox
    assert x1 <= 906 and x2 >= 900
    assert y1 <= 446 and y2 >= 440


def test_candidate_count_is_capped():
    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        min_area_px=2,
        max_candidates_per_frame=3,
    )
    detector.process(_blank_frame(), timestamp=1.0)

    frame = _blank_frame()
    for idx in range(8):
        x = 100 + idx * 140
        frame[200:206, x:x + 6] = 255

    result = detector.process(frame, timestamp=1.0 + 1 / 60)

    assert result.has_detection is True
    assert len(result.candidates) == 3


def test_speed_is_estimated_for_matched_candidates():
    detector = FastMotionDetector(process_width=960, diff_threshold=18)
    detector.process(_blank_frame(), timestamp=1.0)

    first = _blank_frame()
    first[300:306, 600:606] = 255
    first_result = detector.process(first, timestamp=1.0 + 1 / 60)
    first_track_id = first_result.candidates[0].track_id

    second = _blank_frame()
    second[300:306, 640:646] = 255
    result = detector.process(second, timestamp=1.0 + 2 / 60)

    assert result.has_detection is True
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.bbox[0] >= 638
    assert candidate.track_id == first_track_id
    assert candidate.speed_px_per_sec is not None
    assert candidate.speed_px_per_sec > 0


def test_adjacent_projectiles_keep_distinct_stable_tracks_across_frames():
    detector = FastMotionDetector(process_width=960, diff_threshold=18)

    first_frame = [_candidate_at(100.0), _candidate_at(140.0)]
    detector._assign_tracks(first_frame, timestamp=1.0)
    first_ids = [candidate.track_id for candidate in first_frame]
    assert first_ids == [1, 2]

    detector._prev_candidates = first_frame
    detector._prev_timestamp = 1.0
    second_frame = [_candidate_at(105.0), _candidate_at(118.0)]
    detector._assign_tracks(second_frame, timestamp=1.0 + 1 / 60)
    second_ids = [candidate.track_id for candidate in second_frame]

    # Both current candidates are closest to the first previous candidate.
    # The previous track must still be consumed by only one current candidate.
    assert second_ids == first_ids
    assert len(set(second_ids)) == 2

    detector._prev_candidates = second_frame
    detector._prev_timestamp = 1.0 + 1 / 60
    third_frame = [_candidate_at(110.0), _candidate_at(123.0)]
    detector._assign_tracks(third_frame, timestamp=1.0 + 2 / 60)

    assert [candidate.track_id for candidate in third_frame] == first_ids
    assert all(len(candidate.trajectory) == 3 for candidate in third_frame)


def test_many_short_lived_tracks_do_not_leave_permanent_histories():
    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        trajectory_history_length=8,
    )

    seen_track_ids: set[int] = set()
    for frame_index in range(250):
        # Each candidate is farther than the one-frame association radius, so
        # this models a long-running stream of unrelated one-frame projectiles.
        candidate = _candidate_at(float(frame_index * 1000))
        timestamp = 1.0 + frame_index / 60
        detector._assign_tracks([candidate], timestamp=timestamp)
        assert candidate.track_id is not None
        seen_track_ids.add(candidate.track_id)

        assert set(detector._track_histories) == {candidate.track_id}
        assert len(detector._track_histories[candidate.track_id]) == 1

        detector._prev_candidates = [candidate]
        detector._prev_timestamp = timestamp

    assert len(seen_track_ids) == 250
    assert len(detector._track_histories) == 1

    # Once the current frame has no candidate, the previous track cannot be
    # re-associated by this detector and its history is released immediately.
    detector._assign_tracks([], timestamp=6.0)
    assert detector._track_histories == {}


def test_history_cleanup_preserves_active_track_continuity():
    detector = FastMotionDetector(
        process_width=960,
        diff_threshold=18,
        trajectory_history_length=8,
    )

    first = _candidate_at(100.0)
    detector._assign_tracks([first], timestamp=1.0)
    active_track_id = first.track_id
    detector._prev_candidates = [first]
    detector._prev_timestamp = 1.0

    second = _candidate_at(110.0)
    transient = _candidate_at(1000.0)
    detector._assign_tracks([second, transient], timestamp=1.0 + 1 / 60)
    transient_track_id = transient.track_id
    assert second.track_id == active_track_id
    assert transient_track_id != active_track_id
    assert len(second.trajectory) == 2
    assert set(detector._track_histories) == {
        active_track_id,
        transient_track_id,
    }
    detector._prev_candidates = [second, transient]
    detector._prev_timestamp = 1.0 + 1 / 60

    third = _candidate_at(120.0)
    detector._assign_tracks([third], timestamp=1.0 + 2 / 60)

    assert third.track_id == active_track_id
    assert len(third.trajectory) == 3
    assert [point[1] for point in third.trajectory] == [100.0, 110.0, 120.0]
    assert set(detector._track_histories) == {active_track_id}


def test_single_frame_speed_uses_streak_and_fps_hint():
    detector = FastMotionDetector(process_width=960, diff_threshold=18, fps_hint=60)
    detector.process(_blank_frame(), timestamp=1.0)

    frame = _blank_frame()
    frame[400:405, 700:720] = 255
    result = detector.process(frame, timestamp=1.0 + 1 / 60)

    assert result.has_detection is True
    assert result.candidates[0].speed_px_per_sec is not None
    assert result.candidates[0].speed_px_per_sec > 0
