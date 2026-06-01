"""Unit tests for the fast-motion projectile detector."""

from __future__ import annotations

import numpy as np

from argus.prefilter.fast_motion import FastMotionDetector


def _blank_frame() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _gray_frame(value: int) -> np.ndarray:
    return np.full((1080, 1920, 3), value, dtype=np.uint8)


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
    detector.process(first, timestamp=1.0 + 1 / 60)

    second = _blank_frame()
    second[300:306, 640:646] = 255
    result = detector.process(second, timestamp=1.0 + 2 / 60)

    assert result.has_detection is True
    assert any(c.speed_px_per_sec is not None and c.speed_px_per_sec > 0 for c in result.candidates)


def test_single_frame_speed_uses_streak_and_fps_hint():
    detector = FastMotionDetector(process_width=960, diff_threshold=18, fps_hint=60)
    detector.process(_blank_frame(), timestamp=1.0)

    frame = _blank_frame()
    frame[400:405, 700:720] = 255
    result = detector.process(frame, timestamp=1.0 + 1 / 60)

    assert result.has_detection is True
    assert result.candidates[0].speed_px_per_sec is not None
    assert result.candidates[0].speed_px_per_sec > 0
