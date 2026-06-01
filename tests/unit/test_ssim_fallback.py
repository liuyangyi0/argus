from __future__ import annotations

import cv2
import numpy as np

from argus.anomaly.detector import AnomalibDetector


def _prime_ssim(detector: AnomalibDetector, frame: np.ndarray) -> None:
    for _ in range(detector._ssim_baseline_frames):
        result = detector.predict(frame)
        assert result.detection_failed is False
        assert result.is_anomalous is False


def _textured_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    frame = rng.integers(70, 180, (240, 320, 3), dtype=np.uint8)
    for x in range(20, 320, 40):
        cv2.line(frame, (x, 0), (x, 239), (215, 215, 215), 2)
    for y in range(30, 240, 50):
        cv2.line(frame, (0, y), (319, y), (45, 45, 45), 2)
    return frame


def test_ssim_fallback_suppresses_global_camera_shift() -> None:
    detector = AnomalibDetector(
        threshold=0.7,
        image_size=(128, 128),
        ssim_baseline_frames=5,
        ssim_global_change_suppress_fraction=0.06,
    )
    base = _textured_frame()
    _prime_ssim(detector, base)

    shifted = cv2.warpAffine(
        base,
        np.float64([[1.0, 0.0, 6.0], [0.0, 1.0, 0.0]]),
        (base.shape[1], base.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    result = detector.predict(shifted)

    assert result.raw_score is not None and result.raw_score > 0
    assert result.anomaly_map is not None
    assert float(np.mean(result.anomaly_map > 0.5)) >= 0.06
    assert result.anomaly_score == 0.0
    assert result.is_anomalous is False


def test_ssim_fallback_keeps_local_foreign_object_signal() -> None:
    detector = AnomalibDetector(
        threshold=0.7,
        image_size=(128, 128),
        ssim_baseline_frames=5,
        ssim_global_change_suppress_fraction=0.06,
    )
    base = _textured_frame()
    _prime_ssim(detector, base)

    changed = base.copy()
    cv2.rectangle(changed, (132, 96), (180, 128), (15, 15, 15), -1)
    cv2.rectangle(changed, (137, 101), (175, 123), (235, 235, 235), 1)
    result = detector.predict(changed)

    assert result.anomaly_map is not None
    assert float(np.mean(result.anomaly_map > 0.5)) < 0.12
    assert result.anomaly_score >= result.threshold
    assert result.is_anomalous is True


def test_ssim_fallback_suppresses_edge_reflection_band() -> None:
    detector = AnomalibDetector(
        threshold=0.7,
        image_size=(128, 128),
        ssim_baseline_frames=5,
        ssim_global_change_suppress_fraction=0.04,
    )
    base = _textured_frame()
    _prime_ssim(detector, base)

    changed = base.copy()
    band = changed[150:240, :]
    band[:] = cv2.addWeighted(band, 0.45, np.full_like(band, 240), 0.55, 0)
    cv2.rectangle(changed, (0, 185), (319, 239), (245, 245, 245), -1)
    result = detector.predict(changed)

    assert result.raw_score is not None and result.raw_score > 0
    assert result.anomaly_map is not None
    assert result.anomaly_score == 0.0
    assert result.is_anomalous is False
