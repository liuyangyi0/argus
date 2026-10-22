"""Tests for stable model-output evidence exposed by the anomaly detector."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from argus.anomaly.detector import AnomalibDetector


def test_anomalib_preserves_pre_calibration_score_and_pre_minmax_map() -> None:
    detector = AnomalibDetector(threshold=0.7, image_size=(2, 2))
    inferencer_map = np.array([[10.0, 20.0], [15.0, 30.0]], dtype=np.float32)
    detector._engine = SimpleNamespace(
        predict=lambda _rgb: SimpleNamespace(
            pred_score=np.array([0.8], dtype=np.float32),
            anomaly_map=inferencer_map,
        )
    )
    detector._loaded = True
    detector._calibration_scores = np.array([0.1, 0.2, 0.3, 0.9])
    detector._calibration_n = 4

    result = detector._predict_anomalib(np.zeros((4, 4, 3), dtype=np.uint8))

    assert result.detection_failed is False
    assert result.model_score == pytest.approx(0.8)
    assert result.raw_score == pytest.approx(0.8)
    assert result.anomaly_score == pytest.approx(0.75)
    assert result.model_output_space == "anomalib_postprocessed"
    assert result.model_output_comparable is True
    np.testing.assert_array_equal(result.model_anomaly_map, inferencer_map)
    assert result.anomaly_map is not None
    assert float(result.anomaly_map.min()) == pytest.approx(0.0)
    assert float(result.anomaly_map.max()) == pytest.approx(1.0)


def test_nonfinite_model_map_is_exposed_but_never_marked_comparable() -> None:
    detector = AnomalibDetector(threshold=0.7, image_size=(2, 2))
    inferencer_map = np.array([[0.0, np.nan], [np.inf, 2.0]], dtype=np.float32)
    detector._engine = SimpleNamespace(
        predict=lambda _rgb: SimpleNamespace(
            pred_score=np.array([0.8], dtype=np.float32),
            anomaly_map=inferencer_map,
        )
    )
    detector._loaded = True

    result = detector._predict_anomalib(np.zeros((4, 4, 3), dtype=np.uint8))

    assert result.model_output_space == "anomalib_postprocessed"
    assert result.model_output_comparable is False
    assert result.model_anomaly_map is not None
    assert not np.all(np.isfinite(result.model_anomaly_map))
    assert result.anomaly_map is not None
    assert np.all(np.isfinite(result.anomaly_map))


def test_minmax_broken_inner_output_is_not_claimed_as_comparable() -> None:
    torch = pytest.importorskip("torch")
    preprocessed = False

    def pre_processor(tensor):
        nonlocal preprocessed
        preprocessed = True
        return tensor

    class FakeInner:
        def __call__(self, _tensor):
            return SimpleNamespace(
                pred_score=torch.tensor([60.0]),
                anomaly_map=torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]]),
            )

    detector = AnomalibDetector(threshold=0.5, image_size=(2, 2))
    detector._engine = SimpleNamespace(
        model=SimpleNamespace(model=FakeInner(), pre_processor=pre_processor)
    )
    detector._loaded = True
    detector._minmax_broken = True

    result = detector._predict_anomalib(np.zeros((4, 4, 3), dtype=np.uint8))

    assert result.detection_failed is False
    assert preprocessed is True
    assert result.model_score == pytest.approx(60.0)
    assert result.model_output_space == "anomalib_inner_raw"
    assert result.model_output_comparable is False
    np.testing.assert_array_equal(
        result.model_anomaly_map,
        np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
    )
    assert result.anomaly_map is not None
    assert float(result.anomaly_map.min()) == pytest.approx(0.0)
    assert float(result.anomaly_map.max()) == pytest.approx(1.0)


def test_minmax_check_uses_persisted_buffers_not_reset_metrics() -> None:
    post_processor = SimpleNamespace(
        enable_normalization=True,
        image_min=np.array(0.1),
        image_max=np.array(0.9),
        pixel_min=np.array(0.2),
        pixel_max=np.array(1.2),
        _image_min_max_metric=SimpleNamespace(min=np.inf, max=-np.inf),
    )
    detector = AnomalibDetector()
    detector._engine = SimpleNamespace(
        model=SimpleNamespace(post_processor=post_processor)
    )
    detector._minmax_broken = True

    detector._check_minmax_normalization()

    assert detector._minmax_broken is False
    assert detector._model_output_space_hint == "anomalib_postprocessed"


def test_minmax_check_flags_invalid_persisted_pixel_buffers() -> None:
    post_processor = SimpleNamespace(
        enable_normalization=True,
        image_min=np.array(0.1),
        image_max=np.array(0.9),
        pixel_min=np.array(np.nan),
        pixel_max=np.array(np.nan),
    )
    detector = AnomalibDetector()
    detector._engine = SimpleNamespace(
        model=SimpleNamespace(post_processor=post_processor)
    )

    detector._check_minmax_normalization()

    assert detector._minmax_broken is True


def test_disabled_anomalib_normalization_has_separate_comparable_space() -> None:
    detector = AnomalibDetector()
    detector._engine = SimpleNamespace(
        model=SimpleNamespace(
            post_processor=SimpleNamespace(enable_normalization=False)
        )
    )

    detector._check_minmax_normalization()

    assert detector._minmax_broken is False
    assert detector._model_output_space_hint == "anomalib_unscaled"
