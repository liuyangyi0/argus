"""Tests for anomaly detection model ensemble (ANO-005)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argus.anomaly.detector import AnomalyResult
from argus.anomaly.ensemble import DetectorEnsemble, EnsembleConfig, EnsembleResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    score: float,
    threshold: float = 0.7,
    anomaly_map: np.ndarray | None = None,
) -> AnomalyResult:
    """Create an AnomalyResult with sensible defaults."""
    return AnomalyResult(
        anomaly_score=score,
        anomaly_map=anomaly_map,
        is_anomalous=score >= threshold,
        threshold=threshold,
    )


def _make_mock_detector(result: AnomalyResult, load_success: bool = True) -> MagicMock:
    """Create a mock AnomalibDetector that returns a predetermined result."""
    mock = MagicMock()
    mock.predict.return_value = result
    mock.load.return_value = load_success
    mock.is_loaded = load_success
    return mock


def _build_ensemble(
    results: list[AnomalyResult],
    method: str = "mean",
    weights: list[float] | None = None,
    threshold: float = 0.7,
) -> DetectorEnsemble:
    """Build an ensemble pre-loaded with mock detectors."""
    config = EnsembleConfig(
        model_paths=[f"model_{i}.pt" for i in range(len(results))],
        method=method,
        weights=weights,
        threshold=threshold,
    )
    ensemble = DetectorEnsemble(config)
    # Inject mock detectors directly
    ensemble._detectors = [_make_mock_detector(r) for r in results]
    ensemble._loaded = len(results) > 0
    return ensemble


DUMMY_FRAME = np.zeros((256, 256, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. Ensemble with 0 models → always returns normal
# ---------------------------------------------------------------------------

class TestZeroModels:
    def test_no_models_returns_normal(self):
        config = EnsembleConfig(model_paths=[], method="mean")
        ensemble = DetectorEnsemble(config)
        # Don't load — no paths
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_score == 0.0
        assert result.is_anomalous is False
        assert result.individual_scores == []
        assert result.agreement_ratio == 1.0

    def test_no_models_not_loaded(self):
        config = EnsembleConfig(model_paths=[], method="mean")
        ensemble = DetectorEnsemble(config)
        assert ensemble.is_loaded is False
        assert ensemble.model_count == 0


# ---------------------------------------------------------------------------
# 2. Ensemble with 1 model → acts as single detector
# ---------------------------------------------------------------------------

class TestSingleModel:
    def test_single_model_passthrough(self):
        r = _make_result(0.8)
        ensemble = _build_ensemble([r], method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_score == pytest.approx(0.8)
        assert result.is_anomalous is True
        assert len(result.individual_scores) == 1
        assert result.agreement_ratio == 1.0

    def test_single_model_normal(self):
        r = _make_result(0.3)
        ensemble = _build_ensemble([r], method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_score == pytest.approx(0.3)
        assert result.is_anomalous is False


# ---------------------------------------------------------------------------
# 3. Mean fusion: [0.3, 0.7, 0.5] → 0.5
# ---------------------------------------------------------------------------

class TestMeanFusion:
    def test_mean_three_models(self):
        results = [_make_result(0.3), _make_result(0.7), _make_result(0.5)]
        ensemble = _build_ensemble(results, method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_score == pytest.approx(0.5, abs=1e-6)
        assert result.method == "mean"

    def test_mean_below_threshold(self):
        results = [_make_result(0.2), _make_result(0.3), _make_result(0.1)]
        ensemble = _build_ensemble(results, method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_score == pytest.approx(0.2, abs=1e-6)
        assert result.is_anomalous is False


# ---------------------------------------------------------------------------
# 4. Max fusion: [0.3, 0.7, 0.5] → 0.7
# ---------------------------------------------------------------------------

class TestMaxFusion:
    def test_max_three_models(self):
        results = [_make_result(0.3), _make_result(0.7), _make_result(0.5)]
        ensemble = _build_ensemble(results, method="max")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_score == pytest.approx(0.7)
        assert result.method == "max"

    def test_max_picks_highest(self):
        results = [_make_result(0.1), _make_result(0.9)]
        ensemble = _build_ensemble(results, method="max")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_score == pytest.approx(0.9)
        assert result.is_anomalous is True


# ---------------------------------------------------------------------------
# 5. Weighted fusion: [0.3, 0.7] weights [0.8, 0.2] → 0.38
# ---------------------------------------------------------------------------

class TestWeightedFusion:
    def test_weighted_two_models(self):
        results = [_make_result(0.3), _make_result(0.7)]
        ensemble = _build_ensemble(results, method="weighted", weights=[0.8, 0.2])
        result = ensemble.predict(DUMMY_FRAME)

        # (0.3*0.8 + 0.7*0.2) / (0.8+0.2) = (0.24 + 0.14) / 1.0 = 0.38
        assert result.anomaly_score == pytest.approx(0.38, abs=1e-6)
        assert result.is_anomalous is False

    def test_weighted_falls_back_to_mean_on_mismatch(self):
        results = [_make_result(0.3), _make_result(0.7), _make_result(0.5)]
        # Only 2 weights for 3 models → fallback to mean
        ensemble = _build_ensemble(results, method="weighted", weights=[0.5, 0.5])
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_score == pytest.approx(0.5, abs=1e-6)

    def test_weighted_none_weights_falls_back(self):
        results = [_make_result(0.4), _make_result(0.6)]
        ensemble = _build_ensemble(results, method="weighted", weights=None)
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_score == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Vote fusion: 2/3 anomalous → anomalous, score = mean of anomalous
# ---------------------------------------------------------------------------

class TestVoteFusion:
    def test_vote_majority_anomalous(self):
        # threshold=0.7: scores 0.8 and 0.9 are anomalous, 0.3 is not
        results = [_make_result(0.8), _make_result(0.9), _make_result(0.3)]
        ensemble = _build_ensemble(results, method="vote")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.is_anomalous is True
        # Score = mean of anomalous models: (0.8 + 0.9) / 2 = 0.85
        assert result.anomaly_score == pytest.approx(0.85, abs=1e-6)

    # -------------------------------------------------------------------
    # 7. Vote fusion: 1/3 anomalous → normal
    # -------------------------------------------------------------------
    def test_vote_minority_anomalous(self):
        results = [_make_result(0.8), _make_result(0.3), _make_result(0.2)]
        ensemble = _build_ensemble(results, method="vote")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.is_anomalous is False

    def test_vote_no_anomalous_uses_mean(self):
        results = [_make_result(0.1), _make_result(0.2), _make_result(0.3)]
        ensemble = _build_ensemble(results, method="vote")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.is_anomalous is False
        assert result.anomaly_score == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. Agreement ratio
# ---------------------------------------------------------------------------

class TestAgreementRatio:
    def test_all_agree_anomalous(self):
        results = [_make_result(0.8), _make_result(0.9), _make_result(0.75)]
        ensemble = _build_ensemble(results, method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.agreement_ratio == pytest.approx(1.0)

    def test_all_agree_normal(self):
        results = [_make_result(0.1), _make_result(0.2), _make_result(0.3)]
        ensemble = _build_ensemble(results, method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.agreement_ratio == pytest.approx(1.0)

    def test_split_two_thirds(self):
        # 2 anomalous, 1 normal → agreement = 2/3
        results = [_make_result(0.8), _make_result(0.9), _make_result(0.3)]
        ensemble = _build_ensemble(results, method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.agreement_ratio == pytest.approx(2.0 / 3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 9. Heatmap combination averages correctly
# ---------------------------------------------------------------------------

class TestHeatmapCombination:
    def test_heatmap_average(self):
        map1 = np.full((256, 256), 0.2, dtype=np.float32)
        map2 = np.full((256, 256), 0.8, dtype=np.float32)
        results = [
            _make_result(0.5, anomaly_map=map1),
            _make_result(0.5, anomaly_map=map2),
        ]
        ensemble = _build_ensemble(results, method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_map is not None
        assert result.anomaly_map.shape == (256, 256)
        assert np.allclose(result.anomaly_map, 0.5, atol=1e-5)

    def test_heatmap_with_different_sizes(self):
        map1 = np.full((128, 128), 0.4, dtype=np.float32)
        map2 = np.full((64, 64), 0.6, dtype=np.float32)
        results = [
            _make_result(0.5, anomaly_map=map1),
            _make_result(0.5, anomaly_map=map2),
        ]
        ensemble = _build_ensemble(results, method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_map is not None
        # Should be resized to config image_size (256, 256)
        assert result.anomaly_map.shape == (256, 256)
        assert np.allclose(result.anomaly_map, 0.5, atol=1e-5)

    def test_heatmap_none_when_no_maps(self):
        results = [_make_result(0.5), _make_result(0.6)]
        ensemble = _build_ensemble(results, method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_map is None

    def test_heatmap_partial_maps(self):
        """Only one model has a heatmap — should still produce output."""
        hmap = np.full((256, 256), 0.7, dtype=np.float32)
        results = [
            _make_result(0.5, anomaly_map=hmap),
            _make_result(0.5),  # no map
        ]
        ensemble = _build_ensemble(results, method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.anomaly_map is not None
        assert np.allclose(result.anomaly_map, 0.7, atol=1e-5)


# ---------------------------------------------------------------------------
# 10. individual_scores populated correctly
# ---------------------------------------------------------------------------

class TestIndividualScores:
    def test_individual_scores_match(self):
        results = [_make_result(0.1), _make_result(0.5), _make_result(0.9)]
        ensemble = _build_ensemble(results, method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.individual_scores == pytest.approx([0.1, 0.5, 0.9])

    def test_individual_anomalous_flags(self):
        results = [_make_result(0.3), _make_result(0.8)]
        ensemble = _build_ensemble(results, method="mean")
        result = ensemble.predict(DUMMY_FRAME)

        assert result.individual_anomalous == [False, True]


# ---------------------------------------------------------------------------
# 11. EnsembleConfig defaults
# ---------------------------------------------------------------------------

class TestEnsembleConfigDefaults:
    def test_defaults(self):
        config = EnsembleConfig(model_paths=["a.pt", "b.pt"])
        assert config.method == "mean"
        assert config.weights is None
        assert config.image_size == (256, 256)
        assert config.threshold == 0.7

    def test_custom_values(self):
        config = EnsembleConfig(
            model_paths=["x.pt"],
            method="vote",
            weights=[1.0],
            image_size=(512, 512),
            threshold=0.5,
        )
        assert config.method == "vote"
        assert config.weights == [1.0]
        assert config.image_size == (512, 512)
        assert config.threshold == 0.5


# ---------------------------------------------------------------------------
# 12. get_status returns correct info
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_status_before_load(self):
        config = EnsembleConfig(model_paths=["a.pt", "b.pt"], method="max")
        ensemble = DetectorEnsemble(config)
        status = ensemble.get_status()

        assert status["model_count"] == 0
        assert status["loaded"] is False
        assert status["method"] == "max"
        assert status["model_paths"] == ["a.pt", "b.pt"]

    def test_status_after_load(self):
        results = [_make_result(0.5), _make_result(0.6)]
        ensemble = _build_ensemble(results, method="weighted")
        status = ensemble.get_status()

        assert status["model_count"] == 2
        assert status["loaded"] is True
        assert status["method"] == "weighted"


# ---------------------------------------------------------------------------
# 13. Graceful handling when model path doesn't exist
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    @patch("argus.anomaly.ensemble.AnomalibDetector")
    def test_load_with_invalid_paths(self, mock_detector_cls):
        """Models that fail to load are still kept (SSIM fallback)."""
        mock_instance = MagicMock()
        mock_instance.load.return_value = False
        mock_instance.predict.return_value = _make_result(0.0)
        mock_detector_cls.return_value = mock_instance

        config = EnsembleConfig(
            model_paths=["nonexistent1.pt", "nonexistent2.pt"],
        )
        ensemble = DetectorEnsemble(config)
        loaded = ensemble.load()

        assert loaded is False
        assert ensemble.model_count == 2  # detectors created even if load failed

    @patch("argus.anomaly.ensemble.AnomalibDetector")
    def test_partial_load(self, mock_detector_cls):
        """If one model loads and one fails, ensemble still works."""
        good = MagicMock()
        good.load.return_value = True
        good.predict.return_value = _make_result(0.6)

        bad = MagicMock()
        bad.load.return_value = False
        bad.predict.return_value = _make_result(0.0)

        mock_detector_cls.side_effect = [good, bad]

        config = EnsembleConfig(model_paths=["good.pt", "bad.pt"])
        ensemble = DetectorEnsemble(config)
        loaded = ensemble.load()

        assert loaded is True
        assert ensemble.model_count == 2

        result = ensemble.predict(DUMMY_FRAME)
        assert result.anomaly_score == pytest.approx(0.3)  # mean of 0.6 and 0.0


# ---------------------------------------------------------------------------
# 14. EnsembleResult dataclass
# ---------------------------------------------------------------------------

class TestEnsembleResult:
    def test_result_fields(self):
        r = EnsembleResult(
            anomaly_score=0.5,
            anomaly_map=None,
            is_anomalous=False,
            threshold=0.7,
            individual_scores=[0.3, 0.7],
            individual_anomalous=[False, True],
            agreement_ratio=0.5,
            method="mean",
        )
        assert r.anomaly_score == 0.5
        assert r.method == "mean"
        assert len(r.individual_scores) == 2


# ---------------------------------------------------------------------------
# 15. Thread safety — concurrent predict calls
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_predict(self):
        """Ensure predict can be called from multiple threads without error."""
        import concurrent.futures

        results = [_make_result(0.4), _make_result(0.6)]
        ensemble = _build_ensemble(results, method="mean")

        def run_predict():
            return ensemble.predict(DUMMY_FRAME)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_predict) for _ in range(20)]
            for f in futures:
                r = f.result()
                assert r.anomaly_score == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 16. Unknown method falls back to mean
# ---------------------------------------------------------------------------

class TestUnknownMethod:
    def test_unknown_method_uses_mean(self):
        results = [_make_result(0.3), _make_result(0.7)]
        ensemble = _build_ensemble(results, method="unknown_method")
        result = ensemble.predict(DUMMY_FRAME)

        # Falls back to mean
        assert result.anomaly_score == pytest.approx(0.5, abs=1e-6)
