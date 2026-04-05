"""Tests for conformal prediction score calibration."""

import json

import numpy as np
import pytest

from argus.alerts.calibration import CalibrationResult, ConformalCalibrator


class TestConformalCalibrator:

    def test_calibrate_with_normal_distribution(self):
        """正态分布 scores → 阈值递增。"""
        rng = np.random.default_rng(42)
        scores = rng.normal(loc=0.3, scale=0.1, size=1000).clip(0, 1)
        calibrator = ConformalCalibrator()
        result = calibrator.calibrate(scores)

        assert result.info_threshold < result.low_threshold
        assert result.low_threshold < result.medium_threshold
        assert result.medium_threshold < result.high_threshold
        assert result.n_calibration_samples == 1000

    def test_calibrate_minimum_samples(self):
        """< 50 samples → ValueError。"""
        scores = np.array([0.1] * 30)
        calibrator = ConformalCalibrator()
        with pytest.raises(ValueError, match="Need >= 50"):
            calibrator.calibrate(scores)

    def test_threshold_ordering_guaranteed(self):
        """任何 score 分布 → info <= low <= medium <= high。"""
        rng = np.random.default_rng(123)
        # Pathological distribution: all same value
        scores = np.full(100, 0.5)
        calibrator = ConformalCalibrator()
        result = calibrator.calibrate(scores)

        assert result.info_threshold <= result.low_threshold
        assert result.low_threshold <= result.medium_threshold
        assert result.medium_threshold <= result.high_threshold

    def test_save_and_load_roundtrip(self, tmp_path):
        """save → load → 阈值一致。"""
        calibrator = ConformalCalibrator()
        rng = np.random.default_rng(42)
        scores = rng.normal(loc=0.3, scale=0.1, size=200).clip(0, 1)
        result = calibrator.calibrate(scores)

        path = tmp_path / "calibration.json"
        calibrator.save(result, path)
        loaded = calibrator.load(path)

        assert loaded is not None
        assert abs(loaded.info_threshold - result.info_threshold) < 1e-6
        assert abs(loaded.low_threshold - result.low_threshold) < 1e-6
        assert abs(loaded.medium_threshold - result.medium_threshold) < 1e-6
        assert abs(loaded.high_threshold - result.high_threshold) < 1e-6
        assert loaded.n_calibration_samples == result.n_calibration_samples

    def test_load_missing_file(self, tmp_path):
        """文件不存在 → 返回 None。"""
        calibrator = ConformalCalibrator()
        result = calibrator.load(tmp_path / "nonexistent.json")
        assert result is None

    def test_calibrate_with_uniform_scores(self):
        """均匀分布 [0, 0.3] → 阈值全在 0.3 附近。"""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 0.3, size=500)
        calibrator = ConformalCalibrator()
        result = calibrator.calibrate(scores)

        # All thresholds should be near 0.3 (the max of the distribution)
        assert result.info_threshold <= 0.35
        assert result.high_threshold <= 0.40  # with ordering correction
