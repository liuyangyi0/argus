"""Tests for KS drift monitoring (C2)."""

import numpy as np
import pytest

from argus.anomaly.drift import DriftDetector, DriftStatus


class TestDriftDetector:

    def test_no_drift_same_distribution(self):
        """参考和当前来自同一分布 → is_drifted=False。"""
        rng = np.random.default_rng(42)
        ref = rng.normal(0.2, 0.05, 500)
        detector = DriftDetector(reference_scores=ref, check_interval=100)

        # Feed scores from same distribution
        for _ in range(200):
            detector.update(rng.normal(0.2, 0.05))

        status = detector.get_status()
        assert status.is_drifted == False

    def test_drift_detected_shifted_distribution(self):
        """参考 mean=0.1, 当前 mean=0.5 → is_drifted=True。"""
        rng = np.random.default_rng(42)
        ref = rng.normal(0.1, 0.02, 1000)
        detector = DriftDetector(
            reference_scores=ref,
            check_interval=100,
            ks_threshold=0.05,
            p_value_threshold=0.05,
        )

        # Feed very different scores (large shift)
        for _ in range(300):
            detector.update(rng.normal(0.5, 0.02))

        status = detector.get_status()
        assert status.is_drifted == True
        assert status.ks_statistic > 0.05

    def test_ks_2samp_known_values(self):
        """KS statistic for identical distributions should be near 0."""
        a = np.linspace(0, 1, 1000)
        b = np.linspace(0, 1, 1000)
        ks, p = DriftDetector._ks_2samp(a, b)
        assert ks < 0.01
        assert p > 0.5

    def test_insufficient_samples_no_check(self):
        """< 100 samples → 不运行检查。"""
        rng = np.random.default_rng(42)
        ref = rng.normal(0.2, 0.05, 500)
        detector = DriftDetector(reference_scores=ref, check_interval=50)

        # Feed only 50 scores (< 100 minimum)
        for _ in range(50):
            detector.update(rng.normal(0.5, 0.1))

        status = detector.get_status()
        # Should not have run check yet
        assert status.last_check_time == 0.0
