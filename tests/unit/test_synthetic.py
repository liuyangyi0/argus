"""Tests for synthetic data generation and recall evaluation (D3)."""

import cv2
import numpy as np
import pytest

from argus.validation.recall_test import load_synthetic_pairs, evaluate_recall


class TestSyntheticData:

    def test_load_synthetic_pairs(self, tmp_path):
        """Load paired synthetic images and masks."""
        # Create fake synthetic data
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        for i in range(5):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"synthetic_{i:05d}.png"), img)
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.imwrite(str(masks_dir / f"synthetic_{i:05d}_mask.png"), mask)

        pairs = load_synthetic_pairs(tmp_path)
        assert len(pairs) == 5

    def test_load_synthetic_pairs_empty_dir(self, tmp_path):
        """Empty directory returns no pairs."""
        pairs = load_synthetic_pairs(tmp_path)
        assert len(pairs) == 0

    def test_recall_evaluation_runs(self, tmp_path):
        """Recall evaluation completes without error."""
        # Create minimal synthetic data
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        for i in range(3):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"synthetic_{i:05d}.png"), img)
            mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.imwrite(str(masks_dir / f"synthetic_{i:05d}_mask.png"), mask)

        # Mock detector
        class MockDetector:
            def predict(self, frame):
                class Result:
                    anomaly_score = 0.8
                return Result()

        result = evaluate_recall(MockDetector(), tmp_path, threshold=0.5)
        assert result["recall"] == 1.0
        assert result["tp"] == 3
        assert result["fn"] == 0
