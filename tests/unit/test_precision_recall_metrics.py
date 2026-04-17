"""Unit tests for the Phase 1 metrics module (src/argus/anomaly/metrics.py)."""

from __future__ import annotations

import numpy as np
import pytest

from argus.anomaly.metrics import (
    compute_auroc,
    compute_confusion_matrix,
    compute_pr_auc,
    compute_pr_curve,
    compute_precision_recall_f1,
    evaluate_at_threshold,
    find_optimal_threshold,
)


class TestConfusionMatrix:
    def test_basic_counts(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}

    def test_all_correct(self):
        y = np.array([1, 0, 1, 0])
        cm = compute_confusion_matrix(y, y)
        assert cm == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}

    def test_all_zero_pred(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.zeros(4, dtype=int)
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm == {"tp": 0, "fp": 0, "fn": 2, "tn": 2}


class TestPrecisionRecallF1:
    def test_perfect_classifier(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        res = compute_precision_recall_f1(y_true, y_scores, threshold=0.5)
        assert res["precision"] == pytest.approx(1.0)
        assert res["recall"] == pytest.approx(1.0)
        assert res["f1"] == pytest.approx(1.0)

    def test_half_recall(self):
        # 2 positives, but threshold catches only 1
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.3, 0.2, 0.1])
        res = compute_precision_recall_f1(y_true, y_scores, threshold=0.5)
        # tp=1, fp=0, fn=1, tn=2 → P=1.0, R=0.5, F1=0.667
        assert res["precision"] == pytest.approx(1.0)
        assert res["recall"] == pytest.approx(0.5)
        assert res["f1"] == pytest.approx(2 / 3)

    def test_zero_precision_and_recall(self):
        # All predictions negative, all labels negative → no tp/fp/fn → defaults 0
        y_true = np.array([0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3])
        res = compute_precision_recall_f1(y_true, y_scores, threshold=0.5)
        assert res["precision"] == 0.0
        assert res["recall"] == 0.0
        assert res["f1"] == 0.0


class TestAUROC:
    def test_perfect(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert compute_auroc(y_true, y_scores) == pytest.approx(1.0)

    def test_worst(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert compute_auroc(y_true, y_scores) == pytest.approx(0.0)

    def test_random(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=1000)
        y_scores = rng.random(1000)
        auroc = compute_auroc(y_true, y_scores)
        assert 0.4 < auroc < 0.6  # chance-level

    def test_ties(self):
        # With ties and score 0.5 for one pos + one neg → AUROC = 0.5 for that
        # comparison; overall should land at a well-defined value (not NaN).
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.5, 0.5, 0.8, 0.2])
        auroc = compute_auroc(y_true, y_scores)
        # Hand: pos ranks (avg for ties): score 0.8→4, score 0.5→avg(2,3)=2.5
        # pos_rank_sum = 4+2.5 = 6.5; U = 6.5 - 2*3/2 = 3.5; AUROC = 3.5/4 = 0.875
        assert auroc == pytest.approx(0.875)

    def test_empty_class_returns_zero(self):
        y_true = np.array([0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3])
        assert compute_auroc(y_true, y_scores) == 0.0


class TestPRCurve:
    def test_curve_monotone_recall(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        curve = compute_pr_curve(y_true, y_scores)
        # Recall should be non-decreasing as we lower threshold
        recalls = curve["recalls"]
        assert recalls == sorted(recalls)
        assert recalls[0] == 0.0
        assert recalls[-1] == pytest.approx(1.0)

    def test_perfect_curve_pr_auc_one(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert compute_pr_auc(y_true, y_scores) == pytest.approx(1.0)

    def test_empty_positives(self):
        curve = compute_pr_curve(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
        assert curve == {"precisions": [1.0], "recalls": [0.0], "thresholds": [1.0]}


class TestFindOptimalThreshold:
    def test_f1_targets_best_split(self):
        # Perfect separation at 0.5 → optimal threshold in [0.3, 0.7]
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        t = find_optimal_threshold(y_true, y_scores, target="f1")
        assert 0.3 < t <= 0.9
        # Verify the chosen threshold actually gives F1=1.0
        res = compute_precision_recall_f1(y_true, y_scores, t)
        assert res["f1"] == pytest.approx(1.0)

    def test_fpr_bound(self):
        # 100 neg at uniform [0,0.5], 10 pos at 0.8 — ask for FPR ≤ 0.01
        rng = np.random.default_rng(1)
        neg_scores = rng.uniform(0, 0.5, size=100)
        pos_scores = rng.uniform(0.7, 0.9, size=10)
        y_scores = np.concatenate([neg_scores, pos_scores])
        y_true = np.concatenate([np.zeros(100), np.ones(10)]).astype(int)
        t = find_optimal_threshold(y_true, y_scores, target="fpr_0.01")
        # At chosen t, FPR must be ≤ 0.01 (i.e. ≤ 1 false positive out of 100)
        fp = int(((y_true == 0) & (y_scores >= t)).sum())
        assert fp <= 1


class TestEvaluateAtThreshold:
    def test_full_bundle_keys_and_values(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_scores = np.array([0.9, 0.4, 0.2, 0.1, 0.8, 0.3])
        res = evaluate_at_threshold(y_true, y_scores, threshold=0.5)
        for key in [
            "precision", "recall", "f1", "auroc", "pr_auc",
            "confusion_matrix", "threshold", "n_positive", "n_negative",
        ]:
            assert key in res
        assert res["n_positive"] == 3
        assert res["n_negative"] == 3
        # tp=2 (scores 0.9,0.8), fn=1 (score 0.4), fp=0, tn=3 at threshold 0.5
        assert res["confusion_matrix"] == {"tp": 2, "fp": 0, "fn": 1, "tn": 3}
        assert res["precision"] == pytest.approx(1.0)
        assert res["recall"] == pytest.approx(2 / 3)
