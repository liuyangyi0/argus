"""Precision / Recall / F1 / AUROC / PR-AUC metrics for anomaly detection.

Phase 1 评估尺子 — 所有指标用纯 numpy 实现，不引入 sklearn 依赖。
供 training_validator / recall_test / model_compare / scripts/evaluate_regression 共用。

约定:
    y_true: np.ndarray shape (N,) with 0 (normal) / 1 (anomaly)
    y_scores: np.ndarray shape (N,) anomaly scores in [0, 1], higher = more anomalous
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary confusion matrix.

    Returns {tp, fp, fn, tn} as Python ints (JSON-serializable).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_precision_recall_f1(
    y_true: np.ndarray, y_scores: np.ndarray, threshold: float,
) -> dict:
    """Precision / Recall / F1 at a fixed threshold.

    Returns {precision, recall, f1, threshold, confusion_matrix}.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    y_pred = (y_scores >= threshold).astype(np.int64)
    cm = compute_confusion_matrix(y_true, y_pred)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": cm,
    }


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """AUROC via Mann-Whitney U statistic (numpy only, ties handled via avg rank).

    Returns 0.0 if either class is empty.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Rank with ties averaged ("average" rank).
    order = np.argsort(y_scores, kind="mergesort")
    ranks_sorted = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_scores[order]
    i = 0
    N = len(y_scores)
    while i < N:
        j = i
        while j + 1 < N and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based avg of ranks i+1..j+1
        ranks_sorted[i : j + 1] = avg
        i = j + 1
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted

    pos_rank_sum = ranks[y_true == 1].sum()
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_pr_curve(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    """Precision-Recall curve at every distinct threshold.

    Returns {precisions, recalls, thresholds} as lists, ordered by descending threshold.
    Includes the (recall=0, precision=1) anchor at the start so trapezoidal PR-AUC
    integration is well-defined.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    n_pos = int((y_true == 1).sum())
    if n_pos == 0 or len(y_true) == 0:
        return {"precisions": [1.0], "recalls": [0.0], "thresholds": [1.0]}

    # Sort by score desc. At each step we "predict positive" for everything
    # above the current threshold (descending).
    order = np.argsort(-y_scores, kind="mergesort")
    sorted_true = y_true[order]
    sorted_scores = y_scores[order]

    tp_cum = np.cumsum(sorted_true == 1).astype(np.float64)
    fp_cum = np.cumsum(sorted_true == 0).astype(np.float64)
    pred_pos = tp_cum + fp_cum

    precision = np.where(pred_pos > 0, tp_cum / pred_pos, 1.0)
    recall = tp_cum / n_pos

    # Anchor: recall=0, precision=1 (corresponds to threshold > max score)
    precisions = [1.0] + precision.tolist()
    recalls = [0.0] + recall.tolist()
    thresholds = [float(sorted_scores[0]) + 1e-9] + sorted_scores.tolist()

    return {
        "precisions": [float(p) for p in precisions],
        "recalls": [float(r) for r in recalls],
        "thresholds": [float(t) for t in thresholds],
    }


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area under the Precision-Recall curve via trapezoidal integration.

    Manual trapezoidal rule (np.trapz removed in NumPy 2.x, np.trapezoid not
    in 1.x) keeps this version-independent.
    """
    curve = compute_pr_curve(y_true, y_scores)
    recalls = np.asarray(curve["recalls"], dtype=np.float64)
    precisions = np.asarray(curve["precisions"], dtype=np.float64)
    if len(recalls) < 2:
        return 0.0
    order = np.argsort(recalls)
    r = recalls[order]
    p = precisions[order]
    dr = np.diff(r)
    avg_p = (p[:-1] + p[1:]) * 0.5
    return float(np.sum(dr * avg_p))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target: Literal["f1", "fpr_0.1", "fpr_0.01", "fpr_0.001"] = "f1",
) -> float:
    """Search for the threshold optimizing a target metric.

    target="f1": maximize F1
    target="fpr_X": smallest threshold s.t. false-positive-rate <= X
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    if len(y_scores) == 0:
        return 0.5

    candidates = np.unique(y_scores)
    if target == "f1":
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            res = compute_precision_recall_f1(y_true, y_scores, float(t))
            if res["f1"] > best_f1:
                best_f1 = res["f1"]
                best_t = float(t)
        return best_t

    if target.startswith("fpr_"):
        fpr_target = float(target.split("_", 1)[1])
        n_neg = int((y_true == 0).sum())
        if n_neg == 0:
            return 0.5
        # Try thresholds descending from high to low; first one violating FPR
        # bound means the previous one was the tightest we can go.
        best_t = 1.0
        for t in sorted(candidates.tolist(), reverse=True):
            y_pred = (y_scores >= t).astype(np.int64)
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fpr = fp / n_neg
            if fpr <= fpr_target:
                best_t = float(t)
            else:
                break
        return best_t

    return 0.5


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> dict:
    """One-shot: compute precision/recall/f1/auroc/pr_auc/confusion_matrix.

    Used by evaluate_metrics, model_compare, scripts/evaluate_regression.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    prf1 = compute_precision_recall_f1(y_true, y_scores, threshold)
    return {
        "precision": prf1["precision"],
        "recall": prf1["recall"],
        "f1": prf1["f1"],
        "auroc": compute_auroc(y_true, y_scores),
        "pr_auc": compute_pr_auc(y_true, y_scores),
        "confusion_matrix": prf1["confusion_matrix"],
        "threshold": float(threshold),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
    }
