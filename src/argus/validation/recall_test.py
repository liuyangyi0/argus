"""Recall / Precision / F1 / AUROC / PR-AUC evaluation tools.

Phase 1 评估尺子 — 扩展为支持真实人工标注目录结构：
    data/validation/{camera_id}/confirmed/   → 阳性（真实异物）
    data/baselines/{camera_id}/false_positives/ → 阴性（被人工否决的误报帧）

`evaluate_recall` 保持不变（兼容 TRN-005 synthetic 流程）；新增 `evaluate_metrics` 和
`evaluate_real_labeled` 面向真实标注。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import structlog

from argus.anomaly.metrics import evaluate_at_threshold

logger = structlog.get_logger()


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def load_synthetic_pairs(synthetic_dir: Path) -> list[tuple[Path, Path]]:
    """Load synthetic image and mask pairs.

    Returns list of (image_path, mask_path) tuples.
    """
    masks_dir = synthetic_dir / "masks"
    pairs = []
    for img_path in sorted(synthetic_dir.glob("synthetic_*.png")):
        mask_name = img_path.stem + "_mask.png"
        mask_path = masks_dir / mask_name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def evaluate_recall(
    detector,
    synthetic_dir: Path,
    threshold: float = 0.5,
) -> dict:
    """Run detector on synthetic anomaly images and compute recall.

    Args:
        detector: An object with a `predict(frame)` method returning
                  an object with `anomaly_score` attribute.
        synthetic_dir: Directory containing synthetic_*.png and masks/.
        threshold: Score threshold for positive detection.

    Returns:
        Dict with recall, tp, fn, total, and per-image scores.
    """
    pairs = load_synthetic_pairs(synthetic_dir)
    if not pairs:
        logger.warning("recall_test.no_pairs", dir=str(synthetic_dir))
        return {"recall": 0.0, "tp": 0, "fn": 0, "total": 0, "scores": []}

    tp, fn = 0, 0
    scores = []

    for img_path, mask_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        result = detector.predict(img)
        score = result.anomaly_score
        scores.append(score)

        if score >= threshold:
            tp += 1
        else:
            fn += 1

    total = tp + fn
    recall = tp / total if total > 0 else 0.0

    logger.info(
        "recall_test.complete",
        recall=round(recall, 3),
        tp=tp, fn=fn, total=total,
        mean_score=round(float(np.mean(scores)), 4) if scores else 0,
    )

    return {
        "recall": recall,
        "tp": tp,
        "fn": fn,
        "total": total,
        "scores": scores,
    }


def _iter_images(directory: Path) -> Iterable[Path]:
    """Yield image paths (recursively) from a directory, sorted.

    Recurses one level so that both `confirmed/*.jpg` and `confirmed/sub/*.jpg` work;
    stops at depth 2 to avoid pulling in huge nested datasets by accident.
    """
    if not directory.is_dir():
        return
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
            yield p
        elif p.is_dir():
            for q in sorted(p.iterdir()):
                if q.is_file() and q.suffix.lower() in _IMAGE_EXTENSIONS:
                    yield q


def _score_directory(detector, directory: Path, label: int) -> tuple[list[float], list[int]]:
    """Run detector on every image in `directory`, return (scores, labels)."""
    scores: list[float] = []
    labels: list[int] = []
    for img_path in _iter_images(directory):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        try:
            result = detector.predict(frame)
            scores.append(float(result.anomaly_score))
            labels.append(label)
        except Exception as e:
            logger.debug("recall_test.predict_failed", path=str(img_path), error=str(e))
            continue
    return scores, labels


def evaluate_metrics(
    detector,
    positive_dir: Path,
    negative_dir: Path,
    threshold: float = 0.7,
    min_samples_per_class: int = 10,
) -> dict | None:
    """Compute P/R/F1/AUROC/PR-AUC on real labeled directories.

    Args:
        detector: object with .predict(frame) returning .anomaly_score.
        positive_dir: directory of anomaly (label=1) images.
        negative_dir: directory of normal (label=0) images.
        threshold: decision threshold for P/R/F1 (AUROC/PR-AUC are threshold-free).
        min_samples_per_class: skip evaluation if either class has fewer samples.

    Returns:
        Dict from evaluate_at_threshold + {n_positive, n_negative, scores, labels},
        or None if skipped (too few samples).
    """
    pos_scores, pos_labels = _score_directory(detector, positive_dir, 1)
    neg_scores, neg_labels = _score_directory(detector, negative_dir, 0)

    if len(pos_scores) < min_samples_per_class or len(neg_scores) < min_samples_per_class:
        logger.warning(
            "recall_test.insufficient_samples",
            n_positive=len(pos_scores),
            n_negative=len(neg_scores),
            min=min_samples_per_class,
        )
        return None

    y_scores = np.array(pos_scores + neg_scores, dtype=np.float64)
    y_true = np.array(pos_labels + neg_labels, dtype=np.int64)

    metrics = evaluate_at_threshold(y_true, y_scores, threshold)
    metrics["scores"] = y_scores.tolist()
    metrics["labels"] = y_true.tolist()

    logger.info(
        "recall_test.real_labeled_complete",
        n_positive=metrics["n_positive"],
        n_negative=metrics["n_negative"],
        precision=round(metrics["precision"], 3),
        recall=round(metrics["recall"], 3),
        f1=round(metrics["f1"], 3),
        auroc=round(metrics["auroc"], 3),
        pr_auc=round(metrics["pr_auc"], 3),
    )
    return metrics


def evaluate_real_labeled(
    detector,
    camera_id: str,
    data_root: Path = Path("data"),
    threshold: float = 0.7,
    min_samples_per_class: int = 10,
) -> dict | None:
    """Convenience wrapper: use standard project layout for pos/neg dirs.

    Positive  = data/validation/{camera_id}/confirmed/
    Negative  = data/baselines/{camera_id}/false_positives/

    Returns None if either directory is missing or has too few samples.
    """
    positive_dir = data_root / "validation" / camera_id / "confirmed"
    negative_dir = data_root / "baselines" / camera_id / "false_positives"

    if not positive_dir.is_dir() or not negative_dir.is_dir():
        logger.info(
            "recall_test.real_labeled_skipped",
            camera_id=camera_id,
            positive_exists=positive_dir.is_dir(),
            negative_exists=negative_dir.is_dir(),
        )
        return None

    return evaluate_metrics(
        detector,
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        threshold=threshold,
        min_samples_per_class=min_samples_per_class,
    )
