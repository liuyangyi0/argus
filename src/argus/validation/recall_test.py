"""Recall evaluation tool for synthetic anomaly data (D3-2).

Runs a trained detector on synthetic anomaly images and computes recall metrics.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


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
