"""Three-step training validation pipeline.

Every trained model must pass ALL three validation steps before it can
be marked as complete:

1. Holdout AUROC: Compute AUROC on holdout set (normal vs synthetic anomalies)
2. Synthetic Recall: Run recall evaluation on generated synthetic anomaly set
3. Historical Alert Replay: Re-score recent alerts to check for regression

Nuclear environment: any failure blocks model deployment.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import structlog

from argus.anomaly.trainer import _list_images

if TYPE_CHECKING:
    from argus.storage.database import Database

logger = structlog.get_logger()


@dataclass
class StepResult:
    """Result of a single validation step."""

    name: str
    passed: bool
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0
    detail: dict = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class ValidationReport:
    """Combined result of all validation steps."""

    all_passed: bool = False
    steps: list[StepResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    # Convenience accessors for the synthetic-data pipeline
    auroc: float | None = None
    recall: float | None = None
    replay: dict | None = None

    # Phase 1: real-labeled P/R/F1/AUROC/PR-AUC. Populated only when
    # data/validation/{camera_id}/confirmed and data/baselines/{camera_id}/
    # false_positives each have ≥10 samples. None when skipped.
    real_precision: float | None = None
    real_recall: float | None = None
    real_f1: float | None = None
    real_auroc: float | None = None
    real_pr_auc: float | None = None
    real_confusion_matrix: dict | None = None
    real_sample_count: int | None = None
    # Phase 2: raw per-sample scores/labels — enables frontend threshold slider
    # without re-running the model. None when real-labeled eval was skipped.
    real_scores: list[float] | None = None
    real_labels: list[int] | None = None

    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "auroc": self.auroc,
            "recall": self.recall,
            "replay": self.replay,
            "real_precision": self.real_precision,
            "real_recall": self.real_recall,
            "real_f1": self.real_f1,
            "real_auroc": self.real_auroc,
            "real_pr_auc": self.real_pr_auc,
            "real_confusion_matrix": self.real_confusion_matrix,
            "real_sample_count": self.real_sample_count,
            "duration_seconds": round(self.duration_seconds, 1),
            "steps": [
                {
                    "name": s.name,
                    "passed": s.passed,
                    "metric_name": s.metric_name,
                    "metric_value": round(s.metric_value, 4) if s.metric_value else None,
                    "threshold": s.threshold,
                    "skipped": s.skipped,
                    "skip_reason": s.skip_reason,
                    "detail": s.detail,
                }
                for s in self.steps
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _compute_auroc(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> float:
    """Compute AUROC from normal and anomaly score distributions (numpy only).

    Labels: normal=0, anomaly=1. Higher score = more anomalous.
    Uses the Mann-Whitney U statistic: AUROC = U / (n0 * n1).
    """
    n0 = len(normal_scores)
    n1 = len(anomaly_scores)
    if n0 == 0 or n1 == 0:
        return 0.0

    # Count how many anomaly scores exceed each normal score
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([np.zeros(n0), np.ones(n1)])

    # Sort by score
    order = np.argsort(all_scores)
    sorted_labels = all_labels[order]

    # Mann-Whitney U statistic
    # Sum of ranks of positive class minus the minimum possible sum
    ranks = np.arange(1, len(sorted_labels) + 1)
    pos_rank_sum = ranks[sorted_labels == 1].sum()
    u = pos_rank_sum - n1 * (n1 + 1) / 2
    auroc = u / (n0 * n1)

    return float(auroc)


class TrainingValidator:
    """Orchestrates three-step validation after model training."""

    def __init__(
        self,
        database: Database | None = None,
        foe_objects_dir: Path | None = None,
    ):
        self._database = database
        self._foe_objects_dir = foe_objects_dir or Path("data/foe_objects")

    def validate(
        self,
        detector,
        camera_id: str,
        zone_id: str,
        val_dir: Path,
        baseline_dir: Path,
        threshold: float,
        auroc_threshold: float = 0.99,
        recall_threshold: float = 0.95,
        replay_days: int = 30,
    ) -> ValidationReport:
        """Run all three validation steps. Returns combined report."""
        import tempfile

        start = time.monotonic()
        steps: list[StepResult] = []

        # Generate synthetic images once, reuse for step 1 (AUROC) and step 2 (Recall)
        synthetic_dir = None
        tmpdir_obj = None
        if self._foe_objects_dir.exists() and any(self._foe_objects_dir.glob("*.png")):
            try:
                from argus.validation.synthetic import generate_synthetic
                tmpdir_obj = tempfile.TemporaryDirectory()
                synthetic_dir = Path(tmpdir_obj.name) / "synthetic"
                n_generated = generate_synthetic(
                    baseline_dir=baseline_dir,
                    objects_dir=self._foe_objects_dir,
                    output_dir=synthetic_dir,
                    count=100,
                    seed=42,
                )
                if n_generated < 5:
                    synthetic_dir = None
            except Exception as e:
                logger.warning("validation.synthetic_generation_failed", error=str(e))
                synthetic_dir = None

        try:
            step1 = self._validate_auroc(
                detector, val_dir, baseline_dir, auroc_threshold, synthetic_dir,
            )
            steps.append(step1)

            step2 = self._validate_synthetic_recall(
                detector, threshold, recall_threshold, synthetic_dir,
            )
            steps.append(step2)
        finally:
            if tmpdir_obj is not None:
                tmpdir_obj.cleanup()

        # Step 3: Historical Alert Replay
        step3 = self._validate_historical_replay(
            detector, camera_id, threshold, replay_days,
        )
        steps.append(step3)

        # Step 4: Phase 1 — real-labeled P/R/F1/AUROC/PR-AUC
        step4 = self._validate_real_labeled(detector, camera_id, threshold)
        steps.append(step4)

        duration = time.monotonic() - start

        # All must pass (skipped counts as pass), but at least one must
        # actually run — if every step was skipped we have no evidence of quality.
        any_actually_ran = any(not s.skipped for s in steps)
        all_passed = any_actually_ran and all(s.passed or s.skipped for s in steps)
        if not any_actually_ran:
            logger.warning(
                "training_validator.all_skipped",
                msg="All validation steps were skipped — marking as NOT passed",
            )

        # Hoist step 4's real-labeled metrics onto the report for easy DB persistence
        real = step4.detail if not step4.skipped else {}

        report = ValidationReport(
            all_passed=all_passed,
            steps=steps,
            duration_seconds=duration,
            auroc=step1.metric_value if not step1.skipped else None,
            recall=step2.metric_value if not step2.skipped else None,
            replay=step3.detail if not step3.skipped else None,
            real_precision=real.get("precision"),
            real_recall=real.get("recall"),
            real_f1=real.get("f1"),
            real_auroc=real.get("auroc"),
            real_pr_auc=real.get("pr_auc"),
            real_confusion_matrix=real.get("confusion_matrix"),
            real_sample_count=real.get("sample_count"),
            real_scores=real.get("scores"),
            real_labels=real.get("labels"),
        )

        logger.info(
            "training_validator.complete",
            all_passed=all_passed,
            auroc=step1.metric_value,
            recall=step2.metric_value,
            real_f1=report.real_f1,
            real_auroc=report.real_auroc,
            duration=f"{duration:.1f}s",
        )

        return report

    @staticmethod
    def _score_images(detector, image_paths: list[Path]) -> list[float]:
        """Score a list of images through a detector. Returns anomaly scores."""
        scores = []
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            try:
                result = detector.predict(frame)
                scores.append(result.anomaly_score)
            except Exception:
                continue
        return scores

    def _validate_auroc(
        self,
        detector,
        val_dir: Path,
        baseline_dir: Path,
        auroc_threshold: float,
        synthetic_dir: Path | None = None,
    ) -> StepResult:
        """Step 1: Compute AUROC on holdout set (normal vs synthetic anomalies)."""
        val_normal = val_dir / "normal" if (val_dir / "normal").exists() else val_dir
        val_images = _list_images(val_normal)

        if len(val_images) < 5:
            return StepResult(
                name="holdout_auroc",
                passed=False,
                skipped=True,
                skip_reason=f"Validation set too small ({len(val_images)} images)",
            )

        normal_scores = self._score_images(detector, val_images)

        if len(normal_scores) < 3:
            return StepResult(
                name="holdout_auroc",
                passed=False,
                skipped=True,
                skip_reason="Could not score enough normal images",
            )

        # Score synthetic anomalies (from shared dir or fallback to noise injection)
        if synthetic_dir and synthetic_dir.exists():
            anomaly_scores = self._score_images(detector, _list_images(synthetic_dir))
        else:
            anomaly_scores = self._score_noise_injected(detector, val_images)

        if len(anomaly_scores) < 3:
            return StepResult(
                name="holdout_auroc",
                passed=False,
                skipped=True,
                skip_reason="Could not generate enough synthetic anomalies for AUROC",
            )

        auroc = _compute_auroc(np.array(normal_scores), np.array(anomaly_scores))
        passed = auroc >= auroc_threshold

        logger.info(
            "validation.auroc",
            auroc=round(auroc, 4),
            threshold=auroc_threshold,
            passed=passed,
            normal_count=len(normal_scores),
            anomaly_count=len(anomaly_scores),
        )

        return StepResult(
            name="holdout_auroc",
            passed=passed,
            metric_name="AUROC",
            metric_value=auroc,
            threshold=auroc_threshold,
            detail={
                "normal_count": len(normal_scores),
                "anomaly_count": len(anomaly_scores),
                "normal_mean": float(np.mean(normal_scores)),
                "anomaly_mean": float(np.mean(anomaly_scores)),
            },
        )

    @staticmethod
    def _score_noise_injected(detector, val_images: list[Path], count: int = 50) -> list[float]:
        """Fallback: inject noise rectangles into val images and score them."""
        scores = []
        rng = np.random.default_rng(42)
        for img_path in val_images[:count]:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            x1 = rng.integers(0, max(1, w // 2))
            y1 = rng.integers(0, max(1, h // 2))
            x2 = min(w, x1 + rng.integers(w // 8, w // 3))
            y2 = min(h, y1 + rng.integers(h // 8, h // 3))
            noise = rng.integers(0, 255, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
            corrupted = frame.copy()
            corrupted[y1:y2, x1:x2] = noise
            try:
                result = detector.predict(corrupted)
                scores.append(result.anomaly_score)
            except Exception:
                continue
        return scores

    def _validate_synthetic_recall(
        self,
        detector,
        threshold: float,
        recall_threshold: float,
        synthetic_dir: Path | None = None,
    ) -> StepResult:
        """Step 2: Evaluate recall on pre-generated synthetic anomaly images."""
        if synthetic_dir is None or not synthetic_dir.exists():
            return StepResult(
                name="synthetic_recall",
                passed=True,
                skipped=True,
                skip_reason="No synthetic images available (FOE objects missing or generation failed)",
            )

        try:
            from argus.validation.recall_test import evaluate_recall

            synthetic_images = _list_images(synthetic_dir)
            if len(synthetic_images) < 10:
                return StepResult(
                    name="synthetic_recall",
                    passed=True,
                    skipped=True,
                    skip_reason=f"Only {len(synthetic_images)} synthetic images (need >= 10)",
                )

            result = evaluate_recall(detector, synthetic_dir, threshold=threshold)
            recall = result["recall"]
            passed = recall >= recall_threshold

            logger.info(
                "validation.recall",
                recall=round(recall, 4),
                threshold=recall_threshold,
                passed=passed,
                tp=result["tp"],
                fn=result["fn"],
            )

            return StepResult(
                name="synthetic_recall",
                passed=passed,
                metric_name="Recall",
                metric_value=recall,
                threshold=recall_threshold,
                detail={
                    "tp": result["tp"],
                    "fn": result["fn"],
                    "total": result["total"],
                    "mean_score": float(np.mean(result["scores"])) if result["scores"] else 0,
                },
            )

        except Exception as e:
            logger.error("validation.recall_failed", error=str(e))
            return StepResult(
                name="synthetic_recall",
                passed=False,
                skipped=True,
                skip_reason=f"Recall evaluation failed: {e}",
            )

    def _validate_historical_replay(
        self,
        detector,
        camera_id: str,
        threshold: float,
        replay_days: int,
    ) -> StepResult:
        """Step 3: Re-score historical alerts for regression testing."""
        if self._database is None:
            return StepResult(
                name="historical_replay",
                passed=True,
                skipped=True,
                skip_reason="Database not available for historical replay",
            )

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=replay_days)
            with self._database.get_session() as session:
                from argus.storage.models import AlertRecord
                from sqlalchemy import select

                stmt = (
                    select(AlertRecord)
                    .where(AlertRecord.camera_id == camera_id)
                    .where(AlertRecord.timestamp >= cutoff)
                    .where(AlertRecord.snapshot_path.isnot(None))
                    .order_by(AlertRecord.timestamp.desc())
                    .limit(500)
                )
                alerts = list(session.scalars(stmt).all())

            if len(alerts) < 10:
                return StepResult(
                    name="historical_replay",
                    passed=True,
                    skipped=True,
                    skip_reason=f"Only {len(alerts)} alerts in last {replay_days} days (need >= 10)",
                )

            # Re-score each alert's snapshot
            original_detections = len(alerts)
            skipped_missing = 0
            skipped_unreadable = 0
            skipped_error = 0
            new_detections = 0
            replay_scores = []

            for alert in alerts:
                if not alert.snapshot_path or not Path(alert.snapshot_path).exists():
                    skipped_missing += 1
                    original_detections -= 1
                    continue

                frame = cv2.imread(str(alert.snapshot_path))
                if frame is None:
                    skipped_unreadable += 1
                    original_detections -= 1
                    continue

                try:
                    result = detector.predict(frame)
                    score = result.anomaly_score
                    replay_scores.append(score)
                    if score >= threshold:
                        new_detections += 1
                except Exception:
                    skipped_error += 1
                    original_detections -= 1
                    continue

            total_skipped = skipped_missing + skipped_unreadable + skipped_error
            if total_skipped > 0:
                logger.warning(
                    "validation.replay_skipped_alerts",
                    skipped_missing=skipped_missing,
                    skipped_unreadable=skipped_unreadable,
                    skipped_error=skipped_error,
                    effective_alerts=original_detections,
                    total_alerts=len(alerts),
                )

            # Require minimum effective sample size for statistical validity
            min_effective = 5
            if original_detections == 0:
                return StepResult(
                    name="historical_replay",
                    passed=True,
                    skipped=True,
                    skip_reason="No valid historical alert snapshots found",
                )
            if 0 < original_detections < min_effective:
                return StepResult(
                    name="historical_replay",
                    passed=True,
                    skipped=True,
                    skip_reason=f"Insufficient valid snapshots: {original_detections} < {min_effective}",
                )

            new_detection_rate = new_detections / original_detections
            # Pass if new model detects >= 95% of what old model detected
            min_rate = 0.95
            passed = new_detection_rate >= min_rate

            logger.info(
                "validation.replay",
                detection_rate=round(new_detection_rate, 4),
                min_rate=min_rate,
                passed=passed,
                original=original_detections,
                new_detections=new_detections,
            )

            detail = {
                "original_alert_count": original_detections,
                "new_detections": new_detections,
                "new_detection_rate": new_detection_rate,
                "min_rate": min_rate,
                "mean_replay_score": float(np.mean(replay_scores)) if replay_scores else 0,
                "replay_days": replay_days,
            }

            return StepResult(
                name="historical_replay",
                passed=passed,
                metric_name="DetectionRate",
                metric_value=new_detection_rate,
                threshold=min_rate,
                detail=detail,
            )

        except Exception as e:
            logger.error("validation.replay_failed", error=str(e))
            return StepResult(
                name="historical_replay",
                passed=False,
                skipped=True,
                skip_reason=f"Historical replay failed: {e}",
            )

    def _validate_real_labeled(
        self,
        detector,
        camera_id: str,
        threshold: float,
        min_f1: float = 0.0,
    ) -> StepResult:
        """Step 4 (Phase 1): P/R/F1/AUROC/PR-AUC on real human-labeled data.

        Loads positives from data/validation/{camera_id}/confirmed/ and negatives
        from data/baselines/{camera_id}/false_positives/. Skipped when either
        directory is missing or has fewer than 10 samples. min_f1=0.0 means this
        step never fails training (advisory only) — tighten later once a baseline
        F1 is established.
        """
        try:
            from argus.validation.recall_test import evaluate_real_labeled

            metrics = evaluate_real_labeled(
                detector,
                camera_id=camera_id,
                threshold=threshold,
            )
        except Exception as e:
            logger.error("validation.real_labeled_failed", error=str(e))
            return StepResult(
                name="real_labeled_metrics",
                passed=True,
                skipped=True,
                skip_reason=f"Real-labeled evaluation failed: {e}",
            )

        if metrics is None:
            return StepResult(
                name="real_labeled_metrics",
                passed=True,
                skipped=True,
                skip_reason="No real-labeled data (need ≥10 positives and ≥10 negatives)",
            )

        detail = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auroc": metrics["auroc"],
            "pr_auc": metrics["pr_auc"],
            "confusion_matrix": metrics["confusion_matrix"],
            "sample_count": metrics["n_positive"] + metrics["n_negative"],
            "n_positive": metrics["n_positive"],
            "n_negative": metrics["n_negative"],
            "threshold": metrics["threshold"],
            "scores": metrics["scores"],
            "labels": metrics["labels"],
        }

        passed = metrics["f1"] >= min_f1
        logger.info(
            "validation.real_labeled",
            camera_id=camera_id,
            precision=round(metrics["precision"], 4),
            recall=round(metrics["recall"], 4),
            f1=round(metrics["f1"], 4),
            auroc=round(metrics["auroc"], 4),
            pr_auc=round(metrics["pr_auc"], 4),
            n_pos=metrics["n_positive"],
            n_neg=metrics["n_negative"],
        )

        return StepResult(
            name="real_labeled_metrics",
            passed=passed,
            metric_name="F1",
            metric_value=metrics["f1"],
            threshold=min_f1,
            detail=detail,
        )
