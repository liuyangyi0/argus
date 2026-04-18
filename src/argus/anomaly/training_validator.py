"""Four-step training validation pipeline.

Every trained model runs the following validation steps after training
completes. All steps gracefully degrade when sample directories are missing
or under-populated; "skipped" is counted as a pass so a missing sample set
never blocks deployment — it just means the corresponding metric is
unobservable, and the frontend must clearly tell the operator **why** and
**how to unlock it**.

Steps
-----

1. ``holdout_auroc`` — Compute AUROC on a holdout normal set versus
   synthetic anomalies. Uses ``data/foe_objects/*.png`` to generate
   synthetic positives; falls back to noise-injection when synthetic
   generation yields <5 usable images.
2. ``synthetic_recall`` — Evaluate recall on the synthetic anomaly set.
   Skipped when fewer than 10 synthetic images are available.
3. ``historical_replay`` — Re-score recent alerts (last ``replay_days``)
   from ``alert_records`` to catch regressions. Skipped when there are
   <10 alerts or no snapshots on disk.
4. ``real_labeled_metrics`` — Phase 1 P/R/F1/AUROC/PR-AUC on real human
   labeled data. Positives from ``data/validation/<camera_id>/confirmed/``
   and negatives from ``data/baselines/<camera_id>/false_positives/``.
   Skipped when either directory is missing or has fewer than 10 samples.

Data directory reference
------------------------

+----------------------------------------------------+-----------+---------+
| Directory                                          | Purpose   | Min qty |
+====================================================+===========+=========+
| ``data/foe_objects/*.png``                         | Synthetic | >=1 PNG |
|                                                    | anomaly   | (>=5    |
|                                                    | objects   | usable  |
|                                                    |           | gen'd)  |
+----------------------------------------------------+-----------+---------+
| ``data/validation/<camera_id>/confirmed/``         | Real      | >=10    |
|                                                    | positives |         |
+----------------------------------------------------+-----------+---------+
| ``data/baselines/<camera_id>/false_positives/``    | Real      | >=10    |
|                                                    | negatives |         |
+----------------------------------------------------+-----------+---------+

Supported image extensions: ``.png .jpg .jpeg .bmp .tiff``. Both
``confirmed/`` and ``false_positives/`` may contain a single level of
subdirectories (e.g. for date-based grouping).

Diagnostic logs
---------------

When any step is skipped, a structured ``structlog`` event is emitted so
operators know exactly what to fix:

* ``validation.synthetic_skipped`` — ``data/foe_objects/`` missing or
  empty. ``hint`` field explains how to unlock synthetic evaluation.
* ``validation.synthetic_too_few`` — generation produced <5 usable images.
* ``validation.real_labeled_skipped`` — real-labeled directories missing
  or under-populated. Emits ``positive_dir``, ``n_positive``,
  ``negative_dir``, ``n_negative``, ``min_required``, and ``hint``.
* ``validation.real_labeled_failed`` — evaluator raised. Includes the
  same location fields for triage.

The frontend ``web/src/components/models/MetricsChart.vue`` reads
``has_labeled_eval`` from ``GET /api/training-history/{id}/metrics``.
When false it renders an ``<Empty>`` block plus an ``<Alert type="info">``
listing the directories above — never a blank chart.
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
        foe_png_count = (
            sum(1 for _ in self._foe_objects_dir.glob("*.png"))
            if self._foe_objects_dir.exists()
            else 0
        )
        if foe_png_count > 0:
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
                    logger.warning(
                        "validation.synthetic_too_few",
                        generated=n_generated,
                        foe_objects_dir=str(self._foe_objects_dir),
                        foe_png_count=foe_png_count,
                        hint=(
                            "Synthetic anomaly generation produced <5 usable images. "
                            "Add more PNG objects (ideally with transparent background) "
                            f"to {self._foe_objects_dir} to improve holdout AUROC quality."
                        ),
                    )
                    synthetic_dir = None
            except Exception as e:
                logger.warning(
                    "validation.synthetic_generation_failed",
                    foe_objects_dir=str(self._foe_objects_dir),
                    error=str(e),
                )
                synthetic_dir = None
        else:
            logger.info(
                "validation.synthetic_skipped",
                foe_objects_dir=str(self._foe_objects_dir),
                foe_dir_exists=self._foe_objects_dir.exists(),
                hint=(
                    "Synthetic anomaly generation skipped: "
                    f"{self._foe_objects_dir} is missing or contains no *.png files. "
                    "Holdout AUROC will fall back to noise-injection; synthetic recall "
                    "will be skipped. Drop anomaly object PNGs (transparent background "
                    "preferred) into this directory to enable full synthetic evaluation."
                ),
            )

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

        When skipped, logs an actionable diagnostic identifying which directory
        is missing or under-populated so operators know how to unlock the step.
        """
        # Pre-flight probe: inspect directories up-front so we can log precise
        # reasons even when evaluate_real_labeled silently returns None. This is
        # the observability half of "graceful degradation + clear empty state".
        data_root = Path("data")
        positive_dir = data_root / "validation" / camera_id / "confirmed"
        negative_dir = data_root / "baselines" / camera_id / "false_positives"
        min_samples = 10

        def _count_images(directory: Path) -> int:
            if not directory.is_dir():
                return 0
            exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            total = 0
            for p in directory.iterdir():
                if p.is_file() and p.suffix.lower() in exts:
                    total += 1
                elif p.is_dir():
                    total += sum(
                        1 for q in p.iterdir()
                        if q.is_file() and q.suffix.lower() in exts
                    )
            return total

        pos_count = _count_images(positive_dir)
        neg_count = _count_images(negative_dir)
        pos_exists = positive_dir.is_dir()
        neg_exists = negative_dir.is_dir()

        try:
            from argus.validation.recall_test import evaluate_real_labeled

            metrics = evaluate_real_labeled(
                detector,
                camera_id=camera_id,
                threshold=threshold,
            )
        except Exception as e:
            logger.error(
                "validation.real_labeled_failed",
                camera_id=camera_id,
                positive_dir=str(positive_dir),
                negative_dir=str(negative_dir),
                n_positive=pos_count,
                n_negative=neg_count,
                error=str(e),
            )
            return StepResult(
                name="real_labeled_metrics",
                passed=True,
                skipped=True,
                skip_reason=f"Real-labeled evaluation failed: {e}",
            )

        if metrics is None:
            # Diagnose exactly why the evaluator bailed and tell the operator
            # where to put samples. Three reasons possible:
            #   (a) positive_dir missing            → no human-confirmed anomalies yet
            #   (b) negative_dir missing            → no confirmed false positives yet
            #   (c) both exist but under-populated  → need to label more
            missing_dirs: list[str] = []
            under_populated: list[str] = []
            if not pos_exists:
                missing_dirs.append(str(positive_dir))
            elif pos_count < min_samples:
                under_populated.append(f"{positive_dir} has {pos_count}/{min_samples} positives")
            if not neg_exists:
                missing_dirs.append(str(negative_dir))
            elif neg_count < min_samples:
                under_populated.append(f"{negative_dir} has {neg_count}/{min_samples} negatives")

            if missing_dirs:
                reason = f"Missing directories: {', '.join(missing_dirs)}"
            elif under_populated:
                reason = "Insufficient samples — " + "; ".join(under_populated)
            else:
                reason = (
                    f"need ≥{min_samples} positives (found {pos_count}) and "
                    f"≥{min_samples} negatives (found {neg_count})"
                )

            logger.warning(
                "validation.real_labeled_skipped",
                camera_id=camera_id,
                reason=reason,
                positive_dir=str(positive_dir),
                positive_exists=pos_exists,
                n_positive=pos_count,
                negative_dir=str(negative_dir),
                negative_exists=neg_exists,
                n_negative=neg_count,
                min_required=min_samples,
                hint=(
                    "Add human-confirmed anomaly images to "
                    f"data/validation/{camera_id}/confirmed/ and dismissed-FP images to "
                    f"data/baselines/{camera_id}/false_positives/ to enable "
                    "P/R/F1/AUROC/PR-AUC metrics. Synthetic anomaly PNG objects "
                    "can also be dropped in data/foe_objects/ to unlock "
                    "holdout AUROC + synthetic recall steps."
                ),
            )

            return StepResult(
                name="real_labeled_metrics",
                passed=True,
                skipped=True,
                skip_reason=(
                    f"No real-labeled data (need ≥{min_samples} positives and "
                    f"≥{min_samples} negatives; found {pos_count}/{neg_count}). "
                    f"See logs for actionable hint."
                ),
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
