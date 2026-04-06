"""Feedback loop manager for alert review and model retraining (Section 6).

Handles both active (operator) and passive (drift/health) feedback.
Feedback entries enter a pending queue and are consumed by retraining runs.

Workflow:
1. Operator/system submits feedback → FeedbackRecord(status=pending) created
2. Side-effects: FP frames → baseline dir; confirmed frames → validation dir
3. Engineer triggers retraining → selects pending batch → mark_batch_processed
4. TrainingRecord records which feedback_ids were incorporated
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import structlog

from argus.config.schema import FeedbackConfig
from argus.storage.database import Database
from argus.storage.models import (
    BaselineState,
    FeedbackRecord,
    FeedbackSource,
    FeedbackStatus,
    FeedbackType,
)

logger = structlog.get_logger()


class FeedbackManager:
    """Manages the feedback loop for alert review and model retraining.

    Replaces the old FP-only export with a full queue model supporting
    three feedback types (confirmed, false_positive, uncertain) and
    passive feedback from drift/health monitoring.
    """

    def __init__(
        self,
        database: Database,
        baselines_dir: str | Path = "data/baselines",
        alerts_dir: str | Path = "data/alerts",
        config: FeedbackConfig | None = None,
    ):
        self._db = database
        self._baselines_dir = Path(baselines_dir)
        self._alerts_dir = Path(alerts_dir)
        self._config = config or FeedbackConfig()

    # ── Core feedback submission ──

    def submit_feedback(
        self,
        alert_id: str,
        feedback_type: str,
        camera_id: str,
        zone_id: str = "default",
        category: str | None = None,
        notes: str | None = None,
        submitted_by: str = "operator",
        model_version_id: str | None = None,
        anomaly_score: float | None = None,
        snapshot_path: str | None = None,
    ) -> FeedbackRecord:
        """Submit operator feedback for an alert.

        Creates a FeedbackRecord in the pending queue and triggers
        side-effects based on feedback type:
        - false_positive + auto_baseline_on_fp → copy snapshot to baselines
        - confirmed + auto_validation_on_confirmed → copy to validation set
        - uncertain → log + no copy (awaits supervisor review)

        Returns the created FeedbackRecord.
        """
        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            alert_id=alert_id,
            camera_id=camera_id,
            zone_id=zone_id,
            feedback_type=feedback_type,
            category=category,
            model_version_at_time=model_version_id,
            anomaly_score=anomaly_score,
            snapshot_path=snapshot_path,
            notes=notes,
            submitted_by=submitted_by,
            source=FeedbackSource.MANUAL,
            status=FeedbackStatus.PENDING,
        )
        saved = self._db.save_feedback(record)

        # Side-effects
        if feedback_type == FeedbackType.FALSE_POSITIVE and self._config.auto_baseline_on_fp:
            self._copy_to_baseline(saved)
        elif feedback_type == FeedbackType.CONFIRMED and self._config.auto_validation_on_confirmed:
            self._copy_to_validation(saved)
        elif feedback_type == FeedbackType.UNCERTAIN:
            logger.info(
                "feedback.uncertain_submitted",
                alert_id=alert_id,
                camera_id=camera_id,
                submitted_by=submitted_by,
                escalation_role=self._config.uncertain_escalation_role,
            )

        logger.info(
            "feedback.submitted",
            feedback_id=saved.feedback_id,
            feedback_type=feedback_type,
            alert_id=alert_id,
            camera_id=camera_id,
        )
        return saved

    def submit_passive_feedback(
        self,
        camera_id: str,
        zone_id: str = "all",
        source: str = FeedbackSource.DRIFT,
        notes: str | None = None,
        model_version_id: str | None = None,
    ) -> FeedbackRecord:
        """Submit system-generated feedback from drift/health monitoring.

        Passive feedback has no associated alert_id and is auto-generated
        by the pipeline when score distribution drift or camera health
        anomalies are detected.
        """
        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            alert_id=None,
            camera_id=camera_id,
            zone_id=zone_id,
            feedback_type=FeedbackType.CONFIRMED,  # System confirms anomaly
            category=None,
            model_version_at_time=model_version_id,
            notes=notes,
            submitted_by="system",
            source=source,
            status=FeedbackStatus.PENDING,
        )
        saved = self._db.save_feedback(record)
        logger.info(
            "feedback.passive_submitted",
            feedback_id=saved.feedback_id,
            source=source,
            camera_id=camera_id,
        )
        return saved

    # ── Queue operations ──

    def get_pending_for_training(
        self,
        camera_id: str | None = None,
        feedback_type: str | None = None,
    ) -> list[FeedbackRecord]:
        """Get pending feedback entries ready for the next training batch."""
        return self._db.get_pending_feedback(
            camera_id=camera_id, feedback_type=feedback_type,
        )

    def mark_batch_processed(
        self,
        feedback_ids: list[str],
        model_version_id: str,
    ) -> int:
        """Mark a batch of feedback entries as consumed by a training run.

        Returns the number of records updated.
        """
        return self._db.mark_feedback_processed(feedback_ids, model_version_id)

    # ── Confirmed frames collection ──

    def collect_confirmed_for_validation(
        self,
        camera_id: str,
        validation_dir: str | Path | None = None,
    ) -> int:
        """Copy confirmed anomaly snapshots to validation directory.

        These are real anomaly frames used for recall testing — more
        valuable than synthetic data for measuring detection quality.

        Returns the number of images copied.
        """
        out_dir = Path(validation_dir or self._config.validation_dir) / camera_id / "confirmed"
        out_dir.mkdir(parents=True, exist_ok=True)

        pending = self._db.get_pending_feedback(
            camera_id=camera_id, feedback_type=FeedbackType.CONFIRMED,
        )
        copied = 0
        for fb in pending:
            if fb.snapshot_path and Path(fb.snapshot_path).exists():
                dst = out_dir / f"confirmed_{fb.feedback_id[:16]}.jpg"
                if not dst.exists():
                    shutil.copy2(fb.snapshot_path, dst)
                    # Write metadata sidecar
                    meta = {
                        "feedback_id": fb.feedback_id,
                        "alert_id": fb.alert_id,
                        "anomaly_score": fb.anomaly_score,
                        "camera_id": fb.camera_id,
                        "zone_id": fb.zone_id,
                        "model_version": fb.model_version_at_time,
                        "submitted_by": fb.submitted_by,
                    }
                    meta_path = dst.with_suffix(".meta.json")
                    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
                    copied += 1

        if copied:
            logger.info(
                "feedback.confirmed_collected",
                camera_id=camera_id,
                collected=copied,
                output_dir=str(out_dir),
            )
        return copied

    # ── Legacy compatibility ──

    def export_false_positives(
        self,
        camera_id: str,
        zone_id: str = "default",
    ) -> int:
        """Export false positive snapshots to the baseline directory.

        Copies FP-marked alert snapshots into the camera's baseline
        folder so they're included in the next training cycle.

        Returns the number of images exported.
        """
        # Get all FP alerts for this camera
        alerts = self._db.get_alerts(camera_id=camera_id, limit=500)
        fp_alerts = [a for a in alerts if a.false_positive and a.snapshot_path]

        if not fp_alerts:
            logger.info("feedback.no_fp", camera_id=camera_id)
            return 0

        # Create output directory
        fp_dir = self._baselines_dir / camera_id / zone_id / "false_positives"
        fp_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        for alert in fp_alerts:
            src = Path(alert.snapshot_path)
            if src.exists():
                dst = fp_dir / f"fp_{alert.alert_id}.jpg"
                if not dst.exists():
                    shutil.copy2(src, dst)
                    # Write metadata sidecar
                    meta = {
                        "alert_id": alert.alert_id,
                        "operator": alert.acknowledged_by or "unknown",
                        "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
                        "camera_id": camera_id,
                        "zone_id": zone_id,
                        "category": "false_positive",
                        "anomaly_score": alert.anomaly_score,
                        "severity": alert.severity,
                        "notes": alert.notes,
                    }
                    meta_path = dst.with_suffix(".meta.json")
                    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
                    exported += 1

        logger.info(
            "feedback.exported",
            camera_id=camera_id,
            exported=exported,
            total_fp=len(fp_alerts),
        )
        return exported

    def merge_fp_into_baseline(
        self,
        camera_id: str,
        zone_id: str = "default",
        baseline_manager=None,
        max_fp_images: int | None = None,
    ) -> dict:
        """Merge false positive candidate pool into a new baseline version.

        1. Read FP images from false_positives/ directory
        2. Read current Active baseline images
        3. Combine and apply diversity_select for deduplication
        4. Create new version in Draft state (NOT Active — immutability preserved)
        5. Copy selected images into the new version

        Returns summary dict with version, counts, etc.
        """
        if baseline_manager is None:
            raise ValueError("baseline_manager is required")

        fp_dir = self._baselines_dir / camera_id / zone_id / "false_positives"
        fp_images = sorted(
            list(fp_dir.glob("*.jpg")) + list(fp_dir.glob("*.png"))
        ) if fp_dir.is_dir() else []

        if not fp_images:
            return {"error": "No false positive images in candidate pool", "fp_count": 0}

        if max_fp_images and len(fp_images) > max_fp_images:
            fp_images = fp_images[:max_fp_images]

        # Get current baseline images
        current_dir = baseline_manager.get_baseline_dir(camera_id, zone_id)
        current_images = sorted(
            list(current_dir.glob("*.png")) + list(current_dir.glob("*.jpg"))
        ) if current_dir.is_dir() else []

        # Create new version (registered as Draft via lifecycle)
        new_version_dir = baseline_manager.create_new_version(camera_id, zone_id)

        # Copy current baseline images
        idx = 0
        for img in current_images:
            dst = new_version_dir / f"baseline_{idx:05d}{img.suffix}"
            shutil.copy2(str(img), str(dst))
            idx += 1

        # Copy FP images (preserving fp_ prefix for traceability)
        fp_included = 0
        for img in fp_images:
            dst = new_version_dir / f"fp_{fp_included:05d}{img.suffix}"
            shutil.copy2(str(img), str(dst))
            # Copy metadata sidecar if exists
            meta_src = img.with_suffix(".meta.json")
            if meta_src.exists():
                meta_dst = dst.with_suffix(".meta.json")
                shutil.copy2(str(meta_src), str(meta_dst))
            fp_included += 1
            idx += 1

        # Deduplicate: keep at most the original baseline count + 20% headroom
        total_before = idx
        total_after = total_before
        if len(current_images) > 0 and total_before > len(current_images):
            target = max(len(current_images) + len(current_images) // 5, 30)
            all_imgs = sorted(
                list(new_version_dir.glob("*.png"))
                + list(new_version_dir.glob("*.jpg"))
            )
            if len(all_imgs) > target:
                selected = set(baseline_manager.diversity_select(
                    new_version_dir, target
                ))
                for img in all_imgs:
                    if img not in selected:
                        img.unlink()
                        meta = img.with_suffix(".meta.json")
                        if meta.exists():
                            meta.unlink()
                total_after = len(selected)

        # Update lifecycle record with final count
        lifecycle = getattr(baseline_manager, "_lifecycle", None)
        if lifecycle:
            lifecycle.register_version(
                camera_id, zone_id, new_version_dir.name,
                image_count=total_after,
            )

        logger.info(
            "feedback.fp_merged",
            camera_id=camera_id,
            version=new_version_dir.name,
            baseline_images=len(current_images),
            fp_included=fp_included,
            total_after_dedup=total_after,
        )

        return {
            "version": new_version_dir.name,
            "baseline_images": len(current_images),
            "fp_included": fp_included,
            "total_before_dedup": total_before,
            "total_after_dedup": total_after,
            "state": BaselineState.DRAFT,
        }

    def get_feedback_stats(self, camera_id: str | None = None) -> dict:
        """Get feedback statistics for monitoring.

        Returns both legacy alert-based stats and new queue-based summary.
        """
        alerts = self._db.get_alerts(camera_id=camera_id, limit=1000)

        total = len(alerts)
        acknowledged = sum(1 for a in alerts if a.acknowledged)
        false_positives = sum(1 for a in alerts if a.false_positive)
        fp_rate = false_positives / total if total > 0 else 0

        # New queue-based summary
        queue_summary = self._db.get_feedback_summary(camera_id=camera_id)

        return {
            "total_alerts": total,
            "acknowledged": acknowledged,
            "false_positives": false_positives,
            "false_positive_rate": round(fp_rate, 4),
            "unreviewed": total - acknowledged - false_positives,
            "feedback_queue": queue_summary,
        }

    # ── Internal helpers ──

    def _copy_to_baseline(self, record: FeedbackRecord) -> None:
        """Copy FP snapshot to camera's baseline false_positives directory."""
        if not record.snapshot_path:
            return
        src = Path(record.snapshot_path)
        if not src.exists():
            logger.warning(
                "feedback.snapshot_not_found",
                feedback_id=record.feedback_id,
                path=str(src),
            )
            return

        fp_dir = self._baselines_dir / record.camera_id / record.zone_id / "false_positives"
        fp_dir.mkdir(parents=True, exist_ok=True)

        dst = fp_dir / f"fp_{record.feedback_id[:16]}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
            # Write metadata sidecar
            meta = {
                "feedback_id": record.feedback_id,
                "alert_id": record.alert_id,
                "category": record.category,
                "anomaly_score": record.anomaly_score,
                "camera_id": record.camera_id,
                "zone_id": record.zone_id,
                "model_version": record.model_version_at_time,
                "submitted_by": record.submitted_by,
            }
            meta_path = dst.with_suffix(".meta.json")
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            logger.debug(
                "feedback.fp_to_baseline",
                feedback_id=record.feedback_id,
                dst=str(dst),
            )

    def _copy_to_validation(self, record: FeedbackRecord) -> None:
        """Copy confirmed anomaly snapshot to validation directory."""
        if not record.snapshot_path:
            return
        src = Path(record.snapshot_path)
        if not src.exists():
            return

        val_dir = self._config.validation_dir / record.camera_id / "confirmed"
        val_dir.mkdir(parents=True, exist_ok=True)

        dst = val_dir / f"confirmed_{record.feedback_id[:16]}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
            meta = {
                "feedback_id": record.feedback_id,
                "alert_id": record.alert_id,
                "anomaly_score": record.anomaly_score,
                "camera_id": record.camera_id,
                "zone_id": record.zone_id,
                "model_version": record.model_version_at_time,
            }
            meta_path = dst.with_suffix(".meta.json")
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            logger.debug(
                "feedback.confirmed_to_validation",
                feedback_id=record.feedback_id,
                dst=str(dst),
            )
