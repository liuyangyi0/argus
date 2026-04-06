"""False positive feedback loop.

Exports false positive data for model retraining and tracks
feedback statistics to measure system improvement over time.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import structlog

from argus.storage.database import Database
from argus.storage.models import BaselineState

logger = structlog.get_logger()


class FeedbackManager:
    """Manages the false positive feedback loop for model improvement.

    When operators mark alerts as false positives, those frames can be
    added to the normal training set so the model learns to ignore
    similar patterns in the future.

    Workflow:
    1. Operator marks alert as FP via dashboard
    2. FeedbackManager extracts the snapshot from the alert
    3. Snapshot is copied to the camera's baseline directory
    4. Next retraining cycle incorporates the FP frames
    """

    def __init__(
        self,
        database: Database,
        baselines_dir: str | Path = "data/baselines",
        alerts_dir: str | Path = "data/alerts",
    ):
        self._db = database
        self._baselines_dir = Path(baselines_dir)
        self._alerts_dir = Path(alerts_dir)

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

        # Deduplicate: keep at most the original baseline count + 20% headroom for FP additions
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
        """Get feedback statistics for monitoring."""
        alerts = self._db.get_alerts(camera_id=camera_id, limit=1000)

        total = len(alerts)
        acknowledged = sum(1 for a in alerts if a.acknowledged)
        false_positives = sum(1 for a in alerts if a.false_positive)
        fp_rate = false_positives / total if total > 0 else 0

        return {
            "total_alerts": total,
            "acknowledged": acknowledged,
            "false_positives": false_positives,
            "false_positive_rate": round(fp_rate, 4),
            "unreviewed": total - acknowledged - false_positives,
        }
