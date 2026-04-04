"""False positive feedback loop.

Exports false positive data for model retraining and tracks
feedback statistics to measure system improvement over time.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import structlog

from argus.storage.database import Database

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
                    exported += 1

        logger.info(
            "feedback.exported",
            camera_id=camera_id,
            exported=exported,
            total_fp=len(fp_alerts),
        )
        return exported

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
