"""Background retention policy enforcement for recordings and alert evidence.

Periodically scans recording directories and enforces:
- Local storage: delete after ``local_retention_days``
- Archive: move to ``archive_path`` after local retention (if enabled)
- Archive cleanup: delete from archive after ``archive_retention_days``
- Alert evidence: clean up old alert snapshots/recordings
"""

from __future__ import annotations

import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger()


class RetentionManager:
    """Background retention policy enforcement.

    Runs as a daemon thread with a configurable scan interval.

    Parameters
    ----------
    continuous_recording_dir:
        Root directory for continuous recording segments.
    alert_recording_dir:
        Root directory for alert-triggered recordings.
    local_retention_days:
        Days to keep recordings on local storage (default 180).
    archive_enabled:
        Whether to archive instead of deleting.
    archive_path:
        External storage mount point for archived recordings.
    archive_retention_days:
        Days to keep recordings on archive storage (default 360).
    alert_retention_days:
        Days to keep alert evidence (snapshots, heatmaps, clips).
    cleanup_interval_hours:
        Hours between cleanup scans.
    """

    def __init__(
        self,
        continuous_recording_dir: Path,
        alert_recording_dir: Path,
        local_retention_days: int = 180,
        archive_enabled: bool = False,
        archive_path: Path | None = None,
        archive_retention_days: int = 360,
        alert_retention_days: int = 180,
        cleanup_interval_hours: float = 6.0,
    ) -> None:
        self._cont_dir = Path(continuous_recording_dir)
        self._alert_dir = Path(alert_recording_dir)
        self._local_days = local_retention_days
        self._archive_enabled = archive_enabled
        self._archive_path = Path(archive_path) if archive_path else None
        self._archive_days = archive_retention_days
        self._alert_days = alert_retention_days
        self._interval_s = cleanup_interval_hours * 3600.0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="retention-mgr",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "retention.started",
            local_days=self._local_days,
            archive_enabled=self._archive_enabled,
            interval_h=self._interval_s / 3600,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        logger.info("retention.stopped")

    # ------------------------------------------------------------------
    # Cleanup logic
    # ------------------------------------------------------------------

    def run_cleanup(self) -> dict:
        """Execute one cleanup pass.  Returns stats dict."""
        stats = {
            "local_deleted": 0,
            "archived": 0,
            "archive_deleted": 0,
            "alert_deleted": 0,
        }
        stats["local_deleted"] = self._cleanup_local()
        if self._archive_enabled:
            stats["archived"] = self._archive_recordings()
            stats["archive_deleted"] = self._cleanup_archive()
        stats["alert_deleted"] = self._cleanup_alert_evidence()
        logger.info("retention.cleanup_complete", **stats)
        return stats

    def _cleanup_local(self) -> int:
        """Delete local recordings older than retention period."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self._local_days)
        return self._delete_old_date_dirs(self._cont_dir, cutoff)

    def _archive_recordings(self) -> int:
        """Move recordings past local retention to archive."""
        if self._archive_path is None:
            return 0

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self._local_days)
        moved = 0

        if not self._cont_dir.exists():
            return 0

        for date_dir in sorted(self._cont_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            dir_date = self._parse_date_dir(date_dir.name)
            if dir_date is None or dir_date >= cutoff:
                continue

            dest = self._archive_path / date_dir.name
            try:
                dest.mkdir(parents=True, exist_ok=True)
                for item in date_dir.iterdir():
                    shutil.move(str(item), str(dest / item.name))
                    moved += 1
                # Remove empty source dir
                if not any(date_dir.iterdir()):
                    date_dir.rmdir()
            except OSError:
                logger.warning(
                    "retention.archive_error",
                    src=str(date_dir),
                    dest=str(dest),
                    exc_info=True,
                )
        return moved

    def _cleanup_archive(self) -> int:
        """Delete archive recordings past archive retention."""
        if self._archive_path is None or not self._archive_path.exists():
            return 0
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self._archive_days)
        return self._delete_old_date_dirs(self._archive_path, cutoff)

    def _cleanup_alert_evidence(self) -> int:
        """Clean up old alert snapshots and recordings."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self._alert_days)
        return self._delete_old_date_dirs(self._alert_dir, cutoff)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _delete_old_date_dirs(self, root: Path, cutoff: datetime) -> int:
        """Delete date-named subdirectories older than *cutoff*."""
        deleted = 0
        if not root.exists():
            return 0

        for date_dir in sorted(root.iterdir()):
            if not date_dir.is_dir():
                continue
            dir_date = self._parse_date_dir(date_dir.name)
            if dir_date is None or dir_date >= cutoff:
                continue
            try:
                shutil.rmtree(date_dir)
                deleted += 1
                logger.debug("retention.deleted_dir", path=str(date_dir))
            except OSError:
                logger.warning(
                    "retention.delete_error",
                    path=str(date_dir),
                    exc_info=True,
                )
        return deleted

    @staticmethod
    def _parse_date_dir(name: str) -> datetime | None:
        """Parse a YYYY-MM-DD directory name to a datetime."""
        try:
            return datetime.strptime(name, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_cleanup()
            except Exception:
                logger.exception("retention.loop_error")
            self._stop_event.wait(timeout=self._interval_s)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_retention_stats(self) -> dict:
        """Return storage usage and retention statistics."""

        def _dir_size(path: Path) -> int:
            if not path.exists():
                return 0
            total = 0
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
            return total

        local_bytes = _dir_size(self._cont_dir)
        alert_bytes = _dir_size(self._alert_dir)
        archive_bytes = _dir_size(self._archive_path) if self._archive_path else 0

        return {
            "local_storage_bytes": local_bytes,
            "local_storage_gb": round(local_bytes / (1024**3), 2),
            "alert_storage_bytes": alert_bytes,
            "alert_storage_gb": round(alert_bytes / (1024**3), 2),
            "archive_storage_bytes": archive_bytes,
            "archive_storage_gb": round(archive_bytes / (1024**3), 2),
            "local_retention_days": self._local_days,
            "archive_retention_days": self._archive_days,
            "alert_retention_days": self._alert_days,
        }
