"""Background retention policy enforcement for recordings and alert evidence.

Periodically scans recording directories and enforces:
- Local storage: delete after ``local_retention_days``
- Archive: move to ``archive_path`` after local retention (if enabled)
- Archive cleanup: delete from archive after ``archive_retention_days``
- Alert evidence: clean up old alert snapshots/recordings
"""

from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from argus.storage.database import Database

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
    database:
        Optional database manager used to remove alert_recordings metadata when
        alert recording directories are deleted by retention.
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
        database: Database | None = None,
    ) -> None:
        self._cont_dir = Path(continuous_recording_dir)
        self._alert_dir = Path(alert_recording_dir)
        self._local_days = local_retention_days
        self._archive_enabled = archive_enabled
        self._archive_path = Path(archive_path) if archive_path else None
        self._archive_days = archive_retention_days
        self._alert_days = alert_retention_days
        self._interval_s = cleanup_interval_hours * 3600.0
        self._database = database

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
            "emergency_deleted": 0,
        }

        # Emergency disk cleanup: if free space < 1 GB, aggressively delete oldest
        emergency = self._emergency_cleanup_if_needed()
        stats["emergency_deleted"] = emergency

        stats["local_deleted"] = self._cleanup_local()
        if self._archive_enabled:
            stats["archived"] = self._archive_recordings()
            stats["archive_deleted"] = self._cleanup_archive()
        stats["alert_deleted"] = self._cleanup_alert_evidence()
        logger.info("retention.cleanup_complete", **stats)
        return stats

    def _emergency_cleanup_if_needed(self, min_free_gb: float = 1.0) -> int:
        """Delete oldest recordings if disk free space is critically low."""
        try:
            stat = shutil.disk_usage(str(self._cont_dir))
            free_gb = stat.free / (1024 ** 3)
            if free_gb >= min_free_gb:
                return 0

            logger.warning(
                "retention.emergency_cleanup",
                free_gb=round(free_gb, 2),
                threshold_gb=min_free_gb,
            )
            # Delete oldest date directories until free space > threshold
            deleted = 0
            if not self._cont_dir.exists():
                return 0
            today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            for date_dir in sorted(self._cont_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                if self._parse_date_dir(date_dir.name) is None:
                    continue
                # Never delete today's recordings even in emergency
                if date_dir.name == today:
                    continue
                try:
                    shutil.rmtree(date_dir)
                    deleted += 1
                    logger.info("retention.emergency_deleted", path=str(date_dir))
                    stat = shutil.disk_usage(str(self._cont_dir))
                    if stat.free / (1024 ** 3) >= min_free_gb:
                        break
                except OSError:
                    logger.warning("retention.emergency_delete_error", path=str(date_dir), exc_info=True)
            return deleted
        except OSError:
            return 0

    def _cleanup_local(self) -> int:
        """Delete local recordings older than retention period.

        When archiving is enabled, only delete directories that have been
        successfully archived (exist in archive_path) to prevent data loss.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self._local_days)
        if not self._archive_enabled:
            return self._delete_old_date_dirs(self._cont_dir, cutoff)

        # Archive-safe mode: only delete local dirs that exist in archive
        deleted = 0
        if not self._cont_dir.exists():
            return 0
        for date_dir in sorted(self._cont_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            dir_date = self._parse_date_dir(date_dir.name)
            if dir_date is None or dir_date >= cutoff:
                continue
            # Check that archive copy exists before deleting local
            if self._archive_path and (self._archive_path / date_dir.name).exists():
                try:
                    shutil.rmtree(date_dir)
                    deleted += 1
                except OSError:
                    logger.warning("retention.local_delete_error", path=str(date_dir), exc_info=True)
            else:
                logger.warning(
                    "retention.skip_delete_no_archive",
                    path=str(date_dir),
                    msg="Local recording past retention but archive copy missing — keeping",
                )
        return deleted

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
        return self._delete_old_alert_recording_dirs(self._alert_dir, cutoff)

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

    def _delete_old_alert_recording_dirs(self, root: Path, cutoff: datetime) -> int:
        """Delete old alert recording directories and matching DB metadata."""
        deleted = 0
        if not root.exists():
            return 0

        for date_dir in sorted(root.iterdir()):
            if not date_dir.is_dir():
                continue
            dir_date = self._parse_date_dir(date_dir.name)
            if dir_date is None or dir_date >= cutoff:
                continue

            for rec_dir in self._iter_alert_recording_dirs(date_dir):
                alert_id = self._alert_id_from_recording_dir(rec_dir)
                try:
                    if alert_id and not self._delete_alert_recording_metadata(alert_id):
                        continue
                    shutil.rmtree(rec_dir)
                    deleted += 1
                    self._remove_empty_ancestors(rec_dir.parent, stop_at=root)
                    logger.debug(
                        "retention.deleted_alert_recording_dir",
                        path=str(rec_dir),
                        alert_id=alert_id,
                    )
                except OSError:
                    logger.warning(
                        "retention.alert_recording_delete_error",
                        path=str(rec_dir),
                        alert_id=alert_id,
                        exc_info=True,
                    )

            self._remove_empty_ancestors(date_dir, stop_at=root)
        return deleted

    @staticmethod
    def _iter_alert_recording_dirs(date_dir: Path):
        """Yield alert recording directories below a date partition."""
        for camera_or_recording_dir in sorted(date_dir.iterdir()):
            if not camera_or_recording_dir.is_dir():
                continue
            if (camera_or_recording_dir / "metadata.json").exists():
                yield camera_or_recording_dir
                continue
            for rec_dir in sorted(camera_or_recording_dir.iterdir()):
                if rec_dir.is_dir():
                    yield rec_dir

    @staticmethod
    def _alert_id_from_recording_dir(rec_dir: Path) -> str | None:
        """Read alert_id from metadata.json, falling back to directory name."""
        meta_path = rec_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                alert_id = meta.get("alert_id")
                if isinstance(alert_id, str) and alert_id:
                    return alert_id
            except (OSError, json.JSONDecodeError, TypeError):
                logger.debug(
                    "retention.alert_recording_metadata_read_failed",
                    path=str(meta_path),
                    exc_info=True,
                )
        return rec_dir.name or None

    def _delete_alert_recording_metadata(self, alert_id: str) -> bool:
        if self._database is None:
            return True
        delete_fn = getattr(self._database, "delete_alert_recording_metadata", None)
        if delete_fn is None:
            return True
        try:
            delete_fn(alert_id)
            return True
        except Exception:
            logger.warning(
                "retention.alert_recording_metadata_delete_failed",
                alert_id=alert_id,
                exc_info=True,
            )
            return False

    @staticmethod
    def _remove_empty_ancestors(path: Path, stop_at: Path) -> None:
        """Remove empty directories from path upward, stopping before stop_at."""
        try:
            stop_resolved = stop_at.resolve()
        except OSError:
            stop_resolved = stop_at
        current = path
        while True:
            try:
                if current.resolve() == stop_resolved:
                    break
                current.rmdir()
            except OSError:
                break
            parent = current.parent
            if parent == current:
                break
            current = parent

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
