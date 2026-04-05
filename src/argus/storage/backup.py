"""Database and configuration backup/restore.

Provides automated SQLite backup using the backup API (online, non-blocking),
and manages backup lifecycle including rotation and restore.
"""
from __future__ import annotations

import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()


class BackupManager:
    """Manages database and configuration backups."""

    def __init__(
        self,
        database_url: str = "sqlite:///data/db/argus.db",
        backup_dir: str | Path = "data/backups",
        max_backups: int = 10,
    ):
        self._db_path = database_url.replace("sqlite:///", "")
        self._backup_dir = Path(backup_dir)
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._max_backups = max_backups

    def create_backup(self, include_models: bool = False) -> dict:
        """Create a full backup. Returns metadata dict with path, size, duration."""
        start = time.monotonic()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"argus_backup_{ts}"
        backup_path = self._backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)

        result = {"timestamp": ts, "path": str(backup_path), "files": []}

        # 1. SQLite online backup (safe even while database is in use)
        db_backup = backup_path / "argus.db"
        try:
            src = sqlite3.connect(self._db_path)
            dst = sqlite3.connect(str(db_backup))
            src.backup(dst)
            dst.close()
            src.close()
            result["files"].append(str(db_backup))
            logger.info("backup.database_ok", path=str(db_backup))
        except Exception as e:
            logger.error("backup.database_failed", error=str(e))
            result["db_error"] = str(e)

        # 2. Copy configs
        configs_src = Path("configs")
        if configs_src.exists():
            configs_dst = backup_path / "configs"
            shutil.copytree(configs_src, configs_dst, dirs_exist_ok=True)
            result["files"].append(str(configs_dst))

        # 3. Optionally copy models
        if include_models:
            for src_dir_name in ("data/models", "data/exports"):
                src_dir = Path(src_dir_name)
                if src_dir.exists():
                    dst_dir = backup_path / src_dir_name
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                    result["files"].append(str(dst_dir))

        duration = time.monotonic() - start
        result["duration_seconds"] = round(duration, 2)

        # Calculate total size
        total_size = sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file())
        result["size_bytes"] = total_size
        result["size_mb"] = round(total_size / (1024 * 1024), 2)

        logger.info("backup.complete", path=str(backup_path), size_mb=result["size_mb"], duration=f"{duration:.1f}s")

        # Rotate old backups
        self._rotate_backups()

        return result

    def restore_database(self, backup_name: str) -> bool:
        """Restore database from a backup. Returns True on success."""
        backup_path = self._backup_dir / backup_name / "argus.db"
        if not backup_path.exists():
            logger.error("restore.not_found", path=str(backup_path))
            return False

        target = Path(self._db_path)
        # Create a safety copy of current db
        safety = target.with_suffix(".db.pre_restore")
        try:
            if target.exists():
                shutil.copy2(target, safety)
            shutil.copy2(backup_path, target)
            logger.info("restore.complete", source=str(backup_path), target=str(target))
            return True
        except Exception as e:
            logger.error("restore.failed", error=str(e))
            # Attempt to restore safety copy
            if safety.exists():
                shutil.copy2(safety, target)
            return False

    def list_backups(self) -> list[dict]:
        """List all available backups with metadata."""
        backups = []
        if not self._backup_dir.exists():
            return backups

        for entry in sorted(self._backup_dir.iterdir(), reverse=True):
            if not entry.is_dir() or not entry.name.startswith("argus_backup_"):
                continue
            total_size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            has_db = (entry / "argus.db").exists()
            has_configs = (entry / "configs").exists()
            has_models = (entry / "data" / "models").exists()

            # Parse timestamp from name
            ts_str = entry.name.replace("argus_backup_", "")
            try:
                ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                created = ts.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                created = ts_str

            backups.append({
                "name": entry.name,
                "created": created,
                "size_mb": round(total_size / (1024 * 1024), 2),
                "has_db": has_db,
                "has_configs": has_configs,
                "has_models": has_models,
            })
        return backups

    def delete_backup(self, backup_name: str) -> bool:
        """Delete a specific backup."""
        path = self._backup_dir / backup_name
        if not path.exists() or not path.is_dir():
            return False
        shutil.rmtree(path)
        logger.info("backup.deleted", name=backup_name)
        return True

    def _rotate_backups(self) -> None:
        """Remove oldest backups exceeding max_backups."""
        backups = sorted(
            [d for d in self._backup_dir.iterdir() if d.is_dir() and d.name.startswith("argus_backup_")],
            key=lambda p: p.name,
        )
        while len(backups) > self._max_backups:
            oldest = backups.pop(0)
            shutil.rmtree(oldest)
            logger.info("backup.rotated", removed=oldest.name)
