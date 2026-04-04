"""Baseline image management.

Manages the collection, storage, and lifecycle of "normal" reference images
used to train anomaly detection models. Supports automated capture from
live cameras and versioned storage.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


class BaselineManager:
    """Manages baseline (normal) images for anomaly model training.

    Baselines are organized by camera and zone:
        baselines_dir/
            cam_01/
                zone_a/
                    v001/
                        baseline_00000.png
                        ...
                    v002/
                        ...
                    current -> v002  (symlink or marker file)
    """

    def __init__(self, baselines_dir: str | Path):
        self.baselines_dir = Path(baselines_dir)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    def get_baseline_dir(self, camera_id: str, zone_id: str = "default") -> Path:
        """Get the current baseline directory for a camera/zone."""
        base = self.baselines_dir / camera_id / zone_id
        current_marker = base / "current.txt"

        if current_marker.exists():
            version = current_marker.read_text().strip()
            version_dir = base / version
            if version_dir.is_dir():
                return version_dir

        # Fallback: find latest version directory
        versions = sorted(base.glob("v*"), reverse=True)
        if versions:
            return versions[0]

        return base

    def create_new_version(self, camera_id: str, zone_id: str = "default") -> Path:
        """Create a new baseline version directory."""
        base = self.baselines_dir / camera_id / zone_id
        base.mkdir(parents=True, exist_ok=True)

        # Find next version number
        existing = sorted(base.glob("v*"))
        if existing:
            last_num = int(existing[-1].name[1:])
            version = f"v{last_num + 1:03d}"
        else:
            version = "v001"

        version_dir = base / version
        version_dir.mkdir()
        return version_dir

    def set_current_version(self, camera_id: str, zone_id: str, version: str) -> None:
        """Set the current active baseline version."""
        base = self.baselines_dir / camera_id / zone_id
        marker = base / "current.txt"
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(version)
        tmp.replace(marker)
        logger.info("baseline.version_set", camera_id=camera_id, zone_id=zone_id, version=version)

    def save_frame(self, frame: np.ndarray, output_dir: Path, index: int) -> Path:
        """Save a single frame as a baseline image."""
        filename = output_dir / f"baseline_{index:05d}.png"
        cv2.imwrite(str(filename), frame)
        return filename

    def count_images(self, camera_id: str, zone_id: str = "default") -> int:
        """Count baseline images for a camera/zone."""
        base_dir = self.get_baseline_dir(camera_id, zone_id)
        if not base_dir.is_dir():
            return 0
        return len(list(base_dir.glob("*.png"))) + len(list(base_dir.glob("*.jpg")))

    def get_all_baselines(self) -> list[dict]:
        """List all baselines with metadata."""
        results = []
        for camera_dir in sorted(self.baselines_dir.iterdir()):
            if not camera_dir.is_dir():
                continue
            for zone_dir in sorted(camera_dir.iterdir()):
                if not zone_dir.is_dir():
                    continue
                current = self.get_baseline_dir(camera_dir.name, zone_dir.name)
                count = self.count_images(camera_dir.name, zone_dir.name)
                results.append({
                    "camera_id": camera_dir.name,
                    "zone_id": zone_dir.name,
                    "current_version": current.name if current.exists() else None,
                    "image_count": count,
                    "path": str(current),
                })
        return results

    def cleanup_old_versions(
        self, camera_id: str, zone_id: str = "default", keep: int = 3
    ) -> int:
        """Remove old baseline versions, keeping the N most recent."""
        base = self.baselines_dir / camera_id / zone_id
        versions = sorted(base.glob("v*"))

        if len(versions) <= keep:
            return 0

        to_remove = versions[:-keep]
        removed = 0
        for v in to_remove:
            shutil.rmtree(v)
            removed += 1
            logger.info("baseline.removed_old", camera_id=camera_id, zone_id=zone_id, version=v.name)

        return removed
