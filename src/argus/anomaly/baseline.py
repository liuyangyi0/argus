"""Baseline image management.

Manages the collection, storage, and lifecycle of "normal" reference images
used to train anomaly detection models. Supports automated capture from
live cameras and versioned storage.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import structlog

from argus.storage.models import BaselineState

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

    def __init__(self, baselines_dir: str | Path, lifecycle=None):
        self.baselines_dir = Path(baselines_dir)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self._lifecycle = lifecycle

    @staticmethod
    def _resolve_current_version(base: Path) -> Path:
        """Resolve the current version directory from a base path."""
        current_marker = base / "current.txt"
        if current_marker.exists():
            version = current_marker.read_text().strip()
            version_dir = base / version
            if version_dir.is_dir():
                return version_dir
        versions = sorted(base.glob("v*"), reverse=True)
        if versions:
            return versions[0]
        return base

    @staticmethod
    def _create_version_dir(base: Path) -> tuple[Path, str]:
        """Create a new vNNN directory under base, return (path, version_name)."""
        base.mkdir(parents=True, exist_ok=True)
        existing = sorted(base.glob("v*"))
        last_num = 0
        for d in reversed(existing):
            m = re.match(r"^v(\d+)$", d.name)
            if m:
                last_num = int(m.group(1))
                break
        version = f"v{last_num + 1:03d}"
        version_dir = base / version
        version_dir.mkdir()
        return version_dir, version

    def get_baseline_dir(self, camera_id: str, zone_id: str = "default") -> Path:
        """Get the current baseline directory for a camera/zone."""
        return self._resolve_current_version(self.baselines_dir / camera_id / zone_id)

    def create_new_version(self, camera_id: str, zone_id: str = "default") -> Path:
        """Create a new baseline version directory."""
        base = self.baselines_dir / camera_id / zone_id
        version_dir, version = self._create_version_dir(base)
        if self._lifecycle:
            self._lifecycle.register_version(camera_id, zone_id, version)
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
        """Save a single frame as a baseline image.

        Raises ValueError if attempting to write to an Active baseline version
        (immutability guard for nuclear audit compliance).
        """
        if self._lifecycle:
            # Parse camera_id/zone_id/version from path
            try:
                parts = output_dir.relative_to(self.baselines_dir).parts
                if len(parts) >= 3:
                    cam_id, zone_id, version = parts[0], parts[1], parts[2]
                    ver_rec = self._lifecycle.get_version(cam_id, zone_id, version)
                    if ver_rec and ver_rec.state == BaselineState.ACTIVE:
                        raise ValueError(
                            f"Cannot write to Active baseline {cam_id}/{zone_id}/{version}. "
                            "Active baselines are immutable."
                        )
            except ValueError as e:
                if "Cannot write to Active" in str(e):
                    raise
                pass  # Path parsing failed (e.g. not relative), skip guard

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
                state = None
                if self._lifecycle and current.exists():
                    ver_rec = self._lifecycle.get_version(
                        camera_dir.name, zone_dir.name, current.name
                    )
                    state = ver_rec.state if ver_rec else None
                results.append({
                    "camera_id": camera_dir.name,
                    "zone_id": zone_dir.name,
                    "current_version": current.name if current.exists() else None,
                    "image_count": count,
                    "path": str(current),
                    "state": state,
                })
        return results

    def diversity_select(
        self,
        image_dir: Path,
        target_count: int,
        feature_size: tuple[int, int] = (64, 64),
    ) -> list[Path]:
        """Select the most diverse subset of images using k-center greedy.

        Uses color histogram features in LAB space for perceptual diversity.
        Returns paths of selected images, sorted by file name.

        Algorithm (Sener & Savarese, ICLR 2018 simplified):
        1. Compute feature vector for each image (LAB color histogram)
        2. Start with first image as seed
        3. Iteratively add the image farthest from all already-selected images
        4. Stop at target_count
        """
        image_paths = sorted(
            list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        )

        if len(image_paths) <= target_count:
            return image_paths

        # Step 1: Compute features (LAB color histograms, 3 channels x 32 bins = 96-dim)
        features = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                features.append(np.zeros(96))
                continue
            img = cv2.resize(img, feature_size)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            hist = []
            for ch in range(3):
                h = cv2.calcHist([lab], [ch], None, [32], [0, 256])
                h = h.flatten() / (h.sum() + 1e-8)
                hist.append(h)
            features.append(np.concatenate(hist))

        features = np.array(features)  # (N, 96)

        # Step 2-4: K-center greedy
        n = len(features)
        selected = [0]
        min_distances = np.full(n, np.inf)

        for _ in range(target_count - 1):
            last = features[selected[-1]]
            dists = np.linalg.norm(features - last, axis=1)
            min_distances = np.minimum(min_distances, dists)
            min_distances[selected] = -1  # exclude already selected

            next_idx = int(np.argmax(min_distances))
            selected.append(next_idx)

        return [image_paths[i] for i in sorted(selected)]

    def cleanup_old_versions(
        self, camera_id: str, zone_id: str = "default", keep: int = 3
    ) -> int:
        """Remove old baseline versions, keeping the N most recent."""
        base = self.baselines_dir / camera_id / zone_id
        versions = sorted(base.glob("v*"))

        if len(versions) <= keep:
            return 0

        to_remove = versions[:-keep]
        retired_versions: set[str] = set()
        if self._lifecycle:
            all_recs = self._lifecycle.get_versions(camera_id, zone_id)
            retired_versions = {r.version for r in all_recs if r.state == BaselineState.RETIRED}
        removed = 0
        for v in to_remove:
            if v.name in retired_versions:
                logger.debug("baseline.skip_retired", camera_id=camera_id, version=v.name)
                continue
            shutil.rmtree(v)
            removed += 1
            logger.info("baseline.removed_old", camera_id=camera_id, zone_id=zone_id, version=v.name)

        return removed

    # ── Camera Group Methods ──

    def get_group_baseline_dir(self, group_id: str, zone_id: str = "default") -> Path:
        """Get the current baseline directory for a camera group."""
        return self._resolve_current_version(
            self.baselines_dir / "_groups" / group_id / zone_id
        )

    def create_group_version(self, group_id: str, zone_id: str = "default") -> Path:
        """Create a new version directory for a camera group baseline."""
        base = self.baselines_dir / "_groups" / group_id / zone_id
        version_dir, version = self._create_version_dir(base)
        if self._lifecycle:
            self._lifecycle.register_version(
                camera_id=f"group:{group_id}",
                zone_id=zone_id,
                version=version,
                group_id=group_id,
            )
        return version_dir

    def merge_camera_baselines(
        self,
        group_id: str,
        camera_ids: list[str],
        zone_id: str = "default",
        target_count: int | None = None,
    ) -> Path:
        """Merge baselines from multiple cameras into a group version.

        1. Create new group version directory
        2. Copy images from each camera's current baseline
        3. Optionally apply diversity_select for deduplication
        4. Return the group version directory path
        """
        version_dir = self.create_group_version(group_id, zone_id)

        # Collect images from all member cameras
        idx = 0
        for cam_id in camera_ids:
            cam_dir = self.get_baseline_dir(cam_id, zone_id)
            if not cam_dir.is_dir():
                continue
            for img in sorted(
                list(cam_dir.glob("*.png")) + list(cam_dir.glob("*.jpg"))
            ):
                dst = version_dir / f"group_{cam_id}_{idx:05d}{img.suffix}"
                shutil.copy2(str(img), str(dst))
                idx += 1

        # Optionally deduplicate
        if target_count and idx > target_count:
            selected = set(self.diversity_select(version_dir, target_count))
            for img in version_dir.iterdir():
                if img.is_file() and img not in selected:
                    img.unlink()

        # Update image count in lifecycle
        final_count = len(
            list(version_dir.glob("*.png")) + list(version_dir.glob("*.jpg"))
        )
        if self._lifecycle:
            self._lifecycle.register_version(
                camera_id=f"group:{group_id}",
                zone_id=zone_id,
                version=version_dir.name,
                image_count=final_count,
                group_id=group_id,
            )

        logger.info(
            "baseline.group_merged",
            group_id=group_id,
            cameras=camera_ids,
            total_images=final_count,
            version=version_dir.name,
        )

        return version_dir
