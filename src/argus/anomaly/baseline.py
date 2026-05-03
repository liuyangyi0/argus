"""Baseline image management.

Manages the collection, storage, and lifecycle of "normal" reference images
used to train anomaly detection models. Supports automated capture from
live cameras and versioned storage.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
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
        """Resolve the current version directory from a base path.

        Returns the versioned subdirectory (e.g. ``base/v001``), never
        the bare *base* directory itself — that would cause the trainer
        to scan the wrong location and report misleading "图片不足" errors.
        """
        current_marker = base / "current.txt"
        if current_marker.exists():
            version = current_marker.read_text().strip()
            version_dir = base / version
            if version_dir.is_dir():
                return version_dir
        versions = sorted(base.glob("v*"), reverse=True)
        if versions:
            return versions[0]
        v001 = base / "v001"
        if not v001.exists():
            v001.mkdir(parents=True, exist_ok=True)
        return v001

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
        base.mkdir(parents=True, exist_ok=True)
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

    def resolve_dataset_dirs(self, selection) -> list[Path]:
        """Resolve a DatasetSelection to absolute baseline directories (痛点 2).

        Each selection item must point at an existing version directory under
        ``baselines_dir/<camera_id>/<zone_id>/<version>``; missing or empty
        directories raise ``FileNotFoundError`` so the caller fails loudly
        rather than silently producing an empty training set.
        """
        if selection is None:
            raise ValueError("DatasetSelection is required")
        resolved: list[Path] = []
        for item in selection.items:
            path = self.baselines_dir / item.camera_id / item.zone_id / item.version
            if not path.is_dir():
                raise FileNotFoundError(
                    f"Baseline version not found: {item.camera_id}/{item.zone_id}/{item.version}"
                )
            if not (any(path.glob("*.png")) or any(path.glob("*.jpg"))):
                raise FileNotFoundError(
                    f"Baseline version has no images: {item.camera_id}/{item.zone_id}/{item.version}"
                )
            resolved.append(path)
        return resolved

    @staticmethod
    def count_images_multi(dirs: list[Path]) -> int:
        """Sum image counts across multiple baseline directories."""
        total = 0
        for path in dirs:
            if not path.is_dir():
                continue
            total += len(list(path.glob("*.png"))) + len(list(path.glob("*.jpg")))
        return total

    # ── Per-image CRUD (for manual baseline maintenance) ──

    # Accept only baseline_NNNNN.png / .jpg / .jpeg filenames. The 5-digit
    # prefix matches what save_frame() writes; the optional fp_ prefix below
    # is covered by the broader regex because merge_fp_into_baseline uses
    # its own naming. We also accept legacy "img_NNNNN" / "fp_*" patterns
    # so the UI can still enumerate/delete older files — but uploads always
    # use the canonical baseline_NNNNN.{ext} name.
    _IMAGE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.(png|jpg|jpeg)$")
    _CANONICAL_FILENAME_RE = re.compile(r"^baseline_\d{5}\.(png|jpg|jpeg)$")
    _ALLOWED_EXTS = frozenset({"png", "jpg", "jpeg"})

    @staticmethod
    def _safe_filename(filename: str) -> bool:
        """Reject filenames that could escape the version directory."""
        if not filename or len(filename) > 128:
            return False
        if "/" in filename or "\\" in filename or "\x00" in filename:
            return False
        if ".." in filename:
            return False
        return bool(BaselineManager._IMAGE_FILENAME_RE.match(filename))

    def _version_dir(self, camera_id: str, version: str, zone_id: str = "default") -> Path:
        """Resolve the on-disk directory for a specific version (no legacy flat fallback)."""
        return self.baselines_dir / camera_id / zone_id / version

    def _is_active_version(self, camera_id: str, version: str, zone_id: str = "default") -> bool:
        """True if the version is in ACTIVE lifecycle state."""
        if not self._lifecycle:
            return False
        rec = self._lifecycle.get_version(camera_id, zone_id, version)
        return bool(rec and rec.state == BaselineState.ACTIVE)

    def list_images(
        self, camera_id: str, version: str, zone_id: str = "default"
    ) -> list[dict]:
        """List images in a specific baseline version directory.

        Returns:
            List of ``{filename, size_bytes, created_at}`` sorted by filename.
            Empty list if the version directory doesn't exist.
        """
        version_dir = self._version_dir(camera_id, version, zone_id)
        if not version_dir.is_dir():
            return []

        results: list[dict] = []
        for entry in sorted(version_dir.iterdir()):
            if not entry.is_file():
                continue
            suffix = entry.suffix.lower().lstrip(".")
            if suffix not in self._ALLOWED_EXTS:
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            results.append({
                "filename": entry.name,
                "size_bytes": int(stat.st_size),
                "created_at": created_at,
            })
        return results

    def delete_image(
        self,
        camera_id: str,
        version: str,
        filename: str,
        zone_id: str = "default",
    ) -> bool:
        """Delete one image file from a baseline version.

        Safety:
            - Filename must match the allowed pattern (no traversal).
            - ACTIVE versions are immutable — deletion is rejected.

        Returns:
            True on successful deletion; False if filename is invalid,
            file doesn't exist, or the version is ACTIVE.
        """
        if not self._safe_filename(filename):
            logger.warning(
                "baseline.image_delete_rejected_unsafe",
                camera_id=camera_id,
                version=version,
                filename=filename,
            )
            return False

        if self._is_active_version(camera_id, version, zone_id):
            logger.warning(
                "baseline.image_delete_rejected_active",
                camera_id=camera_id,
                version=version,
            )
            return False

        version_dir = self._version_dir(camera_id, version, zone_id)
        target = version_dir / filename
        # Containment check — resolve and confirm parent is the version dir.
        try:
            resolved_target = target.resolve()
            resolved_parent = version_dir.resolve()
        except OSError:
            return False
        if resolved_target.parent != resolved_parent:
            logger.warning(
                "baseline.image_delete_rejected_outside_version_dir",
                camera_id=camera_id,
                version=version,
                filename=filename,
            )
            return False

        if not target.is_file():
            return False

        try:
            target.unlink()
        except OSError as exc:
            logger.warning(
                "baseline.image_delete_failed",
                camera_id=camera_id,
                version=version,
                filename=filename,
                error=str(exc),
            )
            return False

        logger.info(
            "baseline.image_deleted",
            camera_id=camera_id,
            zone_id=zone_id,
            version=version,
            filename=filename,
        )
        return True

    def add_image_from_bytes(
        self,
        camera_id: str,
        version: str,
        data: bytes,
        ext: str,
        zone_id: str = "default",
    ) -> str:
        """Write raw image bytes as a new baseline image.

        Args:
            data: Raw image bytes (png/jpg/jpeg).
            ext: Extension without leading dot, e.g. ``"png"``, ``"jpg"``.

        Returns:
            The new filename (e.g. ``"baseline_00042.png"``).

        Raises:
            ValueError: If the version is ACTIVE, ext is invalid, or the
                byte stream doesn't decode as a real image.
        """
        ext_norm = ext.lower().lstrip(".")
        if ext_norm == "jpeg":
            ext_norm = "jpg"
        if ext_norm not in {"png", "jpg"}:
            raise ValueError(f"不支持的图片格式: {ext}")

        if not data:
            raise ValueError("图片内容为空")

        if self._is_active_version(camera_id, version, zone_id):
            raise ValueError("生产中的基线版本不可上传图片，请先退役")

        version_dir = self._version_dir(camera_id, version, zone_id)
        if not version_dir.is_dir():
            raise ValueError(f"基线版本不存在: {camera_id}/{zone_id}/{version}")

        # Verify the payload decodes as a real image before writing — rejects
        # mislabeled uploads, corrupt files, and non-image junk.
        arr = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("图片内容损坏或格式不受支持")

        # Find next index — scan existing baseline_NNNNN.* files and pick max + 1.
        max_idx = -1
        for existing in version_dir.glob("baseline_*.*"):
            suffix = existing.suffix.lower().lstrip(".")
            if suffix not in self._ALLOWED_EXTS:
                continue
            stem = existing.stem  # e.g. "baseline_00042"
            try:
                idx = int(stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if idx > max_idx:
                max_idx = idx
        next_idx = max_idx + 1
        if next_idx > 99999:
            raise ValueError("基线版本图片数量已达上限")

        filename = f"baseline_{next_idx:05d}.{ext_norm}"
        target = version_dir / filename
        # Atomic write: to temp then rename.
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)

        logger.info(
            "baseline.image_added",
            camera_id=camera_id,
            zone_id=zone_id,
            version=version,
            filename=filename,
            size_bytes=len(data),
        )
        return filename

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
        protected_versions: set[str] = set()
        if self._lifecycle:
            all_recs = self._lifecycle.get_versions(camera_id, zone_id)
            # Protect ACTIVE, VERIFIED, and RETIRED versions from deletion
            protected_states = {BaselineState.ACTIVE, BaselineState.VERIFIED, BaselineState.RETIRED}
            protected_versions = {r.version for r in all_recs if r.state in protected_states}
        removed = 0
        for v in to_remove:
            if v.name in protected_versions:
                logger.debug("baseline.skip_protected", camera_id=camera_id, version=v.name)
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
        """Create a new version directory for a camera group baseline.

        Note: Does NOT auto-register with lifecycle. Callers (e.g.
        merge_camera_baselines) should register after populating the version
        so image_count is accurate.
        """
        base = self.baselines_dir / "_groups" / group_id / zone_id
        version_dir, _version = self._create_version_dir(base)
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
