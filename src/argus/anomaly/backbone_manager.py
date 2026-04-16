"""Backbone Manager: shared DINOv2 backbone with per-camera heads.

Loads the backbone once and shares it across all cameras. Per-camera heads
are small (few MB) and loaded independently. Backbone upgrade atomically
swaps for all cameras.
"""

from __future__ import annotations

import threading
from pathlib import Path

import structlog

from argus.storage.release_pipeline import BackboneIncompatibleError

logger = structlog.get_logger()


class BackboneManager:
    """Manages a single shared DINOv2 backbone for all cameras.

    Thread-safe singleton pattern. The backbone is loaded once and its
    features are shared. Upgrade swaps atomically.
    """

    _instance: BackboneManager | None = None
    _creation_lock = threading.Lock()

    def __init__(self):
        self._backbone = None
        self._backbone_version: str | None = None
        self._backbone_path: Path | None = None
        self._lock = threading.Lock()
        self._loaded = False

    @classmethod
    def get_instance(cls) -> BackboneManager:
        """Get or create the singleton BackboneManager."""
        if cls._instance is not None:
            return cls._instance
        with cls._creation_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def load(self, backbone_path: Path, backbone_version: str) -> bool:
        """Load a DINOv2 backbone. Returns True on success."""
        try:
            import torch

            logger.info(
                "backbone_manager.loading",
                path=str(backbone_path),
                version=backbone_version,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if backbone_path.suffix in (".bin", ".pt", ".pth"):
                backbone = torch.load(backbone_path, map_location=device, weights_only=False)
            else:
                # torch.hub.load always loads to CPU; move after
                backbone = torch.hub.load(
                    "facebookresearch/dinov2", backbone_version.split("-")[0],
                )
                backbone = backbone.to(device)
            backbone.eval()

            with self._lock:
                self._backbone = backbone
                self._backbone_version = backbone_version
                self._backbone_path = backbone_path
                self._loaded = True

            logger.info("backbone_manager.loaded", version=backbone_version, device=device)
            return True

        except Exception as e:
            logger.error("backbone_manager.load_failed", error=str(e))
            return False

    def upgrade(self, new_backbone_path: Path, new_version: str) -> bool:
        """Upgrade backbone atomically. Old backbone kept until swap succeeds.

        All cameras automatically use the new backbone on their next frame.
        """
        with self._lock:
            old_backbone = self._backbone
            old_version = self._backbone_version

        success = self.load(new_backbone_path, new_version)
        if not success and old_backbone is not None:
            with self._lock:
                self._backbone = old_backbone
                self._backbone_version = old_version
            logger.warning("backbone_manager.upgrade_rollback", version=old_version)

        return success

    def get_backbone(self):
        """Get the loaded backbone model. Thread-safe."""
        with self._lock:
            return self._backbone

    @property
    def version(self) -> str | None:
        return self._backbone_version

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def validate_head_compatibility(self, head_backbone_ref: str | None) -> None:
        """Validate a head is compatible with the loaded backbone.

        Raises BackboneIncompatibleError if mismatched.
        """
        if head_backbone_ref is None:
            return

        if self._backbone_version is None:
            raise BackboneIncompatibleError(
                f"Head requires backbone {head_backbone_ref}, but no backbone is loaded"
            )

        if head_backbone_ref != self._backbone_version:
            raise BackboneIncompatibleError(
                f"Head requires backbone {head_backbone_ref}, "
                f"but loaded backbone is {self._backbone_version}"
            )


class HeadDetector:
    """Lightweight per-camera anomaly head that uses shared backbone features.

    Only loads the small head weights (few MB). Relies on BackboneManager
    for feature extraction.
    """

    def __init__(
        self,
        head_path: Path,
        camera_id: str,
        backbone_ref: str | None = None,
        threshold: float = 0.5,
    ):
        self._head_path = head_path
        self._camera_id = camera_id
        self._backbone_ref = backbone_ref
        self._threshold = threshold
        self._head = None
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> bool:
        """Load the head weights. Validates backbone compatibility first."""
        try:
            manager = BackboneManager.get_instance()
            manager.validate_head_compatibility(self._backbone_ref)

            import torch

            head_weights = torch.load(
                self._head_path, map_location="cpu", weights_only=False,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(head_weights, "to") and hasattr(head_weights, "eval"):
                head_weights = head_weights.to(device).eval()

            with self._lock:
                self._head = head_weights
                self._loaded = True

            logger.info(
                "head_detector.loaded",
                camera_id=self._camera_id,
                path=str(self._head_path),
                device=device,
            )
            return True

        except BackboneIncompatibleError:
            raise
        except Exception as e:
            logger.error(
                "head_detector.load_failed",
                camera_id=self._camera_id,
                error=str(e),
            )
            return False

    def hot_reload(self, new_head_path: Path, backbone_ref: str | None = None) -> bool:
        """Hot-reload the head weights. Atomic swap."""
        manager = BackboneManager.get_instance()
        manager.validate_head_compatibility(backbone_ref or self._backbone_ref)

        try:
            import torch

            new_head = torch.load(
                new_head_path, map_location="cpu", weights_only=False,
            )

            with self._lock:
                self._head = new_head
                self._head_path = new_head_path
                if backbone_ref:
                    self._backbone_ref = backbone_ref

            logger.info(
                "head_detector.hot_reloaded",
                camera_id=self._camera_id,
                path=str(new_head_path),
            )
            return True

        except Exception as e:
            logger.error(
                "head_detector.hot_reload_failed",
                camera_id=self._camera_id,
                error=str(e),
            )
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def backbone_ref(self) -> str | None:
        return self._backbone_ref
