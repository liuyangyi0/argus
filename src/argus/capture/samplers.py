from __future__ import annotations

import importlib
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()

# timm model name mapping for DINOv2 backbones
_TIMM_MODEL_MAP: dict[str, str] = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
}

# CLS token feature dimensions per backbone
_FEATURE_DIMS: dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}

# ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_ACTIVE_SAMPLER_DEPENDENCIES: tuple[str, ...] = ("faiss", "timm", "torch")


def get_active_sampler_unavailable_reason() -> str | None:
    """Return a human-readable reason when active sampling dependencies are unavailable."""
    missing: list[str] = []
    for module_name in _ACTIVE_SAMPLER_DEPENDENCIES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)

    if not missing:
        return None

    return f"缺少依赖: {', '.join(missing)}"


def _resolve_model_image_size(model: Any, fallback: int) -> int:
    """Resolve the preferred square input size from a timm model config."""
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", None) or {}
    input_size = cfg.get("input_size")
    if isinstance(input_size, (tuple, list)) and len(input_size) >= 3:
        height = int(input_size[-2])
        width = int(input_size[-1])
        if height > 0 and width > 0 and height == width:
            return height
    return int(fallback)


class BaseSampler(ABC):
    """Abstract base class for frame sampling strategies."""

    @abstractmethod
    def should_accept(self, frame: np.ndarray) -> tuple[bool, dict]:
        """Check if frame should be included.

        Returns:
            Tuple of (accepted, metadata_dict).
        """

    @abstractmethod
    def get_sleep_interval(self) -> float:
        """Seconds to sleep before next frame grab."""

    def on_frame_saved(self, frame: np.ndarray, index: int) -> None:
        """Hook called after frame is accepted and saved."""


class UniformSampler(BaseSampler):
    """Sample frames at uniform time intervals."""

    def __init__(self, duration_hours: float, target_frames: int) -> None:
        self.duration_hours = duration_hours
        self.target_frames = target_frames

    def should_accept(self, frame: np.ndarray) -> tuple[bool, dict]:
        return True, {"strategy": "uniform"}

    def get_sleep_interval(self) -> float:
        return self.duration_hours * 3600 / self.target_frames


class ActiveSampler(BaseSampler):
    """Diversity-aware sampler using DINOv2 features and FAISS cosine similarity."""

    def __init__(
        self,
        diversity_threshold: float = 0.3,
        backbone: str = "dinov2_vits14",
        image_size: int = 224,
        sleep_interval_seconds: float = 1.0,
        cpu_threads: int = 1,
    ) -> None:
        self.diversity_threshold = diversity_threshold
        self.backbone = backbone
        self.image_size = image_size
        self.sleep_interval_seconds = max(0.0, float(sleep_interval_seconds))
        self.cpu_threads = max(1, int(cpu_threads))
        self._feature_dim = _FEATURE_DIMS[backbone]

        # Lazy-loaded on first should_accept call
        self._model: Any = None
        self._faiss_index: Any = None
        self._last_feature: np.ndarray | None = None

    def _ensure_model(self) -> None:
        """Lazy-load the DINOv2 model and FAISS index on first use."""
        if self._model is not None:
            return

        unavailable_reason = get_active_sampler_unavailable_reason()
        if unavailable_reason is not None:
            raise RuntimeError(f"高级采样不可用: {unavailable_reason}")

        import faiss
        import timm
        import torch

        timm_name = _TIMM_MODEL_MAP[self.backbone]
        logger.info(
            "loading_dinov2_model", timm_name=timm_name, backbone=self.backbone
        )
        self._model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        self._model.eval()
        self._torch = torch
        resolved_image_size = _resolve_model_image_size(self._model, self.image_size)
        if resolved_image_size != self.image_size:
            logger.info(
                "active_sampler_image_size_adjusted",
                requested_image_size=self.image_size,
                resolved_image_size=resolved_image_size,
                backbone=self.backbone,
            )
            self.image_size = resolved_image_size

        try:
            torch.set_num_threads(self.cpu_threads)
        except Exception:
            logger.debug("active_sampler_set_num_threads_failed", exc_info=True)

        if hasattr(faiss, "omp_set_num_threads"):
            try:
                faiss.omp_set_num_threads(self.cpu_threads)
            except Exception:
                logger.debug("active_sampler_set_faiss_threads_failed", exc_info=True)

        self._faiss_index = faiss.IndexFlatIP(self._feature_dim)
        logger.info(
            "active_sampler_ready",
            feature_dim=self._feature_dim,
            threshold=self.diversity_threshold,
            sleep_interval_seconds=self.sleep_interval_seconds,
            cpu_threads=self.cpu_threads,
        )

    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame and extract CLS token features.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            L2-normalized feature vector of shape (1, feature_dim) as float32 numpy.
        """
        import cv2

        torch = self._torch

        # Resize and convert BGR -> RGB
        resized = cv2.resize(frame, (self.image_size, self.image_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalization
        normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD

        # HWC -> CHW -> NCHW tensor
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()

        with torch.no_grad():
            features = self._model(tensor)  # (1, feature_dim)

        vec = features.cpu().numpy().astype(np.float32)

        # L2-normalize for cosine similarity via IndexFlatIP
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / (norm + 1e-8)

        return vec

    def should_accept(self, frame: np.ndarray) -> tuple[bool, dict]:
        self._ensure_model()

        vec = self._extract_features(frame)
        self._last_feature = vec

        n = self._faiss_index.ntotal
        if n == 0:
            logger.debug("active_sampler_first_frame")
            return True, {
                "strategy": "active",
                "diversity_score": 1.0,
                "index_size": 0,
            }

        similarities, _ = self._faiss_index.search(vec, k=1)
        similarity = float(similarities[0, 0])
        diversity_score = 1.0 - similarity

        accepted = diversity_score > self.diversity_threshold
        logger.debug(
            "active_sampler_decision",
            accepted=accepted,
            diversity_score=round(diversity_score, 4),
            threshold=self.diversity_threshold,
            index_size=n,
        )
        return accepted, {
            "strategy": "active",
            "diversity_score": diversity_score,
            "index_size": n,
        }

    def on_frame_saved(self, frame: np.ndarray, index: int) -> None:
        if self._last_feature is not None and self._faiss_index is not None:
            self._faiss_index.add(self._last_feature)
            logger.debug(
                "active_sampler_indexed",
                frame_index=index,
                index_size=self._faiss_index.ntotal,
            )

    def get_sleep_interval(self) -> float:
        return self.sleep_interval_seconds


class ScheduledSampler(BaseSampler):
    """Time-window sampler with per-period quotas and optional inner sampler."""

    def __init__(
        self,
        schedule_periods: dict[str, tuple[int, int]],
        frames_per_period: int,
        inner_sampler: BaseSampler | None = None,
        duration_hours: float = 72.0,
    ) -> None:
        self.schedule_periods = schedule_periods
        self.frames_per_period = frames_per_period
        self.inner_sampler = inner_sampler
        self.duration_hours = duration_hours

        # Quota tracker: {period_name: frames_collected}
        self._quota: dict[str, int] = {name: 0 for name in schedule_periods}

    def _current_period(self) -> str | None:
        """Return the name of the currently active period, or None."""
        hour = datetime.now().hour
        for name, (start, end) in self.schedule_periods.items():
            if start <= end:
                # Normal range, e.g. (5, 8) means hours 5, 6, 7
                if start <= hour < end:
                    return name
            else:
                # Overnight range, e.g. (22, 2) means hours 22, 23, 0, 1
                if hour >= start or hour < end:
                    return name
        return None

    def _period_duration_hours(self, name: str) -> float:
        """Compute the duration of a period in hours."""
        start, end = self.schedule_periods[name]
        if start <= end:
            return float(end - start)
        else:
            return float(24 - start + end)

    def _seconds_until_next_period(self) -> float:
        """Compute seconds until the start of the next scheduled period."""
        now = datetime.now()
        current_hour = now.hour
        current_minutes_into_hour = now.minute * 60 + now.second

        best = float("inf")
        for start, _ in self.schedule_periods.values():
            if start > current_hour:
                wait = (start - current_hour) * 3600 - current_minutes_into_hour
            elif start == current_hour:
                # Period starts this hour; if we're not in it, next occurrence
                # is in 24h
                wait = 24 * 3600 - current_minutes_into_hour
            else:
                wait = (24 - current_hour + start) * 3600 - current_minutes_into_hour
            if wait < 0:
                wait += 24 * 3600
            if wait < best:
                best = wait

        return best if best != float("inf") else 60.0

    def should_accept(self, frame: np.ndarray) -> tuple[bool, dict]:
        period = self._current_period()

        if period is None:
            return False, {"strategy": "scheduled", "reason": "outside_window"}

        if self._quota[period] >= self.frames_per_period:
            return False, {
                "strategy": "scheduled",
                "reason": "quota_full",
                "period": period,
            }

        if self.inner_sampler is not None:
            accepted, inner_meta = self.inner_sampler.should_accept(frame)
            meta = {
                "strategy": "scheduled",
                "period": period,
                "quota_used": self._quota[period],
                "quota_limit": self.frames_per_period,
                **inner_meta,
            }
            return accepted, meta

        return True, {
            "strategy": "scheduled",
            "period": period,
            "quota_used": self._quota[period],
            "quota_limit": self.frames_per_period,
        }

    def on_frame_saved(self, frame: np.ndarray, index: int) -> None:
        period = self._current_period()
        if period is not None:
            self._quota[period] += 1
            logger.debug(
                "scheduled_sampler_saved",
                period=period,
                quota_used=self._quota[period],
                quota_limit=self.frames_per_period,
            )
        if self.inner_sampler is not None:
            self.inner_sampler.on_frame_saved(frame, index)

    def get_sleep_interval(self) -> float:
        period = self._current_period()
        if period is not None:
            duration_h = self._period_duration_hours(period)
            return duration_h * 3600 / self.frames_per_period

        wait = self._seconds_until_next_period()
        # Cap at 60s to allow periodic re-checking
        return min(wait, 60.0)
