"""Model training orchestration.

Coordinates the full training pipeline: validate baselines, split train/val,
train model, validate output, evaluate on validation set, generate quality
report, recommend threshold, and record results.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np
import structlog

from argus.anomaly import _trainer_export as _te
from argus.anomaly.artifacts import (
    artifact_metadata,
    exported_artifact,
    inherited_baseline_quality,
    new_artifact_run,
    training_directory,
    write_artifact_manifest,
)
from argus.anomaly.baseline import BaselineManager
from argus.anomaly.baseline_quality import FixedSceneQualityContext, inspect_baseline_quality
from argus.anomaly.scene_training import (
    SceneTrainingInput,
    revalidate_scene_training_input,
)
from argus.core.error_channel import (
    SEVERITY_ERROR,
    get_error_channel,
)
from argus.storage.models import BaselineState

if TYPE_CHECKING:
    from argus.anomaly.dataset_selection import DatasetSelection


_IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")
_SCENE_PROFILE_BASELINE_SCHEMAS = frozenset(
    {
        "argus.scene-profile-baseline.v2",
        "argus.scene-profile-baseline.v3",
    }
)
_SCENE_PROFILE_SPLIT_POLICY = (
    "complete_operator_declared_source_session_grouping"
)
_SCENE_PROFILE_SPLIT_ASSIGNMENT = (
    "lexicographically_greatest_declared_session_id_is_calibration; "
    "all_other_declared_sessions_are_train"
)
_SCENE_PROFILE_SPLIT_ROLES = ("train", "calibration", "diagnostic_only")


@dataclass(frozen=True)
class _ManifestSample:
    path: Path
    sha256: str


def _adjacent_baseline_manifest(directory: Path) -> Path:
    return directory.parent / "manifest.json"


class _DatasetMerger:
    """Symlink (fallback to copy) multiple baseline dirs into one temp root.

    Anomalib's Folder datamodule expects a single root, so multi-version
    training (痛点 2) merges all selected versions into a tmp dir before
    handing it off. Filenames carry a ``{camera}_{zone}_{version}_`` prefix
    to avoid collisions when two captures use the same per-frame index.

    Win11 普通用户态 ``os.symlink`` raises OSError(WinError 1314) without
    Developer Mode; in that case we fall back to ``shutil.copy2`` for the
    rest of the run. Cleanup is guaranteed via ``__exit__``.
    """

    def __init__(self, items: Sequence[tuple[str, Path]]) -> None:
        # items: list of (label_prefix, source_dir) tuples
        self._items = list(items)
        self._tmp: Path | None = None
        self._used_copy = False
        self._linked = 0
        self._copied = 0

    def __enter__(self) -> Path:
        manifest_backed = [
            (label, _adjacent_baseline_manifest(src_dir))
            for label, src_dir in self._items
            if (
                _adjacent_baseline_manifest(src_dir).exists()
                or _adjacent_baseline_manifest(src_dir).is_symlink()
            )
        ]
        if manifest_backed:
            sources = ", ".join(
                f"{label}:{manifest_path}"
                for label, manifest_path in manifest_backed
            )
            raise ValueError(
                "manifest-backed scene-profile baselines cannot be flattened "
                "by _DatasetMerger because that would discard the authoritative "
                f"whole-session split; train them directly instead ({sources})"
            )
        self._tmp = Path(tempfile.mkdtemp(prefix="argus_train_"))
        symlink_supported = True
        for label, src_dir in self._items:
            for pattern in _IMAGE_EXTS:
                for img in src_dir.glob(pattern):
                    target = self._tmp / f"{label}_{img.name}"
                    if symlink_supported:
                        try:
                            os.symlink(img, target)
                            self._linked += 1
                            continue
                        except (OSError, NotImplementedError):
                            # Win11 user-mode falls here; switch to copy for
                            # ALL subsequent files (no point retrying).
                            symlink_supported = False
                            self._used_copy = True
                    try:
                        shutil.copy2(img, target)
                        self._copied += 1
                    except OSError as e:
                        logger.warning(
                            "dataset_merger.skip_failed_image",
                            path=str(img),
                            error=str(e),
                        )
        logger.info(
            "dataset_merger.merged",
            sources=len(self._items),
            linked=self._linked,
            copied=self._copied,
            tmp=str(self._tmp),
        )
        return self._tmp

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tmp is not None and self._tmp.exists():
            shutil.rmtree(self._tmp, ignore_errors=True)


def _build_merger_items(
    selection: "DatasetSelection",
    resolved_dirs: list[Path],
) -> list[tuple[str, Path]]:
    """Pair each resolved dir with a unique filename prefix from selection."""
    items: list[tuple[str, Path]] = []
    for sel_item, src_dir in zip(selection.items, resolved_dirs):
        label = f"{sel_item.camera_id}_{sel_item.zone_id}_{sel_item.version}"
        items.append((label, src_dir))
    return items

logger = structlog.get_logger()

MIN_BASELINE_IMAGES: int = 30

_EMPTY_VAL_STATS: dict = {
    "scores": [],
    "mean": 0.0,
    "std": 0.0,
    "min": 0.0,
    "max": 0.0,
    "p5": 0.0,
    "p25": 0.0,
    "p50": 0.0,
    "p75": 0.0,
    "p95": 0.0,
}

MODEL_INFO = {
    "patchcore": {
        "name": "PatchCore",
        "description": "高精度异常检测，适合静态场景",
        "speed": "中等",
        "epochs": 1,
        "memory": "高",
    },
    "efficient_ad": {
        "name": "EfficientAD",
        "description": "高效异常检测，适合边缘部署",
        "speed": "快",
        "epochs": 70,
        "memory": "低",
    },
    "fastflow": {
        "name": "FastFlow",
        "description": "基于正则化流的实时异常检测",
        "speed": "很快",
        "epochs": 70,
        "memory": "中等",
    },
    "padim": {
        "name": "PaDiM",
        "description": "轻量级异常检测，适合嵌入式设备",
        "speed": "很快",
        "epochs": 1,
        "memory": "低",
    },
    "dinomaly2": {
        "name": "Dinomaly",
        "description": "DINOv2 重建异常检测，支持少样本 (8-shot)",
        "speed": "中等",
        "epochs": 1,
        "memory": "高",
    },
}

# Mapping from user-facing backbone names to anomalib Dinomaly encoder_name values
_DINOMALY_BACKBONE_MAP = {
    "dinov2_vits14": "dinov2reg_vit_small_14",
    "dinov2_vitb14": "dinov2reg_vit_base_14",
    "dinov2_vitl14": "dinov2reg_vit_large_14",
}


def _resolve_dinomaly_backbone(backbone: str) -> str:
    """Resolve user-facing backbone name to anomalib Dinomaly encoder_name.

    If the name is already in anomalib format (e.g. 'dinov2reg_vit_base_14'),
    it is returned as-is.
    """
    return _DINOMALY_BACKBONE_MAP.get(backbone, backbone)


def _training_device_kwargs(backend: str, *, cuda_available: bool) -> dict[str, object]:
    """Resolve explicit device policies without importing or running a model."""
    if backend == "torch-cuda":
        if cuda_available is not True:
            raise RuntimeError("torch-cuda training requires CUDA; CPU fallback is forbidden")
        return {"accelerator": "gpu", "devices": 1}
    if backend == "torch-cpu":
        return {"accelerator": "cpu", "devices": 1}
    return {}


class TrainingStatus(str, Enum):
    IDLE = "idle"
    VALIDATING = "validating"
    SPLITTING = "splitting"
    TRAINING = "training"
    EXPORTING = "exporting"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class QualityReport:
    """Quality assessment of a trained model."""

    grade: str  # A / B / C / F
    score_stats: dict = field(default_factory=dict)
    threshold_recommended: float = 0.0
    suggestions: list[str] = field(default_factory=list)


@dataclass
class TrainingResult:
    status: TrainingStatus
    model_path: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    image_count: int = 0
    train_count: int = 0
    val_count: int = 0
    pre_validation: dict | None = None
    val_stats: dict | None = None
    quality_report: QualityReport | None = None
    threshold_recommended: float | None = None
    output_validation: dict | None = None
    model_version_id: str | None = None
    validation_report: object | None = None  # ValidationReport from training_validator
    calibration_status: str | None = None  # "ok", "skipped", "failed"
    baseline_validation_skipped: bool = False
    training_dir: str | None = None
    export_dir: str | None = None
    run_id: str | None = None
    actual_export_format: str | None = None
    actual_quantization: str | None = None


def _list_images(directory: Path) -> list[Path]:
    """Return sorted list of .png and .jpg files in directory."""
    return sorted(list(directory.glob("*.png")) + list(directory.glob("*.jpg")))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _role_counts(value: object, *, field: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(_SCENE_PROFILE_SPLIT_ROLES):
        raise ValueError(f"scene-profile manifest {field} must contain exact split roles")
    counts: dict[str, int] = {}
    for role in _SCENE_PROFILE_SPLIT_ROLES:
        count = value[role]
        if type(count) is not int or count < 0:
            raise ValueError(f"scene-profile manifest {field}.{role} must be non-negative int")
        counts[role] = count
    return counts


def _scene_profile_manifest_samples(
    baseline_dir: Path,
    *,
    expected_camera_id: str | None,
) -> tuple[list[_ManifestSample], list[_ManifestSample]] | None:
    """Validate a v2/v3 scene-profile manifest and return its exact split.

    Absence preserves the legacy random split. Once an adjacent manifest is
    present, every contract or artifact mismatch is fatal; silently falling
    back to a random split would reintroduce temporal/session leakage.
    """

    manifest_path = _adjacent_baseline_manifest(baseline_dir)
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return None
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("scene-profile manifest must be a regular file")
    if not baseline_dir.is_dir():
        raise ValueError("scene-profile baseline image directory does not exist")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read scene-profile manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("scene-profile manifest root must be an object")
    if manifest.get("schema_version") not in _SCENE_PROFILE_BASELINE_SCHEMAS:
        raise ValueError("unsupported adjacent scene-profile manifest schema")
    if manifest.get("training_ready") is not True:
        raise ValueError("scene-profile manifest is not ready for offline training")
    if manifest.get("release_eligible") is not False:
        raise ValueError("scene-profile manifest release_eligible must be literal false")
    if manifest.get("corpus_evidence_complete") is not True:
        raise ValueError("scene-profile manifest corpus evidence is incomplete")
    camera_id = manifest.get("camera_id")
    if type(camera_id) is not str or not camera_id.strip():
        raise ValueError("scene-profile manifest camera_id is missing")
    if expected_camera_id is not None and camera_id != expected_camera_id:
        raise ValueError("scene-profile manifest camera_id does not match training camera")
    scene_profile_id = manifest.get("scene_profile_id")
    if type(scene_profile_id) is not str or not scene_profile_id.strip():
        raise ValueError("scene-profile manifest scene_profile_id is missing")

    split = manifest.get("split_contract")
    if not isinstance(split, dict):
        raise ValueError("scene-profile manifest split_contract must be an object")
    if split.get("policy") != _SCENE_PROFILE_SPLIT_POLICY:
        raise ValueError("scene-profile manifest split policy is unsupported")
    if split.get("assignment") != _SCENE_PROFILE_SPLIT_ASSIGNMENT:
        raise ValueError("scene-profile manifest split assignment is unsupported")
    if split.get("random_frame_split_allowed") is not False:
        raise ValueError("scene-profile manifest must forbid random frame splitting")
    if split.get("single_session_policy") != "diagnostic_only":
        raise ValueError("scene-profile manifest single-session policy is unsafe")
    source_counts = _role_counts(
        split.get("source_counts"),
        field="split_contract.source_counts",
    )
    sample_counts = _role_counts(
        split.get("sample_counts"),
        field="split_contract.sample_counts",
    )
    if source_counts["diagnostic_only"] != 0 or sample_counts["diagnostic_only"] != 0:
        raise ValueError("scene-profile manifest contains diagnostic-only data")

    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("scene-profile manifest sources must be a non-empty list")
    sources: dict[int, dict[str, object]] = {}
    source_paths: set[str] = set()
    source_hashes: set[str] = set()
    session_roles: dict[str, str] = {}
    actual_source_counts = {role: 0 for role in _SCENE_PROFILE_SPLIT_ROLES}
    for position, source in enumerate(raw_sources):
        if not isinstance(source, dict):
            raise ValueError(f"scene-profile manifest sources[{position}] must be an object")
        source_index = source.get("source_index")
        if type(source_index) is not int or source_index in sources:
            raise ValueError("scene-profile manifest source_index must be a unique integer")
        source_path = source.get("path")
        source_hash = source.get("sha256")
        source_role = source.get("split_role")
        if type(source_path) is not str or not source_path.strip():
            raise ValueError("scene-profile manifest source path is missing")
        if source_path in source_paths:
            raise ValueError("scene-profile manifest contains duplicate source paths")
        if not _is_sha256(source_hash) or source_hash in source_hashes:
            raise ValueError("scene-profile manifest source sha256 is invalid or duplicated")
        if source_role not in {"train", "calibration"}:
            raise ValueError("scene-profile manifest source has an unsafe split role")
        review = source.get("annotation_review_evidence")
        if not isinstance(review, dict) or review.get("complete") is not True:
            raise ValueError("scene-profile manifest source review evidence is incomplete")
        if review.get("source_video_sha256") != source_hash:
            raise ValueError("scene-profile manifest source/review sha256 mismatch")
        if review.get("source_video_sha256_matches") is not True:
            raise ValueError("scene-profile manifest source video identity is unverified")
        if review.get("camera_id") != camera_id or review.get("camera_id_matches") is not True:
            raise ValueError("scene-profile manifest source camera identity is unverified")
        session_id = review.get("source_session_id")
        if (
            type(session_id) is not str
            or not session_id.strip()
            or session_id != session_id.strip()
        ):
            raise ValueError("scene-profile manifest source_session_id is invalid")
        prior_role = session_roles.setdefault(session_id, source_role)
        if prior_role != source_role:
            raise ValueError("scene-profile manifest splits one source session across roles")
        sources[source_index] = {
            "path": source_path,
            "sha256": source_hash,
            "split_role": source_role,
            "source_session_id": session_id,
        }
        source_paths.add(source_path)
        source_hashes.add(source_hash)
        actual_source_counts[source_role] += 1
    if set(sources) != set(range(len(raw_sources))):
        raise ValueError("scene-profile manifest source indexes must be contiguous from zero")
    if actual_source_counts != source_counts:
        raise ValueError("scene-profile manifest source split counts do not match sources")

    declared_sessions = split.get("declared_session_ids")
    expected_sessions = sorted(session_roles)
    if declared_sessions != expected_sessions or len(expected_sessions) < 2:
        raise ValueError("scene-profile manifest requires at least two exact reviewed sessions")
    if split.get("declared_independent_session_count") != len(expected_sessions):
        raise ValueError("scene-profile manifest declared session count is inconsistent")
    calibration_session = expected_sessions[-1]
    for session_id, role in session_roles.items():
        expected_role = "calibration" if session_id == calibration_session else "train"
        if role != expected_role:
            raise ValueError("scene-profile manifest session role assignment is inconsistent")

    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("scene-profile manifest samples must be a non-empty list")
    manifest_root = manifest_path.parent.resolve(strict=True)
    baseline_root = baseline_dir.resolve(strict=True)
    train_samples: list[_ManifestSample] = []
    calibration_samples: list[_ManifestSample] = []
    seen_paths: set[Path] = set()
    seen_names: set[str] = set()
    actual_sample_counts = {role: 0 for role in _SCENE_PROFILE_SPLIT_ROLES}
    for position, sample in enumerate(raw_samples):
        if not isinstance(sample, dict):
            raise ValueError(f"scene-profile manifest samples[{position}] must be an object")
        if sample.get("sample_index") != position or type(sample.get("sample_index")) is not int:
            raise ValueError("scene-profile manifest sample indexes must be contiguous from zero")
        role = sample.get("split_role")
        if role not in {"train", "calibration"}:
            raise ValueError("scene-profile manifest sample has an unsafe split role")
        source_index = sample.get("source_index")
        if type(source_index) is not int or source_index not in sources:
            raise ValueError("scene-profile manifest sample references an unknown source")
        source = sources[source_index]
        if role != source["split_role"]:
            raise ValueError("scene-profile manifest sample/source split role mismatch")
        if sample.get("source_path") != source["path"]:
            raise ValueError("scene-profile manifest sample/source path mismatch")
        if sample.get("source_sha256") != source["sha256"]:
            raise ValueError("scene-profile manifest sample/source sha256 mismatch")

        relative_value = sample.get("path")
        if type(relative_value) is not str or not relative_value or "\\" in relative_value:
            raise ValueError(
                "scene-profile manifest sample path must be canonical POSIX relative path"
            )
        relative = PurePosixPath(relative_value)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_value
            or len(relative.parts) != 2
            or relative.parts[0] != baseline_dir.name
            or relative.parts[1] in {".", ".."}
        ):
            raise ValueError("scene-profile manifest sample path escapes the image directory")
        candidate = manifest_root.joinpath(*relative.parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("scene-profile manifest sample must be a regular existing file")
        resolved = candidate.resolve(strict=True)
        if resolved.parent != baseline_root:
            raise ValueError("scene-profile manifest sample path escapes the image directory")
        if resolved.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("scene-profile manifest sample has unsupported image extension")
        if resolved in seen_paths or resolved.name in seen_names:
            raise ValueError("scene-profile manifest sample path or target name is duplicated")
        expected_hash = sample.get("sha256")
        if not _is_sha256(expected_hash) or _sha256_file(resolved) != expected_hash:
            raise ValueError("scene-profile manifest sample sha256 mismatch")
        manifest_sample = _ManifestSample(path=resolved, sha256=expected_hash)
        if role == "train":
            train_samples.append(manifest_sample)
        else:
            calibration_samples.append(manifest_sample)
        seen_paths.add(resolved)
        seen_names.add(resolved.name)
        actual_sample_counts[role] += 1

    actual_images: set[Path] = set()
    for pattern in _IMAGE_EXTS:
        for image in baseline_dir.glob(pattern):
            if image.is_symlink() or not image.is_file():
                raise ValueError("scene-profile baseline contains a non-regular image")
            actual_images.add(image.resolve(strict=True))
    if actual_images != seen_paths:
        raise ValueError("scene-profile manifest samples do not exactly cover baseline images")
    if not train_samples or not calibration_samples:
        raise ValueError("scene-profile manifest train and calibration roles must be non-empty")
    if actual_sample_counts != sample_counts:
        raise ValueError("scene-profile manifest sample split counts do not match samples")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("accepted_images") != len(raw_samples):
        raise ValueError("scene-profile manifest accepted image count is inconsistent")
    return train_samples, calibration_samples


class ModelTrainer:
    """Orchestrates anomaly model training and export.

    Wraps Anomalib's training API into a simple interface that can be
    called from the scheduler or the dashboard.
    """

    def __init__(
        self,
        baseline_manager: BaselineManager,
        models_dir: str | Path = "data/models",
        exports_dir: str | Path = "data/exports",
        model_registry=None,
        training_validator=None,
        validation_config: dict | None = None,
    ):
        self._baseline_manager = baseline_manager
        self._models_dir = Path(models_dir)
        self._exports_dir = Path(exports_dir)
        self._model_registry = model_registry
        self._registry_binding_lock = threading.Lock()
        self._training_validator = training_validator
        self._validation_config = validation_config or {}
        self._status = TrainingStatus.IDLE
        self._last_result: TrainingResult | None = None

    @property
    def exports_dir(self) -> Path:
        return self._exports_dir

    def bind_model_registry(self, registry) -> None:
        """Attach the authoritative registry before running derived operations."""
        if registry is None:
            raise ValueError("model registry is required")
        with self._registry_binding_lock:
            if self._model_registry is not None and self._model_registry is not registry:
                same_sessions = getattr(type(self._model_registry), "uses_same_sessions_as", None)
                if callable(same_sessions) and same_sessions(self._model_registry, registry):
                    return
                raise ValueError("model registry is already bound to a different database")
            self._model_registry = registry

    @property
    def status(self) -> TrainingStatus:
        return self._status

    @property
    def last_result(self) -> TrainingResult | None:
        return self._last_result

    @staticmethod
    def _free_gpu_for_training() -> None:
        """Aggressively free all GPU memory before training.

        On a 12GB GPU (e.g. RTX 3060), inference models + CUDA OpenCV
        can consume 10GB+.  This method:
        1. Moves YOLO models to CPU
        2. Clears all PyTorch CUDA caches and tensors
        """
        import gc
        import torch

        freed = 0

        # Move shared YOLO models to CPU
        try:
            from argus.person.detector import _shared_yolo_registry
            for name, model in _shared_yolo_registry.items():
                if hasattr(model, "to"):
                    model.to("cpu")
                    freed += 1
                    logger.info("trainer.gpu_freed", model=f"yolo:{name}")
        except Exception:
            pass

        # Force release ALL CUDA tensors and cached allocations
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        total_mb = torch.cuda.mem_get_info()[1] // (1024 * 1024)
        logger.info(
            "trainer.gpu_memory_freed",
            models_moved=freed,
            free_mb=free_mb,
            total_mb=total_mb,
        )

    @staticmethod
    def _restore_gpu_after_training() -> None:
        """Move inference models back to GPU after training completes."""
        try:
            import torch
            if not torch.cuda.is_available():
                return
            from argus.person.detector import _shared_yolo_registry
            for name, model in _shared_yolo_registry.items():
                if hasattr(model, "to"):
                    model.to("cuda:0")
                    logger.info("trainer.gpu_restored", model=f"yolo:{name}")
        except Exception:
            pass

    def _fail(self, start: float, **kwargs) -> TrainingResult:
        """Build a FAILED TrainingResult and update internal state."""
        kwargs.setdefault("duration_seconds", time.monotonic() - start)
        result = TrainingResult(status=TrainingStatus.FAILED, **kwargs)
        self._status = TrainingStatus.FAILED
        self._last_result = result
        return result

    def train(
        self,
        camera_id: str,
        zone_id: str = "default",
        model_type: str = "patchcore",
        image_size: int = 256,
        export_format: str | None = "openvino",
        quantization: str = "fp16",
        calibration_images: int = 100,
        progress_callback: callable | None = None,
        anomaly_config: object | None = None,
        resume_from: str | None = None,
        group_id: str | None = None,
        backbone_checkpoint: str | None = None,
        skip_baseline_validation: bool = False,
        baseline_dir_override: Path | None = None,
        image_count_override: int | None = None,
        baseline_versions_label: str | None = None,
        require_calibration: bool = False,
        scene_training_input: SceneTrainingInput | None = None,
    ) -> TrainingResult:
        """Train an anomaly detection model for a specific camera/zone.

        Full pipeline: validate -> split -> train -> export -> evaluate -> report.

        Args:
            anomaly_config: Optional AnomalyConfig with model-specific parameters
                (e.g. dinomaly_backbone, dinomaly_encoder_layers for dinomaly2).
            baseline_dir_override: Override the resolved baseline directory.
                Used by multi-version training (痛点 2): the caller merges
                several baseline versions into a tmp dir via _DatasetMerger
                and passes that tmp path here. When set, ``image_count_override``
                must also be supplied and the lifecycle gate is skipped (the
                caller has pre-validated each version).
            image_count_override: Image count for the override directory.
            baseline_versions_label: Human/JSON label describing the version
                set (e.g. ``v001+v003``); persisted to TrainingRecord.
            require_calibration: Fail before candidate registration unless a
                deployable runtime artifact has successful conformal calibration.
            scene_training_input: Shared, immutable V3 workflow input. It is
                revalidated before training and again before registration; it
                cannot be combined with a group or legacy/random dataset.
        """
        start = time.monotonic()

        def _progress(pct: int, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        if type(require_calibration) is not bool:
            return self._fail(start, error="require_calibration must be a literal boolean")
        if scene_training_input is not None:
            try:
                scene_training_input = revalidate_scene_training_input(scene_training_input)
                if (
                    not require_calibration or group_id is not None
                    or skip_baseline_validation is not False
                    or camera_id != scene_training_input.camera_id or zone_id != "default"
                    or baseline_dir_override != scene_training_input.images_dir
                    or type(image_count_override) is not int
                    or image_count_override != scene_training_input.image_count
                ):
                    raise ValueError("scene training requires its exact V3 input and mandatory calibration")
            except (OSError, TypeError, ValueError) as exc:
                return self._fail(start, error=f"场景训练输入校验失败: {exc}")

        if baseline_dir_override is not None:
            baseline_dir = baseline_dir_override
        elif group_id:
            baseline_dir = self._baseline_manager.get_group_baseline_dir(group_id, zone_id)
        else:
            baseline_dir = self._baseline_manager.get_baseline_dir(camera_id, zone_id)

        # Lifecycle gate: only Verified or Active baselines can be used for
        # training. Skipped for multi-version override because the caller
        # (job_executor) already validated each constituent version.
        lifecycle = getattr(self._baseline_manager, "_lifecycle", None)
        if lifecycle and baseline_dir.exists() and baseline_dir_override is None:
            version_name = baseline_dir.name
            lifecycle_camera = f"group:{group_id}" if group_id else camera_id
            ver_rec = lifecycle.get_version(lifecycle_camera, zone_id, version_name)
            if ver_rec and ver_rec.state == BaselineState.DRAFT:
                return self._fail(
                    start,
                    error=f"基���版本 {version_name} 处于 Draft 状态，未通过审核，不能用于训练",
                )

        if image_count_override is not None:
            image_count = image_count_override
        elif group_id:
            # Group baselines live under _groups/<id>, not the synthetic
            # registry camera ID group:<id> used for the resulting model.
            image_count = len(_list_images(baseline_dir))
        else:
            image_count = self._baseline_manager.count_images(camera_id, zone_id)

        # Dinomaly2 supports few-shot mode with fewer images
        min_images = MIN_BASELINE_IMAGES
        if model_type == "dinomaly2" and anomaly_config is not None:
            few_shot_min = getattr(anomaly_config, "dinomaly_few_shot_images", 8)
            if image_count < MIN_BASELINE_IMAGES and image_count >= few_shot_min:
                min_images = few_shot_min
                logger.info(
                    "trainer.few_shot_mode",
                    camera_id=camera_id,
                    image_count=image_count,
                    few_shot_min=few_shot_min,
                )

        if image_count < min_images:
            return self._fail(
                start,
                error=f"基线图片不足: {image_count} 张 (需要 >= {min_images})",
                image_count=image_count,
            )

        # Pre-training validation (TRN-001)
        self._status = TrainingStatus.VALIDATING
        _progress(5, "正在验证基线质量...")

        try:
            quality_context = (
                {"scene_training_input": scene_training_input}
                if scene_training_input is not None else {}
            )
            pre_validation = self._validate_baseline_quality(
                baseline_dir, min_images=min_images, **quality_context,
            )
        except (OSError, TypeError, ValueError) as exc:
            return self._fail(start, error=f"基线质量上下文校验失败: {exc}", image_count=image_count)
        if not pre_validation["passed"]:
            if skip_baseline_validation:
                logger.warning(
                    "training.validation_skipped",
                    camera_id=camera_id,
                    errors=pre_validation["errors"],
                    msg="Baseline validation failed but skip_baseline_validation=True",
                )
            else:
                error_msg = "; ".join(pre_validation["errors"])
                return self._fail(
                    start,
                    error=f"基线质量验证失败: {error_msg}",
                    image_count=image_count,
                    pre_validation=pre_validation,
                )

        logger.info(
            "training.started",
            camera_id=camera_id,
            zone_id=zone_id,
            model_type=model_type,
            images=image_count,
            corruption_rate=pre_validation["corruption_rate"],
            near_duplicate_rate=pre_validation["near_duplicate_rate"],
            brightness_std=round(pre_validation["brightness_std"], 2),
            validation_skipped=skip_baseline_validation,
        )

        run = new_artifact_run(self._models_dir, self._exports_dir, camera_id, zone_id)
        output_dir = run.training_dir

        # Train/val split (TRN-002)
        self._status = TrainingStatus.SPLITTING
        _progress(10, "正在切分训练/验证集...")

        try:
            train_dir, val_dir, train_count, val_count = self._split_train_val(
                baseline_dir,
                output_dir,
                expected_camera_id=camera_id,
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.error("training.split_failed", error=str(exc))
            return self._fail(
                start,
                error=f"训练/校准集切分失败: {exc}",
                image_count=image_count,
                pre_validation=pre_validation,
            )

        # Anomalib training
        self._status = TrainingStatus.TRAINING
        _progress(15, f"正在训练 {model_type} 模型 ({train_count} 张训练图片)...")

        common_fail_kwargs = dict(
            image_count=image_count,
            train_count=train_count,
            val_count=val_count,
            pre_validation=pre_validation,
        )

        engine = model = None
        try:
            try:
                engine, model = self._train_anomalib(
                    data_dir=train_dir,
                    output_dir=output_dir,
                    model_type=model_type,
                    image_size=image_size,
                    anomaly_config=anomaly_config,
                    resume_from=resume_from,
                    backbone_checkpoint=backbone_checkpoint,
                )
            except ImportError:
                return self._fail(start, error="anomalib 未安装", **common_fail_kwargs)
            except Exception as e:
                logger.error("training.failed", error=str(e))
                get_error_channel().emit(
                    severity=SEVERITY_ERROR,
                    source="trainer",
                    code="training_failed",
                    message=f"模型训练失败 ({camera_id}/{zone_id}): {e}",
                    context={
                        "camera_id": camera_id,
                        "zone_id": zone_id,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                )
                return self._fail(start, error=str(e), **common_fail_kwargs)

            # Export
            export_path_str = None
            artifact_path = None
            actual_export_format = export_format
            actual_quantization = quantization
            if export_format:
                self._status = TrainingStatus.EXPORTING
                quant_label = f" + {quantization}" if quantization != "fp32" else ""
                _progress(70, f"正在导出 {export_format}{quant_label} 格式...")
                try:
                    export_path_str = str(run.export_dir)
                    export_result = self._export_model(
                        engine=engine,
                        model=model,
                        export_format=export_format,
                        export_path=export_path_str,
                        image_size=image_size,
                        quantization=quantization,
                        val_dir=val_dir,
                        calibration_images=calibration_images,
                    )
                    actual_export_format = export_result["actual_format"]
                    actual_quantization = export_result["actual_quantization"]
                    artifact_path = exported_artifact(
                        run.export_dir, actual_export_format, export_result.get("artifact_path"),
                    )
                    if actual_export_format != export_format:
                        logger.warning(
                            "training.export_format_changed",
                            requested=export_format,
                            actual=actual_export_format,
                        )
                    if actual_quantization != quantization:
                        logger.warning(
                            "training.quantization_changed",
                            requested=quantization,
                            actual=actual_quantization,
                        )
                except Exception as e:
                    logger.error("training.export_failed", error=str(e))
                    return self._fail(
                        start,
                        error=f"模型导出失败: {e}",
                        **common_fail_kwargs,
                    )
        finally:
            # Release training artifacts to free memory — runs on ALL exit paths
            del engine, model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    self._restore_gpu_after_training()
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Output validation + smoke test (TRN-006)
        # Returns (validation_dict, loaded_detector_or_None)
        _progress(80, "正在验证模型输出...")
        output_validation, detector = self._validate_output(
            output_dir,
            Path(export_path_str) if export_path_str else None,
            val_dir,
            artifact_path=artifact_path,
            inference_backend=getattr(anomaly_config, "inference_backend", "auto"),
        )

        if artifact_path is not None and not output_validation["smoke_test_passed"]:
            return self._fail(
                start, error="部署模型冒烟推理失败，未注册模型",
                output_validation=output_validation, **common_fail_kwargs,
            )
        if artifact_path is None and not output_validation["checkpoint_valid"]:
            return self._fail(
                start,
                error="训练未产生有效模型文件",
                output_validation=output_validation,
                **common_fail_kwargs,
            )

        # Post-training validation on val set (TRN-003)
        # Reuses detector from smoke test to avoid redundant model load
        self._status = TrainingStatus.EVALUATING
        _progress(85, f"正在验证集上评估模型 ({val_count} 张)...")

        val_stats = self._validate_on_val_set(val_dir, detector=detector)
        if artifact_path is not None:
            try:
                scores = np.asarray(val_stats.get("scores", []), dtype=float)
                valid_scores = scores.size > 0 and bool(np.isfinite(scores).all())
            except (TypeError, ValueError):
                valid_scores = False
            if not valid_scores:
                return self._fail(
                    start, error="部署模型未产生有效有限验证分数，未注册模型",
                    output_validation=output_validation, **common_fail_kwargs,
                )
        if require_calibration and scene_training_input is not None:
            # Every frozen calibration image must produce exactly one finite
            # score. A skipped decode cannot be hidden by having >=50 scores
            # left over, nor may a truncated/capped evaluation claim success.
            try:
                calibration_scores = np.asarray(val_stats.get("scores", []), dtype=float)
                expected_count = scene_training_input.calibration_count
                complete_calibration = (
                    expected_count >= 50
                    and type(val_count) is int and val_count == expected_count
                    and calibration_scores.ndim == 1
                    and calibration_scores.size == expected_count
                    and bool(np.isfinite(calibration_scores).all())
                )
            except (TypeError, ValueError):
                complete_calibration = False
            if not complete_calibration:
                return self._fail(
                    start, error="场景校准未完整覆盖冻结校准集，未注册模型",
                    calibration_status="failed", output_validation=output_validation,
                    **common_fail_kwargs,
                )

        # Threshold recommendation (TRN-005)
        threshold_recommended = self._recommend_threshold(val_stats)

        # Quality report (TRN-004)
        quality_report = self._generate_quality_report(val_stats, threshold_recommended)
        for warning in pre_validation.get("warnings", []):
            if warning not in quality_report.suggestions:
                quality_report.suggestions.append(warning)

        # Three-step validation pipeline (if configured)
        validation_report = None
        if self._training_validator is not None and detector is not None:
            _progress(90, "正在执行三步验证（AUROC/合成Recall/历史回放）...")
            try:
                validation_report = self._training_validator.validate(
                    detector=detector,
                    camera_id=camera_id,
                    zone_id=zone_id,
                    val_dir=val_dir,
                    baseline_dir=baseline_dir,
                    threshold=threshold_recommended,
                    auroc_threshold=self._validation_config.get(
                        "validation_auroc_threshold", 0.85
                    ),
                    recall_threshold=self._validation_config.get(
                        "validation_recall_threshold", 0.80
                    ),
                    replay_days=self._validation_config.get(
                        "historical_replay_days", 30
                    ),
                )
                if not validation_report.all_passed:
                    quality_report.grade = "F"
                    quality_report.suggestions.append(
                        "三步验证未通过 — 模型不得部署"
                    )
                    logger.warning(
                        "training.validation_failed",
                        camera_id=camera_id,
                        auroc=validation_report.auroc,
                        recall=validation_report.recall,
                    )
            except Exception as e:
                logger.error("training.validation_error", error=str(e))

        _progress(95, f"训练完成 — 质量等级: {quality_report.grade}")

        # Conformal calibration (A2-3): run on val set scores if detector available.
        # Must run before TrainingResult construction so the final status is captured
        # in the result dataclass instead of a local variable that gets discarded.
        calibration_status: str | None = "skipped"
        if detector is not None and val_stats.get("scores"):
            try:
                from argus.alerts.calibration import ConformalCalibrator

                cal_scores = np.array(val_stats["scores"])
                if len(cal_scores) >= 50:
                    calibrator = ConformalCalibrator()
                    cal_result = calibrator.calibrate(cal_scores)
                    cal_path = output_dir / "calibration.json"
                    calibrator.save(
                        cal_result, cal_path, sorted_scores=np.sort(cal_scores)
                    )
                    logger.info("trainer.calibration_saved", path=str(cal_path))
                    synced_paths = self._sync_calibration_to_export(
                        cal_path,
                        Path(export_path_str) if export_path_str else None,
                    )
                    if synced_paths:
                        logger.info(
                            "trainer.calibration_export_synced",
                            paths=[str(path) for path in synced_paths],
                        )
                    calibration_status = "ok"
                else:
                    logger.warning(
                        "trainer.calibration_skipped",
                        reason=f"Not enough scores for calibration ({len(cal_scores)} < 50)",
                    )
            except Exception as e:
                calibration_status = "failed"
                logger.warning("trainer.calibration_failed", error=str(e))

        # Freeze the deployment bundle before registry hashing. A failed
        # calibration must not leave a registered, partially updated candidate.
        if require_calibration and (artifact_path is None or calibration_status != "ok"):
            return self._fail(
                start, error="场景训练必须完成部署模型校准，未注册模型",
                calibration_status=calibration_status,
                **common_fail_kwargs,
            )
        if scene_training_input is not None:
            try:
                scene_training_input = revalidate_scene_training_input(scene_training_input)
            except (OSError, TypeError, ValueError) as exc:
                return self._fail(
                    start, error=f"场景训练输入在训练期间发生变化，未注册模型: {exc}",
                    calibration_status=calibration_status,
                    **common_fail_kwargs,
                )
        model_version_id = None
        if artifact_path is not None:
            if calibration_status == "failed":
                return self._fail(start, error="模型校准失败，未注册模型", **common_fail_kwargs)
            try:
                metadata = write_artifact_manifest(
                    run, artifact_path, model_type=model_type,
                    calibration_status=calibration_status,
                    baseline_quality=pre_validation,
                )
                if self._model_registry is not None:
                    model_version_id = self._model_registry.register(
                        model_path=artifact_path,
                        baseline_dir=baseline_dir,
                        camera_id=camera_id,
                        model_type=model_type,
                        training_params={
                            "image_size": image_size,
                            "export_format": actual_export_format,
                            "quantization": actual_quantization,
                            "train_count": train_count,
                            "val_count": val_count,
                            "image_count": image_count,
                            "quality_grade": quality_report.grade,
                            "zone_id": zone_id,
                            "run_id": run.run_id,
                            "training_dir": str(output_dir),
                            "export_dir": str(run.export_dir),
                            "artifact_manifest": metadata,
                            "baseline_quality": pre_validation,
                            **({"scene_workflow": dict(scene_training_input.lineage)}
                               if scene_training_input is not None else {}),
                        },
                    )
            except Exception as e:
                logger.error("training.model_registry_failed", error=str(e))
                return self._fail(start, error=f"模型产物登记失败: {e}", **common_fail_kwargs)

        # Strip raw scores from val_stats to avoid bloating DB/JSON storage.
        # Percentile statistics are already computed and sufficient for reporting.
        if val_stats.get("scores"):
            del val_stats["scores"]

        duration = time.monotonic() - start
        result = TrainingResult(
            status=TrainingStatus.COMPLETE,
            model_path=str(artifact_path) if artifact_path else None,
            duration_seconds=duration,
            image_count=image_count,
            train_count=train_count,
            val_count=val_count,
            pre_validation=pre_validation,
            val_stats=val_stats,
            quality_report=quality_report,
            threshold_recommended=threshold_recommended,
            output_validation=output_validation,
            model_version_id=model_version_id,
            validation_report=validation_report,
            calibration_status=calibration_status,
            baseline_validation_skipped=skip_baseline_validation and not pre_validation.get("passed", True),
            training_dir=str(output_dir),
            export_dir=export_path_str,
            run_id=run.run_id,
            actual_export_format=actual_export_format,
            actual_quantization=actual_quantization,
        )
        self._status = TrainingStatus.COMPLETE
        self._last_result = result

        logger.info(
            "training.complete",
            camera_id=camera_id,
            duration=f"{duration:.1f}s",
            model_path=str(output_dir),
            quality_grade=quality_report.grade,
            threshold=round(threshold_recommended, 3),
        )

        # Cleanup split directory
        split_dir = output_dir / "_split"
        if split_dir.exists():
            try:
                shutil.rmtree(split_dir)
            except OSError as e:
                try:
                    files = [f for f in split_dir.rglob("*") if f.is_file()]
                    n_files = len(files)
                    size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
                except Exception:
                    n_files, size_mb = -1, -1.0
                logger.warning(
                    "trainer.split_cleanup_failed",
                    path=str(split_dir),
                    error=str(e),
                    files=n_files,
                    size_mb=round(size_mb, 1),
                )

        return result

    # ── Incremental training from active learning labels ──

    def incremental_train(
        self,
        camera_id: str,
        zone_id: str = "default",
        model_type: str = "patchcore",
        labeled_entries: list | None = None,
        database: object | None = None,
        progress_callback: callable | None = None,
        anomaly_config: object | None = None,
    ) -> TrainingResult:
        """Incremental retraining using active learning labels.

        Normal-labeled frames are merged into the baselines directory,
        then a full retraining is triggered. The labeled entries are
        tracked so they aren't re-used in future incremental runs.

        Args:
            labeled_entries: List of LabelingQueueRecord objects with labels.
                If None and database provided, fetches untrained labeled entries.
            database: Database instance for fetching/marking entries.
        """
        start = time.monotonic()

        def _progress(pct: int, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        _progress(5, "正在收集主动学习标注数据...")

        # Fetch labeled entries from DB if not provided
        if labeled_entries is None and database is not None:
            labeled_entries = database.get_labeled_entries(
                camera_id=camera_id,
                trained_into="",
            )

        if not labeled_entries:
            return self._fail(start, error="没有可用的标注数据用于增量训练")

        normal_frames = [e for e in labeled_entries if e.label == "normal"]
        anomaly_frames = [e for e in labeled_entries if e.label == "anomaly"]

        logger.info(
            "incremental_train.starting",
            camera_id=camera_id,
            normal_count=len(normal_frames),
            anomaly_count=len(anomaly_frames),
        )

        # Copy normal-labeled frames to baselines directory
        _progress(10, f"正在合并 {len(normal_frames)} 张正常样本到基线...")
        baselines_dir = self._baseline_manager.get_baseline_dir(camera_id, zone_id)
        baselines_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        copy_errors = 0
        for entry in normal_frames:
            src = Path(entry.frame_path)
            if src.exists():
                dst = baselines_dir / f"al_{src.name}"
                if not dst.exists():
                    try:
                        shutil.copy2(str(src), str(dst))
                        copied += 1
                    except OSError as e:
                        copy_errors += 1
                        logger.warning(
                            "incremental_train.copy_failed",
                            src=str(src), error=str(e),
                        )
        if copy_errors > 0 and copied == 0:
            logger.error(
                "incremental_train.all_copies_failed",
                total=len(normal_frames), errors=copy_errors,
            )

        logger.info(
            "incremental_train.baselines_merged",
            camera_id=camera_id,
            copied=copied,
            total_normal=len(normal_frames),
        )

        # Run full training with the merged baselines
        _progress(20, "正在使用合并后的基线启动模型训练...")
        result = self.train(
            camera_id=camera_id,
            zone_id=zone_id,
            model_type=model_type,
            progress_callback=lambda pct, msg: _progress(20 + int(pct * 0.75), msg),
            anomaly_config=anomaly_config,
            skip_baseline_validation=True,
        )

        # Mark entries as consumed by this training run
        if database is not None and result.status == TrainingStatus.COMPLETE:
            entry_ids = [e.id for e in labeled_entries]
            version_id = result.model_version_id or f"incremental-{camera_id}-{int(time.time())}"
            database.mark_labeling_trained(entry_ids, trained_into=version_id)
            _progress(100, f"增量训练完成 — 模型版本: {version_id}")
        else:
            _progress(100, f"增量训练失败: {result.error or '未知错误'}")

        return result

    # ── TRN-001: Pre-training validation ──

    def _validate_baseline_quality(
        self, baseline_dir: Path, min_images: int = MIN_BASELINE_IMAGES,
        *, scene_training_input: SceneTrainingInput | None = None,
    ) -> dict:
        """Select static input rules only after server-side strict V3 validation.

        An adjacent manifest or client-selected mode is not authority. The
        context is resolved again here, then rechecked after scanning. Existing
        split-copy hashes and the final pre-registration revalidation remain.
        """
        context = None
        verified = None
        if scene_training_input is not None:
            from argus.preprocessing.scene_guard import SceneGuardConfig

            verified = revalidate_scene_training_input(scene_training_input)
            if Path(baseline_dir).absolute() != verified.images_dir.absolute():
                raise ValueError("static quality input is not the verified scene image directory")
            manifest_path = verified.root / "manifest.json"
            with manifest_path.open("rb") as stream:
                raw = stream.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != verified.lineage["manifest_sha256"]:
                raise ValueError("static quality manifest changed or exceeds its metadata budget")
            manifest = json.loads(raw)
            if manifest.get("schema_version") != "argus.scene-profile-baseline.v3":
                raise ValueError("static quality requires its exact strict V3 baseline")
            try:
                split = manifest["split_contract"]
                context = FixedSceneQualityContext(
                    config=SceneGuardConfig(**manifest["parameters"]["scene_guard"]),
                    expected_shape=(manifest["reference"]["height"], manifest["reference"]["width"]),
                    sample_count=verified.image_count, training_count=verified.training_count,
                    calibration_count=verified.calibration_count,
                    source_session_count=len(split["declared_session_ids"]),
                    accepted_span_seconds=sum(split["accepted_time_span_seconds"][role]
                        for role in ("train", "calibration")),
                )
            except (KeyError, TypeError) as exc:
                raise ValueError("static quality context lacks complete frozen scene metadata") from exc
        report = inspect_baseline_quality(_list_images(baseline_dir), min_images, fixed_scene=context)
        if verified is not None:
            revalidate_scene_training_input(verified)
            report["scene_workflow"] = dict(verified.lineage)
        return report

    # ── TRN-002: Auto train/val split ──

    @staticmethod
    def _split_train_val(
        baseline_dir: Path,
        output_dir: Path,
        seed: int = 42,
        *,
        expected_camera_id: str | None = None,
    ) -> tuple[Path, Path, int, int]:
        """Split normal images into training and calibration/validation sets.

        A validated scene-profile baseline v2/v3 manifest is authoritative and is
        copied exactly by whole-session roles. Only a missing adjacent manifest
        permits the legacy seeded 80/20 random split.

        Files are physically copied to ensure portability.
        Returns (train_dir, val_dir, train_count, val_count).
        val_dir contains a 'normal' subdirectory with the actual images.
        """
        manifest_samples = _scene_profile_manifest_samples(
            baseline_dir,
            expected_camera_id=expected_camera_id,
        )
        expected_hashes: dict[Path, str] = {}
        if manifest_samples is not None:
            train_manifest_samples, calibration_manifest_samples = manifest_samples
            train_images = [item.path for item in train_manifest_samples]
            val_images = [item.path for item in calibration_manifest_samples]
            expected_hashes = {
                item.path: item.sha256
                for item in train_manifest_samples + calibration_manifest_samples
            }
            split_strategy = "scene_profile_manifest_v2_or_v3"
        else:
            images = _list_images(baseline_dir)
            if len(images) < 5:
                raise ValueError(
                    f"基线图片太少，无法拆分训练/验证集: {len(images)} 张 (至少需要 5 张)"
                )
            rng = random.Random(seed)
            shuffled = list(images)
            rng.shuffle(shuffled)
            split_idx = max(2, int(len(shuffled) * 0.8))
            train_images = shuffled[:split_idx]
            val_images = shuffled[split_idx:] or shuffled[-1:]
            split_strategy = "legacy_seeded_random_80_20"

        split_dir = output_dir / "_split"
        train_dir = split_dir / "train" / "normal"
        val_dir = split_dir / "val" / "normal"

        if split_dir.exists():
            shutil.rmtree(split_dir)

        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        try:
            for source_dir, destination_dir in (
                (train_images, train_dir),
                (val_images, val_dir),
            ):
                for image in source_dir:
                    target = destination_dir / image.name
                    shutil.copy2(str(image), str(target))
                    expected_hash = expected_hashes.get(image)
                    if expected_hash is not None and _sha256_file(target) != expected_hash:
                        raise ValueError(
                            "scene-profile sample changed while creating the split"
                        )
        except BaseException:
            shutil.rmtree(split_dir, ignore_errors=True)
            raise

        logger.info(
            "training.split_complete",
            train=len(train_images),
            val=len(val_images),
            total=len(train_images) + len(val_images),
            strategy=split_strategy,
        )
        return train_dir.parent, val_dir.parent, len(train_images), len(val_images)

    # ── TRN-003: Post-training validation ──

    def _validate_on_val_set(
        self, val_dir: Path, *, detector=None
    ) -> dict:
        """Run inference on validation set and collect score distribution.

        If detector is provided, reuses it instead of loading the model again.
        """
        val_normal = val_dir / "normal" if (val_dir / "normal").exists() else val_dir
        val_images = _list_images(val_normal)

        if not val_images:
            logger.warning("training.no_val_images", val_dir=str(val_dir))
            return dict(_EMPTY_VAL_STATS)

        if detector is None:
            logger.warning("training.no_detector_for_validation")
            return dict(_EMPTY_VAL_STATS)

        scores: list[float] = []
        try:
            for img_path in val_images:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                result = detector.predict(frame)
                if getattr(result, "detection_failed", False) is True:
                    raise RuntimeError("Inference failed while collecting validation scores")
                if not np.isfinite(result.anomaly_score):
                    raise RuntimeError("Non-finite validation score")
                scores.append(result.anomaly_score)
        except Exception as e:
            logger.error("training.val_inference_failed", error=str(e))
            return dict(_EMPTY_VAL_STATS)

        if not scores:
            return dict(_EMPTY_VAL_STATS)

        arr = np.array(scores)
        p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95]).tolist()
        return {
            "scores": scores,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95,
        }

    # ── TRN-004: Quality report ──

    @staticmethod
    def _generate_quality_report(
        val_stats: dict, threshold_recommended: float
    ) -> QualityReport:
        """Generate a quality report based on validation score distribution."""
        mean = val_stats.get("mean", 0.0)
        std = val_stats.get("std", 0.0)
        max_score = val_stats.get("max", 0.0)
        suggestions: list[str] = []

        if mean < 0.3 and std < 0.1 and max_score < threshold_recommended * 0.8:
            grade = "A"
        elif mean < 0.4 and std < 0.15 and max_score < threshold_recommended:
            grade = "B"
        elif mean < 0.5 and max_score < threshold_recommended:
            grade = "C"
        else:
            grade = "F"

        if not val_stats.get("scores"):
            suggestions.append("验证集为空，无法评估模型质量")
            return QualityReport(
                grade="F",
                score_stats=val_stats,
                threshold_recommended=threshold_recommended,
                suggestions=suggestions,
            )

        if grade in ("F", "C"):
            suggestions.append("建议补充同一固定机位下业务允许的正常工况；不要移动机位或焦距制造多样性")
        if std > 0.15:
            suggestions.append("基线图片质量不一致，检查是否有异常值或场景变化")
        if max_score > 0 and max_score > threshold_recommended * 0.7:
            suggestions.append("验证集最高分接近阈值，存在误报风险")
        if mean > 0.4:
            suggestions.append("模型在正常图片上的平均分偏高，可能欠拟合")
        if grade == "A" and not suggestions:
            suggestions.append("仅说明已观察正常工况下的拟合较好；仍需独立业务验收、场景绑定、人工审批和可信 soak，不能直接部署")

        return QualityReport(
            grade=grade,
            score_stats=val_stats,
            threshold_recommended=threshold_recommended,
            suggestions=suggestions,
        )

    # ── TRN-005: Threshold recommendation ──

    @staticmethod
    def _recommend_threshold(val_stats: dict) -> float:
        """Recommend an anomaly threshold based on validation scores.

        Formula: max(mean + 2.5*std, max_score * 1.05), capped at [0.1, 0.95].
        """
        if not val_stats.get("scores"):
            return 0.7

        mean = val_stats.get("mean", 0.0)
        std = val_stats.get("std", 0.0)
        max_score = val_stats.get("max", 0.0)

        threshold = max(mean + 2.5 * std, max_score * 1.05)
        threshold = min(threshold, 0.95)
        threshold = max(threshold, 0.1)
        return round(threshold, 4)

    # ── TRN-006: Output validation ──

    def _validate_output(
        self, output_dir: Path, export_path: Path | None, val_dir: Path,
        artifact_path: Path | None = None,
        *, inference_backend: str = "auto",
    ) -> tuple[dict, object | None]:
        """Validate training output: checkpoint integrity, export files, smoke test.

        Returns (validation_dict, loaded_detector_or_None) so the caller
        can reuse the loaded detector for val-set evaluation.
        """
        result = {
            "checkpoint_valid": False,
            "export_valid": False,
            "smoke_test_passed": False,
            "inference_latency_ms": None,
            "errors": [],
        }
        loaded_detector = None

        # Search export directory first (inference-ready formats), then training output
        model_file = artifact_path
        if model_file is None and export_path and export_path.exists():
            model_file = self._find_best_model_file(export_path)
        if model_file is None:
            model_file = self._find_best_model_file(output_dir)
        if model_file is None:
            result["errors"].append("未找到模型文件")
        elif model_file.stat().st_size == 0:
            result["errors"].append("模型文件大小为 0")
        else:
            result["checkpoint_valid"] = True

        if export_path and export_path.exists():
            export_files = (
                list(export_path.rglob("*.xml"))
                + list(export_path.rglob("*.onnx"))
                + list(export_path.rglob("*.pt"))
            )
            if export_files and all(f.stat().st_size > 0 for f in export_files):
                result["export_valid"] = True
            else:
                result["errors"].append("导出文件缺失或为空")
        elif export_path:
            result["errors"].append(f"导出目录不存在: {export_path}")

        # Smoke inference test — keep detector loaded for reuse
        if model_file:
            val_normal = val_dir / "normal" if (val_dir / "normal").exists() else val_dir
            test_images = _list_images(val_normal)
            if test_images:
                try:
                    from argus.anomaly.detector import AnomalibDetector

                    # Validation/calibration must observe this artifact's
                    # native scores, never inherit a shared ancestor's old
                    # calibration or calibrate already calibrated scores.
                    detector = AnomalibDetector(
                        model_path=model_file, threshold=0.5, enable_calibration=False,
                        inference_backend=inference_backend,
                    )
                    if detector.load():
                        frame = cv2.imread(str(test_images[0]))
                        if frame is not None:
                            t0 = time.monotonic()
                            pred = detector.predict(frame)
                            latency = (time.monotonic() - t0) * 1000
                            if (
                                getattr(pred, "detection_failed", False) is not True
                                and np.isfinite(pred.anomaly_score)
                                and 0 <= pred.anomaly_score <= 1
                            ):
                                result["smoke_test_passed"] = True
                                result["inference_latency_ms"] = round(latency, 1)
                                loaded_detector = detector
                            else:
                                result["errors"].append(f"推理结果异常: score={pred.anomaly_score}")
                except Exception as e:
                    result["errors"].append(f"冒烟测试失败: {e}")

        return result, loaded_detector

    @staticmethod
    def _sync_calibration_to_export(
        calibration_path: Path,
        export_path: Path | None,
    ) -> list[Path]:
        """Copy calibration beside exported model files for detector discovery.

        ``AnomalibDetector`` searches for ``calibration.json`` from the model
        file's directory upward. Training writes calibration into the model
        output directory, while deployment usually loads the OpenVINO export.
        Keep the export self-contained so loading ``.../model.xml`` cannot
        silently run uncalibrated.
        """
        if export_path is None or not export_path.exists():
            return []
        if not calibration_path.exists():
            return []

        target_dirs = {export_path}
        for model_file in export_path.rglob("*"):
            if model_file.suffix.lower() in {".xml", ".onnx", ".pt"}:
                target_dirs.add(model_file.parent)

        copied: list[Path] = []
        source_resolved = calibration_path.resolve()
        for target_dir in sorted(target_dirs):
            target = target_dir / "calibration.json"
            if target.exists() and target.resolve() == source_resolved:
                copied.append(target)
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(calibration_path, target)
            copied.append(target)

        return copied

    # ── TRN-008: Model A/B comparison ──

    def compare_models(
        self, old_model_path: str | Path, new_model_path: str | Path, val_dir: str | Path
    ) -> dict:
        """Compare two models on the same validation set.

        Returns score distributions, latency comparison, and recommendation.
        """
        from argus.anomaly.detector import AnomalibDetector

        val_path = Path(val_dir)
        val_normal = val_path / "normal" if (val_path / "normal").exists() else val_path
        val_images = _list_images(val_normal)

        if not val_images:
            return {"error": "验证集为空", "recommendation": "inconclusive"}

        results = {}
        detectors: list[AnomalibDetector] = []
        try:
            for label, model_path in [("old", Path(old_model_path)), ("new", Path(new_model_path))]:
                detector = AnomalibDetector(model_path=model_path, threshold=0.5)
                detectors.append(detector)
                if not detector.load():
                    return {"error": f"无法加载{label}模型: {model_path}", "recommendation": "inconclusive"}

                scores = []
                total_latency = 0.0
                for img_path in val_images:
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        continue
                    t0 = time.monotonic()
                    pred = detector.predict(frame)
                    total_latency += time.monotonic() - t0
                    scores.append(pred.anomaly_score)

                arr = np.array(scores) if scores else np.array([0.0])
                avg_latency = (total_latency / len(scores) * 1000) if scores else 0.0

                results[label] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "max": float(arr.max()),
                    "p95": float(np.percentile(arr, 95)),
                    "latency_ms": round(avg_latency, 1),
                }
        finally:
            for d in detectors:
                try:
                    d.unload()
                except Exception:
                    pass
            del detectors
            gc.collect()

        old_s = results["old"]
        new_s = results["new"]
        if new_s["mean"] <= old_s["mean"] and new_s["max"] <= old_s["max"]:
            recommendation = "deploy"
            reason = "新模型在验证集上的平均分和最高分均不高于旧模型"
        elif new_s["mean"] > old_s["mean"] * 1.2:
            recommendation = "keep_old"
            reason = "新模型平均分显著高于旧模型，建议保留旧模型"
        else:
            recommendation = "review"
            reason = "新旧模型表现接近，建议人工评估"

        return {
            "old": old_s,
            "new": new_s,
            "delta_mean": round(new_s["mean"] - old_s["mean"], 4),
            "delta_max": round(new_s["max"] - old_s["max"], 4),
            "delta_latency_ms": round(new_s["latency_ms"] - old_s["latency_ms"], 1),
            "recommendation": recommendation,
            "reason": reason,
        }

    # ── Internal helpers ──

    @staticmethod
    def _find_best_model_file(model_dir: Path) -> Path | None:
        """Find the best model file in a training output directory.

        Preference: .xml (OpenVINO) > .onnx > .pt (Torch) > .ckpt (Lightning)
        """
        if model_dir.is_file():
            return model_dir if model_dir.suffix in {".xml", ".onnx", ".pt", ".ckpt"} else None
        for pattern in ("**/*.xml", "**/*.onnx", "**/*.pt", "**/*.ckpt"):
            files = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                return files[0]
        return None

    def _train_anomalib(
        self,
        data_dir: Path,
        output_dir: Path,
        model_type: str,
        image_size: int,
        anomaly_config: object | None = None,
        resume_from: str | None = None,
        backbone_checkpoint: str | None = None,
    ) -> tuple:
        """Execute Anomalib training. Raises ImportError if anomalib is not installed.

        Returns (engine, model) so the caller can use them for export.
        data_dir should be the train split directory (containing a 'normal' subdirectory).

        Args:
            anomaly_config: Optional AnomalyConfig for model-specific parameters.
        """
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage
        import torch

        # Explicit deployment policies apply to training too. Lightning's
        # default accelerator="auto" would otherwise silently train on CPU
        # when a required CUDA device is unavailable.
        backend = getattr(anomaly_config, "inference_backend", "auto")
        device_kwargs = _training_device_kwargs(backend, cuda_available=torch.cuda.is_available())

        # Free GPU memory used by inference models (YOLO, anomaly detector)
        # so training has enough VRAM. Models are reloaded after training.
        if torch.cuda.is_available():
            self._free_gpu_for_training()

        resize_transform = Compose([
            ToImage(),
            Resize((image_size, image_size)),
            ToDtype(torch.float32, scale=True),
        ])

        # Auto-scale batch size based on available GPU memory.
        # DINOv2-based models (Dinomaly) need ~4GB base + ~200MB/sample.
        # PatchCore needs ~2GB base + ~100MB/sample.
        batch_size = 32
        if torch.cuda.is_available():
            free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
            if free_mb < 4096:
                batch_size = 4
            elif free_mb < 8192:
                batch_size = 8
            elif free_mb < 12288:
                batch_size = 16
            logger.info("trainer.batch_size_auto", batch_size=batch_size, free_mb=free_mb)

        datamodule = Folder(
            name="baseline",
            root=str(data_dir),
            normal_dir="normal",
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=0,
            augmentations=resize_transform,
        )

        if model_type == "efficient_ad":
            from anomalib.models import EfficientAd
            model = EfficientAd()
        elif model_type == "fastflow":
            from anomalib.models import Fastflow
            model = Fastflow(
                backbone="resnet18",  # Lightweight for real-time
                flow_steps=8,
            )
        elif model_type == "padim":
            from anomalib.models import Padim
            model = Padim(
                backbone="resnet18",
                layers=["layer1", "layer2", "layer3"],
            )
        elif model_type == "dinomaly2":
            # Dinomaly2 uses DINOv2 backbone with reconstruction-based anomaly detection
            try:
                from anomalib.models import Dinomaly

                # Build constructor kwargs from anomaly_config
                dinomaly_kwargs: dict = {}
                if anomaly_config is not None:
                    backbone = getattr(anomaly_config, "dinomaly_backbone", None)
                    if backbone:
                        dinomaly_kwargs["encoder_name"] = _resolve_dinomaly_backbone(backbone)
                    encoder_layers = getattr(anomaly_config, "dinomaly_encoder_layers", None)
                    if encoder_layers:
                        dinomaly_kwargs["target_layers"] = encoder_layers

                model = Dinomaly(**dinomaly_kwargs)
                logger.info(
                    "trainer.dinomaly_created",
                    **{k: str(v) for k, v in dinomaly_kwargs.items()},
                )

                if backbone_checkpoint:
                    try:
                        import torch
                        state_dict = torch.load(
                            backbone_checkpoint, map_location="cpu", weights_only=True,
                        )
                        # The selected backbone is part of the model's
                        # provenance. Partial/zero-key loads must not silently
                        # train and register a different pretrained encoder.
                        model.model.encoder.load_state_dict(state_dict, strict=True)
                        logger.info(
                            "trainer.custom_backbone_loaded",
                            path=backbone_checkpoint,
                        )
                    except Exception as e:
                        logger.error(
                            "trainer.custom_backbone_failed",
                            path=backbone_checkpoint,
                            error=str(e),
                        )
                        raise RuntimeError(
                            f"Selected backbone checkpoint could not be loaded: {backbone_checkpoint}"
                        ) from e
            except ImportError:
                raise ImportError(
                    "Dinomaly 模型在当前 anomalib 版本中不可用。"
                    "请升级 anomalib (pip install anomalib>=1.2) 或选择其他模型类型 (patchcore/efficient_ad)。"
                )
        else:  # patchcore (default)
            from anomalib.models import Patchcore
            model = Patchcore(
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                coreset_sampling_ratio=0.1,
            )

        # FastFlow and EfficientAD need multiple epochs; PatchCore/Padim/Dinomaly need 1
        if model_type in ("efficient_ad", "fastflow"):
            max_epochs = 70
        else:
            max_epochs = 1  # PatchCore, Padim, and Dinomaly only need feature extraction

        # Early stopping + checkpointing for multi-epoch models
        callbacks = []
        if max_epochs > 1:
            try:
                from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

                callbacks.append(EarlyStopping(
                    monitor="image_AUROC",
                    patience=10,
                    mode="max",
                    min_delta=0.001,
                ))
                callbacks.append(ModelCheckpoint(
                    monitor="image_AUROC",
                    mode="max",
                    save_top_k=1,
                    filename="best-{epoch}-{image_AUROC:.4f}",
                ))
                logger.info("trainer.callbacks_enabled", model_type=model_type, max_epochs=max_epochs)
            except ImportError:
                logger.warning("trainer.callbacks_unavailable", msg="EarlyStopping/ModelCheckpoint not available")

        # H-1: Add a timer callback to enforce training timeout
        try:
            from datetime import timedelta as _td

            from lightning.pytorch.callbacks import Timer

            # Multi-epoch models get 2h, single-epoch feature extractors get 30min
            timeout_seconds = 7200 if max_epochs > 1 else 1800
            timer = Timer(duration=_td(seconds=timeout_seconds))
            callbacks.append(timer)
            logger.info(
                "trainer.timeout_set",
                timeout_seconds=timeout_seconds,
                max_epochs=max_epochs,
            )
        except ImportError:
            logger.warning("trainer.timer_unavailable", msg="Lightning Timer callback not available")

        engine = Engine(
            default_root_dir=str(output_dir),
            max_epochs=max_epochs,
            callbacks=callbacks if callbacks else None,
            **device_kwargs,
        )

        # Checkpoint resumption
        fit_kwargs = {"model": model, "datamodule": datamodule}
        if resume_from:
            ckpt = Path(resume_from)
            if not ckpt.exists():
                logger.warning("trainer.checkpoint_not_found", path=resume_from)
            elif ckpt.stat().st_size < 1024:
                logger.warning(
                    "trainer.checkpoint_too_small",
                    path=resume_from,
                    size=ckpt.stat().st_size,
                )
            else:
                fit_kwargs["ckpt_path"] = resume_from
                logger.info("trainer.resuming_from_checkpoint", path=resume_from)

        engine.fit(**fit_kwargs)

        return engine, model

    # ── Export & Quantize ──
    # The implementations live in argus.anomaly._trainer_export. They are
    # re-mounted as static methods at the bottom of this module so existing
    # call sites (ModelTrainer._export_model / _assert_openvino_static /
    # _quantize_int8) continue to work unchanged.

    # ── Re-export & Re-calibrate ──

    @staticmethod
    def _find_checkpoint(model_dir: Path, *, legacy_only: bool = False) -> Path | None:
        """Find the best Lightning checkpoint in a model directory."""
        if model_dir.is_file():
            return model_dir if model_dir.suffix == ".ckpt" else None
        ckpts = sorted(
            (
                path for path in model_dir.rglob("*.ckpt")
                if path.is_file() and (
                    not legacy_only
                    or not any(part.startswith("run-") for part in path.relative_to(model_dir).parts)
                )
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if legacy_only and len(ckpts) > 1:
            raise ValueError("旧模型目录包含多个checkpoint，请指定精确checkpoint或版本清单")
        return ckpts[0] if ckpts else None

    @staticmethod
    def _instantiate_model(model_type: str):
        """Instantiate an Anomalib model class by type name."""
        if model_type == "efficient_ad":
            from anomalib.models import EfficientAd
            return EfficientAd()
        elif model_type == "fastflow":
            from anomalib.models import Fastflow
            return Fastflow(backbone="resnet18", flow_steps=8)
        elif model_type == "padim":
            from anomalib.models import Padim
            return Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
        elif model_type == "dinomaly2":
            try:
                from anomalib.models import Dinomaly
                return Dinomaly()
            except ImportError:
                from anomalib.models import Patchcore
                return Patchcore(
                    backbone="wide_resnet50_2",
                    layers=["layer2", "layer3"],
                    coreset_sampling_ratio=0.1,
                )
        else:  # patchcore (default)
            from anomalib.models import Patchcore
            return Patchcore(
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                coreset_sampling_ratio=0.1,
            )

    def reexport_model(
        self,
        model_dir: Path,
        export_format: str = "openvino",
        quantization: str = "fp16",
        model_type: str = "patchcore",
        image_size: int = 256,
        calibration_images: int = 100,
        camera_id: str | None = None,
        zone_id: str | None = None,
    ) -> dict:
        """Re-export a model from its checkpoint, fitting PostProcessor MinMax.

        This is the definitive fix for the MinMax normalization bug (min=inf,
        max=-inf) — engine.export() internally calls post_processor.fit().

        Args:
            model_dir: Path to the model training output directory.
            export_format: Target format: openvino, onnx, or torch.
            quantization: fp16, fp32, or int8 (int8 only for openvino).
            model_type: Anomalib model type for class instantiation.
            image_size: Input image size used during training.
            calibration_images: Number of images for INT8 calibration.

        Returns:
            Dict with status, export_path, and format.
        """
        source_path = Path(model_dir)
        metadata = artifact_metadata(source_path.parent if source_path.suffix == ".ckpt" else source_path)
        source_training_dir = training_directory(
            source_path.parent if source_path.suffix == ".ckpt" else source_path
        )
        if metadata and not source_training_dir.resolve().is_relative_to(self._models_dir.resolve()):
            return {"status": "error", "error": "模型清单中的训练目录超出配置的模型根目录"}
        try:
            ckpt = self._find_checkpoint(
                source_path if source_path.suffix == ".ckpt" else source_training_dir,
                legacy_only=not metadata,
            )
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        if ckpt is None:
            return {"status": "error", "error": "未找到 checkpoint (.ckpt) 文件"}
        if self._model_registry is None:
            return {"status": "error", "error": "模型注册服务不可用，不能创建候选版本"}

        # Use explicit IDs if provided, else derive from model_dir path structure:
        # data/models/{camera_id}/{zone_id}
        parts = source_training_dir.parts
        if zone_id is None:
            zone_id = metadata.get("zone_id") or (parts[-1] if len(parts) >= 2 else "default")
        if camera_id is None:
            camera_id = metadata.get("camera_id") or (parts[-2] if len(parts) >= 2 else "unknown")

        try:
            from anomalib.engine import Engine

            run = new_artifact_run(self._models_dir, self._exports_dir, camera_id, zone_id)
            export_path = str(run.export_dir)
            # Lightning may write beside its checkpoint while exporting. Copy
            # it into the new run so a published source stays untouched.
            derived_checkpoint = run.training_dir / "source.ckpt"
            shutil.copy2(ckpt, derived_checkpoint)
            model = self._instantiate_model(model_type)
            export_kwargs = {}
            if export_format in {"openvino", "onnx"}:
                # Force static ONNX graph (dynamic_axes={}) so the resulting
                # OpenVINO IR has a fixed [1,3,H,W] input shape.  See
                # _export_model for the full rationale — post-export
                # ov.Model.reshape() SIGSEGVs in OpenVINO 2026.1 on DINOv2
                # backbones, so static-at-export is the only safe path.
                export_kwargs["onnx_kwargs"] = {
                    "dynamo": False,
                    "dynamic_axes": {},
                }
                export_kwargs["input_size"] = (image_size, image_size)

            engine = Engine(default_root_dir=str(run.training_dir), max_epochs=1)
            returned = engine.export(
                model=model,
                export_type=export_format,
                export_root=export_path,
                ckpt_path=str(derived_checkpoint),
                **export_kwargs,
            )
            artifact = exported_artifact(run.export_dir, export_format, returned)
            if export_format == "openvino":
                self._assert_openvino_static(run.export_dir)
            baseline_dir = self._baseline_manager.get_baseline_dir(camera_id, zone_id)
            actual_quantization = quantization
            if quantization == "int8" and export_format == "openvino":
                if not self._quantize_int8(
                    export_path=run.export_dir,
                    val_dir=baseline_dir if baseline_dir.is_dir() else None,
                    calibration_images=calibration_images,
                ):
                    actual_quantization = "fp32"
            elif export_format != "openvino":
                actual_quantization = "fp32"

            # Re-export can change normalization or quantization. Thresholds
            # from the source artifact must not be copied onto new scores.
            calibration_status = "skipped"
            calibration_dir = (
                baseline_dir / "normal" if (baseline_dir / "normal").is_dir() else baseline_dir
            )
            if len(_list_images(calibration_dir)) >= 50:
                from argus.alerts.calibration import ConformalCalibrator
                from argus.anomaly.detector import AnomalibDetector

                detector = AnomalibDetector(model_path=artifact, threshold=0.5, enable_calibration=False)
                if not detector.load():
                    raise RuntimeError("Re-exported model failed calibration load")
                if getattr(detector, "_minmax_broken", False):
                    raise RuntimeError("Re-exported model still has invalid normalization parameters")
                scores = self._validate_on_val_set(calibration_dir, detector=detector).get("scores", [])
                if len(scores) < 50:
                    raise RuntimeError("Re-exported model produced insufficient calibration scores")
                calibrator = ConformalCalibrator()
                score_array = np.asarray(scores)
                cal_result = calibrator.calibrate(score_array)
                cal_path = run.training_dir / "calibration.json"
                calibrator.save(cal_result, cal_path, sorted_scores=np.sort(score_array))
                self._sync_calibration_to_export(cal_path, run.export_dir)
                calibration_status = "ok"
            source_quality = inherited_baseline_quality(metadata, source_model_path=str(source_path))
            manifest = write_artifact_manifest(
                run, artifact, model_type=model_type,
                calibration_status=calibration_status,
                source_model_path=str(source_path),
                baseline_quality=source_quality,
            )
            version_id = self._model_registry.register(
                model_path=artifact,
                baseline_dir=baseline_dir,
                camera_id=camera_id,
                model_type=model_type,
                training_params={
                    "zone_id": zone_id,
                    "run_id": run.run_id,
                    "image_size": image_size,
                    "export_format": export_format,
                    "quantization": actual_quantization,
                    "derived_operation": "reexport",
                    "baseline_quality": source_quality,
                    "artifact_manifest": manifest,
                },
                registered_by="model_reexport",
            )
        except Exception as e:
            logger.error("reexport.failed", error=str(e))
            return {"status": "error", "error": f"导出失败: {e}"}

        return {
            "status": "ok",
            "export_path": export_path,
            "model_path": str(artifact),
            "model_version_id": version_id,
            "run_id": run.run_id,
            "format": export_format,
            "quantization": actual_quantization,
        }

    def recalibrate_model(
        self,
        model_dir: Path,
        baseline_dir: Path,
        camera_id: str,
        zone_id: str = "default",
        model_type: str = "patchcore",
    ) -> dict:
        """Re-calibrate score normalization using baseline images.

        Creates a new candidate with conformal thresholds in calibration.json.
        Broken PostProcessor normalization requires re-export first; an
        in-memory sigmoid fit is not a persistent deployment artifact.

        Args:
            model_dir: Path to the model training output directory.
            baseline_dir: Path to baseline images for the camera/zone.
            camera_id: Camera identifier.
            zone_id: Zone identifier.

        Returns:
            Dict with status, n_samples, and calibration thresholds.
        """
        if self._model_registry is None:
            return {"status": "error", "error": "模型注册服务不可用，不能创建候选版本"}
        source_path = Path(model_dir)
        metadata = artifact_metadata(source_path)
        source_training_dir = training_directory(source_path)
        if metadata and not source_training_dir.resolve().is_relative_to(self._models_dir.resolve()):
            return {"status": "error", "error": "模型清单中的训练目录超出配置的模型根目录"}
        model_file = None
        if source_path.is_file() and source_path.suffix in {".pt", ".xml", ".onnx"}:
            model_file = source_path
        elif metadata.get("artifact_path"):
            model_file = Path(metadata["artifact_path"])
        else:
            # Legacy directories are supported only when the runtime artifact
            # is unambiguous. Never pick a new run by mtime from a shared root.
            legacy_export = self._exports_dir / camera_id / zone_id
            candidates = []
            for root in (source_training_dir, legacy_export):
                if root.is_dir():
                    candidates.extend(
                        file for file in root.rglob("*")
                        if file.is_file() and file.suffix in {".pt", ".xml", ".onnx"}
                        and not any(part.startswith("run-") for part in file.relative_to(root).parts)
                        and not file.stem.endswith("_fp32")
                    )
            if len(candidates) == 1:
                model_file = candidates[0]
        if model_file is None:
            return {"status": "error", "error": "未找到唯一已导出的模型文件，请指定精确模型路径"}

        if not baseline_dir.is_dir():
            return {"status": "error", "error": f"基线目录不存在: {baseline_dir}"}

        try:
            from argus.anomaly.detector import AnomalibDetector

            run = new_artifact_run(self._models_dir, self._exports_dir, camera_id, zone_id)
            # Copy only the selected runtime bundle, not a camera-wide tree.
            # Its .bin and local sidecars remain beside the new exact file.
            bundle_files = [model_file]
            if model_file.suffix == ".xml":
                bundle_files.append(model_file.with_suffix(".bin"))
            bundle_files.extend(
                model_file.parent / name
                for name in ("metadata.json", "calibration.json", "scene_profile.json")
            )
            for source in bundle_files:
                if source.is_file():
                    shutil.copy2(source, run.export_dir / source.name)
            derived_artifact = run.export_dir / model_file.name
            checkpoint = self._find_checkpoint(source_training_dir, legacy_only=not metadata)
            if checkpoint is not None:
                shutil.copy2(checkpoint, run.training_dir / "source.ckpt")
            detector = AnomalibDetector(
                model_path=derived_artifact, threshold=0.5, enable_calibration=False,
            )
            if not detector.load():
                return {"status": "error", "error": "模型加载失败"}
            if getattr(detector, "_minmax_broken", False):
                return {"status": "error", "error": "归一化参数无效，请先重新导出模型再校准"}

            # Collect anomaly scores from baseline images
            scores = []
            for img_path in sorted(baseline_dir.iterdir()):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                prediction = detector.predict(frame)
                if getattr(prediction, "detection_failed", False) is True:
                    raise RuntimeError("Inference failed while collecting calibration scores")
                score = float(prediction.anomaly_score)
                if np.isfinite(score):
                    scores.append(score)

            if len(scores) < 50:
                return {
                    "status": "error",
                    "error": f"基线图片不足: {len(scores)} 个有效分数 (需要 >= 50)",
                }

            # Conformal calibration
            from argus.alerts.calibration import ConformalCalibrator

            cal_scores = np.array(scores)
            calibrator = ConformalCalibrator()
            cal_result = calibrator.calibrate(cal_scores)

            for save_dir in (run.training_dir, run.export_dir):
                calibrator.save(
                    cal_result, save_dir / "calibration.json", sorted_scores=np.sort(cal_scores),
                )

            # Broken normalization is rejected above. In-memory sigmoid
            # fitting does not persist across reload and cannot certify a new
            # candidate artifact as calibrated.
            sigmoid_recalibrated = False

            source_quality = inherited_baseline_quality(metadata, source_model_path=str(source_path))
            manifest = write_artifact_manifest(
                run, derived_artifact, model_type=model_type,
                calibration_status="ok", source_model_path=str(source_path),
                baseline_quality=source_quality,
            )
            version_id = self._model_registry.register(
                model_path=derived_artifact,
                baseline_dir=baseline_dir,
                camera_id=camera_id,
                model_type=model_type,
                training_params={
                    "zone_id": zone_id,
                    "run_id": run.run_id,
                    "derived_operation": "recalibrate",
                    "baseline_quality": source_quality,
                    "artifact_manifest": manifest,
                },
                registered_by="model_recalibrate",
            )

            logger.info(
                "recalibrate.complete",
                camera_id=camera_id,
                zone_id=zone_id,
                n_samples=len(scores),
                sigmoid_recalibrated=sigmoid_recalibrated,
            )

            return {
                "status": "ok",
                "model_path": str(derived_artifact),
                "export_path": str(run.export_dir),
                "model_version_id": version_id,
                "run_id": run.run_id,
                "n_samples": len(scores),
                "thresholds": {
                    "info": round(cal_result.info_threshold, 4),
                    "low": round(cal_result.low_threshold, 4),
                    "medium": round(cal_result.medium_threshold, 4),
                    "high": round(cal_result.high_threshold, 4),
                },
                "sigmoid_recalibrated": sigmoid_recalibrated,
            }

        except Exception as e:
            logger.error("recalibrate.failed", error=str(e))
            return {"status": "error", "error": f"校准失败: {e}"}


# Re-mount export/quantize helpers as ModelTrainer static methods so existing
# call sites (ModelTrainer._export_model, ModelTrainer._assert_openvino_static,
# ModelTrainer._quantize_int8) keep working after the implementation moved to
# argus.anomaly._trainer_export. See that module for the implementations.
ModelTrainer._export_model = staticmethod(_te.export_model)
ModelTrainer._assert_openvino_static = staticmethod(_te.assert_openvino_static)
ModelTrainer._quantize_int8 = staticmethod(_te.quantize_int8)
