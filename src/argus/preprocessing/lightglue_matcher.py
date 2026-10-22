"""Offline ALIKED/LightGlue correspondences for the SceneGuard comparison tool.

This module never imports the optional Torch/LightGlue dependency at import time.
Constructing a matcher is an explicit, independent-process operation. It cannot
change a production SceneGuard decision, profile, model, or recovery counter.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import re
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

LIGHTGLUE_SOURCE_COMMIT = "eb42fee2d71449efb0aa5c10549752b5d75384d8"
ALIKED_WEIGHTS_URL = (
    "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth"
)
LIGHTGLUE_WEIGHTS_URL = (
    "https://github.com/cvg/LightGlue/releases/download/"
    "v0.1_arxiv/aliked_lightglue.pth"
)
_CONSTRUCTOR_LOCK = threading.Lock()


class LightGlueError(RuntimeError):
    """An explicit failure; callers must record it instead of treating it as a match."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LightGlueMatcherConfig:
    extractor_weights: Path
    extractor_sha256: str
    matcher_weights: Path
    matcher_sha256: str
    device: str = "cuda:0"
    max_keypoints: int = 2048
    process_width: int = 960
    filter_threshold: float = 0.1

    def __post_init__(self) -> None:
        for name in ("extractor_sha256", "matcher_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value):
                raise LightGlueError("invalid_config", f"{name} must be a SHA256 hex digest")
            object.__setattr__(self, name, value.lower())
        for name in ("extractor_weights", "matcher_weights"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", self.device):
            raise LightGlueError("invalid_config", "device must be cpu, cuda, or cuda:N")
        if self.device == "cuda":
            object.__setattr__(self, "device", "cuda:0")
        for name, maximum in (("max_keypoints", 20000), ("process_width", 8192)):
            value = getattr(self, name)
            minimum = 1 if name == "max_keypoints" else 64
            if (
                isinstance(value, bool) or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise LightGlueError("invalid_config", f"{name} must be in [{minimum}, {maximum}]")
        if (
            isinstance(self.filter_threshold, bool)
            or not isinstance(self.filter_threshold, (int, float))
            or not np.isfinite(self.filter_threshold)
            or not 0 <= self.filter_threshold <= 1
        ):
            raise LightGlueError("invalid_config", "filter_threshold must be finite and in [0, 1]")


@dataclass(frozen=True)
class CorrespondenceMatches:
    """One-to-one pixel coordinates in each original, unresized input image."""

    frame_points: np.ndarray
    reference_points: np.ndarray
    reference_keypoints: int
    frame_keypoints: int
    raw_matches: int
    elapsed_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


def _verified_weight_bytes(path: Path, expected_sha256: str) -> bytes:
    try:
        if not path.is_file():
            raise OSError("not a regular file")
        content = path.read_bytes()
    except OSError as exc:
        raise LightGlueError("weights_unavailable", f"Cannot read local weights: {path}") from exc
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise LightGlueError("weights_sha256_mismatch", f"Local weights SHA256 mismatch: {path}")
    return content


def _checkpoint(torch: Any, content: bytes) -> dict[str, Any]:
    """Deserialize only tensors from the exact bytes whose digest was checked."""
    try:
        state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LightGlueError(
            "invalid_checkpoint", "Local weights cannot be safely deserialized",
        ) from exc
    if (
        not isinstance(state, Mapping)
        or not state
        or any(
            not isinstance(key, str) or not torch.is_tensor(value)
            for key, value in state.items()
        )
    ):
        raise LightGlueError(
            "invalid_checkpoint", "Weights must be a nonempty tensor state dictionary",
        )
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise LightGlueError("invalid_checkpoint", "Weights contain nonfinite tensor values")
    return dict(state)


@contextmanager
def _local_weight_loader(torch: Any, states: dict[str, dict[str, Any]]) -> Iterator[list[str]]:
    """Intercept the two upstream constructor requests without any network fallback.

    Upstream ALIKED has no local-path constructor. This narrow, locked patch is
    only for the independent comparison command, never the Argus service process.
    """
    calls: list[str] = []

    def load(url: str, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if url not in states:
            raise LightGlueError(
                "offline_weights_required", f"Unexpected upstream weights URL: {url}",
            )
        calls.append(url)
        return dict(states[url])

    def refuse_download(*_args: Any, **_kwargs: Any) -> None:
        raise LightGlueError("offline_weights_required", "Model weight downloads are disabled")

    with _CONSTRUCTOR_LOCK:
        previous_loader = torch.hub.load_state_dict_from_url
        previous_download = torch.hub.download_url_to_file
        torch.hub.load_state_dict_from_url = load
        torch.hub.download_url_to_file = refuse_download
        try:
            yield calls
        finally:
            torch.hub.load_state_dict_from_url = previous_loader
            torch.hub.download_url_to_file = previous_download


def _strict_matcher_state(matcher: Any, state: dict[str, Any]) -> None:
    """Validate all learned keys, including the upstream legacy key migration."""
    renamed: dict[str, Any] = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("self_attn", "cross_attn"):
            match = re.match(rf"^{prefix}\.(\d+)\.(.+)$", key)
            if match:
                new_key = f"transformers.{match[1]}.{prefix}.{match[2]}"
                break
        if new_key in renamed:
            raise LightGlueError("invalid_checkpoint", "Colliding legacy matcher checkpoint keys")
        renamed[new_key] = value
    current = matcher.state_dict()
    # The official constructor derives this nonlearned buffer from layer index.
    # Older official weights omit it. No trained parameter may be filled in.
    if "confidence_thresholds" in current and "confidence_thresholds" not in renamed:
        renamed["confidence_thresholds"] = current["confidence_thresholds"]
    try:
        matcher.load_state_dict(renamed, strict=True)
    except Exception as exc:
        raise LightGlueError(
            "incomplete_checkpoint", "Matcher weights do not exactly cover its architecture",
        ) from exc


def _verified_source_provenance() -> dict[str, Any]:
    """Reject unreviewed packages before importing their Python initialization.

    PEP 610 provenance is an installation prerequisite, not a signature or a
    claim that the installed tree has not subsequently been modified.
    """
    try:
        distribution = importlib.metadata.distribution("lightglue")
        direct_url = distribution.read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError as exc:
        raise LightGlueError(
            "dependency_missing", "Install the pinned optional lightglue extra on the test machine",
        ) from exc
    except OSError as exc:
        raise LightGlueError(
            "source_unverified", "Cannot read LightGlue installation provenance",
        ) from exc
    try:
        provenance = json.loads(direct_url or "null")
        source = urlsplit(provenance["url"])
        vcs = provenance["vcs_info"]
        installed_commit = vcs["commit_id"]
        approved = (
            source.scheme == "https" and source.netloc.lower() == "github.com"
            and source.path.lower().removesuffix(".git") == "/cvg/lightglue"
            and not source.query and not source.fragment
            and vcs["vcs"] == "git" and installed_commit == LIGHTGLUE_SOURCE_COMMIT
        )
    except (KeyError, ValueError, TypeError, AttributeError):
        approved = False
    if not approved:
        raise LightGlueError(
            "source_unverified",
            "LightGlue must have official pinned Git installation provenance; "
            "reinstall the optional extra in the isolated comparison environment",
        )
    try:
        spec = importlib.util.find_spec("lightglue")
        expected_origin = Path(distribution.locate_file("lightglue/__init__.py")).resolve(
            strict=True,
        )
        if (
            spec is None or spec.origin is None
            or Path(spec.origin).resolve(strict=True) != expected_origin
        ):
            raise ValueError("the import origin differs from the installed distribution")
    except (OSError, ValueError, TypeError, AttributeError, ImportError) as exc:
        raise LightGlueError(
            "source_unverified",
            "LightGlue import path is missing or shadows its pinned installation",
        ) from exc
    return {
        "lightglue_version": str(distribution.version),
        "source_commit_expected": LIGHTGLUE_SOURCE_COMMIT,
        "source_commit_installed": installed_commit,
        "source_commit_verified": True,
        "source_url_installed": provenance["url"],
        "source_import_origin": str(expected_origin),
        "installed_tree_integrity_verified": False,
    }


def _runtime_versions(lightglue: Any, torch: Any, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        **provenance,
        "lightglue_version": str(
            getattr(lightglue, "__version__", None) or provenance["lightglue_version"]
        ),
        "torch_version": str(torch.__version__),
        "extractor": "aliked-n16",
        "matcher_weights_version": "v0.1_arxiv",
    }


def _single_batch_numpy(value: Any, ndim: int, name: str) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise LightGlueError("invalid_correspondences", f"{name} must contain one image batch")
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == ndim + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != ndim:
        raise LightGlueError("invalid_correspondences", f"Unexpected {name} dimensions")
    return array


def _image_points(features: Mapping[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    if "keypoints" not in features:
        raise LightGlueError("invalid_correspondences", "Extractor did not return keypoints")
    points = _single_batch_numpy(features["keypoints"], 2, "keypoints")
    if points.shape[1:] != (2,) or not (
        np.issubdtype(points.dtype, np.floating) or np.issubdtype(points.dtype, np.integer)
    ):
        raise LightGlueError("invalid_correspondences", "Keypoints must be numeric Nx2")
    points = points.astype(np.float64, copy=True)
    if (
        not np.isfinite(points).all()
        or (points < 0).any()
        or (points[:, 0] >= shape[1]).any()
        or (points[:, 1] >= shape[0]).any()
    ):
        raise LightGlueError(
            "invalid_correspondences", "Keypoints are nonfinite or outside the input image",
        )
    return points


def _require_tensor_device(value: Any, device: str, name: str) -> None:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise LightGlueError("invalid_correspondences", f"{name} must contain one image batch")
        value = value[0]
    if not hasattr(value, "detach") or not hasattr(value, "device"):
        raise LightGlueError("invalid_correspondences", f"{name} must be an inference tensor")
    if str(value.device) != device:
        raise LightGlueError(
            "device_mismatch", f"{name} was returned on {value.device}, expected {device}",
        )


class LightGlueMatcher:
    """Explicitly initialized local models and a bounded one-reference cache."""

    def __init__(self, config: LightGlueMatcherConfig) -> None:
        self.config = config
        extractor_bytes = _verified_weight_bytes(config.extractor_weights, config.extractor_sha256)
        matcher_bytes = _verified_weight_bytes(config.matcher_weights, config.matcher_sha256)
        provenance = _verified_source_provenance()
        try:
            torch = importlib.import_module("torch")
            lightglue = importlib.import_module("lightglue")
        except ImportError as exc:
            raise LightGlueError(
                "dependency_missing",
                "Install the optional lightglue extra on the test machine",
            ) from exc
        except Exception as exc:
            raise LightGlueError(
                "initialization_failed", "Optional LightGlue dependencies could not initialize",
            ) from exc
        self._torch = torch
        try:
            self._device = torch.device(config.device)
            if self._device.type == "cuda":
                if not torch.cuda.is_available() or self._device.index >= torch.cuda.device_count():
                    raise LightGlueError(
                        "device_unavailable",
                        f"Requested CUDA device is unavailable: {config.device}",
                    )
            extractor_state = _checkpoint(torch, extractor_bytes)
            matcher_state = _checkpoint(torch, matcher_bytes)
            with _local_weight_loader(torch, {
                ALIKED_WEIGHTS_URL: extractor_state,
                LIGHTGLUE_WEIGHTS_URL: matcher_state,
            }) as calls:
                extractor = lightglue.ALIKED(
                    model_name="aliked-n16", max_num_keypoints=config.max_keypoints,
                )
                matcher = lightglue.LightGlue(
                    features="aliked", filter_threshold=config.filter_threshold,
                    flash=False, mp=False,
                )
            if calls != [ALIKED_WEIGHTS_URL, LIGHTGLUE_WEIGHTS_URL]:
                raise LightGlueError(
                    "offline_weights_required",
                    "Upstream constructors did not load both authorized local states",
                )
            extractor.load_state_dict(extractor_state, strict=True)
            _strict_matcher_state(matcher, matcher_state)
            self._extractor = extractor.eval().to(self._device)
            self._matcher = matcher.eval().to(self._device)
        except LightGlueError:
            raise
        except Exception as exc:
            raise LightGlueError(
                "initialization_failed", "ALIKED/LightGlue local model initialization failed",
            ) from exc
        self._versions = _runtime_versions(lightglue, torch, provenance)
        self._lock = threading.Lock()
        self._reference_key: str | None = None
        self._reference_features: Mapping[str, Any] | None = None
        self._reference_points: np.ndarray | None = None

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (
            not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 32
        ):
            raise LightGlueError(
                "invalid_frame", "Expected a BGR uint8 HxWx3 image at least 32 pixels per side",
            )
        original = np.ascontiguousarray(image).copy()
        height, width = original.shape[:2]
        if width > self.config.process_width:
            target_height = int(round(height * self.config.process_width / width))
            if target_height < 32:
                raise LightGlueError(
                    "invalid_frame", "Resized image height would be below 32 pixels",
                )
            processed = cv2.resize(
                original, (self.config.process_width, target_height), interpolation=cv2.INTER_AREA,
            )
        else:
            processed = original
        return original, processed

    def _extract(self, image: np.ndarray) -> Mapping[str, Any]:
        rgb_chw = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))
        tensor = self._torch.from_numpy(rgb_chw).to(
            device=self._device, dtype=self._torch.float32,
        ) / 255.0
        # ALIKED's upstream default silently resizes to 1024. All resizing here
        # is explicit so exported points retain an auditable pixel transform.
        features = self._extractor.extract(tensor, resize=None)
        if not isinstance(features, Mapping):
            raise LightGlueError(
                "invalid_correspondences", "Extractor output must be a feature mapping",
            )
        for name in ("keypoints", "descriptors", "image_size"):
            if name not in features:
                raise LightGlueError("invalid_correspondences", f"Extractor did not return {name}")
            _require_tensor_device(features[name], str(self._device), name)
        points = _image_points(features, image.shape)
        descriptors = _single_batch_numpy(features["descriptors"], 2, "descriptors")
        image_size = _single_batch_numpy(features["image_size"], 1, "image_size")
        if (
            len(points) > self.config.max_keypoints
            or descriptors.shape != (len(points), 128)
            or not np.issubdtype(descriptors.dtype, np.floating)
            or not np.isfinite(descriptors).all()
        ):
            raise LightGlueError(
                "invalid_correspondences",
                "ALIKED descriptors must be finite Nx128 features aligned with keypoints",
            )
        if (
            image_size.shape != (2,)
            or not (
                np.issubdtype(image_size.dtype, np.floating)
                or np.issubdtype(image_size.dtype, np.integer)
            )
            or not np.isfinite(image_size).all()
            or not np.array_equal(image_size, [image.shape[1], image.shape[0]])
        ):
            raise LightGlueError(
                "invalid_correspondences",
                "Extractor image_size must describe the actual processed image",
            )
        return features

    def _synchronize(self) -> None:
        if self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)

    @staticmethod
    def _restore_points(
        points: np.ndarray, original: np.ndarray, processed: np.ndarray,
    ) -> np.ndarray:
        scale = np.array([
            original.shape[1] / processed.shape[1], original.shape[0] / processed.shape[0],
        ])
        # OpenCV's resize uses pixel centers, matching LightGlue's own inverse
        # preprocessor transform. Width/height rounding is handled separately.
        restored = (points + 0.5) * scale - 0.5
        if (
            not np.isfinite(restored).all() or (restored < 0).any()
            or (restored[:, 0] >= original.shape[1]).any()
            or (restored[:, 1] >= original.shape[0]).any()
        ):
            raise LightGlueError(
                "invalid_correspondences", "Restored keypoints are outside the original image",
            )
        return np.ascontiguousarray(restored, dtype=np.float64)

    def match(self, reference_bgr: np.ndarray, frame_bgr: np.ndarray) -> CorrespondenceMatches:
        reference, reference_small = self._prepare(reference_bgr)
        frame, frame_small = self._prepare(frame_bgr)
        identity = hashlib.sha256()
        identity.update(str(reference.shape).encode("ascii"))
        identity.update(reference.tobytes())
        identity.update(repr(self.config).encode("utf-8"))
        reference_key = identity.hexdigest()
        with self._lock:
            try:
                self._synchronize()
                started = time.perf_counter()
                reference_cached = self._reference_key == reference_key
                with self._torch.inference_mode():
                    if not reference_cached:
                        features = self._extract(reference_small)
                        points = _image_points(features, reference_small.shape)
                        self._reference_features = features
                        self._reference_points = points
                        self._reference_key = reference_key
                    reference_features = self._reference_features
                    reference_points = self._reference_points
                    if reference_features is None or reference_points is None:
                        raise LightGlueError(
                            "inference_failed", "Reference feature cache is unavailable",
                        )
                    frame_features = self._extract(frame_small)
                    frame_points = _image_points(frame_features, frame_small.shape)
                    if len(reference_points) == 0 or len(frame_points) == 0:
                        pairs = np.empty((0, 2), dtype=np.int64)
                        scores = np.empty(0, dtype=np.float64)
                    else:
                        output = self._matcher({
                            "image0": reference_features, "image1": frame_features,
                        })
                        if (
                            not isinstance(output, Mapping)
                            or not {"matches", "scores"} <= output.keys()
                        ):
                            raise LightGlueError(
                                "invalid_correspondences", "Matcher output lacks matches or scores",
                            )
                        for name in ("matches", "scores"):
                            _require_tensor_device(output[name], str(self._device), name)
                        pairs = _single_batch_numpy(output["matches"], 2, "matches")
                        scores = _single_batch_numpy(output["scores"], 1, "scores")
                    if (
                        pairs.shape[1:] != (2,) or not np.issubdtype(pairs.dtype, np.integer)
                        or (pairs < 0).any() or (pairs[:, 0] >= len(reference_points)).any()
                        or (pairs[:, 1] >= len(frame_points)).any()
                        or len(np.unique(pairs[:, 0])) != len(pairs)
                        or len(np.unique(pairs[:, 1])) != len(pairs)
                    ):
                        raise LightGlueError(
                            "invalid_correspondences",
                            "Matches must be unique, in-range integer index pairs",
                        )
                    if (
                        len(scores) != len(pairs) or not (
                            np.issubdtype(scores.dtype, np.floating)
                            or np.issubdtype(scores.dtype, np.integer)
                        )
                        or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
                    ):
                        raise LightGlueError(
                            "invalid_correspondences",
                            "Match scores must be finite probabilities aligned with pairs",
                        )
                    raw_matches = len(pairs)
                    accepted = scores > self.config.filter_threshold
                    pairs, scores = pairs[accepted], scores[accepted]
                    reference_matched = self._restore_points(
                        reference_points[pairs[:, 0]], reference, reference_small,
                    )
                    frame_matched = self._restore_points(
                        frame_points[pairs[:, 1]], frame, frame_small,
                    )
                    if (
                        len(np.unique(reference_matched, axis=0)) != len(pairs)
                        or len(np.unique(frame_matched, axis=0)) != len(pairs)
                    ):
                        raise LightGlueError(
                            "invalid_correspondences", "Matched pixel coordinates must be unique",
                        )
                self._synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            except LightGlueError:
                raise
            except Exception as exc:
                raise LightGlueError(
                    "inference_failed", "ALIKED/LightGlue correspondence inference failed",
                ) from exc
        metadata = {
            **self._versions,
            "device": str(self._device),
            "diagnostic_cpu": self._device.type == "cpu",
            "extractor_sha256": self.config.extractor_sha256,
            "matcher_sha256": self.config.matcher_sha256,
            "max_keypoints": self.config.max_keypoints,
            "process_width": self.config.process_width,
            "filter_threshold": self.config.filter_threshold,
            "extractor_resize": None,
            "reference_cached": reference_cached,
            "reference_content_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
            "reference_size": [reference.shape[1], reference.shape[0]],
            "reference_processed_size": [reference_small.shape[1], reference_small.shape[0]],
            "frame_size": [frame.shape[1], frame.shape[0]],
            "frame_processed_size": [frame_small.shape[1], frame_small.shape[0]],
        }
        return CorrespondenceMatches(
            frame_points=frame_matched, reference_points=reference_matched,
            reference_keypoints=len(reference_points), frame_keypoints=len(frame_points),
            raw_matches=raw_matches, elapsed_ms=elapsed_ms, metadata=metadata,
            scores=np.ascontiguousarray(scores, dtype=np.float64),
        )
