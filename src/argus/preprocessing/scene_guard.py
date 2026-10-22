"""Fixed-reference camera pose guard based on local feature geometry.

``SceneGuard`` decides whether an incoming camera frame still represents a
previously approved scene.  Unlike the vibration aligner, it never learns or
refreshes its reference from runtime frames: the reference must be supplied
explicitly by the caller.  This prevents a moved camera or an abnormal scene
from silently becoming the new normal.

The guard uses ORB (the CPU-friendly default) or SIFT keypoints, Lowe-ratio
matching, and a USAC_MAGSAC homography with a RANSAC fallback.  A homography is
accepted only when its match count, inlier ratio, spatial coverage, projection
error, overlap, and geometric transform are plausible.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np


class SceneState(str, Enum):
    """Operational state reported by :class:`SceneGuard`."""

    CALIBRATING = "calibrating"
    STABLE = "stable"
    ALIGNABLE = "alignable"
    MOVED = "moved"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class SceneGuardConfig:
    """Thresholds for fixed-reference scene verification.

    The field names intentionally match the public application configuration
    planned for the scene guard, so ``SceneGuardConfig(**config.model_dump())``
    and ``SceneGuard(reference_frame=frame, **mapping)`` require no adapter.
    Shift thresholds are expressed in original-frame pixels.
    """

    process_width: int = 960
    feature_type: str = "ORB"
    min_matches: int = 24
    min_inliers: int = 12
    min_inlier_ratio: float = 0.40
    max_reprojection_error_px: float = 3.0
    min_coverage_ratio: float = 0.06
    min_overlap_ratio: float = 0.72
    stable_max_shift_px: float = 3.0
    stable_max_corner_displacement_px: float = 3.0
    stable_max_rotation_deg: float = 0.10
    stable_max_scale_delta: float = 0.003
    alignable_max_shift_px: float = 100.0
    alignable_max_rotation_deg: float = 12.0
    alignable_max_scale_delta: float = 0.18
    consecutive_matches: int = 3

    # Feature/quality controls are deliberately conservative defaults.  They
    # remain configurable for low-texture cameras without changing the API.
    max_features: int = 3000
    ratio_test: float = 0.78
    ransac_threshold_px: float = 3.0
    max_anisotropy: float = 1.20
    max_perspective: float = 0.035
    min_frame_keypoints: int = 20
    min_brightness: float = 4.0
    max_brightness: float = 251.0
    min_contrast_std: float = 6.0
    min_blur_variance: float = 8.0
    return_aligned_frame: bool = False

    def __post_init__(self) -> None:
        feature_type = self.feature_type.upper()
        if feature_type not in {"ORB", "SIFT"}:
            raise ValueError("feature_type must be 'ORB' or 'SIFT'")
        object.__setattr__(self, "feature_type", feature_type)

        positive = {
            "process_width": self.process_width,
            "min_matches": self.min_matches,
            "min_inliers": self.min_inliers,
            "max_reprojection_error_px": self.max_reprojection_error_px,
            "min_coverage_ratio": self.min_coverage_ratio,
            "min_overlap_ratio": self.min_overlap_ratio,
            "stable_max_shift_px": self.stable_max_shift_px,
            "stable_max_corner_displacement_px": (
                self.stable_max_corner_displacement_px
            ),
            "alignable_max_shift_px": self.alignable_max_shift_px,
            "consecutive_matches": self.consecutive_matches,
            "max_features": self.max_features,
            "ransac_threshold_px": self.ransac_threshold_px,
            "max_anisotropy": self.max_anisotropy,
            "min_frame_keypoints": self.min_frame_keypoints,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        fractions = {
            "min_inlier_ratio": self.min_inlier_ratio,
            "min_coverage_ratio": self.min_coverage_ratio,
            "min_overlap_ratio": self.min_overlap_ratio,
            "ratio_test": self.ratio_test,
        }
        for name, value in fractions.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")

        if self.stable_max_shift_px > self.alignable_max_shift_px:
            raise ValueError("stable_max_shift_px cannot exceed alignable_max_shift_px")
        if self.min_inliers > self.min_matches:
            raise ValueError("min_inliers cannot exceed min_matches")
        if self.stable_max_rotation_deg > self.alignable_max_rotation_deg:
            raise ValueError(
                "stable_max_rotation_deg cannot exceed alignable_max_rotation_deg"
            )
        if self.stable_max_scale_delta > self.alignable_max_scale_delta:
            raise ValueError(
                "stable_max_scale_delta cannot exceed alignable_max_scale_delta"
            )


@dataclass(frozen=True, slots=True)
class SceneMatchMetrics:
    """Auditable evidence used to classify a frame."""

    reference_available: bool = False
    reference_keypoints: int = 0
    frame_keypoints: int = 0
    raw_matches: int = 0
    good_matches: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    reference_coverage_ratio: float = 0.0
    frame_coverage_ratio: float = 0.0
    coverage_ratio: float = 0.0
    reprojection_error_px: float = math.inf
    overlap_ratio: float = 0.0
    shift_px: float = math.inf
    max_corner_displacement_px: float = math.inf
    rotation_deg: float = math.inf
    scale: float = math.nan
    scale_delta: float = math.inf
    anisotropy: float = math.inf
    perspective: float = math.inf
    brightness: float = 0.0
    contrast_std: float = 0.0
    blur_variance: float = 0.0
    transform_reasonable: bool = False
    estimator: str = "none"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly scalar metrics."""
        values = asdict(self)
        return {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in values.items()
        }


@dataclass(frozen=True, slots=True)
class SceneGuardResult:
    """Scene classification plus optional registration output."""

    state: SceneState
    raw_state: SceneState
    reason: str
    metrics: SceneMatchMetrics
    aligned_frame: np.ndarray | None = None
    homography: np.ndarray | None = None

    @property
    def usable(self) -> bool:
        """Whether downstream detection may safely use the fixed-scene model."""
        return self.state in {SceneState.STABLE, SceneState.ALIGNABLE}


class SceneGuard:
    """Verify camera pose against an explicitly approved reference frame.

    Parameters use the same names as :class:`SceneGuardConfig`.  Passing a
    ``config`` is convenient internally, while keyword arguments make direct
    mapping from YAML/Pydantic straightforward.  ``reference_frame`` may be
    omitted, but :meth:`evaluate` then always returns ``UNKNOWN`` until
    :meth:`set_reference` is explicitly called.
    """

    def __init__(
        self,
        reference_frame: np.ndarray | None = None,
        *,
        config: SceneGuardConfig | None = None,
        process_width: int = 960,
        feature_type: str = "ORB",
        min_matches: int = 24,
        min_inliers: int = 12,
        min_inlier_ratio: float = 0.40,
        max_reprojection_error_px: float = 3.0,
        min_coverage_ratio: float = 0.06,
        min_overlap_ratio: float = 0.72,
        stable_max_shift_px: float = 3.0,
        stable_max_corner_displacement_px: float = 3.0,
        stable_max_rotation_deg: float = 0.10,
        stable_max_scale_delta: float = 0.003,
        alignable_max_shift_px: float = 100.0,
        alignable_max_rotation_deg: float = 12.0,
        alignable_max_scale_delta: float = 0.18,
        consecutive_matches: int = 3,
        return_aligned_frame: bool = False,
        **advanced: Any,
    ) -> None:
        if config is not None and advanced:
            raise ValueError("advanced keyword options cannot be combined with config")
        self.config = config or SceneGuardConfig(
            process_width=process_width,
            feature_type=feature_type,
            min_matches=min_matches,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
            max_reprojection_error_px=max_reprojection_error_px,
            min_coverage_ratio=min_coverage_ratio,
            min_overlap_ratio=min_overlap_ratio,
            stable_max_shift_px=stable_max_shift_px,
            stable_max_corner_displacement_px=stable_max_corner_displacement_px,
            stable_max_rotation_deg=stable_max_rotation_deg,
            stable_max_scale_delta=stable_max_scale_delta,
            alignable_max_shift_px=alignable_max_shift_px,
            alignable_max_rotation_deg=alignable_max_rotation_deg,
            alignable_max_scale_delta=alignable_max_scale_delta,
            consecutive_matches=consecutive_matches,
            return_aligned_frame=return_aligned_frame,
            **advanced,
        )

        self._detector, self._norm = self._create_detector(self.config)
        self._reference_frame: np.ndarray | None = None
        self._reference_gray: np.ndarray | None = None
        self._reference_keypoints: list[cv2.KeyPoint] = []
        self._reference_descriptors: np.ndarray | None = None
        self._reference_scale = (1.0, 1.0)

        self._confirmed_state = SceneState.UNKNOWN
        self._pending_state: SceneState | None = None
        self._pending_count = 0
        self._safe_streak = 0

        if reference_frame is not None:
            self.set_reference(reference_frame)

    @staticmethod
    def _create_detector(
        config: SceneGuardConfig,
    ) -> tuple[cv2.Feature2D, int]:
        if config.feature_type == "SIFT":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("this OpenCV build does not provide SIFT")
            return cv2.SIFT_create(nfeatures=config.max_features), cv2.NORM_L2
        return (
            cv2.ORB_create(
                nfeatures=config.max_features,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=19,
                fastThreshold=12,
            ),
            cv2.NORM_HAMMING,
        )

    @property
    def has_reference(self) -> bool:
        return self._reference_frame is not None

    @property
    def reference_frame(self) -> np.ndarray | None:
        """Return a defensive copy of the approved reference frame."""
        if self._reference_frame is None:
            return None
        return self._reference_frame.copy()

    @property
    def state(self) -> SceneState:
        if not self.has_reference:
            return SceneState.UNKNOWN
        return self._confirmed_state

    def set_reference(self, reference_frame: np.ndarray) -> None:
        """Explicitly install an approved reference; runtime frames are never used."""
        self._validate_frame(reference_frame)
        gray_full = self._to_gray(reference_frame)
        gray, scale = self._resize_for_processing(gray_full)
        keypoints, descriptors = self._extract_features(gray)
        if descriptors is None or len(keypoints) < self.config.min_frame_keypoints:
            raise ValueError(
                "reference frame has insufficient local features "
                f"({len(keypoints)} < {self.config.min_frame_keypoints})"
            )

        self._reference_frame = reference_frame.copy()
        self._reference_gray = gray
        self._reference_scale = scale
        self._reference_keypoints = keypoints
        self._reference_descriptors = descriptors
        self._confirmed_state = SceneState.CALIBRATING
        self._pending_state = None
        self._pending_count = 0
        self._safe_streak = 0

    def clear_reference(self) -> None:
        """Remove the approved reference without adopting a replacement."""
        self._reference_frame = None
        self._reference_gray = None
        self._reference_keypoints = []
        self._reference_descriptors = None
        self._reference_scale = (1.0, 1.0)
        self._confirmed_state = SceneState.UNKNOWN
        self._pending_state = None
        self._pending_count = 0
        self._safe_streak = 0

    def require_reconfirmation(self) -> None:
        """Keep the approved reference but require a fresh safe-frame streak.

        Reconnects and stream-identity changes cross a capture trust boundary:
        the next frame may come from a different device even when its
        resolution is unchanged.  This method deliberately does not replace
        or relearn the reference; it only resets the release debounce.
        """
        if not self.has_reference:
            self._confirmed_state = SceneState.UNKNOWN
        else:
            self._confirmed_state = SceneState.CALIBRATING
        self._pending_state = None
        self._pending_count = 0
        self._safe_streak = 0

    def evaluate(
        self,
        frame: np.ndarray,
        *,
        return_aligned: bool | None = None,
    ) -> SceneGuardResult:
        """Classify ``frame`` and optionally warp it into reference coordinates."""
        self._validate_frame(frame)
        want_aligned = (
            self.config.return_aligned_frame
            if return_aligned is None
            else bool(return_aligned)
        )

        if not self.has_reference:
            return SceneGuardResult(
                state=SceneState.UNKNOWN,
                raw_state=SceneState.UNKNOWN,
                reason="no approved reference frame",
                metrics=SceneMatchMetrics(reference_available=False),
            )

        assert self._reference_frame is not None
        assert self._reference_descriptors is not None

        if frame.shape[:2] != self._reference_frame.shape[:2]:
            return self._result(
                SceneState.MOVED,
                "frame resolution differs from the approved scene reference",
                SceneMatchMetrics(
                    reference_available=True,
                    reference_keypoints=len(self._reference_keypoints),
                ),
            )

        gray_full = self._to_gray(frame)
        gray, frame_scale = self._resize_for_processing(gray_full)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        base = {
            "reference_available": True,
            "reference_keypoints": len(self._reference_keypoints),
            "brightness": brightness,
            "contrast_std": contrast,
            "blur_variance": blur,
        }

        quality_reason = self._blocked_reason(brightness, contrast, blur)
        if quality_reason is not None:
            metrics = SceneMatchMetrics(**base)
            return self._result(SceneState.BLOCKED, quality_reason, metrics)

        frame_keypoints, frame_descriptors = self._extract_features(gray)
        base["frame_keypoints"] = len(frame_keypoints)
        if (
            frame_descriptors is None
            or len(frame_keypoints) < self.config.min_frame_keypoints
        ):
            metrics = SceneMatchMetrics(**base)
            return self._result(
                SceneState.BLOCKED,
                "frame is occluded, defocused, or has too few local features",
                metrics,
            )

        matcher = cv2.BFMatcher(self._norm, crossCheck=False)
        try:
            knn_matches = matcher.knnMatch(
                frame_descriptors,
                self._reference_descriptors,
                k=2,
            )
        except cv2.error:
            knn_matches = []

        # A single reference feature may otherwise attract several frame
        # features.  Retaining the lowest-distance match makes the geometry
        # less vulnerable to repeated textures.
        unique_matches: dict[int, cv2.DMatch] = {}
        for pair in knn_matches:
            if len(pair) != 2:
                continue
            best, second = pair
            if best.distance >= self.config.ratio_test * second.distance:
                continue
            previous = unique_matches.get(best.trainIdx)
            if previous is None or best.distance < previous.distance:
                unique_matches[best.trainIdx] = best
        good_matches = list(unique_matches.values())
        base.update(raw_matches=len(knn_matches), good_matches=len(good_matches))

        if len(good_matches) < self.config.min_matches:
            metrics = SceneMatchMetrics(**base)
            return self._result(
                SceneState.UNKNOWN,
                f"only {len(good_matches)} reliable feature matches",
                metrics,
            )

        frame_points_small = np.float32(
            [frame_keypoints[match.queryIdx].pt for match in good_matches]
        )
        reference_points_small = np.float32(
            [self._reference_keypoints[match.trainIdx].pt for match in good_matches]
        )
        homography_small, mask, estimator = self._find_homography(
            frame_points_small,
            reference_points_small,
        )
        base["estimator"] = estimator
        if homography_small is None or mask is None:
            metrics = SceneMatchMetrics(**base)
            return self._result(
                SceneState.UNKNOWN,
                "feature matches do not form a consistent homography",
                metrics,
            )

        homography = self._to_full_resolution_homography(
            homography_small,
            frame_scale,
            self._reference_scale,
        )
        inlier_mask = mask.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inlier_count / max(1, len(good_matches))

        frame_points = self._points_to_full(frame_points_small, frame_scale)
        reference_points = self._points_to_full(
            reference_points_small,
            self._reference_scale,
        )
        frame_inliers = frame_points[inlier_mask]
        reference_inliers = reference_points[inlier_mask]
        frame_coverage = self._coverage(frame_inliers, frame.shape[:2])
        reference_coverage = self._coverage(
            reference_inliers,
            self._reference_frame.shape[:2],
        )
        coverage = min(frame_coverage, reference_coverage)
        reprojection_error = self._reprojection_error(
            frame_inliers,
            reference_inliers,
            homography,
        )
        overlap = self._overlap_ratio(
            homography,
            frame.shape[:2],
            self._reference_frame.shape[:2],
        )
        transform = self._transform_metrics(
            homography,
            frame.shape[:2],
            self._reference_frame.shape[:2],
        )
        transform_reason = self._unreasonable_transform_reason(
            shift_px=transform["shift_px"],
            rotation_deg=transform["rotation_deg"],
            scale_delta=transform["scale_delta"],
            anisotropy=transform["anisotropy"],
            perspective=transform["perspective"],
            overlap_ratio=overlap,
        )

        metrics = SceneMatchMetrics(
            **base,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            reference_coverage_ratio=reference_coverage,
            frame_coverage_ratio=frame_coverage,
            coverage_ratio=coverage,
            reprojection_error_px=reprojection_error,
            overlap_ratio=overlap,
            transform_reasonable=transform_reason is None,
            **transform,
        )

        evidence_failures: list[str] = []
        minimum_inliers = max(
            self.config.min_inliers,
            math.ceil(self.config.min_matches * self.config.min_inlier_ratio),
        )
        if inlier_count < minimum_inliers:
            evidence_failures.append(f"inliers {inlier_count} < {minimum_inliers}")
        if inlier_ratio < self.config.min_inlier_ratio:
            evidence_failures.append(
                f"inlier ratio {inlier_ratio:.3f} < {self.config.min_inlier_ratio:.3f}"
            )
        if coverage < self.config.min_coverage_ratio:
            evidence_failures.append(
                f"coverage {coverage:.3f} < {self.config.min_coverage_ratio:.3f}"
            )
        if reprojection_error > self.config.max_reprojection_error_px:
            evidence_failures.append(
                "reprojection error "
                f"{reprojection_error:.2f}px > {self.config.max_reprojection_error_px:.2f}px"
            )
        if evidence_failures:
            return self._result(
                SceneState.UNKNOWN,
                "; ".join(evidence_failures),
                metrics,
                homography=homography,
            )

        if transform_reason is not None:
            return self._result(
                SceneState.MOVED,
                transform_reason,
                metrics,
                homography=homography,
            )

        is_stable = (
            metrics.shift_px <= self.config.stable_max_shift_px
            and metrics.max_corner_displacement_px
            <= self.config.stable_max_corner_displacement_px
            and abs(metrics.rotation_deg) <= self.config.stable_max_rotation_deg
            and metrics.scale_delta <= self.config.stable_max_scale_delta
        )
        raw_state = SceneState.STABLE if is_stable else SceneState.ALIGNABLE
        reason = "scene matches approved camera pose" if is_stable else "scene can be registered"
        aligned = None
        if want_aligned:
            height, width = self._reference_frame.shape[:2]
            aligned = cv2.warpPerspective(
                frame,
                homography,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        return self._result(
            raw_state,
            reason,
            metrics,
            aligned_frame=aligned,
            homography=homography,
        )

    def _result(
        self,
        raw_state: SceneState,
        reason: str,
        metrics: SceneMatchMetrics,
        *,
        aligned_frame: np.ndarray | None = None,
        homography: np.ndarray | None = None,
    ) -> SceneGuardResult:
        state, progress = self._debounce(raw_state)
        if state is SceneState.CALIBRATING:
            reason = (
                f"awaiting {raw_state.value} confirmation "
                f"({progress}/{self.config.consecutive_matches}): {reason}"
            )
        return SceneGuardResult(
            state=state,
            raw_state=raw_state,
            reason=reason,
            metrics=metrics,
            aligned_frame=aligned_frame,
            homography=None if homography is None else homography.copy(),
        )

    def _debounce(self, candidate: SceneState) -> tuple[SceneState, int]:
        safe_states = {SceneState.STABLE, SceneState.ALIGNABLE}
        required = self.config.consecutive_matches

        # Unsafe evidence isolates the old model immediately.  Debouncing a
        # moved/unknown/blocked frame would knowingly permit detection against
        # the wrong scene for N more frames.
        if candidate not in safe_states:
            self._confirmed_state = candidate
            self._pending_state = None
            self._pending_count = 0
            self._safe_streak = 0
            return candidate, 1

        if required == 1:
            self._confirmed_state = candidate
            self._pending_state = None
            self._pending_count = 0
            self._safe_streak = 1
            return candidate, 1

        # Once released, STABLE and ALIGNABLE are both geometrically safe; a
        # label change between them must not restart calibration forever.
        if self._confirmed_state in safe_states:
            self._confirmed_state = candidate
            self._pending_state = None
            self._pending_count = 0
            self._safe_streak = required
            return candidate, required

        # After startup or any unsafe frame, require N consecutive safe frames.
        # STABLE/ALIGNABLE may alternate while contributing to the same streak.
        self._safe_streak += 1
        self._pending_state = candidate
        self._pending_count = self._safe_streak
        if self._safe_streak >= required:
            self._confirmed_state = candidate
            self._pending_state = None
            self._pending_count = 0
            self._safe_streak = required
            return candidate, required

        return SceneState.CALIBRATING, self._safe_streak

    def _find_homography(
        self,
        source: np.ndarray,
        destination: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        methods: list[tuple[int, str]] = []
        if hasattr(cv2, "USAC_MAGSAC"):
            methods.append((cv2.USAC_MAGSAC, "USAC_MAGSAC"))
        methods.append((cv2.RANSAC, "RANSAC"))

        for method, name in methods:
            try:
                homography, mask = cv2.findHomography(
                    source,
                    destination,
                    method,
                    self.config.ransac_threshold_px,
                    maxIters=10_000,
                    confidence=0.999,
                )
            except cv2.error:
                continue
            if homography is not None and mask is not None and np.all(np.isfinite(homography)):
                return homography, mask, name
        return None, None, "none"

    def _unreasonable_transform_reason(
        self,
        *,
        shift_px: float,
        rotation_deg: float,
        scale_delta: float,
        anisotropy: float,
        perspective: float,
        overlap_ratio: float,
    ) -> str | None:
        failures: list[str] = []
        if shift_px > self.config.alignable_max_shift_px:
            failures.append(
                f"shift {shift_px:.1f}px > {self.config.alignable_max_shift_px:.1f}px"
            )
        if abs(rotation_deg) > self.config.alignable_max_rotation_deg:
            failures.append(
                "rotation "
                f"{abs(rotation_deg):.2f}deg > {self.config.alignable_max_rotation_deg:.2f}deg"
            )
        if scale_delta > self.config.alignable_max_scale_delta:
            failures.append(
                "scale delta "
                f"{scale_delta:.3f} > {self.config.alignable_max_scale_delta:.3f}"
            )
        if anisotropy > self.config.max_anisotropy:
            failures.append(
                f"anisotropy {anisotropy:.3f} > {self.config.max_anisotropy:.3f}"
            )
        if perspective > self.config.max_perspective:
            failures.append(
                f"perspective {perspective:.4f} > {self.config.max_perspective:.4f}"
            )
        if overlap_ratio < self.config.min_overlap_ratio:
            failures.append(
                f"overlap {overlap_ratio:.3f} < {self.config.min_overlap_ratio:.3f}"
            )
        return "; ".join(failures) if failures else None

    @staticmethod
    def _transform_metrics(
        homography: np.ndarray,
        frame_shape: tuple[int, int],
        reference_shape: tuple[int, int],
    ) -> dict[str, float]:
        frame_h, frame_w = frame_shape
        ref_h, ref_w = reference_shape
        center = np.float32([[[frame_w / 2.0, frame_h / 2.0]]])
        sample = np.float32(
            [[[frame_w / 2.0, frame_h / 2.0], [frame_w / 2.0 + 1, frame_h / 2.0],
              [frame_w / 2.0, frame_h / 2.0 + 1]]]
        )
        projected_center = cv2.perspectiveTransform(center, homography)[0, 0]
        frame_corners = np.float32(
            [[[0.0, 0.0], [float(frame_w), 0.0],
              [float(frame_w), float(frame_h)], [0.0, float(frame_h)]]]
        )
        reference_corners = np.float32(
            [[0.0, 0.0], [float(ref_w), 0.0],
             [float(ref_w), float(ref_h)], [0.0, float(ref_h)]]
        )
        projected_corners = cv2.perspectiveTransform(
            frame_corners,
            homography,
        )[0]
        max_corner_displacement = float(
            np.max(np.linalg.norm(projected_corners - reference_corners, axis=1))
        )
        projected = cv2.perspectiveTransform(sample, homography)[0]
        dx = projected[1] - projected[0]
        dy = projected[2] - projected[0]
        scale_x = float(np.linalg.norm(dx))
        scale_y = float(np.linalg.norm(dy))
        scale = (scale_x + scale_y) / 2.0
        anisotropy = max(scale_x, scale_y) / max(1e-9, min(scale_x, scale_y))
        rotation = math.degrees(math.atan2(float(dx[1]), float(dx[0])))
        reference_center = np.float32([ref_w / 2.0, ref_h / 2.0])
        shift = float(np.linalg.norm(projected_center - reference_center))
        normalized = homography / homography[2, 2]
        diagonal = math.hypot(frame_w, frame_h)
        perspective = float(math.hypot(normalized[2, 0], normalized[2, 1]) * diagonal)
        return {
            "shift_px": shift,
            "max_corner_displacement_px": max_corner_displacement,
            "rotation_deg": rotation,
            "scale": scale,
            "scale_delta": abs(scale - 1.0),
            "anisotropy": anisotropy,
            "perspective": perspective,
        }

    @staticmethod
    def _coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
        if len(points) < 3:
            return 0.0
        hull = cv2.convexHull(points.astype(np.float32))
        area = abs(float(cv2.contourArea(hull)))
        height, width = shape
        return min(1.0, area / max(1.0, float(height * width)))

    @staticmethod
    def _reprojection_error(
        source: np.ndarray,
        destination: np.ndarray,
        homography: np.ndarray,
    ) -> float:
        if len(source) == 0:
            return math.inf
        projected = cv2.perspectiveTransform(source[:, None, :], homography)[:, 0, :]
        errors = np.linalg.norm(projected - destination, axis=1)
        return float(np.mean(errors))

    @staticmethod
    def _overlap_ratio(
        homography: np.ndarray,
        frame_shape: tuple[int, int],
        reference_shape: tuple[int, int],
    ) -> float:
        """Return the fraction of reference pixels backed by valid frame data.

        This deliberately measures the support of the same perspective warp
        used for registration.  Clipping the projected corner polygon is not
        equivalent at raster boundaries and can under-count even an identity
        transform.  A constant-zero border also keeps replicated alignment
        pixels from being reported as real scene overlap.
        """
        frame_h, frame_w = frame_shape
        ref_h, ref_w = reference_shape
        if (
            frame_h <= 0
            or frame_w <= 0
            or ref_h <= 0
            or ref_w <= 0
            or homography.shape != (3, 3)
            or not np.all(np.isfinite(homography))
        ):
            return 0.0

        # 255 preserves fractional edge support under bilinear interpolation;
        # averaging the warped mask makes the metric independently auditable.
        valid_source = np.full((frame_h, frame_w), 255, dtype=np.uint8)
        try:
            valid_reference = cv2.warpPerspective(
                valid_source,
                homography,
                (ref_w, ref_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        except cv2.error:
            return 0.0

        ratio = float(cv2.mean(valid_reference)[0]) / 255.0
        return min(1.0, max(0.0, ratio))

    @staticmethod
    def _points_to_full(
        points: np.ndarray,
        scale: tuple[float, float],
    ) -> np.ndarray:
        full = points.astype(np.float32).copy()
        full[:, 0] /= scale[0]
        full[:, 1] /= scale[1]
        return full

    @staticmethod
    def _to_full_resolution_homography(
        homography: np.ndarray,
        frame_scale: tuple[float, float],
        reference_scale: tuple[float, float],
    ) -> np.ndarray:
        frame_to_small = np.diag([frame_scale[0], frame_scale[1], 1.0])
        reference_to_small = np.diag(
            [reference_scale[0], reference_scale[1], 1.0]
        )
        full = np.linalg.inv(reference_to_small) @ homography @ frame_to_small
        return full / full[2, 2]

    def _resize_for_processing(
        self,
        gray: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        height, width = gray.shape
        if width <= self.config.process_width:
            return gray, (1.0, 1.0)
        target_width = self.config.process_width
        target_height = max(1, round(height * target_width / width))
        resized = cv2.resize(
            gray,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        return resized, (target_width / width, target_height / height)

    def _extract_features(
        self,
        gray: np.ndarray,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        return list(keypoints), descriptors

    def _blocked_reason(
        self,
        brightness: float,
        contrast: float,
        blur: float,
    ) -> str | None:
        if brightness < self.config.min_brightness:
            return "camera frame is black or severely underexposed"
        if brightness > self.config.max_brightness:
            return "camera frame is white or severely overexposed"
        if contrast < self.config.min_contrast_std:
            return "camera frame is occluded or has insufficient contrast"
        if blur < self.config.min_blur_variance:
            return "camera frame is severely defocused"
        return None

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"unsupported frame shape {frame.shape}")
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty ndarray")
        if frame.ndim not in {2, 3}:
            raise ValueError(f"frame must be 2D or 3D, got shape {frame.shape}")


__all__ = [
    "SceneGuard",
    "SceneGuardConfig",
    "SceneGuardResult",
    "SceneMatchMetrics",
    "SceneState",
]
