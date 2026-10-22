"""Offline LightGlue scene-pose comparison with no production authority.

The approved reference and recovery debounce belong to this instance only.
This class neither calls the production guard nor installs an aligned frame
into a detection pipeline.  It reuses stateless geometry measurements so the
comparison keeps the existing pixel units and scene thresholds.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from argus.preprocessing.scene_guard import (
    SceneGuard,
    SceneGuardConfig,
    SceneGuardResult,
    SceneMatchMetrics,
    SceneState,
)

if TYPE_CHECKING:
    from argus.preprocessing.lightglue_matcher import (
        CorrespondenceMatches,
        LightGlueMatcher,
    )


class LightGlueSceneGuard:
    """Compare matched points against one immutable, approved scene reference.

    ``SceneGuardResult.usable`` describes only this offline experiment.  It
    must never be used to release a camera, approve a profile, or feed a warp
    to a detector.  ``return_aligned_frame`` is deliberately ignored.

    LightGlue has its own feature extractor: construction does not require
    the reference to have any ORB/SIFT features.  Matcher initialization and
    local checkpoint validation are the responsibility of ``matcher``.
    """

    def __init__(
        self,
        reference_frame: np.ndarray,
        *,
        config: SceneGuardConfig,
        matcher: LightGlueMatcher,
    ) -> None:
        self._validate_config(config)
        self.config = config
        self._matcher = matcher
        self._reference_frame = self._as_bgr(reference_frame)
        self._reference_frame.setflags(write=False)
        self._reference_scale = self._processing_scale(self._reference_frame.shape[:2])
        self._confirmed_state = SceneState.CALIBRATING
        self._safe_streak = 0
        self._last_match_metadata: dict[str, Any] = {}
        self._last_match_elapsed_ms: float | None = None
        self._last_evaluation_elapsed_ms: float | None = None

    @staticmethod
    def _validate_config(config: SceneGuardConfig) -> None:
        for field in fields(config):
            value = getattr(config, field.name)
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")

    @property
    def reference_frame(self) -> np.ndarray:
        """Return a copy; runtime frames can never replace the reference."""
        return self._reference_frame.copy()

    @property
    def state(self) -> SceneState:
        return self._confirmed_state

    @property
    def last_match_metadata(self) -> dict[str, Any]:
        """Return detached matcher evidence for the most recent evaluation."""
        return copy.deepcopy(self._last_match_metadata)

    @property
    def last_match_elapsed_ms(self) -> float | None:
        return self._last_match_elapsed_ms

    @property
    def last_evaluation_elapsed_ms(self) -> float | None:
        return self._last_evaluation_elapsed_ms

    def require_reconfirmation(self) -> None:
        """Reset only the experimental recovery streak, retaining the reference."""
        self._confirmed_state = SceneState.CALIBRATING
        self._safe_streak = 0

    def evaluate(self, frame: np.ndarray) -> SceneGuardResult:
        """Return independent pose evidence; errors immediately clear its streak."""
        started = time.perf_counter()
        self._last_match_metadata = {}
        self._last_match_elapsed_ms = None
        try:
            return self._evaluate(frame)
        except Exception as exc:
            # Missing dependencies, corrupt outputs, OpenCV failures, and model
            # failures are observable failures, never a successful comparison.
            code = getattr(exc, "code", "comparison_failed")
            self._last_match_metadata = {
                **self._last_match_metadata,
                "error_code": str(code),
                "error_type": type(exc).__name__,
            }
            return self._result(
                SceneState.UNKNOWN,
                f"LightGlue comparison failed ({code}: {type(exc).__name__})",
                SceneMatchMetrics(reference_available=True),
            )
        finally:
            self._last_evaluation_elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0

    def _evaluate(self, frame: np.ndarray) -> SceneGuardResult:
        frame_bgr = self._as_bgr(frame)
        if frame_bgr.shape[:2] != self._reference_frame.shape[:2]:
            return self._result(
                SceneState.MOVED,
                "frame resolution differs from the approved scene reference",
                SceneMatchMetrics(reference_available=True),
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_scale = self._processing_scale(gray.shape)
        if frame_scale != (1.0, 1.0):
            gray = cv2.resize(
                gray,
                (self.config.process_width, max(1, round(gray.shape[0] * frame_scale[1]))),
                interpolation=cv2.INTER_AREA,
            )
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        base: dict[str, Any] = {
            "reference_available": True,
            "brightness": brightness,
            "contrast_std": contrast,
            "blur_variance": blur,
        }
        quality_reason = self._blocked_reason(brightness, contrast, blur)
        if quality_reason is not None:
            return self._result(
                SceneState.BLOCKED, quality_reason, SceneMatchMetrics(**base)
            )

        # A separate copy prevents an injected matcher from mutating either the
        # approved reference or the caller's frame, including between samples.
        matches = self._matcher.match(self._reference_frame.copy(), frame_bgr.copy())
        frame_points, reference_points = self._validate_matches(matches)
        base.update(
            reference_keypoints=int(matches.reference_keypoints),
            frame_keypoints=int(matches.frame_keypoints),
            raw_matches=int(matches.raw_matches),
            good_matches=len(frame_points),
        )
        self._last_match_metadata = copy.deepcopy(matches.metadata)
        elapsed = float(matches.elapsed_ms)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("matcher elapsed_ms must be finite and nonnegative")
        self._last_match_elapsed_ms = elapsed

        if min(matches.reference_keypoints, matches.frame_keypoints) < (
            self.config.min_frame_keypoints
        ):
            return self._result(
                SceneState.BLOCKED,
                "reference or frame has too few LightGlue extractor keypoints",
                SceneMatchMetrics(**base),
            )
        if len(frame_points) < self.config.min_matches:
            return self._result(
                SceneState.UNKNOWN,
                f"only {len(frame_points)} reliable feature matches",
                SceneMatchMetrics(**base),
            )

        # The classic RANSAC threshold is defined in resized processing pixels.
        # Matching itself stays at the adapter's chosen extraction resolution;
        # its returned points have already been mapped to full-resolution pixels.
        frame_points_small = np.float32(frame_points * np.asarray(frame_scale))
        reference_points_small = np.float32(
            reference_points * np.asarray(self._reference_scale)
        )
        small_homography, mask, estimator = self._find_homography(
            frame_points_small, reference_points_small
        )
        base["estimator"] = estimator
        if small_homography is None or mask is None:
            return self._result(
                SceneState.UNKNOWN,
                "feature matches do not form a consistent homography",
                SceneMatchMetrics(**base),
            )
        self._validate_homography(small_homography, gray.shape)
        homography = SceneGuard._to_full_resolution_homography(
            small_homography, frame_scale, self._reference_scale
        )
        self._validate_homography(homography, frame_bgr.shape[:2])
        inlier_mask = np.asarray(mask).reshape(-1)
        if len(inlier_mask) != len(frame_points) or not np.all(
            np.isin(inlier_mask, [0, 1])
        ):
            raise ValueError("homography estimator returned an invalid inlier mask")
        inlier_mask = inlier_mask.astype(bool)
        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inlier_count / max(1, len(frame_points))
        frame_inliers = frame_points[inlier_mask].astype(np.float32)
        reference_inliers = reference_points[inlier_mask].astype(np.float32)
        frame_coverage = SceneGuard._coverage(frame_inliers, frame_bgr.shape[:2])
        reference_coverage = SceneGuard._coverage(
            reference_inliers, self._reference_frame.shape[:2]
        )
        coverage = min(frame_coverage, reference_coverage)
        reprojection_error = SceneGuard._reprojection_error(
            frame_inliers, reference_inliers, homography
        )
        overlap = SceneGuard._overlap_ratio(
            homography, frame_bgr.shape[:2], self._reference_frame.shape[:2]
        )
        transform = SceneGuard._transform_metrics(
            homography, frame_bgr.shape[:2], self._reference_frame.shape[:2]
        )
        if not all(
            math.isfinite(value)
            for value in [coverage, reprojection_error, overlap, *transform.values()]
        ):
            raise ValueError("homography metrics contain non-finite values")
        transform_reason = self._unreasonable_transform_reason(transform, overlap)
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

        failures: list[str] = []
        minimum_inliers = max(
            self.config.min_inliers,
            math.ceil(self.config.min_matches * self.config.min_inlier_ratio),
        )
        if inlier_count < minimum_inliers:
            failures.append(f"inliers {inlier_count} < {minimum_inliers}")
        if inlier_ratio < self.config.min_inlier_ratio:
            failures.append(
                f"inlier ratio {inlier_ratio:.3f} < {self.config.min_inlier_ratio:.3f}"
            )
        if coverage < self.config.min_coverage_ratio:
            failures.append(
                f"coverage {coverage:.3f} < {self.config.min_coverage_ratio:.3f}"
            )
        if reprojection_error > self.config.max_reprojection_error_px:
            failures.append(
                "reprojection error "
                f"{reprojection_error:.2f}px > {self.config.max_reprojection_error_px:.2f}px"
            )
        if failures:
            return self._result(
                SceneState.UNKNOWN, "; ".join(failures), metrics, homography
            )
        if transform_reason is not None:
            return self._result(SceneState.MOVED, transform_reason, metrics, homography)

        stable = (
            metrics.shift_px <= self.config.stable_max_shift_px
            and metrics.max_corner_displacement_px
            <= self.config.stable_max_corner_displacement_px
            and abs(metrics.rotation_deg) <= self.config.stable_max_rotation_deg
            and metrics.scale_delta <= self.config.stable_max_scale_delta
        )
        return self._result(
            SceneState.STABLE if stable else SceneState.ALIGNABLE,
            "scene matches approved camera pose" if stable else "scene can be registered",
            metrics,
            homography,
        )

    @staticmethod
    def _as_bgr(frame: np.ndarray) -> np.ndarray:
        SceneGuard._validate_frame(frame)
        if (
            not np.issubdtype(frame.dtype, np.number)
            or np.issubdtype(frame.dtype, np.complexfloating)
            or not np.all(np.isfinite(frame))
        ):
            raise ValueError("frame must contain finite real pixels")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.shape[2] != 3:
            raise ValueError(f"unsupported frame shape {frame.shape}")
        return np.ascontiguousarray(frame).copy()

    def _processing_scale(self, shape: tuple[int, int]) -> tuple[float, float]:
        height, width = shape
        if width <= self.config.process_width:
            return 1.0, 1.0
        target_width = self.config.process_width
        target_height = max(1, round(height * target_width / width))
        return target_width / width, target_height / height

    def _blocked_reason(self, brightness: float, contrast: float, blur: float) -> str | None:
        if brightness < self.config.min_brightness:
            return "camera frame is black or severely underexposed"
        if brightness > self.config.max_brightness:
            return "camera frame is white or severely overexposed"
        if contrast < self.config.min_contrast_std:
            return "camera frame is occluded or has insufficient contrast"
        if blur < self.config.min_blur_variance:
            return "camera frame is severely defocused"
        return None

    def _validate_matches(
        self, matches: CorrespondenceMatches
    ) -> tuple[np.ndarray, np.ndarray]:
        validated: list[np.ndarray] = []
        for value in (matches.frame_points, matches.reference_points):
            points = np.asarray(value)
            if not (
                np.issubdtype(points.dtype, np.floating)
                or np.issubdtype(points.dtype, np.integer)
            ):
                raise ValueError("matched points must contain real numeric pixels")
            points = points.astype(np.float64, copy=True)
            if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
                raise ValueError("matched points must be finite Nx2 arrays")
            if len(np.unique(points, axis=0)) != len(points):
                raise ValueError("matched points must have one-to-one coordinates")
            height, width = self._reference_frame.shape[:2]
            if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(
                points[:, 1] >= height
            ):
                raise ValueError("matched points fall outside the original image")
            validated.append(points)
        frame_points, reference_points = validated
        if len(frame_points) != len(reference_points):
            raise ValueError("matched point arrays differ in length")
        for name in ("reference_keypoints", "frame_keypoints", "raw_matches"):
            value = getattr(matches, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < len(frame_points)
            ):
                raise ValueError(f"invalid matcher count: {name}")
        if matches.raw_matches > min(matches.reference_keypoints, matches.frame_keypoints):
            raise ValueError("raw one-to-one matches exceed available keypoints")
        scores = np.asarray(matches.scores)
        if (
            scores.shape != (len(frame_points),)
            or not (
                np.issubdtype(scores.dtype, np.floating)
                or np.issubdtype(scores.dtype, np.integer)
            )
            or not np.all(np.isfinite(scores))
            or np.any(scores < 0) or np.any(scores > 1)
        ):
            raise ValueError("match scores must be finite probabilities aligned with coordinates")
        if not isinstance(matches.metadata, dict):
            raise ValueError("matcher metadata must be a dictionary")
        return frame_points, reference_points

    def _find_homography(
        self, source: np.ndarray, destination: np.ndarray
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
            if (
                homography is not None
                and mask is not None
                and np.all(np.isfinite(homography))
            ):
                return homography, mask, name
        return None, None, "none"

    @staticmethod
    def _validate_homography(homography: np.ndarray, shape: tuple[int, int]) -> None:
        if (
            homography.shape != (3, 3)
            or not np.all(np.isfinite(homography))
            or abs(homography[2, 2]) <= 1e-12
        ):
            raise ValueError("homography is non-finite or has no finite origin")
        normalized = homography / homography[2, 2]
        # A reflected or collapsed image is not a plausible camera pose.  Also
        # reject a projective horizon crossing any part of the frame rectangle.
        determinant = float(np.linalg.det(normalized))
        if not math.isfinite(determinant) or determinant <= 1e-12:
            raise ValueError("homography is singular or reverses orientation")
        height, width = shape
        corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
        if np.any(corners @ normalized[2, :] <= 1e-9):
            raise ValueError("homography crosses the image projective horizon")

    def _unreasonable_transform_reason(
        self, transform: dict[str, float], overlap: float
    ) -> str | None:
        limits = (
            ("shift", transform["shift_px"], self.config.alignable_max_shift_px, "px"),
            (
                "rotation",
                abs(transform["rotation_deg"]),
                self.config.alignable_max_rotation_deg,
                "deg",
            ),
            ("scale delta", transform["scale_delta"], self.config.alignable_max_scale_delta, ""),
            ("anisotropy", transform["anisotropy"], self.config.max_anisotropy, ""),
            ("perspective", transform["perspective"], self.config.max_perspective, ""),
        )
        failures = [
            f"{name} {value:.4f}{unit} > {limit:.4f}{unit}"
            for name, value, limit, unit in limits
            if value > limit
        ]
        if overlap < self.config.min_overlap_ratio:
            failures.append(f"overlap {overlap:.3f} < {self.config.min_overlap_ratio:.3f}")
        return "; ".join(failures) if failures else None

    def _result(
        self,
        raw_state: SceneState,
        reason: str,
        metrics: SceneMatchMetrics,
        homography: np.ndarray | None = None,
    ) -> SceneGuardResult:
        safe_states = {SceneState.STABLE, SceneState.ALIGNABLE}
        required = self.config.consecutive_matches
        if raw_state not in safe_states:
            self._confirmed_state = raw_state
            self._safe_streak = 0
            state = raw_state
        elif self._confirmed_state in safe_states:
            self._confirmed_state = raw_state
            self._safe_streak = required
            state = raw_state
        else:
            self._safe_streak += 1
            if self._safe_streak >= required:
                self._confirmed_state = raw_state
                self._safe_streak = required
                state = raw_state
            else:
                state = SceneState.CALIBRATING
                reason = (
                    f"awaiting {raw_state.value} confirmation "
                    f"({self._safe_streak}/{required}): {reason}"
                )
        return SceneGuardResult(
            state=state,
            raw_state=raw_state,
            reason=reason,
            metrics=metrics,
            aligned_frame=None,
            homography=None if homography is None else homography.copy(),
        )


__all__ = ["LightGlueSceneGuard"]
