"""Tests for fixed-reference camera pose guarding."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from argus.preprocessing.scene_guard import (
    SceneGuard,
    SceneGuardConfig,
    SceneState,
)


def _feature_scene(seed: int = 7, size: tuple[int, int] = (480, 640)) -> np.ndarray:
    """Create a deterministic, widely textured BGR scene for geometric tests."""
    rng = np.random.default_rng(seed)
    height, width = size
    frame = np.full((height, width, 3), 35, dtype=np.uint8)

    for _ in range(180):
        x = int(rng.integers(15, width - 15))
        y = int(rng.integers(15, height - 15))
        radius = int(rng.integers(3, 12))
        color = tuple(int(value) for value in rng.integers(70, 250, 3))
        cv2.circle(frame, (x, y), radius, color, -1)

    for _ in range(50):
        start = tuple(int(value) for value in rng.integers([0, 0], [width, height]))
        end = tuple(int(value) for value in rng.integers([0, 0], [width, height]))
        color = tuple(int(value) for value in rng.integers(80, 255, 3))
        cv2.line(frame, start, end, color, 1)

    cv2.putText(
        frame,
        f"SCENE-{seed}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        3,
    )
    return frame


def _translate(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = frame.shape[:2]
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _rotate(frame: np.ndarray, angle: float) -> np.ndarray:
    height, width = frame.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _scale_about_center(frame: np.ndarray, scale: float) -> np.ndarray:
    height, width = frame.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), 0.0, scale)
    return cv2.warpAffine(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


class TestExplicitReference:
    def test_no_reference_is_unknown_and_never_learns_runtime_frame(self):
        guard = SceneGuard(consecutive_matches=1)
        frame = _feature_scene()

        first = guard.evaluate(frame)
        second = guard.evaluate(frame)

        assert first.state is SceneState.UNKNOWN
        assert second.state is SceneState.UNKNOWN
        assert not first.metrics.reference_available
        assert not guard.has_reference
        assert guard.reference_frame is None

    def test_reference_property_returns_a_defensive_copy(self):
        reference = _feature_scene()
        guard = SceneGuard(reference, consecutive_matches=1)

        exported = guard.reference_frame
        assert exported is not None
        exported[:] = 0

        assert guard.reference_frame is not None
        assert np.any(guard.reference_frame)
        assert guard.evaluate(reference).state is SceneState.STABLE

    def test_clear_reference_does_not_adopt_next_frame(self):
        reference = _feature_scene()
        guard = SceneGuard(reference, consecutive_matches=1)
        guard.clear_reference()

        result = guard.evaluate(reference)

        assert result.state is SceneState.UNKNOWN
        assert not guard.has_reference


class TestSceneGeometry:
    def test_overlap_is_valid_warp_coverage_not_a_constant(self):
        shape = (480, 640)
        identity = np.eye(3, dtype=np.float64)
        near_identity = np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        large_translation = np.array(
            [[1.0, 0.0, 400.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        identity_overlap = SceneGuard._overlap_ratio(identity, shape, shape)
        near_identity_overlap = SceneGuard._overlap_ratio(
            near_identity,
            shape,
            shape,
        )
        moved_overlap = SceneGuard._overlap_ratio(
            large_translation,
            shape,
            shape,
        )

        assert identity_overlap == pytest.approx(1.0, abs=1e-9)
        assert near_identity_overlap == pytest.approx(
            ((shape[1] - 1) * (shape[0] - 1)) / (shape[1] * shape[0]),
            abs=1e-6,
        )
        assert 0.99 < near_identity_overlap < identity_overlap
        assert moved_overlap == pytest.approx(
            (shape[1] - 400) / shape[1],
            abs=1e-6,
        )
        assert moved_overlap < 0.5

    def test_identity_is_stable_with_auditable_metrics(self):
        reference = _feature_scene()
        guard = SceneGuard(reference, consecutive_matches=1)

        result = guard.evaluate(reference)
        metrics = result.metrics.to_dict()

        assert result.state is SceneState.STABLE
        assert result.raw_state is SceneState.STABLE
        assert result.usable
        assert metrics["good_matches"] >= guard.config.min_matches
        assert metrics["inlier_count"] >= guard.config.min_matches
        assert metrics["inlier_ratio"] > 0.95
        assert metrics["coverage_ratio"] > guard.config.min_coverage_ratio
        assert metrics["reprojection_error_px"] < 0.1
        assert metrics["overlap_ratio"] > 0.99
        assert metrics["max_corner_displacement_px"] < 0.1
        assert metrics["transform_reasonable"] is True
        assert metrics["estimator"] in {"USAC_MAGSAC", "RANSAC"}

    def test_translation_is_alignable_and_returns_registered_frame(self):
        reference = _feature_scene()
        shifted = _translate(reference, dx=18, dy=9)
        guard = SceneGuard(reference, consecutive_matches=1)

        result = guard.evaluate(shifted, return_aligned=True)

        assert result.state is SceneState.ALIGNABLE
        assert result.aligned_frame is not None
        assert result.homography is not None
        assert result.metrics.shift_px == pytest.approx(np.hypot(18, 9), abs=2.0)

        margin = 35
        reference_center = reference[margin:-margin, margin:-margin].astype(np.float32)
        shifted_center = shifted[margin:-margin, margin:-margin].astype(np.float32)
        aligned_center = result.aligned_frame[margin:-margin, margin:-margin].astype(
            np.float32
        )
        unaligned_error = float(np.mean(np.abs(shifted_center - reference_center)))
        aligned_error = float(np.mean(np.abs(aligned_center - reference_center)))
        assert aligned_error < unaligned_error * 0.25

    def test_small_rotation_is_alignable(self):
        reference = _feature_scene()
        rotated = _rotate(reference, angle=5.0)
        guard = SceneGuard(reference, consecutive_matches=1)

        result = guard.evaluate(rotated)

        assert result.state is SceneState.ALIGNABLE
        assert abs(result.metrics.rotation_deg) == pytest.approx(5.0, abs=0.7)
        assert result.metrics.overlap_ratio > 0.90
        assert result.metrics.transform_reasonable

    @pytest.mark.parametrize(
        ("changed", "metric"),
        [
            (lambda frame: _rotate(frame, angle=0.8), "rotation_deg"),
            (lambda frame: _scale_about_center(frame, scale=1.02), "scale_delta"),
        ],
    )
    def test_center_anchored_geometry_change_is_not_unwarped_stable(
        self,
        changed,
        metric: str,
    ):
        reference = _feature_scene(size=(1080, 1920))
        guard = SceneGuard(reference, process_width=960, consecutive_matches=1)

        result = guard.evaluate(changed(reference))

        assert result.state is SceneState.ALIGNABLE
        assert result.metrics.max_corner_displacement_px > 3.0
        assert getattr(result.metrics, metric) > 0.003

    def test_resolution_change_is_moved_without_adopting_new_geometry(self):
        reference = _feature_scene()
        resized = cv2.resize(reference, (320, 240))
        guard = SceneGuard(reference, consecutive_matches=1)

        result = guard.evaluate(resized)

        assert result.state is SceneState.MOVED
        assert result.usable is False
        assert "resolution" in result.reason

    def test_large_rotation_of_same_scene_is_moved(self):
        reference = _feature_scene()
        rotated = _rotate(reference, angle=20.0)
        guard = SceneGuard(
            reference,
            consecutive_matches=1,
            alignable_max_rotation_deg=10.0,
        )

        result = guard.evaluate(rotated, return_aligned=True)

        assert result.state is SceneState.MOVED
        assert result.raw_state is SceneState.MOVED
        assert result.metrics.inlier_ratio > 0.8
        assert not result.metrics.transform_reasonable
        assert "rotation" in result.reason
        # An out-of-policy transform is diagnostic only, never supplied to the
        # detector as if it were an approved registration.
        assert result.aligned_frame is None


class TestUnsafeScenes:
    def test_unrelated_textured_scene_is_unknown(self):
        reference = _feature_scene(seed=1)
        unrelated = _feature_scene(seed=99)
        guard = SceneGuard(reference, consecutive_matches=1)

        result = guard.evaluate(unrelated)

        assert result.state is SceneState.UNKNOWN
        assert result.raw_state is SceneState.UNKNOWN
        assert not result.usable
        assert (
            result.metrics.inlier_ratio < guard.config.min_inlier_ratio
            or result.metrics.coverage_ratio < guard.config.min_coverage_ratio
        )

    @pytest.mark.parametrize("value", [0, 127, 255])
    def test_full_occlusion_is_blocked(self, value: int):
        reference = _feature_scene()
        occluded = np.full_like(reference, value)
        guard = SceneGuard(reference, consecutive_matches=1)

        result = guard.evaluate(occluded)

        assert result.state is SceneState.BLOCKED
        assert result.raw_state is SceneState.BLOCKED
        assert not result.usable
        assert result.aligned_frame is None


class TestDebouncing:
    def test_requires_consecutive_matching_states(self):
        reference = _feature_scene()
        guard = SceneGuard(reference, consecutive_matches=3)

        first = guard.evaluate(reference)
        second = guard.evaluate(reference)
        third = guard.evaluate(reference)

        assert first.raw_state is SceneState.STABLE
        assert first.state is SceneState.CALIBRATING
        assert second.state is SceneState.CALIBRATING
        assert third.state is SceneState.STABLE

    def test_transition_is_fail_safe_until_confirmed(self):
        reference = _feature_scene()
        guard = SceneGuard(reference, consecutive_matches=2)
        assert guard.evaluate(reference).state is SceneState.CALIBRATING
        assert guard.evaluate(reference).state is SceneState.STABLE

        blocked = np.zeros_like(reference)
        first_blocked = guard.evaluate(blocked)
        first_recovery = guard.evaluate(reference)
        recovered = guard.evaluate(reference)

        assert first_blocked.raw_state is SceneState.BLOCKED
        assert first_blocked.state is SceneState.BLOCKED
        assert first_recovery.state is SceneState.CALIBRATING
        assert recovered.state is SceneState.STABLE

    def test_stable_and_alignable_share_one_safe_recovery_streak(self):
        reference = _feature_scene()
        shifted = _translate(reference, dx=18, dy=9)
        guard = SceneGuard(reference, consecutive_matches=3)

        first = guard.evaluate(reference)
        second = guard.evaluate(shifted)
        third = guard.evaluate(reference)

        assert first.state is SceneState.CALIBRATING
        assert first.raw_state is SceneState.STABLE
        assert second.state is SceneState.CALIBRATING
        assert second.raw_state is SceneState.ALIGNABLE
        assert third.state is SceneState.STABLE

    def test_reconfirmation_keeps_reference_but_requires_fresh_streak(self):
        reference = _feature_scene()
        guard = SceneGuard(reference, consecutive_matches=2)
        assert guard.evaluate(reference).state is SceneState.CALIBRATING
        assert guard.evaluate(reference).state is SceneState.STABLE

        guard.require_reconfirmation()

        assert guard.has_reference
        assert guard.state is SceneState.CALIBRATING
        assert guard.evaluate(reference).state is SceneState.CALIBRATING
        assert guard.evaluate(reference).state is SceneState.STABLE


class TestConfiguration:
    def test_config_names_map_directly_to_constructor(self):
        guard = SceneGuard(
            process_width=640,
            feature_type="orb",
            min_matches=20,
            min_inlier_ratio=0.5,
            max_reprojection_error_px=2.5,
            min_coverage_ratio=0.08,
            min_overlap_ratio=0.8,
            stable_max_shift_px=4,
            stable_max_rotation_deg=0.5,
            stable_max_scale_delta=0.01,
            alignable_max_shift_px=80,
            alignable_max_rotation_deg=8,
            alignable_max_scale_delta=0.1,
            consecutive_matches=2,
        )

        assert guard.config == SceneGuardConfig(
            process_width=640,
            feature_type="ORB",
            min_matches=20,
            min_inlier_ratio=0.5,
            max_reprojection_error_px=2.5,
            min_coverage_ratio=0.08,
            min_overlap_ratio=0.8,
            stable_max_shift_px=4,
            stable_max_rotation_deg=0.5,
            stable_max_scale_delta=0.01,
            alignable_max_shift_px=80,
            alignable_max_rotation_deg=8,
            alignable_max_scale_delta=0.1,
            consecutive_matches=2,
        )

    @pytest.mark.parametrize("feature_type", ["ORB", "SIFT"])
    def test_both_supported_feature_types_can_match_identity(self, feature_type: str):
        reference = _feature_scene()
        guard = SceneGuard(
            reference,
            feature_type=feature_type,
            consecutive_matches=1,
        )

        assert guard.evaluate(reference).state is SceneState.STABLE

    def test_rejects_invalid_threshold_order(self):
        with pytest.raises(ValueError, match="stable_max_shift_px"):
            SceneGuardConfig(
                stable_max_shift_px=20,
                alignable_max_shift_px=10,
            )
