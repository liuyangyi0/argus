"""Independent LightGlue scene comparison; no model weights are loaded here."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from argus.preprocessing.scene_guard import SceneGuard, SceneGuardConfig, SceneState
from argus.preprocessing.scene_guard_shadow import LightGlueSceneGuard


def _frame(shape: tuple[int, int] = (80, 120)) -> np.ndarray:
    return np.random.default_rng(913).integers(20, 235, (*shape, 3), dtype=np.uint8)


def _points(shape: tuple[int, int] = (80, 120)) -> np.ndarray:
    height, width = shape
    xs, ys = np.meshgrid(np.linspace(8, width - 9, 6), np.linspace(8, height - 9, 5))
    return np.column_stack([xs.ravel(), ys.ravel()])


def _matches(
    reference: np.ndarray | None = None, frame: np.ndarray | None = None
) -> SimpleNamespace:
    reference = _points() if reference is None else reference
    frame = reference.copy() if frame is None else frame
    return SimpleNamespace(
        reference_points=reference,
        frame_points=frame,
        reference_keypoints=len(reference),
        frame_keypoints=len(frame),
        raw_matches=len(frame),
        scores=np.ones(len(frame)),
        elapsed_ms=12.5,
        metadata={"extractor": "aliked", "nested": {"checkpoint": "local"}},
    )


class _Matcher:
    def __init__(self, matches: SimpleNamespace | None = None) -> None:
        self.matches = _matches() if matches is None else matches
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.error: Exception | None = None

    def match(self, reference: np.ndarray, frame: np.ndarray) -> SimpleNamespace:
        self.calls.append((reference.copy(), frame.copy()))
        if self.error is not None:
            raise self.error
        return self.matches


def _config(**overrides) -> SceneGuardConfig:
    return replace(
        SceneGuardConfig(
            process_width=64,
            min_matches=8,
            min_inliers=4,
            min_frame_keypoints=8,
            consecutive_matches=1,
        ),
        **overrides,
    )


def _guard(
    matcher: _Matcher | None = None,
    reference: np.ndarray | None = None,
    **overrides,
) -> LightGlueSceneGuard:
    return LightGlueSceneGuard(
        _frame() if reference is None else reference,
        config=_config(**overrides),
        matcher=_Matcher() if matcher is None else matcher,
    )


def test_reference_does_not_require_orb_or_sift_features(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the shadow guard must not construct classic features")

    monkeypatch.setattr(SceneGuard, "_create_detector", forbidden)
    guard = _guard()

    assert guard.evaluate(_frame()).state is SceneState.STABLE


def test_low_texture_reference_is_not_rejected_at_construction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("reference extraction belongs to the LightGlue matcher")

    monkeypatch.setattr(SceneGuard, "_extract_features", forbidden)
    reference = np.full((80, 120, 3), 80, dtype=np.uint8)
    matcher = _Matcher()
    guard = _guard(matcher, reference)

    # This fake matcher isolates the geometric comparison from extractor quality.
    result = guard.evaluate(_frame())

    assert len(matcher.calls) == 1
    assert result.raw_state is SceneState.STABLE


def test_reference_and_metadata_are_defensive_copies():
    reference = _frame()
    original = reference.copy()
    matcher = _Matcher()
    guard = _guard(matcher, reference)
    reference[:] = 0
    exported = guard.reference_frame
    exported[:] = 255

    result = guard.evaluate(_frame())
    metadata = guard.last_match_metadata
    metadata["nested"]["checkpoint"] = "changed"
    matcher.matches.metadata["nested"]["checkpoint"] = "changed again"

    assert result.raw_state is SceneState.STABLE
    assert np.array_equal(guard.reference_frame, original)
    assert np.array_equal(matcher.calls[0][0], original)
    assert guard.last_match_metadata["nested"]["checkpoint"] == "local"
    assert guard.last_match_elapsed_ms == 12.5
    assert guard.last_evaluation_elapsed_ms is not None
    assert guard.last_evaluation_elapsed_ms >= 0


def test_matcher_cannot_mutate_caller_frame_or_fixed_reference():
    class _MutatingMatcher(_Matcher):
        def match(self, reference, frame):
            reference[:] = 0
            frame[:] = 0
            return self.matches

    reference = _frame()
    frame = _frame()
    guard = _guard(_MutatingMatcher(), reference)

    assert guard.evaluate(frame).raw_state is SceneState.STABLE
    assert np.array_equal(guard.reference_frame, reference)
    assert np.array_equal(frame, _frame())


def test_no_aligned_frame_even_when_classic_config_requests_one():
    guard = _guard(return_aligned_frame=True)

    result = guard.evaluate(_frame())

    assert result.state is SceneState.STABLE
    assert result.aligned_frame is None
    assert result.homography is not None


def test_resolution_change_never_calls_matcher_or_adopts_scene():
    matcher = _Matcher()
    guard = _guard(matcher)
    reference = guard.reference_frame

    result = guard.evaluate(_frame((40, 60)))

    assert result.state is SceneState.MOVED
    assert matcher.calls == []
    assert np.array_equal(guard.reference_frame, reference)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((80, 120, 3), dtype=np.uint8),
        np.full((80, 120, 3), 255, dtype=np.uint8),
        np.full((80, 120, 3), 100, dtype=np.uint8),
    ],
)
def test_image_quality_blocks_before_matcher(frame):
    matcher = _Matcher()
    guard = _guard(matcher)

    result = guard.evaluate(frame)

    assert result.state is SceneState.BLOCKED
    assert not result.usable
    assert matcher.calls == []
    assert guard.last_match_elapsed_ms is None


@pytest.mark.parametrize(
    "invalid",
    [None, np.empty((0, 0)), np.zeros((80, 120, 2)), np.full((80, 120), np.nan)],
)
def test_invalid_frame_is_unknown_and_resets_safe_streak(invalid):
    guard = _guard(consecutive_matches=2)
    assert guard.evaluate(_frame()).state is SceneState.CALIBRATING

    result = guard.evaluate(invalid)

    assert result.state is SceneState.UNKNOWN
    assert guard.evaluate(_frame()).state is SceneState.CALIBRATING
    assert guard.evaluate(_frame()).state is SceneState.STABLE


def test_matcher_failure_does_not_retain_previous_safe_state_or_metadata():
    matcher = _Matcher()
    guard = _guard(matcher, consecutive_matches=2)
    guard.evaluate(_frame())
    assert guard.evaluate(_frame()).state is SceneState.STABLE
    matcher.error = RuntimeError("model load failed")

    failed = guard.evaluate(_frame())

    assert failed.raw_state is SceneState.UNKNOWN
    assert failed.state is SceneState.UNKNOWN
    assert guard.last_match_elapsed_ms is None
    assert guard.last_match_metadata == {
        "error_code": "comparison_failed",
        "error_type": "RuntimeError",
    }
    matcher.error = None
    assert guard.evaluate(_frame()).state is SceneState.CALIBRATING


def test_reconfirmation_and_other_guard_streaks_are_independent():
    first = _guard(consecutive_matches=2)
    second = _guard(consecutive_matches=2)
    first.evaluate(_frame())
    assert first.evaluate(_frame()).state is SceneState.STABLE
    assert second.evaluate(_frame()).state is SceneState.CALIBRATING

    first.require_reconfirmation()

    assert first.state is SceneState.CALIBRATING
    assert first.evaluate(_frame()).state is SceneState.CALIBRATING
    assert second.evaluate(_frame()).state is SceneState.STABLE


def test_alternating_safe_states_share_only_the_shadow_recovery_streak():
    matcher = _Matcher()
    guard = _guard(matcher, consecutive_matches=3)
    assert guard.evaluate(_frame()).state is SceneState.CALIBRATING
    matcher.matches = _matches(_points(), _points() + [5.0, 2.0])

    alignable = guard.evaluate(_frame())

    assert alignable.raw_state is SceneState.ALIGNABLE
    assert alignable.state is SceneState.CALIBRATING
    matcher.matches = _matches()
    assert guard.evaluate(_frame()).state is SceneState.STABLE


def test_full_resolution_translation_keeps_full_pixel_threshold_units():
    reference = _points()
    shifted = reference + [5.0, 2.0]
    guard = _guard(_Matcher(_matches(reference, shifted)))

    result = guard.evaluate(_frame())

    assert result.state is SceneState.ALIGNABLE
    assert result.metrics.shift_px == pytest.approx(np.hypot(5.0, 2.0), abs=0.01)
    assert result.metrics.reprojection_error_px < 0.01
    np.testing.assert_allclose(
        result.homography,
        [[1.0, 0.0, -5.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
        atol=0.01,
    )


def test_homography_estimation_uses_classic_processing_pixel_scale(monkeypatch):
    shape = (1081, 1920)
    reference = _points(shape)
    shifted = reference + [1.0, 1.0]
    matcher = _Matcher(_matches(reference, shifted))
    guard = _guard(matcher, _frame(shape), process_width=960)
    captured = []

    def estimate(source, destination):
        captured.append((source.copy(), destination.copy()))
        sx, sy = 0.5, 540 / 1081
        return (
            np.array([[1.0, 0.0, -sx], [0.0, 1.0, -sy], [0.0, 0.0, 1.0]]),
            np.ones((len(source), 1), dtype=np.uint8),
            "fixture",
        )

    monkeypatch.setattr(guard, "_find_homography", estimate)

    result = guard.evaluate(_frame(shape))

    assert result.state is SceneState.STABLE
    np.testing.assert_allclose(captured[0][0], shifted * [0.5, 540 / 1081], rtol=1e-6)
    np.testing.assert_allclose(captured[0][1], reference * [0.5, 540 / 1081], rtol=1e-6)
    assert result.metrics.shift_px == pytest.approx(np.sqrt(2), abs=0.01)


def test_beyond_alignment_limit_is_moved():
    reference = _points((200, 400))
    # All correspondences stay inside both original images.
    reference[:, 0] = reference[:, 0] * 0.35
    shifted = reference + [130.0, 0.0]
    guard = _guard(_Matcher(_matches(reference, shifted)), _frame((200, 400)))

    result = guard.evaluate(_frame((200, 400)))

    assert result.state is SceneState.MOVED
    assert result.metrics.shift_px == pytest.approx(130.0, abs=0.1)


def test_insufficient_reference_lightglue_features_are_reported_as_blocked():
    points = _points()[:5]
    guard = _guard(_Matcher(_matches(points)))

    result = guard.evaluate(_frame())

    assert result.state is SceneState.BLOCKED
    assert result.metrics.reference_keypoints == 5


def test_clustered_matches_are_unknown_even_with_identity_homography():
    points = _points() * 0.1 + [50.0, 30.0]
    guard = _guard(_Matcher(_matches(points)))

    result = guard.evaluate(_frame())

    assert result.state is SceneState.UNKNOWN
    assert result.metrics.good_matches == 30
    assert result.metrics.coverage_ratio < guard.config.min_coverage_ratio
    assert "coverage" in result.reason


@pytest.mark.parametrize(
    "mutation", [
        "nan", "outside", "duplicate", "wrong_count", "wrong_shape", "complex",
        "nan_scores", "wrong_scores_shape", "complex_scores", "impossible_raw_count",
    ]
)
def test_invalid_correspondences_never_become_stable(mutation):
    matches = _matches()
    if mutation == "nan":
        matches.frame_points[0, 0] = np.nan
    elif mutation == "outside":
        matches.reference_points[0, 0] = 120.0
    elif mutation == "duplicate":
        matches.frame_points[1] = matches.frame_points[0]
    elif mutation == "wrong_count":
        matches.frame_keypoints = 1
    elif mutation == "wrong_shape":
        matches.reference_points = np.zeros((30, 3))
    elif mutation == "complex":
        matches.frame_points = matches.frame_points.astype(np.complex128) + 1j
    elif mutation == "nan_scores":
        matches.scores[0] = np.nan
    elif mutation == "wrong_scores_shape":
        matches.scores = np.ones((len(matches.scores), 1))
    elif mutation == "complex_scores":
        matches.scores = matches.scores.astype(complex) + 1j
    elif mutation == "impossible_raw_count":
        matches.raw_matches = 100_000
    guard = _guard(_Matcher(matches))

    result = guard.evaluate(_frame())

    assert result.state is SceneState.UNKNOWN
    assert not result.usable


@pytest.mark.parametrize(
    "homography",
    [
        np.full((3, 3), np.nan),
        np.zeros((3, 3)),
        np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 79.0], [0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.1, 0.0, 1.0]]),
    ],
)
def test_nonfinite_degenerate_reflected_or_horizon_homography_is_unknown(
    monkeypatch, homography
):
    guard = _guard()
    monkeypatch.setattr(
        guard,
        "_find_homography",
        lambda *args: (homography, np.ones((30, 1), dtype=np.uint8), "fixture"),
    )

    result = guard.evaluate(_frame())

    assert result.state is SceneState.UNKNOWN
    assert result.aligned_frame is None


def test_homography_inlier_mask_must_match_correspondences(monkeypatch):
    guard = _guard()
    monkeypatch.setattr(
        guard,
        "_find_homography",
        lambda *args: (np.eye(3), np.ones((2, 1), dtype=np.uint8), "fixture"),
    )

    assert guard.evaluate(_frame()).state is SceneState.UNKNOWN


def test_nan_geometry_metric_is_not_accepted(monkeypatch):
    guard = _guard()
    monkeypatch.setattr(SceneGuard, "_reprojection_error", lambda *args: float("nan"))

    assert guard.evaluate(_frame()).state is SceneState.UNKNOWN


def test_same_static_geometry_measurements_as_classic_guard():
    guard = _guard()

    result = guard.evaluate(_frame())

    assert result.homography is not None
    assert result.metrics.overlap_ratio == pytest.approx(
        SceneGuard._overlap_ratio(result.homography, (80, 120), (80, 120))
    )
    assert result.metrics.reference_coverage_ratio == pytest.approx(
        SceneGuard._coverage(_points().astype(np.float32), (80, 120))
    )


def test_nonfinite_threshold_config_is_rejected():
    with pytest.raises(ValueError, match="must be finite"):
        _guard(min_brightness=float("nan"))
