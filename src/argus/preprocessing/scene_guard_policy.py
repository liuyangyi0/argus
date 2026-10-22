"""One parameter mapping for runtime and strict scene-baseline construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from argus.preprocessing.scene_guard import SceneGuardConfig

if TYPE_CHECKING:
    from argus.config.schema import SceneGuardConfig as SceneGuardPolicy


def runtime_scene_guard_config(policy: SceneGuardPolicy) -> SceneGuardConfig:
    """Preserve the production mapping without substituting weaker CV defaults."""
    return SceneGuardConfig(
        process_width=policy.process_width,
        feature_type=policy.feature_type,
        min_matches=policy.min_matches,
        min_inliers=policy.min_inliers,
        min_inlier_ratio=policy.min_inlier_ratio,
        max_reprojection_error_px=policy.max_reprojection_error_px,
        min_coverage_ratio=policy.min_coverage_ratio,
        min_overlap_ratio=policy.min_overlap_ratio,
        stable_max_shift_px=policy.stable_max_shift_px,
        stable_max_corner_displacement_px=policy.stable_max_corner_displacement_px,
        stable_max_rotation_deg=policy.stable_max_rotation_deg,
        stable_max_scale_delta=policy.stable_max_scale_delta,
        alignable_max_shift_px=policy.alignable_max_shift_px,
        alignable_max_rotation_deg=policy.alignable_max_rotation_deg,
        alignable_max_scale_delta=policy.alignable_max_scale_delta,
        consecutive_matches=policy.consecutive_matches,
        return_aligned_frame=False,
    )
