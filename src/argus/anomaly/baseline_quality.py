"""Versioned input-quality decisions, separate from release authorization.

The fixed-scene context is a measurement context, not an approval token.
Only the trainer's server-validated V3 workflow path may select it for training.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from argus.preprocessing.scene_guard import SceneGuardConfig

LEGACY_POLICY = "argus.baseline-quality.legacy.v1"
STATIC_SCENE_POLICY = "argus.baseline-quality.scene-static.v1"


@dataclass(frozen=True, slots=True)
class FixedSceneQualityContext:
    config: SceneGuardConfig
    expected_shape: tuple[int, int]
    sample_count: int
    training_count: int
    calibration_count: int
    source_session_count: int
    accepted_span_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.config, SceneGuardConfig):
            raise ValueError("fixed-scene quality requires a frozen SceneGuard config")
        if (not isinstance(self.expected_shape, tuple) or len(self.expected_shape) != 2
                or any(type(x) is not int or x <= 0 for x in self.expected_shape)):
            raise ValueError("fixed-scene quality requires positive reference dimensions")
        for name, minimum in (("sample_count", 150), ("training_count", 100),
                              ("calibration_count", 50), ("source_session_count", 2)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"fixed-scene quality {name} is below its strict minimum")
        if self.sample_count != self.training_count + self.calibration_count:
            raise ValueError("fixed-scene sample counts do not match the complete split")
        if (type(self.accepted_span_seconds) not in (int, float)
                or not math.isfinite(self.accepted_span_seconds) or self.accepted_span_seconds < 300):
            raise ValueError("fixed-scene quality requires at least 300 seconds of accepted coverage")
        if any(isinstance(value, float) and not math.isfinite(value)
               for value in (getattr(self.config, field) for field in self.config.__dataclass_fields__)):
            raise ValueError("fixed-scene quality config must contain finite thresholds")


def inspect_baseline_quality(
    images: list[Path], min_images: int, *, fixed_scene: FixedSceneQualityContext | None = None,
) -> dict:
    """Inspect bytes/pixels without changing samples or granting workflow approval.

    Legacy callers retain their existing diversity errors. For a strict scene
    workflow, temporal sameness is a coverage limitation, not image corruption.
    All frozen samples, including the complete calibration split, remain intact.
    """
    if fixed_scene is not None and not isinstance(fixed_scene, FixedSceneQualityContext):
        raise TypeError("fixed_scene must be a FixedSceneQualityContext")
    total = len(images)
    errors: list[str] = []
    error_codes: list[str] = []
    warnings: list[str] = []
    warning_codes: list[str] = []
    if fixed_scene is not None:
        warning_codes.extend(("observed_conditions_only", "temporal_dependence_unverified",
                              "frozen_capture_not_excluded_by_pixels"))
        warnings.extend((
            "仅覆盖已观察的固定场景正常工况，不证明其他光照或作业工况已覆盖。",
            "连续视频帧不是已证实独立的统计样本；完整校准不等于时序泛化保证。",
            "稳定像素与单调时间戳不能单独证明现场未冻结，仍需来源复核与独立验收。",
        ))
    if total < min_images:
        error_codes.append("insufficient_images")
        errors.append(f"图片数量不足: {total} (需要 >= {min_images})")
    if fixed_scene is not None and total != fixed_scene.sample_count:
        error_codes.append("scene_sample_count_mismatch")
        errors.append("场景样本数量与完整冻结训练/校准集不一致")
    if fixed_scene is not None and len(set(images)) != total:
        if "scene_sample_count_mismatch" not in error_codes:
            error_codes.append("scene_sample_count_mismatch")
        errors.append("场景样本路径重复，不能用重复列表增加冻结样本数")

    corrupted = quality_failures = 0
    brightness: list[float] = []
    hashes: list[np.ndarray] = []
    quality_examples: list[dict[str, str]] = []
    # Keep the historical short-circuit for ordinary insufficient datasets.
    to_scan = images if total >= min_images or fixed_scene is not None else []
    for path in to_scan:
        try:
            frame = cv2.imread(str(path))
            if (frame is None or not isinstance(frame, np.ndarray) or frame.size == 0
                    or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3):
                raise ValueError("unreadable or unsupported image")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(gray.mean()))
            small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            hashes.append((small > small.mean()).flatten().astype(np.uint8))
            if fixed_scene is not None:
                cfg = fixed_scene.config
                reason = None
                if frame.shape[:2] != fixed_scene.expected_shape:
                    reason = "reference_dimensions_mismatch"
                check = gray
                if gray.shape[1] > cfg.process_width:
                    check = cv2.resize(gray, (cfg.process_width,
                        max(1, round(gray.shape[0] * cfg.process_width / gray.shape[1]))),
                        interpolation=cv2.INTER_AREA)
                level = float(check.mean())
                contrast = float(check.std())
                blur = float(cv2.Laplacian(check, cv2.CV_64F).var())
                if level < cfg.min_brightness or level > cfg.max_brightness:
                    reason = reason or "exposure_out_of_range"
                elif contrast < cfg.min_contrast_std:
                    reason = reason or "insufficient_contrast"
                elif blur < cfg.min_blur_variance:
                    reason = reason or "severe_blur"
                if reason:
                    quality_failures += 1
                    if len(quality_examples) < 10:
                        quality_examples.append({"image": path.name, "reason": reason})
        except (OSError, ValueError, cv2.error):
            corrupted += 1

    corruption_rate = corrupted / total if total else 0.0
    if corrupted and (fixed_scene is not None or corruption_rate >= 0.1):
        error_codes.append("corrupt_images")
        errors.append(f"损坏率过高: {corruption_rate:.1%} ({'冻结场景不允许任何损坏样本' if fixed_scene else '上限 10%'})")
    if quality_failures:
        error_codes.append("scene_frame_quality_failed")
        errors.append(f"场景逐帧画质检查失败: {quality_failures} 张（尺寸、曝光、对比度或模糊）")
    brightness_std = float(np.std(brightness)) if brightness else 0.0
    near_duplicate_count = 0
    if len(hashes) > 1:
        matrix = np.asarray(hashes)
        duplicate_flags = np.zeros(len(matrix), dtype=bool)
        for index in range(len(matrix)):
            if duplicate_flags[index]:
                continue
            duplicates = np.sum(matrix[index] != matrix[index + 1:], axis=1) <= 5
            new_duplicates = duplicates & ~duplicate_flags[index + 1:]
            near_duplicate_count += int(new_duplicates.sum())
            duplicate_flags[index + 1:] |= duplicates
    near_duplicate_rate = near_duplicate_count / total if total else 0.0
    if total >= min_images or fixed_scene is not None:
        if brightness_std <= 2.0:
            message = f"亮度多样性不足: std={brightness_std:.1f}（固定场景可能合法，仅代表窄工况）"
            if fixed_scene is not None:
                warnings.append(message)
                warning_codes.append("low_temporal_brightness_variation")
            else:
                errors.append(f"亮度多样性不足: std={brightness_std:.1f} (需要 > 2.0)")
                error_codes.append("low_temporal_brightness_variation")
        if near_duplicate_rate >= 0.8:
            if fixed_scene is not None:
                warnings.append(f"感知近似重复率高: {near_duplicate_rate:.1%}，不等于损坏或独立样本充足。")
                warning_codes.append("high_perceptual_redundancy")
            else:
                errors.append(f"近似重复率过高: {near_duplicate_rate:.1%} (上限 80%)")
                error_codes.append("high_perceptual_redundancy")
    integrity_errors = {"insufficient_images", "scene_sample_count_mismatch", "corrupt_images", "scene_frame_quality_failed"}
    return {
        "passed": not errors, "errors": errors, "error_codes": error_codes,
        "warnings": warnings, "warning_codes": warning_codes,
        "policy_version": STATIC_SCENE_POLICY if fixed_scene else LEGACY_POLICY,
        "scope": "observed_fixed_scene_only" if fixed_scene else "legacy_baseline",
        "integrity_passed": not integrity_errors.intersection(error_codes),
        "sample_count": total, "readable_count": len(hashes),
        "corruption_rate": corruption_rate, "near_duplicate_rate": near_duplicate_rate,
        "near_duplicate_group_count": len(hashes) - near_duplicate_count,
        "brightness_std": brightness_std, "scene_frame_quality_failure_count": quality_failures,
        "scene_frame_quality_examples": quality_examples,
        "coverage_claim": "observed_conditions_only", "independent_sample_count": None,
        "production_readiness": "not_evaluated",
        "source_session_count": fixed_scene.source_session_count if fixed_scene else None,
        "accepted_span_seconds": fixed_scene.accepted_span_seconds if fixed_scene else None,
    }
