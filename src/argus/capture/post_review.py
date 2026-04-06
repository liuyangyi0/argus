"""Post-capture review: score baseline frames with current model to flag outliers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


def post_capture_review(
    version_dir: Path,
    camera_id: str,
    models_dir: Path,
    exports_dir: Path,
    flag_percentile: float = 0.99,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict:
    """Score all captured frames with the current anomaly model and flag outliers.

    Args:
        version_dir: Directory containing captured baseline PNGs.
        camera_id: Camera ID to find the corresponding model.
        models_dir: Root models directory (data/models).
        exports_dir: Root exports directory (data/exports) for OpenVINO models.
        flag_percentile: Frames above this percentile are flagged (0.99 = top 1%).
        progress_callback: Optional progress reporting function.

    Returns:
        Dict with 'flagged_frames', 'score_stats', 'threshold', 'skipped' (if no model).
    """
    try:
        return _run_review(
            version_dir, camera_id, models_dir, exports_dir,
            flag_percentile, progress_callback,
        )
    except Exception as exc:
        logger.error("post_review.failed", error=str(exc), camera=camera_id)
        return {"skipped": True, "reason": f"Review failed: {exc}"}


def _find_model(camera_id: str, models_dir: Path, exports_dir: Path) -> Path | None:
    """Locate the best available model file for the camera.

    Prefers OpenVINO (.xml) from exports_dir, falls back to PyTorch (.ckpt)
    from models_dir.  Returns the newest match or None.
    """
    # OpenVINO export (preferred for CPU inference)
    ov_candidates = sorted(
        exports_dir.glob(f"{camera_id}/**/model.xml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if ov_candidates:
        return ov_candidates[0]

    # PyTorch checkpoint fallback
    pt_candidates = sorted(
        models_dir.glob(f"{camera_id}/**/model.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pt_candidates:
        return pt_candidates[0]

    return None


def _run_review(
    version_dir: Path,
    camera_id: str,
    models_dir: Path,
    exports_dir: Path,
    flag_percentile: float,
    progress_callback: Callable[[int, str], None] | None,
) -> dict:
    # --- Locate model ---
    model_path = _find_model(camera_id, models_dir, exports_dir)
    if model_path is None:
        return {"skipped": True, "reason": f"No trained model found for {camera_id}"}

    # --- Load OpenVINO model ---
    try:
        import openvino as ov
    except ImportError:
        logger.warning("post_review.no_openvino", msg="OpenVINO not installed, skipping review")
        return {"skipped": True, "reason": "OpenVINO runtime not available"}

    core = ov.Core()
    ov_model = core.read_model(str(model_path))
    compiled = core.compile_model(ov_model, "CPU")

    # Determine expected input shape (N, C, H, W)
    input_layer = compiled.input(0)
    _, _, model_h, model_w = input_layer.shape
    output_layer = compiled.output(0)

    # --- Collect frame paths ---
    frames = sorted(version_dir.glob("*.png"))
    if not frames:
        return {"skipped": True, "reason": "No PNG frames found in version directory"}

    total = len(frames)
    scores: list[float] = []
    filenames: list[str] = []
    skipped_frames = 0

    for idx, frame_path in enumerate(frames):
        try:
            img = cv2.imread(str(frame_path))
            if img is None:
                logger.warning("post_review.read_failed", path=str(frame_path))
                skipped_frames += 1
                continue

            # Preprocess: resize, BGR->RGB, float32 [0,1], NCHW
            resized = cv2.resize(img, (model_w, model_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # (1, C, H, W)

            result = compiled({input_layer: blob})
            output = result[output_layer]

            # The output may be a scalar score, a map, or a batch — take max
            score = float(np.max(output))
            if not np.isfinite(score):
                logger.warning("post_review.nan_score", path=frame_path.name)
                skipped_frames += 1
                continue

            scores.append(score)
            filenames.append(frame_path.name)

        except Exception as exc:
            logger.warning("post_review.frame_error", path=frame_path.name, error=str(exc))
            skipped_frames += 1
            continue

        if progress_callback is not None:
            pct = int((idx + 1) / total * 100)
            progress_callback(pct, f"Scored {idx + 1}/{total}")

    if not scores:
        return {"skipped": True, "reason": "All frames failed inference"}

    # --- Flag outliers ---
    scores_arr = np.array(scores, dtype=np.float64)
    threshold = float(np.percentile(scores_arr, flag_percentile * 100))

    flagged: list[dict] = []
    for fname, score in zip(filenames, scores):
        if score >= threshold:
            flagged.append({"filename": fname, "score": round(score, 6)})
            # Update sidecar JSON if it exists
            _update_sidecar(version_dir / fname, score)

    review = {
        "total_frames": total,
        "scored_frames": len(scores),
        "skipped_frames": skipped_frames,
        "flagged_count": len(flagged),
        "flag_percentile": flag_percentile,
        "threshold_score": round(threshold, 6),
        "score_stats": {
            "min": round(float(scores_arr.min()), 6),
            "max": round(float(scores_arr.max()), 6),
            "mean": round(float(scores_arr.mean()), 6),
            "std": round(float(scores_arr.std()), 6),
        },
        "flagged_frames": flagged,
    }

    # Persist review.json
    review_path = version_dir / "review.json"
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    logger.info(
        "post_review.complete",
        camera=camera_id,
        total=total,
        flagged=len(flagged),
        threshold=round(threshold, 6),
    )

    return review


def _update_sidecar(frame_path: Path, score: float) -> None:
    """Add flagged/anomaly_score fields to the frame's sidecar JSON (if it exists)."""
    sidecar = frame_path.with_suffix(".json")
    if not sidecar.exists():
        return
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["flagged"] = True
        data["anomaly_score"] = round(score, 6)
        sidecar.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("post_review.sidecar_update_failed", path=str(sidecar), error=str(exc))
