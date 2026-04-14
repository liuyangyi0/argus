"""Camera calibration management API routes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from argus.dashboard.api_response import api_not_found, api_success, api_validation_error

logger = structlog.get_logger()
router = APIRouter()


def _calibration_dir(request: Request) -> Path:
    """Get calibration directory from app config."""
    config = getattr(request.app.state, "config", None)
    if config and hasattr(config, "physics"):
        return Path(config.physics.calibration_dir)
    return Path("data/calibration")


@router.get("/{camera_id}")
async def get_calibration_status(request: Request, camera_id: str) -> JSONResponse:
    """Get calibration status for a camera."""
    cal_dir = _calibration_dir(request)
    cal_file = cal_dir / camera_id / "calibration.json"

    if not cal_file.exists():
        return api_success({
            "camera_id": camera_id,
            "calibrated": False,
            "message": "No calibration file found",
        })

    data = json.loads(cal_file.read_text())
    return api_success({
        "camera_id": camera_id,
        "calibrated": True,
        "reprojection_error": data.get("reprojection_error", 0),
        "image_size": data.get("image_size", [0, 0]),
    })


@router.post("/{camera_id}/upload")
async def upload_calibration(
    request: Request,
    camera_id: str,
    file: UploadFile = File(...),
) -> JSONResponse:
    """Upload a pre-computed calibration JSON file."""
    if not file.filename or not file.filename.endswith(".json"):
        return api_validation_error("File must be a JSON file")

    # Security: validate camera_id (no path traversal)
    import re
    if not re.match(r"^[a-zA-Z0-9_-]+$", camera_id):
        return api_validation_error("Invalid camera_id")

    content = await file.read()
    try:
        data = json.loads(content)
        # Validate required fields and matrix shapes
        import numpy as np

        for key in ("camera_matrix", "dist_coeffs", "rotation_matrix", "translation_vector"):
            if key not in data:
                return api_validation_error(f"Missing required field: {key}")

        K = np.array(data["camera_matrix"], dtype=np.float64)
        R = np.array(data["rotation_matrix"], dtype=np.float64)
        t = np.array(data["translation_vector"], dtype=np.float64)
        dist = np.array(data["dist_coeffs"], dtype=np.float64)

        if K.shape != (3, 3):
            return api_validation_error("camera_matrix must be 3x3")
        if R.shape != (3, 3):
            return api_validation_error("rotation_matrix must be 3x3")
        if t.size < 3:
            return api_validation_error("translation_vector must have 3 elements")
        if np.any(np.isnan(K)) or np.any(np.isinf(K)):
            return api_validation_error("camera_matrix contains NaN/Inf")
        if abs(np.linalg.det(R) - 1.0) > 0.1:
            return api_validation_error("rotation_matrix is not orthogonal")

        data["camera_id"] = camera_id
    except json.JSONDecodeError:
        return api_validation_error("Invalid JSON content")

    cal_dir = _calibration_dir(request) / camera_id
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_file = cal_dir / "calibration.json"
    cal_file.write_text(json.dumps(data, indent=2))

    logger.info("calibration.uploaded", camera_id=camera_id, path=str(cal_file))
    return api_success({
        "camera_id": camera_id,
        "calibrated": True,
        "path": str(cal_file),
    })


@router.post("/{camera_id}/compute")
async def compute_calibration(request: Request, camera_id: str) -> JSONResponse:
    """Compute calibration from captured checkerboard images.

    Expects checkerboard images in data/calibration/{camera_id}/frames/.
    """
    cal_dir = _calibration_dir(request) / camera_id
    frames_dir = cal_dir / "frames"

    if not frames_dir.exists():
        return api_not_found("No calibration frames directory found")

    frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if len(frame_files) < 5:
        return api_validation_error(f"Need ≥5 checkerboard frames, found {len(frame_files)}")

    def _run_calibration():
        import cv2
        from argus.physics.calibration import CalibrationTool

        tool = CalibrationTool(camera_id=camera_id)
        added = 0
        for fp in frame_files:
            frame = cv2.imread(str(fp))
            if frame is not None and tool.add_frame(frame):
                added += 1

        if added < 5:
            return None, f"Only {added} frames had detectable corners (need ≥5)"

        data = tool.calibrate()
        out_path = cal_dir / "calibration.json"
        tool.save(data, out_path)
        return data.to_dict(), None

    result, error = await asyncio.to_thread(_run_calibration)
    if error:
        return api_validation_error(error)

    return api_success({
        "camera_id": camera_id,
        "calibrated": True,
        "reprojection_error": result.get("reprojection_error", 0),
        "frames_used": len(frame_files),
    })


@router.get("/{camera_id}/verify")
async def verify_calibration(request: Request, camera_id: str) -> JSONResponse:
    """Verify calibration by projecting known world points to pixels.

    Returns a grid of world→pixel mappings for overlay on live feed.
    """
    cal_dir = _calibration_dir(request)
    cal_file = cal_dir / camera_id / "calibration.json"

    if not cal_file.exists():
        return api_not_found("No calibration file found")

    from argus.physics.calibration import CameraCalibration

    config = getattr(request.app.state, "config", None)
    pool_z = config.physics.pool_surface_z_mm if config and hasattr(config, "physics") else 0.0
    cal = CameraCalibration.from_file(cal_file, pool_surface_z_mm=pool_z)

    # Generate a verification grid (1m spacing on pool surface)
    grid_points = []
    for x_m in range(-5, 6):
        for y_m in range(-3, 4):
            x_mm = x_m * 1000.0
            y_mm = y_m * 1000.0
            px, py = cal.world_to_pixel(x_mm, y_mm, pool_z)
            grid_points.append({
                "world_x_mm": x_mm,
                "world_y_mm": y_mm,
                "pixel_x": round(px, 1),
                "pixel_y": round(py, 1),
            })

    cam_pos = cal.camera_position_world
    return api_success({
        "camera_id": camera_id,
        "camera_position": {"x_mm": cam_pos.x_mm, "y_mm": cam_pos.y_mm, "z_mm": cam_pos.z_mm},
        "grid_points": grid_points,
        "pool_surface_z_mm": pool_z,
    })
