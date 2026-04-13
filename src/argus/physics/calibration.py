"""Camera calibration for pixel-to-world coordinate mapping.

Provides intrinsic/extrinsic camera parameter management and
pixel↔real-world coordinate conversion using the pinhole camera model.
Calibration can be performed interactively from checkerboard images
or loaded from a pre-computed JSON file.

The pixel_to_world mapping projects a 2-D image point onto the pool
water-surface plane (known Z in world coordinates) via ray–plane
intersection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class CalibrationData:
    """Stored calibration parameters for a single camera."""

    camera_id: str
    # Intrinsic matrix (3x3)
    camera_matrix: np.ndarray
    # Distortion coefficients (5-element or 8-element)
    dist_coeffs: np.ndarray
    # Rotation matrix (3x3) — world-to-camera
    rotation_matrix: np.ndarray
    # Translation vector (3x1) — world-to-camera
    translation_vector: np.ndarray
    # Reprojection error from calibration (pixels)
    reprojection_error: float = 0.0
    # Image resolution used during calibration
    image_size: tuple[int, int] = (1920, 1080)

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "translation_vector": self.translation_vector.tolist(),
            "reprojection_error": self.reprojection_error,
            "image_size": list(self.image_size),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationData:
        return cls(
            camera_id=data["camera_id"],
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64),
            rotation_matrix=np.array(data["rotation_matrix"], dtype=np.float64),
            translation_vector=np.array(data["translation_vector"], dtype=np.float64),
            reprojection_error=data.get("reprojection_error", 0.0),
            image_size=tuple(data.get("image_size", [1920, 1080])),
        )


@dataclass
class WorldPoint:
    """A point in the real-world coordinate system (millimetres)."""

    x_mm: float
    y_mm: float
    z_mm: float


class CameraCalibration:
    """Pixel↔world coordinate mapping using calibrated camera parameters.

    Loads calibration from a JSON file and provides forward/inverse
    projection between image pixels and the 3-D world coordinate system.

    The ``pixel_to_world`` method intersects the camera ray with the
    water-surface plane at a configurable Z coordinate.
    """

    def __init__(
        self,
        calibration_data: CalibrationData,
        pool_surface_z_mm: float = 0.0,
    ) -> None:
        self._data = calibration_data
        self._pool_z = pool_surface_z_mm

        # Precompute projection matrix P = K @ [R | t]
        self._K = self._data.camera_matrix
        self._R = self._data.rotation_matrix
        self._t = self._data.translation_vector.reshape(3, 1)
        self._Rt = np.hstack([self._R, self._t])  # 3x4
        self._P = self._K @ self._Rt  # 3x4 projection matrix

        # Camera position in world coords: C = -R^T @ t
        self._cam_pos = -self._R.T @ self._t  # 3x1

        # Inverse of K for undistort→normalize
        self._K_inv = np.linalg.inv(self._K)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path, pool_surface_z_mm: float = 0.0) -> CameraCalibration:
        """Load calibration from a JSON file."""
        path = Path(path)
        data = CalibrationData.from_dict(json.loads(path.read_text()))
        logger.info(
            "calibration.loaded",
            camera_id=data.camera_id,
            reproj_error=round(data.reprojection_error, 3),
        )
        return cls(data, pool_surface_z_mm=pool_surface_z_mm)

    @property
    def data(self) -> CalibrationData:
        return self._data

    @property
    def camera_position_world(self) -> WorldPoint:
        """Camera centre in world coordinates."""
        c = self._cam_pos.flatten()
        return WorldPoint(x_mm=float(c[0]), y_mm=float(c[1]), z_mm=float(c[2]))

    def pixel_to_world(self, px: float, py: float, z_mm: float | None = None) -> WorldPoint:
        """Project a pixel onto the world plane at *z_mm*.

        Uses ray–plane intersection: the ray from the camera centre
        through the undistorted pixel is intersected with the horizontal
        plane Z = *z_mm* (defaults to pool surface).
        """
        z_plane = z_mm if z_mm is not None else self._pool_z

        # Undistort the pixel
        pts = np.array([[[px, py]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, self._K, self._data.dist_coeffs, P=self._K)
        u, v = undist[0, 0]

        # Ray direction in camera frame
        ray_cam = self._K_inv @ np.array([u, v, 1.0])
        # Transform to world frame
        ray_world = self._R.T @ ray_cam  # direction in world
        origin = self._cam_pos.flatten()  # camera position in world

        # Intersect with Z = z_plane
        dz = ray_world[2]
        if abs(dz) < 1e-9:
            # Ray is parallel to the plane — return projection at camera Z
            return WorldPoint(x_mm=float(origin[0]), y_mm=float(origin[1]), z_mm=z_plane)

        t_param = (z_plane - origin[2]) / dz
        world = origin + t_param * ray_world

        return WorldPoint(x_mm=float(world[0]), y_mm=float(world[1]), z_mm=z_plane)

    def world_to_pixel(self, x_mm: float, y_mm: float, z_mm: float) -> tuple[float, float]:
        """Project a world point back to image pixel coordinates."""
        pt = np.array([x_mm, y_mm, z_mm, 1.0])
        projected = self._P @ pt  # homogeneous image coords
        if abs(projected[2]) < 1e-9:
            return 0.0, 0.0
        px = projected[0] / projected[2]
        py = projected[1] / projected[2]
        return float(px), float(py)

    def pixel_distance_to_mm(self, px1: float, py1: float, px2: float, py2: float) -> float:
        """Compute real-world distance between two pixels on the pool surface."""
        w1 = self.pixel_to_world(px1, py1)
        w2 = self.pixel_to_world(px2, py2)
        dx = w2.x_mm - w1.x_mm
        dy = w2.y_mm - w1.y_mm
        return float(np.sqrt(dx * dx + dy * dy))


class CalibrationTool:
    """Interactive checkerboard calibration from camera frames.

    Wraps OpenCV's ``findChessboardCorners`` / ``calibrateCamera`` pipeline.
    Results are saved as JSON for use by :class:`CameraCalibration`.
    """

    def __init__(
        self,
        camera_id: str,
        board_size: tuple[int, int] = (9, 6),
        square_size_mm: float = 80.0,
    ) -> None:
        self._camera_id = camera_id
        self._board_size = board_size
        self._square_mm = square_size_mm
        self._obj_points: list[np.ndarray] = []
        self._img_points: list[np.ndarray] = []
        self._image_size: tuple[int, int] | None = None

    def add_frame(self, frame: np.ndarray) -> bool:
        """Detect checkerboard corners in *frame* and store them.

        Returns True if corners were found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        self._image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, self._board_size, None)
        if not found:
            return False

        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Object points in 3-D: (0,0,0), (sq,0,0), (2*sq,0,0), ...
        obj_p = np.zeros((self._board_size[0] * self._board_size[1], 3), np.float32)
        obj_p[:, :2] = (
            np.mgrid[0 : self._board_size[0], 0 : self._board_size[1]]
            .T.reshape(-1, 2)
            * self._square_mm
        )

        self._obj_points.append(obj_p)
        self._img_points.append(corners)
        logger.info("calibration_tool.frame_added", count=len(self._obj_points))
        return True

    @property
    def frame_count(self) -> int:
        return len(self._obj_points)

    def calibrate(self) -> CalibrationData:
        """Run calibration and return the result.

        Requires at least 5 frames with detected corners.
        """
        if len(self._obj_points) < 5:
            raise ValueError(f"Need ≥5 frames with corners, have {len(self._obj_points)}")

        if self._image_size is None:
            raise ValueError("No frames added")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._obj_points,
            self._img_points,
            self._image_size,
            None,
            None,
        )

        # Use first frame's extrinsics as reference (can be overridden later)
        R, _ = cv2.Rodrigues(rvecs[0])
        t = tvecs[0]

        data = CalibrationData(
            camera_id=self._camera_id,
            camera_matrix=mtx,
            dist_coeffs=dist,
            rotation_matrix=R,
            translation_vector=t,
            reprojection_error=ret,
            image_size=self._image_size,
        )
        logger.info(
            "calibration_tool.calibrated",
            reproj_error=round(ret, 4),
            frames_used=len(self._obj_points),
        )
        return data

    def save(self, data: CalibrationData, path: str | Path) -> None:
        """Save calibration data to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data.to_dict(), indent=2))
        logger.info("calibration_tool.saved", path=str(path))

    def draw_corners(self, frame: np.ndarray) -> np.ndarray:
        """Draw detected corners on a frame (for verification UI)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        found, corners = cv2.findChessboardCorners(gray, self._board_size, None)
        out = frame.copy()
        if found:
            cv2.drawChessboardCorners(out, self._board_size, corners, found)
        return out
