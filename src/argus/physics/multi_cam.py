"""Multi-camera 3-D triangulation.

Combines observations from two or more calibrated cameras to produce
true 3-D world positions via Direct Linear Transform (DLT)
triangulation.  Integrates with the existing
:class:`~argus.core.correlation.CrossCameraCorrelator` for detection
matching across cameras.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import structlog

from argus.physics.calibration import CameraCalibration, WorldPoint

logger = structlog.get_logger()


@dataclass
class CameraObservation:
    """A single-camera observation of a detected object."""

    camera_id: str
    pixel_x: float
    pixel_y: float
    timestamp: float
    confidence: float = 1.0


@dataclass
class TriangulationResult:
    """Result of multi-camera triangulation."""

    world_point: WorldPoint
    num_cameras: int
    reprojection_error_px: float  # mean reprojection error across cameras
    camera_ids: list[str]


class MultiCameraTriangulator:
    """Triangulate 3-D positions from 2+ calibrated camera observations.

    Uses OpenCV's ``triangulatePoints`` for stereo pairs and extends
    to N cameras via DLT (Direct Linear Transform) for 3+ views.

    Parameters
    ----------
    calibrations:
        Mapping from camera_id to CameraCalibration instances.
    max_time_delta:
        Maximum timestamp difference (seconds) between observations
        for them to be considered simultaneous.
    """

    def __init__(
        self,
        calibrations: dict[str, CameraCalibration],
        max_time_delta: float = 0.05,
    ) -> None:
        self._calibrations = calibrations
        self._max_dt = max_time_delta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def triangulate(self, observations: list[CameraObservation]) -> TriangulationResult | None:
        """Compute 3-D world position from multiple camera observations.

        Requires at least 2 observations from different calibrated cameras.
        If more than 2 are provided, uses DLT with all views.
        """
        # Filter to calibrated cameras and verify timing
        valid = [
            obs for obs in observations
            if obs.camera_id in self._calibrations
        ]
        if len(valid) < 2:
            return None

        # Check timestamps are close enough
        ts = [obs.timestamp for obs in valid]
        if max(ts) - min(ts) > self._max_dt:
            logger.warning(
                "triangulation.time_delta_exceeded",
                delta=round(max(ts) - min(ts), 4),
                max_delta=self._max_dt,
            )

        if len(valid) == 2:
            return self._triangulate_stereo(valid[0], valid[1])
        else:
            return self._triangulate_dlt(valid)

    def refraction_correct(
        self,
        world_point: WorldPoint,
        entry_z_mm: float,
        n_water: float = 1.33,
    ) -> WorldPoint:
        """Apply Snell's law correction for air→water interface.

        Objects observed through the water surface appear displaced
        due to refraction.  This shifts the estimated position to
        account for the refractive index change at *entry_z_mm*.
        """
        if world_point.z_mm >= entry_z_mm:
            # Object is above water, no correction
            return world_point

        depth_below_surface = entry_z_mm - world_point.z_mm
        # Apparent depth is shallower by factor 1/n_water
        correction_factor = 1.0 - 1.0 / n_water
        corrected_z = world_point.z_mm - depth_below_surface * correction_factor

        return WorldPoint(
            x_mm=world_point.x_mm,
            y_mm=world_point.y_mm,
            z_mm=corrected_z,
        )

    # ------------------------------------------------------------------
    # Stereo triangulation (2 cameras)
    # ------------------------------------------------------------------

    def _triangulate_stereo(
        self, obs_a: CameraObservation, obs_b: CameraObservation,
    ) -> TriangulationResult | None:
        cal_a = self._calibrations[obs_a.camera_id]
        cal_b = self._calibrations[obs_b.camera_id]

        # Undistort points
        pts_a = self._undistort(obs_a, cal_a)
        pts_b = self._undistort(obs_b, cal_b)

        # Projection matrices
        P_a = cal_a._P  # 3x4
        P_b = cal_b._P

        # Triangulate
        pts4d = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)  # 4xN
        pts3d = pts4d[:3] / pts4d[3]  # dehomogenise → 3x1

        world = WorldPoint(
            x_mm=float(pts3d[0, 0]),
            y_mm=float(pts3d[1, 0]),
            z_mm=float(pts3d[2, 0]),
        )

        # Compute reprojection error
        reproj_a = cal_a.world_to_pixel(world.x_mm, world.y_mm, world.z_mm)
        reproj_b = cal_b.world_to_pixel(world.x_mm, world.y_mm, world.z_mm)
        err_a = np.hypot(reproj_a[0] - obs_a.pixel_x, reproj_a[1] - obs_a.pixel_y)
        err_b = np.hypot(reproj_b[0] - obs_b.pixel_x, reproj_b[1] - obs_b.pixel_y)
        mean_err = float((err_a + err_b) / 2.0)

        return TriangulationResult(
            world_point=world,
            num_cameras=2,
            reprojection_error_px=round(mean_err, 3),
            camera_ids=[obs_a.camera_id, obs_b.camera_id],
        )

    # ------------------------------------------------------------------
    # DLT triangulation (3+ cameras)
    # ------------------------------------------------------------------

    def _triangulate_dlt(
        self, observations: list[CameraObservation],
    ) -> TriangulationResult | None:
        """Direct Linear Transform triangulation from N ≥ 2 views.

        Constructs the DLT matrix A (2N × 4) and solves via SVD.
        """
        A_rows = []
        for obs in observations:
            cal = self._calibrations[obs.camera_id]
            pt = self._undistort(obs, cal).flatten()
            u, v = pt[0], pt[1]
            P = cal._P  # 3x4
            A_rows.append(u * P[2] - P[0])
            A_rows.append(v * P[2] - P[1])

        A = np.array(A_rows)  # (2N, 4)

        # SVD: solution is the last column of V^T
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]  # homogeneous 4-vector
        if abs(X[3]) < 1e-12:
            return None
        X = X / X[3]

        world = WorldPoint(x_mm=float(X[0]), y_mm=float(X[1]), z_mm=float(X[2]))

        # Mean reprojection error
        errors = []
        camera_ids = []
        for obs in observations:
            cal = self._calibrations[obs.camera_id]
            rp = cal.world_to_pixel(world.x_mm, world.y_mm, world.z_mm)
            err = np.hypot(rp[0] - obs.pixel_x, rp[1] - obs.pixel_y)
            errors.append(err)
            camera_ids.append(obs.camera_id)

        return TriangulationResult(
            world_point=world,
            num_cameras=len(observations),
            reprojection_error_px=round(float(np.mean(errors)), 3),
            camera_ids=camera_ids,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _undistort(obs: CameraObservation, cal: CameraCalibration) -> np.ndarray:
        """Undistort a single pixel observation."""
        pts = np.array([[[obs.pixel_x, obs.pixel_y]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, cal._K, cal._data.dist_coeffs, P=cal._K)
        return undist[0]  # shape (1, 2)
