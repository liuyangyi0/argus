import { api, u } from './client'
import type { TrajectoryFit } from '../types/api'

export interface TrajectoryPrimary {
  speed_ms: number | null
  speed_px_per_sec: number | null
  trajectory_model: string | null
  origin: { x_mm: number; y_mm: number; z_mm: number } | null
  landing: { x_mm: number; y_mm: number; z_mm: number } | null
}

export interface TrajectoryResponse {
  alert_id: string
  primary: TrajectoryPrimary | null
  trajectories: TrajectoryFit[]
  classification: { label: string; confidence: number } | null
}

export interface ActiveTrack {
  track_id: number
  centroid_x: number
  centroid_y: number
  consecutive_frames: number
  max_score: number
  area_px: number
  trajectory_length: number
}

export interface ActiveTracksResponse {
  camera_id: string
  tracks: ActiveTrack[]
}

export const getAlertTrajectory = (alertId: string): Promise<TrajectoryResponse> =>
  api.get(`/physics/${alertId}/trajectory`).then(u)

export const getActiveTracks = (cameraId: string): Promise<ActiveTracksResponse> =>
  api.get(`/physics/${cameraId}/active-tracks`).then(u)
