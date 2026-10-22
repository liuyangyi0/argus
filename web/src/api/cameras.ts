import type {
  ApiResponse,
  CameraDetailPayload,
  CameraModeMutation,
  CamerasPayload,
  ConnectionTestResult,
  PipelineMode,
  SystemModeSummary,
} from '../types/api'
import { api, unwrap, u } from './client'

export interface CameraWallItem {
  camera_id: string
  name: string
  status: string
  model_version?: string
  fps?: number
  current_score: number
  score_sparkline: number[]
  alert_count_today: number
  active_alert: {
    alert_id: string
    severity: string
    anomaly_score?: number
    timestamp?: number
    created_at?: number
    description?: string
  } | null
  degradation: string | null
  model_backend?: string | null
  expected_model_backend?: string | null
  capture_status?: {
    degradation_reason?: string | null
    fast_motion?: {
      degradation_reason?: string | null
    } | null
  } | null
  frames_dropped?: number
  backpressured?: boolean
}

export interface UsbCameraDevice {
  index: number
  name: string
  width: number
  height: number
}

export interface CameraConfigPayload {
  camera_id: string
  name: string
  region_id: number | null
  source: string
  protocol: string
  fps_target: number
  resolution: number[]
  gige_exposure?: number | null
  gige_gain?: number | null
  gige_pixel_format?: string | null
  gige_capture_script?: string | null
}

// ── Cameras ──
export const getCameras = () =>
  api.get<ApiResponse<CamerasPayload>>('/cameras/json').then(unwrap)
export const getCameraDetail = (id: string) =>
  api.get<ApiResponse<CameraDetailPayload>>(`/cameras/${id}/detail/json`).then(unwrap)
export const startCamera = (id: string) => api.post(`/cameras/${id}/start`).then(u)
export const stopCamera = (id: string) => api.post(`/cameras/${id}/stop`).then(u)
export const setCameraMode = (id: string, mode: PipelineMode) =>
  api.post<ApiResponse<CameraModeMutation>>(`/cameras/${id}/mode`, { mode }).then(unwrap)
export const getUsbDevices = () =>
  api
    .get<ApiResponse<{ devices: UsbCameraDevice[] }>>('/cameras/usb-devices')
    .then(unwrap)
export const addCamera = (data: FormData) => api.post('/cameras', data).then(u)
export const updateCamera = (id: string, data: FormData) =>
  api
    .put<ApiResponse<{ message?: string }>>(`/cameras/${id}`, data)
    .then(unwrap)
export const getCameraConfig = (id: string) =>
  api
    .get<ApiResponse<CameraConfigPayload>>(`/cameras/${id}/config`)
    .then(unwrap)
export const deleteCamera = (id: string) => api.delete(`/cameras/${id}`).then(u)

// ── Camera connectivity test (痛点 8) ──
export const testCameraConnection = (id: string) =>
  api.post<ApiResponse<ConnectionTestResult>>(`/cameras/${id}/test-connection`).then(unwrap)
export const testCameraConnectionDraft = (
  payload: { source: string | number; protocol?: string },
) =>
  api
    .post<ApiResponse<ConnectionTestResult>>('/cameras/test-connection-draft', payload)
    .then(unwrap)

// ── System mode summary (痛点 4) ──
export const getSystemModeSummary = () =>
  api.get<ApiResponse<SystemModeSummary>>('/system/mode_summary').then(unwrap)

// ── Video Wall ──
export const getWallStatus = () =>
  api
    .get<ApiResponse<{ cameras: CameraWallItem[] }>>('/cameras/wall/status')
    .then(unwrap)
