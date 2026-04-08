import axios, { type AxiosResponse } from 'axios'
import type {
  ApiResponse,
  CamerasPayload,
  AlertsPayload,
  HealthPayload,
  TasksPayload,
} from '../types/api'
import { ApiError } from '../types/api'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const body = err.response?.data
    // {code, msg, data} envelope format
    const msg = body?.msg
      // Legacy format fallback
      ?? body?.error
      ?? err.message
      ?? '请求失败'
    console.error('[API]', msg)
    return Promise.reject(err)
  }
)

/**
 * Unwrap the unified response envelope {code, msg, data}.
 * code=0: success, returns data directly.
 * code>0: throws ApiError.
 */
function unwrap<T>(res: AxiosResponse<ApiResponse<T>>): T {
  const body = res.data
  if (body.code === 0) {
    return body.data
  }
  throw new ApiError(body.code, body.msg, body.data)
}

// ── Cameras (P0 — envelope) ──
export const getCameras = () =>
  api.get<ApiResponse<CamerasPayload>>('/cameras/json').then(unwrap)
export const getCameraDetail = (id: string) =>
  api.get<ApiResponse<Record<string, any>>>(`/cameras/${id}/detail/json`).then(unwrap)
export const startCamera = (id: string) =>
  api.post<ApiResponse<Record<string, any>>>(`/cameras/${id}/start`).then(unwrap)
export const stopCamera = (id: string) =>
  api.post<ApiResponse<Record<string, any>>>(`/cameras/${id}/stop`).then(unwrap)
export const getUsbDevices = () =>
  api.get<ApiResponse<{ devices: any[] }>>('/cameras/usb-devices').then(unwrap)

// ── Streaming (go2rtc) ──
export const getStreamInfo = (id: string) => api.get(`/streaming/${id}`)
export const getStreams = () => api.get('/streaming')
export const registerStream = (id: string) => api.post(`/streaming/${id}/register`)

// ── Alerts (P0 — envelope) ──
export const getAlerts = (params?: Record<string, any>) =>
  api.get<ApiResponse<AlertsPayload>>('/alerts/json', { params }).then(unwrap)
export const getAlert = (id: string) => api.get(`/alerts/${id}/detail`)
export const acknowledgeAlert = (id: string) => api.post(`/alerts/${id}/acknowledge`)
export const markFalsePositive = (id: string) => api.post(`/alerts/${id}/false-positive`)
export const updateAlertWorkflow = (id: string, data: Record<string, any>) =>
  api.post<ApiResponse<Record<string, any>>>(`/alerts/${id}/workflow`, data).then(unwrap)
export const bulkAcknowledge = (ids: string[]) =>
  api.post<ApiResponse<{ count: number; message: string }>>('/alerts/bulk-acknowledge', { alert_ids: ids }).then(unwrap)
export const bulkFalsePositive = (ids: string[]) =>
  api.post<ApiResponse<{ count: number; message: string }>>('/alerts/bulk-false-positive', { alert_ids: ids }).then(unwrap)

// ── System (P0 — envelope) ──
export const getHealth = () =>
  api.get<ApiResponse<HealthPayload>>('/system/health').then(unwrap)

// ── Baselines ──
export const getBaselines = () => api.get('/baseline/list/json')
export const startCapture = (data: FormData) => api.post('/baseline/job', data)
export const optimizeBaseline = (data: { camera_id: string; zone_id?: string; target_ratio?: number }) =>
  api.post('/baseline/optimize/json', data)
export const previewOptimize = (params: { camera_id: string; zone_id?: string; target_ratio?: number }) =>
  api.get('/baseline/optimize/preview', { params })

// ── Advanced Baseline Capture Job ──
export const startCaptureJob = (data: FormData) => api.post('/baseline/job', data)
export const pauseCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/pause`)
export const resumeCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/resume`)
export const abortCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/abort`)
export const getCaptureJobStatus = (taskId: string) => api.get(`/baseline/job/${taskId}`)

// ── False Positive Merge ──
export const mergeFalsePositives = (data: { camera_id: string; zone_id?: string; max_fp_images?: number }) =>
  api.post('/baseline/merge-fp', data)

// ── Camera Groups ──
export const getCameraGroups = () => api.get('/baseline/groups/json')
export const mergeGroupBaseline = (data: { group_id: string; zone_id?: string; target_count?: number }) =>
  api.post('/baseline/groups/merge', data)

// ── Baseline Lifecycle ──
export const getBaselineVersions = (params: { camera_id: string; zone_id?: string }) =>
  api.get('/baseline/versions/json', { params })
export const verifyBaseline = (data: { camera_id: string; zone_id?: string; version: string; verified_by: string; verified_by_secondary?: string }) =>
  api.post('/baseline/verify', data)
export const activateBaseline = (data: { camera_id: string; zone_id?: string; version: string; user?: string }) =>
  api.post('/baseline/activate-baseline', data)
export const retireBaseline = (data: { camera_id: string; zone_id?: string; version: string; user?: string; reason?: string }) =>
  api.post('/baseline/retire', data)
export const deleteBaselineVersion = (data: { camera_id: string; zone_id?: string; version: string; user?: string }) =>
  api.post('/baseline/version/delete', data)

// ── Training ──
export const startTraining = (data: FormData) => api.post('/baseline/train', data)
export const getTrainingHistory = (params?: Record<string, any>) =>
  api.get('/baseline/training-history/json', { params })
export const compareModels = (data: { old_record_id: number; new_record_id: number }) =>
  api.post('/baseline/compare', data, { timeout: 120000 })

// ── Models ──
export const getModels = () => api.get('/baseline/models/json')
export const deployModel = (data: Record<string, any>) => api.post('/baseline/deploy', data)
export const getModelRegistry = (cameraId?: string) =>
  api.get('/models/json', { params: cameraId ? { camera_id: cameraId } : {} })
export const activateModel = (versionId: string) =>
  api.post(`/models/${versionId}/activate`)
export const rollbackModel = (versionId: string) =>
  api.post(`/models/${versionId}/rollback`)
export const deleteModel = (versionId: string) =>
  api.delete(`/models/${versionId}`)
export const deleteModelByPath = (modelPath: string) =>
  api.delete('/baseline/models/by-path', { data: { model_path: modelPath } })
export const reexportModel = (versionId: string, data: { export_format?: string; quantization?: string }) =>
  api.post(`/models/${versionId}/reexport`, data, { timeout: 300000 })
export const recalibrateModel = (versionId: string) =>
  api.post(`/models/${versionId}/recalibrate`, {}, { timeout: 120000 })

// ── Release Pipeline ──
export const promoteModel = (versionId: string, data: { target_stage: string; triggered_by: string; reason?: string; canary_camera_id?: string }) =>
  api.post(`/models/${versionId}/promote`, data)
export const retireModel = (versionId: string, data?: { triggered_by?: string; reason?: string }) =>
  api.post(`/models/${versionId}/retire`, data || {})
export const getStageHistory = (versionId: string) =>
  api.get(`/models/${versionId}/stage-history`)
export const getVersionEvents = (params?: { camera_id?: string; limit?: number }) =>
  api.get('/models/events/list', { params })
export const getShadowReport = (versionId: string, params?: { camera_id?: string; days?: number }) =>
  api.get(`/models/${versionId}/shadow-report`, { params })
export const getBackboneStatus = () => api.get('/models/backbone/status')

// ── Batch Inference ──
export const batchInference = (cameraId: string, imagePaths: string[]) =>
  api.post('/models/batch-inference', { camera_id: cameraId, image_paths: imagePaths })

// ── Training Jobs ──
export const getTrainingJobs = (params?: Record<string, any>) =>
  api.get('/training-jobs/json', { params })
export const getTrainingJob = (jobId: string) =>
  api.get(`/training-jobs/${jobId}`)
export const createTrainingJob = (data: {
  job_type: string; camera_id?: string; zone_id?: string;
  model_type?: string; trigger_type?: string; triggered_by?: string;
  hyperparameters?: Record<string, any>;
}) => api.post('/training-jobs/', data)
export const confirmTrainingJob = (jobId: string, data?: { confirmed_by?: string }) =>
  api.post(`/training-jobs/${jobId}/confirm`, data || {})
export const rejectTrainingJob = (jobId: string, data?: { rejected_by?: string; reason?: string }) =>
  api.post(`/training-jobs/${jobId}/reject`, data || {})
export const getBackbones = () => api.get('/training-jobs/backbones/json')

// ── Tasks ──
export const getTasks = () => api.get('/tasks/json')
export const dismissTask = (taskId: string) => api.delete(`/tasks/${taskId}`)

// ── Config ──
export const reloadConfig = () => api.post('/config/reload')

// ── Users ──
export const getUsers = () => api.get('/users/json')
export const createUser = (data: { username: string; password: string; role: string; display_name: string }) =>
  api.post('/users/json', data)
export const deleteUser = (username: string) => api.delete(`/users/${username}/json`)
export const toggleUserActive = (username: string) =>
  api.post(`/users/${username}/toggle-active/json`)

// ── Audit ──
export const getAuditLogs = (params?: { page?: number; page_size?: number; user?: string; action?: string }) =>
  api.get('/audit/json', { params })

// ── Replay (FR-033) ──
export const getReplayMetadata = (alertId: string) =>
  api.get(`/replay/${alertId}/metadata`)
export const getReplaySignals = (alertId: string) =>
  api.get(`/replay/${alertId}/signals`)
export const getReplayFrameUrl = (alertId: string, index: number) =>
  `/api/replay/${alertId}/frame/${index}`
export const getReplayHeatmapUrl = (alertId: string, index: number) =>
  `/api/replay/${alertId}/heatmap/${index}`
export const getReplayReference = (alertId: string, params?: { date?: string; time?: string }) =>
  api.get(`/replay/${alertId}/reference`, { params })
export const pinReplayFrame = (alertId: string, data: { index: number; label: string }) =>
  api.post(`/replay/${alertId}/pin-frame`, data)

// ── Video Wall (envelope) ──
export const getWallStatus = () =>
  api.get<ApiResponse<{ cameras: any[] }>>('/cameras/wall/status').then(unwrap)

// ── Degradation ──
export const getDegradationActive = () => api.get('/degradation/active')
export const getDegradationSummary = () => api.get('/degradation/summary')
export const getDegradationHistory = (days?: number) =>
  api.get('/degradation/history', { params: { days } })

// ── Audio Alerts ──
export const getAudioAlerts = () => api.get('/config/audio-alerts')
export const updateAudioAlerts = (data: any) => api.put('/config/audio-alerts', data)

export default api
