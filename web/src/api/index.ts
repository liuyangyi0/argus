import axios, { type AxiosResponse } from 'axios'
import type {
  ApiResponse,
  CamerasPayload,
  AlertsPayload,
  HealthPayload,
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
    const msg = body?.msg ?? body?.error ?? err.message ?? '请求失败'
    console.error('[API]', msg)
    return Promise.reject(err)
  }
)

/**
 * Unwrap the unified response envelope {code, msg, data}.
 * code=0: success — returns data directly.
 * code>0: throws ApiError.
 */
function unwrap<T>(res: AxiosResponse<ApiResponse<T>>): T {
  const body = res.data
  if (body.code === 0) return body.data
  throw new ApiError(body.code, body.msg, body.data)
}

/** Shorthand for generic unwrap */
const u = (res: AxiosResponse<ApiResponse<any>>) => unwrap<any>(res)

// ── Cameras ──
export const getCameras = () =>
  api.get<ApiResponse<CamerasPayload>>('/cameras/json').then(unwrap)
export const getCameraDetail = (id: string) =>
  api.get(`/cameras/${id}/detail/json`).then(u)
export const startCamera = (id: string) => api.post(`/cameras/${id}/start`).then(u)
export const stopCamera = (id: string) => api.post(`/cameras/${id}/stop`).then(u)
export const getUsbDevices = () => api.get('/cameras/usb-devices').then(u)

// ── Streaming (go2rtc) ──
export const getStreamInfo = (id: string) => api.get(`/streaming/${id}`).then(u)
export const getStreams = () => api.get('/streaming').then(u)
export const registerStream = (id: string) => api.post(`/streaming/${id}/register`).then(u)

// ── Alerts ──
export const getAlerts = (params?: Record<string, any>) =>
  api.get<ApiResponse<AlertsPayload>>('/alerts/json', { params }).then(unwrap)
export const getAlert = (id: string) => api.get(`/alerts/${id}/detail`)
export const acknowledgeAlert = (id: string) => api.post(`/alerts/${id}/acknowledge`)
export const markFalsePositive = (id: string) => api.post(`/alerts/${id}/false-positive`)
export const updateAlertWorkflow = (id: string, data: Record<string, any>) =>
  api.post(`/alerts/${id}/workflow`, data).then(u)
export const bulkAcknowledge = (ids: string[]) =>
  api.post('/alerts/bulk-acknowledge', { alert_ids: ids }).then(u)
export const bulkFalsePositive = (ids: string[]) =>
  api.post('/alerts/bulk-false-positive', { alert_ids: ids }).then(u)

// ── System ──
export const getHealth = () =>
  api.get<ApiResponse<HealthPayload>>('/system/health').then(unwrap)

// ── Baselines ──
export const getBaselines = () => api.get('/baseline/list/json').then(u)
export const startCapture = (data: FormData) => api.post('/baseline/job', data).then(u)
export const optimizeBaseline = (data: { camera_id: string; zone_id?: string; target_ratio?: number }) =>
  api.post('/baseline/optimize/json', data).then(u)
export const previewOptimize = (params: { camera_id: string; zone_id?: string; target_ratio?: number }) =>
  api.get('/baseline/optimize/preview', { params }).then(u)

// ── Advanced Baseline Capture Job ──
export const startCaptureJob = (data: FormData) => api.post('/baseline/job', data).then(u)
export const pauseCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/pause`).then(u)
export const resumeCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/resume`).then(u)
export const abortCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/abort`).then(u)
export const getCaptureJobStatus = (taskId: string) => api.get(`/baseline/job/${taskId}`).then(u)

// ── False Positive Merge ──
export const mergeFalsePositives = (data: { camera_id: string; zone_id?: string; max_fp_images?: number }) =>
  api.post('/baseline/merge-fp', data).then(u)

// ── Camera Groups ──
export const getCameraGroups = () => api.get('/baseline/groups/json').then(u)
export const mergeGroupBaseline = (data: { group_id: string; zone_id?: string; target_count?: number }) =>
  api.post('/baseline/groups/merge', data).then(u)

// ── Baseline Lifecycle ──
export const getBaselineVersions = (params: { camera_id: string; zone_id?: string }) =>
  api.get('/baseline/versions/json', { params }).then(u)
export const verifyBaseline = (data: { camera_id: string; zone_id?: string; version: string; verified_by: string; verified_by_secondary?: string }) =>
  api.post('/baseline/verify', data).then(u)
export const activateBaseline = (data: { camera_id: string; zone_id?: string; version: string; user?: string }) =>
  api.post('/baseline/activate-baseline', data).then(u)
export const retireBaseline = (data: { camera_id: string; zone_id?: string; version: string; user?: string; reason?: string }) =>
  api.post('/baseline/retire', data).then(u)
export const deleteBaselineVersion = (data: { camera_id: string; zone_id?: string; version: string; user?: string }) =>
  api.post('/baseline/version/delete', data).then(u)

// ── Training ──
export const startTraining = (data: FormData) => api.post('/baseline/train', data).then(u)
export const getTrainingHistory = (params?: Record<string, any>) =>
  api.get('/baseline/training-history/json', { params }).then(u)
export const compareModels = (data: { old_record_id: number; new_record_id: number }) =>
  api.post('/baseline/compare', data, { timeout: 120000 }).then(u)

// ── Models ──
export const getModels = () => api.get('/baseline/models/json').then(u)
export const deployModel = (data: Record<string, any>) => api.post('/baseline/deploy', data).then(u)
export const getModelRegistry = (cameraId?: string) =>
  api.get('/models/json', { params: cameraId ? { camera_id: cameraId } : {} }).then(u)
export const activateModel = (versionId: string) =>
  api.post(`/models/${versionId}/activate`).then(u)
export const rollbackModel = (versionId: string) =>
  api.post(`/models/${versionId}/rollback`).then(u)
export const deleteModel = (versionId: string) => api.delete(`/models/${versionId}`).then(u)
export const deleteModelByPath = (modelPath: string) =>
  api.delete('/baseline/models/by-path', { data: { model_path: modelPath } }).then(u)
export const reexportModel = (versionId: string, data: { export_format?: string; quantization?: string }) =>
  api.post(`/models/${versionId}/reexport`, data, { timeout: 300000 }).then(u)
export const recalibrateModel = (versionId: string) =>
  api.post(`/models/${versionId}/recalibrate`, {}, { timeout: 120000 }).then(u)

// ── Release Pipeline ──
export const promoteModel = (versionId: string, data: { target_stage: string; triggered_by: string; reason?: string; canary_camera_id?: string }) =>
  api.post(`/models/${versionId}/promote`, data).then(u)
export const retireModel = (versionId: string, data?: { triggered_by?: string; reason?: string }) =>
  api.post(`/models/${versionId}/retire`, data || {}).then(u)
export const getStageHistory = (versionId: string) =>
  api.get(`/models/${versionId}/stage-history`).then(u)
export const getVersionEvents = (params?: { camera_id?: string; limit?: number }) =>
  api.get('/models/events/list', { params }).then(u)
export const getShadowReport = (versionId: string, params?: { camera_id?: string; days?: number }) =>
  api.get(`/models/${versionId}/shadow-report`, { params }).then(u)
export const getBackboneStatus = () => api.get('/models/backbone/status').then(u)

// ── Batch Inference ──
export const batchInference = (cameraId: string, imagePaths: string[]) =>
  api.post('/models/batch-inference', { camera_id: cameraId, image_paths: imagePaths }).then(u)

// ── Training Jobs ──
export const getTrainingJobs = (params?: Record<string, any>) =>
  api.get('/training-jobs/json', { params }).then(u)
export const getTrainingJob = (jobId: string) => api.get(`/training-jobs/${jobId}`).then(u)
export const createTrainingJob = (data: {
  job_type: string; camera_id?: string; zone_id?: string;
  model_type?: string; trigger_type?: string; triggered_by?: string;
  hyperparameters?: Record<string, any>;
}) => api.post('/training-jobs/', data).then(u)
export const confirmTrainingJob = (jobId: string, data?: { confirmed_by?: string }) =>
  api.post(`/training-jobs/${jobId}/confirm`, data || {}).then(u)
export const rejectTrainingJob = (jobId: string, data?: { rejected_by?: string; reason?: string }) =>
  api.post(`/training-jobs/${jobId}/reject`, data || {}).then(u)
export const getBackbones = () => api.get('/training-jobs/backbones/json').then(u)

// ── Tasks ──
export const getTasks = () => api.get('/tasks/json').then(u)
export const dismissTask = (taskId: string) => api.delete(`/tasks/${taskId}`)

// ── Config ──
export const reloadConfig = () => api.post('/config/reload').then(u)

// ── Users ──
export const getUsers = () => api.get('/users/json').then(u)
export const createUser = (data: { username: string; password: string; role: string; display_name: string }) =>
  api.post('/users/json', data).then(u)
export const deleteUser = (username: string) => api.delete(`/users/${username}/json`).then(u)
export const toggleUserActive = (username: string) =>
  api.post(`/users/${username}/toggle-active/json`).then(u)

// ── Audit ──
export const getAuditLogs = (params?: { page?: number; page_size?: number; user?: string; action?: string }) =>
  api.get('/audit/json', { params }).then(u)

// ── Replay (FR-033) ──
export const getReplayMetadata = (alertId: string) =>
  api.get(`/replay/${alertId}/metadata`).then(u)
export const getReplaySignals = (alertId: string) =>
  api.get(`/replay/${alertId}/signals`).then(u)
export const getReplayFrameUrl = (alertId: string, index: number) =>
  `/api/replay/${alertId}/frame/${index}`
export const getReplayHeatmapUrl = (alertId: string, index: number) =>
  `/api/replay/${alertId}/heatmap/${index}`
export const getReplayReference = (alertId: string, params?: { date?: string; time?: string }) =>
  api.get(`/replay/${alertId}/reference`, { params }).then(u)
export const pinReplayFrame = (alertId: string, data: { index: number; label: string }) =>
  api.post(`/replay/${alertId}/pin-frame`, data).then(u)

// ── Video Wall ──
export const getWallStatus = () =>
  api.get<ApiResponse<{ cameras: any[] }>>('/cameras/wall/status').then(unwrap)

// ── Degradation ──
export const getDegradationActive = () => api.get('/degradation/active').then(u)
export const getDegradationSummary = () => api.get('/degradation/summary').then(u)
export const getDegradationHistory = (days?: number) =>
  api.get('/degradation/history', { params: { days } }).then(u)

// ── Audio Alerts ──
export const getAudioAlerts = () => api.get('/config/audio-alerts').then(u)
export const updateAudioAlerts = (data: any) => api.put('/config/audio-alerts', data).then(u)

export default api
