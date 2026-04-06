import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.error || err.message || '请求失败'
    console.error('[API]', msg)
    return Promise.reject(err)
  }
)

// ── Cameras ──
export const getCameras = () => api.get('/cameras/json')
export const startCamera = (id: string) => api.post(`/cameras/${id}/start`)
export const stopCamera = (id: string) => api.post(`/cameras/${id}/stop`)

// ── Alerts ──
export const getAlerts = (params?: Record<string, any>) => api.get('/alerts/json', { params })
export const getAlert = (id: string) => api.get(`/alerts/${id}/detail`)
export const acknowledgeAlert = (id: string) => api.post(`/alerts/${id}/acknowledge`)
export const markFalsePositive = (id: string) => api.post(`/alerts/${id}/false-positive`)
export const updateAlertWorkflow = (id: string, data: Record<string, any>) =>
  api.post(`/alerts/${id}/workflow`, data)
export const bulkAcknowledge = (ids: string[]) =>
  api.post('/alerts/bulk-acknowledge', { alert_ids: ids })
export const bulkFalsePositive = (ids: string[]) =>
  api.post('/alerts/bulk-false-positive', { alert_ids: ids })

// ── System ──
export const getHealth = () => api.get('/system/health')

// ── Baselines ──
export const getBaselines = () => api.get('/baseline/list/json')
export const startCapture = (data: FormData) => api.post('/baseline/capture', data)
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

export default api
