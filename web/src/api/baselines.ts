import { api, u } from './client'

// ── Baselines ──
export const getBaselines = () => api.get('/baseline/list/json').then(u)
export const optimizeBaseline = (data: { camera_id: string; zone_id?: string; target_ratio?: number }) =>
  api.post('/baseline/optimize/json', data).then(u)
export const previewOptimize = (params: { camera_id: string; zone_id?: string; target_ratio?: number }) =>
  api.get('/baseline/optimize/preview', { params }).then(u)

// ── Capture Jobs ──
export const startCaptureJob = (data: FormData) => api.post('/baseline/job', data).then(u)
/** @deprecated Use startCaptureJob instead */
export const startCapture = startCaptureJob
export const pauseCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/pause`).then(u)
export const resumeCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/resume`).then(u)
export const abortCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/abort`).then(u)

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
