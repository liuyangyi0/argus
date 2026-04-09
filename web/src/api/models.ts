import { api, u } from './client'

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

// ── A/B Comparison ──
export const getABScores = (versionId: string, params?: { camera_id?: string; days?: number; limit?: number }) =>
  api.get(`/models/${versionId}/ab-scores`, { params }).then(u)
export const getABDistribution = (versionId: string, params?: { camera_id?: string; days?: number; bins?: number }) =>
  api.get(`/models/${versionId}/ab-distribution`, { params }).then(u)
export const runABLiveCompare = (cameraId: string, shadowVersionId: string) =>
  api.post('/models/ab-live-compare', { camera_id: cameraId, shadow_version_id: shadowVersionId }, { timeout: 60000 }).then(u)
