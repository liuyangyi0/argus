import type { ApiResponse, HealthPayload, ModelHealthStatus } from '../types/api'
import { api, unwrap, u } from './client'

// ── Health ──
export const getHealth = () =>
  api.get<ApiResponse<HealthPayload>>('/system/health').then(unwrap)

// ── Per-model live health ──
export const getModelsStatus = () =>
  api
    .get<ApiResponse<{ models: ModelHealthStatus[] }>>('/models/status')
    .then(unwrap)
    .then((payload) => payload.models)

// ── Config ──
export const reloadConfig = () => api.post('/config/reload').then(u)

// ── Storage & Retention ──
export const getStorageInfo = () => api.get('/config/storage/info').then(u)
export const cleanupAlerts = () => api.post('/config/cleanup').then(u)

// ── Backup ──
export const createBackup = () => api.post('/backup/create').then(u)

// ── Degradation ──
export const getDegradationActive = () => api.get('/degradation/active').then(u)
export const getDegradationSummary = () => api.get('/degradation/summary').then(u)
export const getDegradationHistory = (days?: number) =>
  api.get('/degradation/history', { params: { days } }).then(u)

// ── Audio Alerts ──
export const getAudioAlerts = () => api.get('/config/audio-alerts').then(u)
export const updateAudioAlerts = (data: any) => api.put('/config/audio-alerts', data).then(u)

// ── Drift & Camera Health ──
export const getDriftStatus = () => api.get('/system/drift').then(u)
export const getCameraHealth = () => api.get('/system/camera-health').then(u)

// ── Classifier (open-vocabulary detector) ──
export interface ClassifierConfigPayload {
  enabled: boolean
  model_name: string
  min_anomaly_score_to_classify: number
  vocabulary: string[]
  high_risk_labels: string[]
  low_risk_labels: string[]
  suppress_labels: string[]
  custom_vocabulary_path: string | null
  runtime: {
    total_pipelines: number
    pipelines_attached: number
    pipelines_loaded: number
  }
}

export const getClassifierConfig = () =>
  api.get<ApiResponse<ClassifierConfigPayload>>('/config/classifier').then(unwrap)

// ── Imaging (multi-modal: polarization + NIR) ──
export interface ImagingConfigPayload {
  enabled: boolean
  mode: 'visible_only' | 'polarization' | 'polarization_nir'
  camera_sdk: 'opencv' | 'arena' | 'spinnaker'
  nir_strobe_enabled: boolean
  polarization_processing: boolean
  fusion_channels: number
  deglare_method: 'stokes' | 'min_intensity'
  dolp_threshold: number
  runtime: {
    total_pipelines: number
    pipelines_with_imaging: number
  }
}

export const getImagingConfig = () =>
  api.get<ApiResponse<ImagingConfigPayload>>('/config/imaging').then(unwrap)

// ── Segmenter (SAM2 instance segmentation) ──
export interface SegmenterConfigPayload {
  enabled: boolean
  model_size: string
  max_points: number
  min_anomaly_score: number
  min_mask_area_px: number
  timeout_seconds: number
  runtime: {
    total_pipelines: number
    pipelines_attached: number
    pipelines_loaded: number
  }
}

export const getSegmenterConfig = () =>
  api.get<ApiResponse<SegmenterConfigPayload>>('/config/segmenter').then(unwrap)

// ── Cross-camera correlation ──
export interface CrossCameraOverlapPair {
  camera_a: string
  camera_b: string
  homography: number[][]  // 3x3
}

export interface CrossCameraConfigPayload {
  enabled: boolean
  corroboration_threshold: number
  max_age_seconds: number
  uncorroborated_severity_downgrade: number
  overlap_pairs: CrossCameraOverlapPair[]
  runtime: {
    total_pipelines: number
    correlator_present: boolean
  }
}

export const getCrossCameraConfig = () =>
  api.get<ApiResponse<CrossCameraConfigPayload>>('/config/cross-camera').then(unwrap)

export interface CrossCameraUpdatePayload {
  overlap_pairs?: CrossCameraOverlapPair[]
  corroboration_threshold?: number
  max_age_seconds?: number
  uncorroborated_severity_downgrade?: number
}

export const updateCrossCameraConfig = (payload: CrossCameraUpdatePayload) =>
  api.put('/config/cross-camera/pairs', payload).then(u)

export interface SegmenterParamsUpdate {
  max_points?: number
  min_anomaly_score?: number
  min_mask_area_px?: number
  timeout_seconds?: number
}

export const updateSegmenterParams = (payload: SegmenterParamsUpdate) =>
  api.put('/config/segmenter/params', payload).then(u)

export interface ClassifierVocabularyUpdate {
  vocabulary: string[]
  high_risk_labels?: string[]
  low_risk_labels?: string[]
  suppress_labels?: string[]
}

export const updateClassifierVocabulary = (payload: ClassifierVocabularyUpdate) =>
  api.put('/config/classifier/vocabulary', payload).then(u)

export const toggleModule = (key: string, value: boolean) =>
  api.post('/config/modules', { key, value }).then(u)

// ── Config management ──
export const saveConfig = () => api.post('/config/save').then(u)
export const updateThresholds = (data: any) => api.post('/config/thresholds', data).then(u)
export const updateDetectionParams = (data: any) => api.post('/config/detection-params', data).then(u)
export const reloadModel = (data: { camera_id: string; model_path: string }) =>
  api.post('/config/reload-model', data).then(u)
export const clearLock = (cameraId: string) => api.post(`/config/clear-lock/${cameraId}`).then(u)
export const restartCamera = (cameraId: string) =>
  api.post(`/config/camera/${cameraId}/restart`).then(u)

// ── Webhook ──
// Email/SMS/template surface was removed in 2026-05; only webhook remains.
export const updateNotifications = (data: any) => api.post('/config/notifications', data).then(u)
export const testWebhook = () => api.post('/config/test-webhook').then(u)

// ── Backup management ──
export const listBackups = () => api.get('/backup/list/json').then(u)
export const restoreBackup = (name: string) => {
  const form = new URLSearchParams()
  form.append('backup_name', name)
  return api.post('/backup/restore', form).then(u)
}
export const deleteBackup = (name: string) => api.delete(`/backup/${name}`).then(u)
