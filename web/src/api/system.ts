import type {
  ApiResponse,
  DetectionParamsResponse,
  HealthPayload,
  ModelHealthStatus,
} from '../types/api'
import { api, unwrap, u } from './client'

export type DegradationLevel = 'info' | 'warning' | 'moderate' | 'severe'

export interface DegradationSummaryEvent {
  event_id: string
  level: DegradationLevel
  category: string
  camera_id: string | null
  title: string
  impact: string
  action: string
  started_at: number
  resolved_at?: number | null
}

export interface DegradationSummaryPayload {
  active_count: number
  max_level: DegradationLevel | null
  events: DegradationSummaryEvent[]
}

export interface CameraStatusCollection {
  cameras: Array<Record<string, unknown>>
}

export interface BackupInfo {
  name?: string
  created?: string
  size_mb?: number
  has_database?: boolean
  has_configs?: boolean
  has_models?: boolean
  [key: string]: unknown
}

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
export const cleanupAlerts = () =>
  api
    .post<ApiResponse<{ deleted: number; files: number }>>('/config/cleanup')
    .then(unwrap)

// ── Backup ──
export const createBackup = () => api.post('/backup/create').then(u)

// ── Degradation ──
export const getDegradationActive = () => api.get('/degradation/active').then(u)
export const getDegradationSummary = () =>
  api
    .get<ApiResponse<DegradationSummaryPayload>>('/degradation/summary')
    .then(unwrap)
export const getDegradationHistory = (days?: number) =>
  api
    .get<ApiResponse<{ items: DegradationSummaryEvent[] }>>('/degradation/history', {
      params: { days },
    })
    .then(unwrap)

// ── Anomaly head degradation (per-pipeline simplex fallback) ──
// Distinct from the global degradation bar (`/degradation/*`): this surfaces
// the specific case where the anomaly model crashed or failed to load and the
// pipeline silently fell back to the simplex safety channel. Operators must
// be told because alert severity is auto-downgraded in this mode.
export interface AnomalyDegradationCameraStatus {
  camera_id: string
  degraded: boolean
  reason: string | null
  since: number | null
}

export interface AnomalyDegradationStatus {
  anomaly: {
    degraded: boolean
    reason: string | null
    since: number | null
    cameras: AnomalyDegradationCameraStatus[]
  }
}

export const getAnomalyDegradation = () =>
  api.get<ApiResponse<AnomalyDegradationStatus>>('/system/anomaly-degradation').then(unwrap)

// ── Audio Alerts ──
export interface AudioAlertLevelConfig {
  enabled?: boolean
  sound?: string
  voice_template?: string
}

export interface AudioAlertsUpdate {
  low: AudioAlertLevelConfig
  medium: AudioAlertLevelConfig
  high: AudioAlertLevelConfig
}

export const getAudioAlerts = () => api.get('/config/audio-alerts').then(u)
export const updateAudioAlerts = (data: AudioAlertsUpdate) =>
  api.put('/config/audio-alerts', data).then(u)

// ── Drift & Camera Health ──
export const getDriftStatus = () =>
  api.get<ApiResponse<CameraStatusCollection>>('/system/drift').then(unwrap)
export const getCameraHealth = () =>
  api.get<ApiResponse<CameraStatusCollection>>('/system/camera-health').then(unwrap)

// ── Classifier (open-vocabulary detector) ──
export interface AuxiliaryRuntimeDetail {
  camera_id: string
  state: string
  backend: string | null
  loaded: boolean
  ready: boolean
  degraded: boolean
  reason?: string | null
  busy?: boolean
}

export interface AuxiliaryRuntimeSummary {
  total_pipelines: number
  pipelines_attached: number
  pipelines_loaded: number
  // Optional while older servers are upgraded: "loaded" alone does not prove readiness.
  pipelines_ready?: number
  pipelines_degraded?: number
  pipelines_failed?: number
  details?: AuxiliaryRuntimeDetail[]
}

export interface ClassifierConfigPayload {
  enabled: boolean
  model_name: string
  min_anomaly_score_to_classify: number
  vocabulary: string[]
  high_risk_labels: string[]
  low_risk_labels: string[]
  suppress_labels: string[]
  custom_vocabulary_path: string | null
  runtime: AuxiliaryRuntimeSummary
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
  runtime: AuxiliaryRuntimeSummary
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

export interface ModuleToggleResult {
  key: string
  value: boolean
  restart_required: boolean
  hot_reloaded: number
  hot_reloadable: boolean
  hot_reload_failed: number
  pipelines_seen: number
  persisted: boolean
  all_applied?: boolean
}

export const getModuleStates = (): Promise<Record<string, boolean>> =>
  api.get<ApiResponse<Record<string, boolean>>>('/config/modules').then(unwrap)

export const toggleModule = (key: string, value: boolean): Promise<ModuleToggleResult> =>
  api
    .post<ApiResponse<ModuleToggleResult>>('/config/modules', { key, value })
    .then(unwrap)

// ── Config management ──
export interface ThresholdUpdate {
  anomaly_threshold?: number
  info_threshold?: number
  low_threshold?: number
  medium_threshold?: number
  high_threshold?: number
}

export interface DetectionParamsUpdate {
  anomaly_threshold?: number
  sev_info?: number
  sev_low?: number
  sev_medium?: number
  sev_high?: number
  temp_gap?: number
  temp_overlap?: number
  supp_zone?: number
}

export interface NotificationUpdate {
  webhook_enabled?: boolean
  webhook_url?: string
  webhook_timeout?: number
}

export const saveConfig = () => api.post('/config/save').then(u)
export const updateThresholds = (data: ThresholdUpdate) =>
  api.post('/config/thresholds', data).then(u)
export const updateDetectionParams = (data: DetectionParamsUpdate) =>
  api
    .post<ApiResponse<DetectionParamsResponse>>('/config/detection-params', data)
    .then(unwrap)
export const reloadModel = (data: { camera_id: string; model_path: string }) =>
  api.post('/config/reload-model', data).then(u)
export const clearLock = (cameraId: string) => api.post(`/config/clear-lock/${cameraId}`).then(u)
export const restartCamera = (cameraId: string) =>
  api.post(`/config/camera/${cameraId}/restart`).then(u)

// ── Webhook ──
// Email/SMS/template surface was removed in 2026-05; only webhook remains.
export const updateNotifications = (data: NotificationUpdate) =>
  api.post('/config/notifications', data).then(u)
export const testWebhook = () => api.post('/config/test-webhook').then(u)

// ── Backup management ──
export const listBackups = () =>
  api
    .get<ApiResponse<{ backups: BackupInfo[] }>>('/backup/list/json')
    .then(unwrap)
export const restoreBackup = (name: string) => {
  const form = new URLSearchParams()
  form.append('backup_name', name)
  return api.post('/backup/restore', form).then(u)
}
export const deleteBackup = (name: string) => api.delete(`/backup/${name}`).then(u)
