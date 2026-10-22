import type {
  ReleaseReadinessGate as GeneratedReleaseReadinessGate,
  ReleaseReadinessResponse as GeneratedReleaseReadinessResponse,
} from './generated-openapi'

// ── Unified API Response Envelope: {code, msg, data} ──

export interface ApiResponse<T = unknown> {
  code: number
  msg: string
  data: T
}

// code=0 is success, code>0 is error

// ── Error Codes (mirrors backend ErrorCode) ──

export const ErrorCode = {
  SUCCESS: 0,
  VALIDATION_ERROR: 40000,
  CONFIRMATION_REQUIRED: 40001,
  AUTH_REQUIRED: 40100,
  FORBIDDEN: 40300,
  NOT_FOUND: 40400,
  METHOD_NOT_ALLOWED: 40500,
  CONFLICT: 40900,
  UNPROCESSABLE_ENTITY: 42200,
  RATE_LIMITED: 42900,
  INTERNAL_ERROR: 50000,
  SERVICE_UNAVAILABLE: 50300,
} as const
export type ErrorCode = (typeof ErrorCode)[keyof typeof ErrorCode]

// ── ApiError class for structured error handling ──

export interface ApiErrorInit {
  httpStatus?: number | null
  businessCode?: number | null
  message: string
  details?: unknown
  /** Raw server payload retained for legacy call sites that read `error.data`. */
  raw?: unknown
}

export class ApiError extends Error {
  /** HTTP transport status. `null` means the failure did not reach HTTP. */
  readonly httpStatus: number | null
  /** Stable backend business code from the `{code,msg,data}` envelope. */
  readonly businessCode: number | null
  readonly details?: unknown

  /**
   * @deprecated Use `httpStatus` or `businessCode`. For compatibility this is
   * the non-2xx HTTP status when present, otherwise the business code.
   */
  readonly code: number
  /** @deprecated Use `details`; retained for existing replay/error consumers. */
  readonly data?: unknown

  constructor(code: number, message: string, data?: unknown)
  constructor(init: ApiErrorInit)
  constructor(codeOrInit: number | ApiErrorInit, message = '请求失败', data?: unknown) {
    const init: ApiErrorInit = typeof codeOrInit === 'number'
      ? {
          httpStatus: codeOrInit > 0 && codeOrInit < 1000 ? codeOrInit : null,
          businessCode: codeOrInit >= 1000 ? codeOrInit : null,
          message,
          details: data,
          raw: data,
        }
      : codeOrInit

    super(init.message)
    this.name = 'ApiError'
    this.httpStatus = init.httpStatus ?? null
    this.businessCode = init.businessCode ?? null
    this.details = init.details
    const failedHttpStatus = this.httpStatus !== null
      && (this.httpStatus < 200 || this.httpStatus >= 300)
      ? this.httpStatus
      : null
    this.code = failedHttpStatus ?? this.businessCode ?? this.httpStatus ?? -1
    this.data = init.raw ?? init.details
  }
}

export interface WsEvent<T = unknown> {
  schema_version: 1
  event_id: string
  topic: string
  type: string
  resource_id: string
  revision: number
  occurred_at: string
  data: T
}

// ── Domain Types ──

export type PipelineMode =
  | 'active'
  | 'maintenance'
  | 'learning'
  | 'collection'
  | 'training'

export interface CameraHealth {
  status?: string
  last_heartbeat?: string | null
  recent_errors?: string[] | null
  fps?: number | null
  [k: string]: unknown
}

export interface CameraSummary {
  camera_id: string
  name: string
  region_id?: number | null
  region_name?: string | null
  connected: boolean
  running: boolean
  pipeline_mode?: PipelineMode | null
  health?: CameraHealth | null
  capture_status?: CameraCaptureStatus | null
  stats: CameraStats | null
}

export interface FastMotionStatus {
  enabled: boolean
  active: boolean
  process_width: number
  min_runtime_fps: number
  required_resolution: [number, number] | number[]
  current_fps?: number | null
  degraded: boolean
  degradation_reason?: string | null
}

export interface CameraCaptureStatus {
  backend?: string | null
  requested_pixel_format?: string | null
  pixel_format?: string | null
  requested_resolution?: [number, number] | number[] | null
  actual_resolution?: [number, number] | number[] | null
  requested_fps?: number | null
  reported_fps?: number | null
  actual_fps?: number | null
  degraded: boolean
  degradation_reason?: string | null
  fast_motion?: FastMotionStatus | null
}

export interface ConnectionTestResult {
  ok: boolean
  latency_ms?: number
  resolution?: [number, number]
  error?: string
}

export interface SystemModeSummary {
  cameras: Record<string, PipelineMode>
  global_state: 'normal' | 'capturing' | 'training' | 'maintenance'
}

export interface CameraModeMutation {
  camera_id: string
  previous_mode: PipelineMode | null
  pipeline_mode: PipelineMode
}

export interface BaselineCollection {
  camera_id: string
  zone_id: string
  version: string
  image_count: number
  status: 'ok' | 'partial' | 'failed' | string
  captured_at?: string | null
  session_label?: string
  acceptance_rate?: number | null
  error?: string | null
  state?: string | null
  is_current?: boolean
}

export interface DatasetSelectionItem {
  camera_id: string
  zone_id: string
  version: string
  session_label?: string
}

export interface DatasetSelection {
  items: DatasetSelectionItem[]
  total_frames?: number
}

export interface ConfigApplyResult {
  changed: boolean
  hot_reloaded: boolean
  applied: number
  total: number
}

export interface DetectionParamsResponse {
  pipelines_updated: number
  anomaly_threshold: ConfigApplyResult
  severity: ConfigApplyResult
  temporal: ConfigApplyResult
  suppression: ConfigApplyResult
}

export interface FeedbackStats {
  fp_count: number
  confirmed_count: number
  fp_rate: number
}

export interface CameraStats {
  frames_captured: number
  frames_analyzed: number
  anomalies_detected: number
  alerts_emitted: number
  avg_latency_ms: number
}

export interface ZoneVertex {
  x: number
  y: number
}

export interface CameraZone {
  zone_id: string
  zone_type: 'include' | 'exclude'
  vertices: ZoneVertex[]
  name?: string
  priority?: string
  [key: string]: unknown
}

export interface SceneGuardStatus {
  enabled: boolean
  usable: boolean
  state: string
  raw_state?: string | null
  reason?: string | null
  profile_id?: string | null
  bound_model_version?: string | null
  bound_model_artifact_sha256?: string | null
  runtime_model_artifact_sha256?: string | null
  reference_available?: boolean
  policy_recovery_count?: number
  required_recovery_count?: number | null
  last_checked_frame?: number | null
  [key: string]: unknown
}

export interface CameraSourceDiagnostics {
  declared_source?: string | null
  runtime_source?: string | null
  go2rtc_stream?: string | null
  fps_observed?: number | null
  last_frame_index?: number | null
  last_snapshot_frame_index?: number | null
  last_snapshot_ts?: number | null
  last_preview_stream_frame_index?: number | null
  source_consistency?: 'direct' | 'go2rtc_redirect' | 'mismatch' | string | null
  [key: string]: unknown
}

export interface CameraDetectorStatus {
  low_light?: boolean
  last_brightness?: number | null
  ssim_calibration_blocked?: boolean
  exposure_recovering?: boolean
  detection_limited?: boolean
  detection_limited_reason?: string | null
  [key: string]: unknown
}

export interface CameraRuntimeDetail {
  pipeline_mode?: PipelineMode | null
  anomaly_locked?: boolean
  scene_guard?: SceneGuardStatus | null
  [key: string]: unknown
}

export interface CameraLifecycleStage {
  name?: string
  status: 'active' | 'completed' | 'pending' | 'failed' | string
  [key: string]: unknown
}

export interface CameraDetailPayload {
  camera_id: string
  name: string
  connected: boolean
  running: boolean
  model_backend?: string | null
  expected_model_backend?: string | null
  stats?: CameraStats | null
  stages?: CameraLifecycleStage[] | null
  zones?: CameraZone[]
  config?: Record<string, unknown> | null
  runtime?: CameraRuntimeDetail | null
  runner?: Record<string, unknown> | null
  detector?: CameraDetectorStatus | null
  health?: Record<string, unknown> | null
  candidate_diagnostics?: unknown[]
  source_diagnostics?: CameraSourceDiagnostics | null
  scene_guard?: SceneGuardStatus | null
  [key: string]: unknown
}

export interface AlertEvidenceJson {
  detection_type?: string | null
  frame_number?: number | null
  [key: string]: unknown
}

export interface CandidateEvidenceEvent {
  candidate_id?: string | null
  candidate_type: string
  frame_index?: number | null
  decision: string
  suppression_reason?: string | null
  [key: string]: unknown
}

export interface AlertSummary {
  alert_id: string
  timestamp: string
  camera_id: string
  zone_id: string
  severity: 'low' | 'medium' | 'high' | 'info'
  anomaly_score: number
  detection_type?: string | null
  evidence_summary?: string | null
  evidence_json?: AlertEvidenceJson | null
  suppression_reason?: string | null
  frame_number?: number | null
  acknowledged: boolean
  false_positive: boolean
  has_recording: boolean
  replay_ready?: boolean
  recording_status: string | null
  recording_unavailable_reason?: string | null
  workflow_status: string
  notes: string
  // Open-vocabulary classifier output (null when classifier is disabled
  // or when the region's score fell below the classify threshold).
  classification_label?: string | null
  classification_confidence?: number | null
  // True when the classifier's label promoted severity (e.g. a "gun"
  // detected in a low-severity anomaly bumps it to high).
  severity_adjusted_by_classifier?: boolean
  // Cross-camera corroboration (stage 2.7). ``corroborated`` is true when
  // another camera with overlapping FoV confirmed the anomaly at the
  // transformed location, false when the partner camera rejected it, and
  // null when cross-camera correlation was disabled for this pipeline.
  corroborated?: boolean | null
  correlation_partner?: string | null
  // SAM2 instance segmentation results (stage 2.4). `segmentation_objects`
  // is a list of per-object dicts (bbox / area_px / centroid / confidence)
  // — the raw masks are never persisted. All three fields are null when the
  // segmenter is disabled or returned an empty result.
  segmentation_count?: number | null
  segmentation_total_area_px?: number | null
  segmentation_objects?: SegmentationObject[] | null
  // Event grouping
  event_group_id?: string | null
  event_group_count?: number | null
  // Image paths
  snapshot_path?: string | null
  heatmap_path?: string | null
  // Physics enrichment
  speed_ms?: number | null
  speed_px_per_sec?: number | null
  trajectory_model?: string | null
  origin_x_mm?: number | null
  origin_y_mm?: number | null
  origin_z_mm?: number | null
  landing_x_mm?: number | null
  landing_y_mm?: number | null
  landing_z_mm?: number | null
  // Per-track fitted trajectories (multi-target rendering)
  trajectories?: TrajectoryFit[] | null
  // Raw centroid history (flat list for simple overlays)
  trajectory_points?: Array<{t: number, x: number, y: number}> | null
  // Alert category classification
  category?: string | null
  // Workflow
  assigned_to?: string | null
  resolved_at?: string | null
  model_version_id?: string | null
  candidate_evidence?: {
    event_count?: number
    trigger_frame_index?: number | null
    events?: CandidateEvidenceEvent[]
    trigger_window?: number
    truncated?: boolean
  } | null
}

export interface SegmentationObject {
  bbox: [number, number, number, number]  // x, y, w, h
  area_px: number
  centroid: [number, number]
  confidence: number
}

export interface TrajectoryFitPoint {
  x_mm: number
  y_mm: number
  z_mm: number
  x_px: number | null
  y_px: number | null
}

export interface TrajectoryFit {
  track_id: number
  is_primary: boolean
  model_type: string
  r_squared: number
  speed_ms: number | null
  speed_px_per_sec: number | null
  origin: TrajectoryFitPoint
  landing: TrajectoryFitPoint
}

export interface ModelInputQualityStatus {
  low_light_enabled?: boolean
  low_light?: boolean
  low_light_threshold?: number | null
  last_brightness?: number | null
  exposure_recovering?: boolean
  exposure_recovery_remaining_seconds?: number
  detection_limited?: boolean
  detection_limited_reason?: string | null
  fast_motion_suppressed?: boolean
  ssim_calibration_blocked?: boolean
  ssim_calibration_blocked_reason?: string | null
}

// Live health record for one detector on one camera
// (aggregated at /api/models/status). backend values depend on the detector:
//   anomaly: "torch-cuda" | "torch-cpu" | "openvino" | "ssim-fallback" | "disabled" | "none"
//   yolo:    "cuda" | "cpu" | "disabled" | "none"
export interface ModelHealthStatus {
  name: string
  camera_id: string
  loaded: boolean
  backend: string
  model_path: string | null
  image_size: [number, number] | null
  last_error: string | null
  last_error_ts: number | null
  consecutive_failures: number
  total_inferences: number
  total_failures: number
  last_success_ts: number | null
  extra: Record<string, unknown> & {
    input_quality?: ModelInputQualityStatus
    expected_backend?: string
  }
}

// Persisted operator-marked clip range on the replay timeline (FR-033).
export interface ReplayClip {
  start_index: number
  end_index: number
  label: string
  created_at: string
}

export interface TaskInfo {
  task_id: string
  task_type: string
  camera_id: string | null
  status: 'pending' | 'running' | 'complete' | 'failed' | 'paused' | 'aborted'
  progress: number
  message: string
  error: string | null
  result: Record<string, unknown> | null
}

export interface HealthInfo {
  status: 'healthy' | 'degraded' | 'unhealthy'
  uptime_seconds: number
  total_alerts: number
  cameras: HealthCamera[]
  platform: string
  python_version: string
}

export interface HealthCamera {
  camera_id: string
  connected: boolean
  frames_captured: number
  avg_latency_ms: number
}

export type UserRole = 'admin' | 'engineer' | 'operator' | 'viewer'

export interface UserInfo {
  username: string
  role: UserRole
  display_name: string
  active: boolean
  last_login: string | null
  created_at: string | null
}

export interface AuditEntry {
  id: number
  timestamp: string | null
  user: string
  action: string
  target_type: string
  target_id: string
  details: string
  ip_address: string
}

// ── Baseline Types ──

export interface BaselineInfo {
  camera_id: string
  version: string
  image_count: number
  session_label: string
  status: string
  state: string | null
}

export interface BaselineVersionInfo {
  id: number
  camera_id: string
  zone_id: string
  version: string
  state: string
  image_count: number
  verified_by: string | null
  verified_at: string | null
  verified_by_secondary: string | null
  activated_at: string | null
  retired_at: string | null
  retirement_reason: string | null
  group_id: string | null
  created_at: string | null
}

// ── Training Types ──

export interface TrainingRecord {
  id: number
  camera_id: string
  zone_id: string
  model_type: string
  export_format: string | null
  baseline_version: string
  baseline_count: number
  train_count: number
  val_count: number
  pre_validation_passed: boolean
  corruption_rate: number | null
  near_duplicate_rate: number | null
  brightness_std: number | null
  val_score_mean: number | null
  val_score_std: number | null
  val_score_max: number | null
  val_score_p95: number | null
  quality_grade: string | null
  threshold_recommended: number | null
  // Phase 1: real-labeled P/R/F1/AUROC/PR-AUC
  val_precision: number | null
  val_recall: number | null
  val_f1: number | null
  val_auroc: number | null
  val_pr_auc: number | null
  val_confusion_matrix: string | null  // JSON-encoded {tp,fp,fn,tn}
  val_real_sample_count: number | null
  model_path: string | null
  export_path: string | null
  checkpoint_valid: boolean | null
  export_valid: boolean | null
  smoke_test_passed: boolean | null
  inference_latency_ms: number | null
  status: string
  error: string | null
  duration_seconds: number
  trained_at: string | null
  created_at: string | null
}

// Phase 2: per-record metrics payload for MetricsChart
export interface TrainingMetricsResponse {
  record_id: number
  has_labeled_eval: boolean
  message?: string
  scores?: number[]
  labels?: number[]
  threshold_used?: number
  optimal_f1_threshold?: number
  sample_count?: number
  metrics_at_threshold?: {
    precision: number
    recall: number
    f1: number
    auroc: number
    pr_auc: number
    confusion_matrix: { tp: number; fp: number; fn: number; tn: number }
    threshold: number
    n_positive: number
    n_negative: number
  }
  pr_curve?: {
    precisions: number[]
    recalls: number[]
    thresholds: number[]
  }
  stored_confusion_matrix?: { tp: number; fp: number; fn: number; tn: number } | null
  stored_metrics?: {
    precision: number | null
    recall: number | null
    f1: number | null
    auroc: number | null
    pr_auc: number | null
  }
}

export interface TrainingValidationReport {
  all_passed?: boolean
  auroc?: number | null
  recall?: number | null
  replay?: {
    new_detection_rate?: number | null
    [key: string]: unknown
  } | null
  [key: string]: unknown
}

export interface TrainingJobInfo {
  id: number
  job_id: string
  job_type: string
  camera_id: string | null
  zone_id: string
  model_type: string | null
  trigger_type: string | null
  triggered_by: string | null
  confirmation_required: boolean
  confirmed_by: string | null
  confirmed_at: string | null
  status: string
  base_model_version: string | null
  dataset_version: string | null
  hyperparameters: string | Record<string, unknown> | null
  metrics: string | Record<string, unknown> | null
  artifacts_path: string | null
  validation_report: string | TrainingValidationReport | null
  model_version_id: string | null
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  duration_seconds: number | null
  error: string | null
}

// ── Model Types ──

export type ModelRuntimeState = 'applied' | 'waiting' | 'failed'

export interface ModelInfo {
  id: number
  model_version_id: string
  camera_id: string
  model_type: string
  model_hash: string
  data_hash: string
  code_version: string | null
  training_params: string | null
  calibration_thresholds: string | null
  backbone_version_id: string | null
  created_at: string | null
  is_active: boolean
  stage: string
  component_type: string
  model_path: string | null
  canary_camera_id: string | null
  release_revision?: number | null
  runtime_state?: ModelRuntimeState | null
  latest_release_operation?: ReleaseOperation | null
}

export interface ModelVersionEvent {
  id: number
  timestamp: string | null
  camera_id: string
  from_version: string | null
  to_version: string
  from_stage: string | null
  to_stage: string
  triggered_by: string
  reason: string | null
  warmup_latency_ms: number | null
  sha256_verified: boolean
}

// ── Degradation Types ──

export interface DegradationEvent {
  event_id: string
  level: string
  category: string
  camera_id: string | null
  title: string
  impact: string
  action: string
  started_at: number
  resolved_at: number | null
}

// ── Response Payload Shapes (used as T in ApiResponse<T>) ──

export interface CamerasPayload { cameras: CameraSummary[] }
export interface AlertQueryParams {
  camera_id?: string
  severity?: string
  limit?: number
  offset?: number
}

export interface AlertsPayload { alerts: AlertSummary[]; total: number }
export interface TasksPayload { tasks: TaskInfo[] }
export interface HealthPayload extends HealthInfo {}
export interface UsersPayload { users: UserInfo[] }
export interface AuditPayload { entries: AuditEntry[] }

export interface ReportEvidenceStats {
  total_alerts: number
  alerts_with_snapshot: number
  alerts_with_heatmap: number
  alerts_with_recording: number
  evidence_complete_count: number
  snapshot_rate: number
  heatmap_rate: number
  recording_rate: number
  evidence_complete_rate: number
}

export interface ReportStats {
  total_alerts: number
  by_severity: Record<'high' | 'medium' | 'low' | 'info', number>
  false_positive_count: number
  false_positive_rate: number
  acknowledged_count: number
  acknowledged_rate: number
  evidence: ReportEvidenceStats
}

// ── Model API Response Shapes ──

export interface ModelRegistryResponse {
  models: ModelInfo[]
  total: number
}

export interface ModelRuntimeMutationResponse {
  activated: string
  camera_id: string
  runtime_synced: boolean
  runtime_state?: ModelRuntimeState | null
  model?: ModelInfo
}

export type ModelReleaseRuntimeState = ModelRuntimeState | Record<string, unknown> | null

export type ReleaseOperationStatus = 'pending' | 'applied' | 'failed'

export interface ReleaseOperation {
  operation_id: string
  idempotency_key?: string | null
  model_version_id: string
  camera_id: string
  from_stage: string
  target_stage: string
  expected_revision: number
  applied_revision: number | null
  status: ReleaseOperationStatus
  requested_by: string
  reason?: string | null
  runtime_state?: ModelReleaseRuntimeState
  error?: string | null
  requested_at?: string | null
  completed_at?: string | null
}

export interface ModelReleaseMutationResponse {
  operation: ReleaseOperation
  model: ModelInfo
  runtime_synced: boolean
  runtime_state?: ModelReleaseRuntimeState
}

export interface ReleaseOperationResponse {
  operation: ReleaseOperation
}

/** OpenAPI-generated server-authoritative production release state. */
export type ReleaseReadinessGate = GeneratedReleaseReadinessGate
export type ReleaseReadiness = GeneratedReleaseReadinessResponse
export type ReleaseReadinessStatus = ReleaseReadiness['status']

export interface TrainingJobsResponse {
  jobs: TrainingJobInfo[]
  total: number
  pending_count: number
}

export interface ShadowReportResponse {
  total_samples: number
  production_alert_rate: number
  shadow_alert_rate: number
  false_positive_delta: number
  alert_rate_delta?: number
  abs_alert_rate_delta?: number
  avg_score_divergence: number
  avg_shadow_latency_ms: number
  gate_report?: {
    ready_for_canary: boolean
    checks: Array<{
      name: string
      label: string
      value: number
      limit: number
      passed: boolean
      enabled: boolean
    }>
  }
}

export interface ABScoresResponse {
  scores: Array<{ t: string; production: number | null; shadow: number; shadow_alert?: boolean; prod_alert?: boolean }>
  total: number
  shadow_version_id: string
}

export interface ABDistributionResponse {
  bin_edges: number[]
  production_counts: number[]
  shadow_counts: number[]
  total_samples: number
}

export interface ABLiveCompareResponse {
  production_score?: number
  production_heatmap?: string | null
  production_latency_ms?: number
  production_error?: string
  shadow_score?: number
  shadow_heatmap?: string | null
  shadow_latency_ms?: number
  shadow_error?: string
  original_frame?: string
}

export interface BatchInferenceResponse {
  results: Array<{
    path: string
    score: number
    is_anomalous: boolean
    error?: string
  }>
  scored: number
  total: number
}

// ── Multi-camera Storyboard (synchronous replay across correlated cameras) ──

/** One camera entry in a storyboard — the frontend needs `trigger_offset_s`
 *  to align each camera's local time against the shared master clock. */
export interface StoryboardCamera {
  alert_id: string
  camera_id: string
  /** Epoch seconds — only kept for UI debug/labelling, not used for sync. */
  trigger_timestamp: number
  /** Seconds relative to the primary alert's trigger_timestamp.
   *  Primary camera is 0; other cameras are positive/negative deltas. */
  trigger_offset_s: number
  metadata_url: string
  video_url: string
  signals_url: string
}

export interface StoryboardResponse {
  primary_alert_id: string
  cameras: StoryboardCamera[]
  count: number
}
