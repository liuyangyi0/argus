// ── Unified API Response Envelope: {code, msg, data} ──

export interface ApiResponse<T = any> {
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
  CONFLICT: 40900,
  INTERNAL_ERROR: 50000,
  SERVICE_UNAVAILABLE: 50300,
} as const
export type ErrorCode = (typeof ErrorCode)[keyof typeof ErrorCode]

// ── ApiError class for structured error handling ──

export class ApiError extends Error {
  code: number
  data?: unknown

  constructor(code: number, msg: string, data?: unknown) {
    super(msg)
    this.name = 'ApiError'
    this.code = code
    this.data = data
  }
}

// ── Domain Types ──

export interface CameraSummary {
  camera_id: string
  name: string
  connected: boolean
  running: boolean
  stats: CameraStats | null
}

export interface CameraStats {
  frames_captured: number
  frames_analyzed: number
  anomalies_detected: number
  alerts_emitted: number
  avg_latency_ms: number
}

export interface AlertSummary {
  alert_id: string
  timestamp: string
  camera_id: string
  zone_id: string
  severity: 'low' | 'medium' | 'high' | 'info'
  anomaly_score: number
  acknowledged: boolean
  false_positive: boolean
  has_recording: boolean
  recording_status: string | null
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
  extra: Record<string, unknown>
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

export interface UserInfo {
  username: string
  role: 'admin' | 'operator' | 'viewer'
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
  validation_report: string | Record<string, any> | null
  model_version_id: string | null
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  duration_seconds: number | null
  error: string | null
}

// ── Model Types ──

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
export interface AlertsPayload { alerts: AlertSummary[]; total?: number }
export interface TasksPayload { tasks: TaskInfo[] }
export interface HealthPayload extends HealthInfo {}
export interface UsersPayload { users: UserInfo[] }
export interface AuditPayload { entries: AuditEntry[] }

// ── Model API Response Shapes ──

export interface ModelRegistryResponse {
  models: ModelInfo[]
  total: number
}

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
  avg_score_divergence: number
  avg_shadow_latency_ms: number
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
