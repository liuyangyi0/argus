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
}

export interface TaskInfo {
  task_id: string
  task_type: string
  camera_id: string | null
  status: 'pending' | 'running' | 'complete' | 'failed'
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

// ── Response Payload Shapes (used as T in ApiResponse<T>) ──

export interface CamerasPayload { cameras: CameraSummary[] }
export interface AlertsPayload { alerts: AlertSummary[]; total?: number }
export interface TasksPayload { tasks: TaskInfo[] }
export interface HealthPayload extends HealthInfo {}
export interface UsersPayload { users: UserInfo[] }
export interface AuditPayload { entries: AuditEntry[] }
