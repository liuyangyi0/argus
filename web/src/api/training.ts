import { api, u, unwrap } from './client'
import type {
  ApiResponse,
  DatasetSelection,
  TrainingJobInfo,
  TrainingJobsResponse,
  TrainingMetricsResponse,
  TrainingRecord,
} from '../types/api'

type QueryParams = Record<string, string | number | boolean | undefined>

export interface BackboneInfo {
  backbone_version_id: string
  backbone_type: string
  is_active: boolean
  created_at: string | null
  [key: string]: unknown
}

export type TrainingJobsPayload = Omit<TrainingJobsResponse, 'total'> & {
  total?: number
}

// ── Training ──
export const getTrainingHistory = (params?: QueryParams) =>
  api
    .get<ApiResponse<{ records: TrainingRecord[] }>>('/baseline/training-history/json', { params })
    .then(unwrap)
export const compareModels = (data: { old_record_id: number; new_record_id: number }) =>
  api.post('/baseline/compare', data, { timeout: 120000 }).then(u)
// Phase 2: real-labeled metrics for a single training record
export const getTrainingMetrics = (recordId: number): Promise<TrainingMetricsResponse> =>
  api
    .get<ApiResponse<TrainingMetricsResponse>>(`/baseline/training-history/${recordId}/metrics`)
    .then(unwrap)

// ── Training Jobs ──
export const getTrainingJobs = (params?: QueryParams) =>
  api
    .get<ApiResponse<TrainingJobsPayload>>('/training-jobs/json', { params })
    .then(unwrap)
export const getTrainingJob = (jobId: string) =>
  api
    .get<ApiResponse<TrainingJobInfo>>(`/training-jobs/${jobId}`)
    .then(unwrap)
export const createTrainingJob = (data: {
  job_type: string; camera_id?: string; zone_id?: string;
  model_type?: string; trigger_type?: string; triggered_by?: string;
  hyperparameters?: Record<string, unknown>; dataset_selection?: DatasetSelection;
}) => api.post('/training-jobs/', data).then(u)
export const confirmTrainingJob = (jobId: string, data?: { confirmed_by?: string }) =>
  api.post(`/training-jobs/${jobId}/confirm`, data || {}).then(u)
export const rejectTrainingJob = (jobId: string, data?: { rejected_by?: string; reason?: string }) =>
  api.post(`/training-jobs/${jobId}/reject`, data || {}).then(u)
export const getBackbones = () =>
  api
    .get<ApiResponse<{ backbones: BackboneInfo[] }>>('/training-jobs/backbones/json')
    .then(unwrap)
