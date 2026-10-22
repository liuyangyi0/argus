import type { ApiResponse } from '../types/api'
import { api, u, unwrap } from './client'

export interface LabelingEntry {
  id: number
  camera_id: string
  frame_path: string
  anomaly_score: number
  entropy: number
  status: string
  [key: string]: unknown
}

export interface LabelingStats {
  total: number
  pending: number
  labeled: number
  skipped: number
  by_label: Record<string, number>
  [key: string]: unknown
}

// ── Labeling Queue ──
export const getLabelingQueue = (params?: {
  camera_id?: string
  status?: string
  limit?: number
  offset?: number
}) => api
  .get<ApiResponse<{ entries: LabelingEntry[] }>>('/labeling/queue', { params })
  .then(unwrap)

export const labelEntry = (entryId: number, data: { label: 'normal' | 'anomaly'; labeled_by?: string }) =>
  api.post(`/labeling/${entryId}/label`, data).then(u)

export const skipEntry = (entryId: number) =>
  api.post(`/labeling/${entryId}/skip`).then(u)

export const getLabelingStats = (cameraId?: string) =>
  api
    .get<ApiResponse<LabelingStats>>('/labeling/stats', {
      params: cameraId ? { camera_id: cameraId } : {},
    })
    .then(unwrap)

export const triggerRetrain = (data?: { camera_id?: string; model_type?: string; triggered_by?: string }) =>
  api
    .post<ApiResponse<{ job_id: string }>>(
      '/labeling/trigger-retrain',
      data || {},
      { timeout: 60000 },
    )
    .then(unwrap)

export const getLabelingImage = (entryId: number) =>
  `/api/labeling/${entryId}/image`
