import { api, u } from './client'

// ── Labeling Queue ──
export const getLabelingQueue = (params?: {
  camera_id?: string
  status?: string
  limit?: number
  offset?: number
}) => api.get('/labeling/queue', { params }).then(u)

export const labelEntry = (entryId: number, data: { label: 'normal' | 'anomaly'; labeled_by?: string }) =>
  api.post(`/labeling/${entryId}/label`, data).then(u)

export const skipEntry = (entryId: number) =>
  api.post(`/labeling/${entryId}/skip`).then(u)

export const getLabelingStats = (cameraId?: string) =>
  api.get('/labeling/stats', { params: cameraId ? { camera_id: cameraId } : {} }).then(u)

export const triggerRetrain = (data?: { camera_id?: string; model_type?: string; triggered_by?: string }) =>
  api.post('/labeling/trigger-retrain', data || {}, { timeout: 60000 }).then(u)

export const getLabelingImage = (entryId: number) =>
  `/api/labeling/${entryId}/image`
