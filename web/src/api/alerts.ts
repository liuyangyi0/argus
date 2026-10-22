import type {
  AlertQueryParams,
  AlertSummary,
  ApiResponse,
  AlertsPayload,
  FeedbackStats,
  ModelInfo,
  ModelRegistryResponse,
} from '../types/api'
import type { AlertBulkMutationResponse } from '../types/generated-openapi'
import { api, unwrap, u } from './client'

export interface AlertWorkflowUpdate {
  status: string
  notes?: string
  category?: string
  assigned_to?: string
  confirmed?: boolean
  from_detail_view?: boolean
}

export interface AlertAnnotation {
  x: number
  y: number
  width: number
  height: number
  label: string
}

export const getAlerts = (params?: AlertQueryParams) =>
  api.get<ApiResponse<AlertsPayload>>('/alerts/json', { params }).then(unwrap)
export const getAlert = (id: string) => api.get(`/alerts/${id}/detail`).then(u)
export const acknowledgeAlert = (id: string) => api.post(`/alerts/${id}/acknowledge`).then(u)
export const markFalsePositive = (id: string) => api.post(`/alerts/${id}/false-positive`).then(u)
// 痛点 10: explicit "this WAS a real anomaly" — feeds confirmed feedback to validation set
export const confirmRealAnomaly = (id: string) =>
  api.post(`/alerts/${id}/confirm-anomaly`).then(u)
export const getFeedbackStats = (camera_id?: string) =>
  api
    .get<ApiResponse<FeedbackStats>>('/alerts/feedback-stats', {
      params: camera_id ? { camera_id } : undefined,
    })
    .then(unwrap)
export const updateAlertWorkflow = (id: string, data: AlertWorkflowUpdate) =>
  api.post(`/alerts/${id}/workflow`, data).then(u)
export const bulkAcknowledge = (ids: string[]) =>
  api
    .post<ApiResponse<AlertBulkMutationResponse>>('/alerts/bulk-acknowledge', { alert_ids: ids })
    .then(unwrap)
export const bulkFalsePositive = (ids: string[]) =>
  api
    .post<ApiResponse<AlertBulkMutationResponse>>('/alerts/bulk-false-positive', {
      alert_ids: ids,
    })
    .then(unwrap)
export const deleteAlert = (id: string) => api.delete(`/alerts/${id}`).then(u)
export const bulkDeleteAlerts = (ids: string[]) =>
  api
    .post<ApiResponse<AlertBulkMutationResponse>>('/alerts/bulk-delete', { alert_ids: ids })
    .then(unwrap)
export const saveAnnotations = (alertId: string, annotations: AlertAnnotation[]) =>
  api.post(`/alerts/${alertId}/annotations`, { annotations }).then(u)
export const getAlertGroup = (eventGroupId: string) =>
  api
    .get<ApiResponse<{ alerts: AlertSummary[] }>>(`/alerts/group/${eventGroupId}`)
    .then(unwrap)

export async function downloadEvidencePackage(alertId: string): Promise<void> {
  const res = await api.get(`/alerts/${alertId}/evidence.zip`, {
    responseType: 'blob',
  })

  const blob: Blob = res.data
  if (blob.type.includes('application/json')) {
    const text = await blob.text()
    const body = JSON.parse(text)
    throw new Error(body.msg || body.detail || '导出证据包失败')
  }

  const disposition = res.headers['content-disposition'] || ''
  const match = disposition.match(/filename="?([^";\s]+)"?/)
  const filename = match?.[1] || `${alertId}_evidence.zip`

  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// C-AlertTrainingLink: resolve a model_version_id to its registry record so the
// alert detail panel can show camera / stage / created_at next to the link.
// Returns null when the version isn't in the registry (e.g. retired record
// pruned, model id stamped on the alert before registry was populated).
export const getModelVersionInfo = async (modelVersionId: string) => {
  if (!modelVersionId) return null
  try {
    const res = await api
      .get<ApiResponse<ModelRegistryResponse>>('/models/json')
      .then(unwrap)
    return res.models.find((model: ModelInfo) => (
      model.model_version_id === modelVersionId
    )) || null
  } catch {
    return null
  }
}
