import type { ApiResponse, AlertsPayload } from '../types/api'
import { api, unwrap, u } from './client'

export const getAlerts = (params?: Record<string, any>) =>
  api.get<ApiResponse<AlertsPayload>>('/alerts/json', { params }).then(unwrap)
export const getAlert = (id: string) => api.get(`/alerts/${id}/detail`).then(u)
export const acknowledgeAlert = (id: string) => api.post(`/alerts/${id}/acknowledge`).then(u)
export const markFalsePositive = (id: string) => api.post(`/alerts/${id}/false-positive`).then(u)
export const updateAlertWorkflow = (id: string, data: Record<string, any>) =>
  api.post(`/alerts/${id}/workflow`, data).then(u)
export const bulkAcknowledge = (ids: string[]) =>
  api.post('/alerts/bulk-acknowledge', { alert_ids: ids }).then(u)
export const bulkFalsePositive = (ids: string[]) =>
  api.post('/alerts/bulk-false-positive', { alert_ids: ids }).then(u)
export const deleteAlert = (id: string) => api.delete(`/alerts/${id}`).then(u)
export const bulkDeleteAlerts = (ids: string[]) =>
  api.post('/alerts/bulk-delete', { alert_ids: ids }).then(u)
