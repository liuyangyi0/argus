import type { ApiResponse, HealthPayload } from '../types/api'
import { api, unwrap, u } from './client'

// ── Health ──
export const getHealth = () =>
  api.get<ApiResponse<HealthPayload>>('/system/health').then(unwrap)

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

// ── Config management ──
export const saveConfig = () => api.post('/config/save').then(u)
export const updateThresholds = (data: any) => api.post('/config/thresholds', data).then(u)
export const updateDetectionParams = (data: any) => api.post('/config/detection-params', data).then(u)
export const reloadModel = (data: { camera_id: string; model_path: string }) =>
  api.post('/config/reload-model', data).then(u)
export const clearLock = (cameraId: string) => api.post(`/config/clear-lock/${cameraId}`).then(u)
export const restartCamera = (cameraId: string) =>
  api.post(`/config/camera/${cameraId}/restart`).then(u)

// ── Webhook / Notifications ──
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
