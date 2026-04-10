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
