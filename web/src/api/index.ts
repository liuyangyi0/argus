import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.error || err.message || '请求失败'
    console.error('[API]', msg)
    return Promise.reject(err)
  }
)

// ── Cameras ──
export const getCameras = () => api.get('/cameras/json')
export const startCamera = (id: string) => api.post(`/cameras/${id}/start`)
export const stopCamera = (id: string) => api.post(`/cameras/${id}/stop`)

// ── Alerts ──
export const getAlerts = (params?: Record<string, any>) => api.get('/alerts/json', { params })
export const getAlert = (id: string) => api.get(`/alerts/${id}/detail`)
export const acknowledgeAlert = (id: string) => api.post(`/alerts/${id}/acknowledge`)
export const markFalsePositive = (id: string) => api.post(`/alerts/${id}/false-positive`)
export const updateAlertWorkflow = (id: string, data: Record<string, any>) =>
  api.post(`/alerts/${id}/workflow`, data)
export const bulkAcknowledge = (ids: string[]) =>
  api.post('/alerts/bulk-acknowledge', { alert_ids: ids })
export const bulkFalsePositive = (ids: string[]) =>
  api.post('/alerts/bulk-false-positive', { alert_ids: ids })

// ── System ──
export const getHealth = () => api.get('/system/health')

// ── Training ──
export const startTraining = (data: Record<string, any>) => api.post('/baseline/train', data)
export const deployModel = (data: Record<string, any>) => api.post('/baseline/deploy', data)

// ── Config ──
export const reloadConfig = () => api.post('/config/reload')

export default api
