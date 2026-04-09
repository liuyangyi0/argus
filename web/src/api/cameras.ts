import type { ApiResponse, CamerasPayload } from '../types/api'
import { api, unwrap, u } from './client'

// ── Cameras ──
export const getCameras = () =>
  api.get<ApiResponse<CamerasPayload>>('/cameras/json').then(unwrap)
export const getCameraDetail = (id: string) =>
  api.get(`/cameras/${id}/detail/json`).then(u)
export const startCamera = (id: string) => api.post(`/cameras/${id}/start`).then(u)
export const stopCamera = (id: string) => api.post(`/cameras/${id}/stop`).then(u)
export const getUsbDevices = () => api.get('/cameras/usb-devices').then(u)
export const addCamera = (data: FormData) => api.post('/cameras', data).then(u)
export const deleteCamera = (id: string) => api.delete(`/cameras/${id}`).then(u)

// ── Streaming (go2rtc) ──
export const getStreamInfo = (id: string) => api.get(`/streaming/${id}`).then(u)
export const getStreams = () => api.get('/streaming').then(u)
export const registerStream = (id: string) => api.post(`/streaming/${id}/register`).then(u)

// ── Video Wall ──
export const getWallStatus = () =>
  api.get<ApiResponse<{ cameras: any[] }>>('/cameras/wall/status').then(unwrap)
