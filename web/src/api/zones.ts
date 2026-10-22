import type { ApiResponse, CameraZone } from '../types/api'
import { api, unwrap, u } from './client'

export interface ZoneCreateRequest {
  camera_id: string
  zone_id: string
  name: string
  polygon: [number, number][]
  zone_type?: 'include' | 'exclude'
  priority?: 'standard' | 'critical' | 'low_priority'
  anomaly_threshold?: number
}

export const createZone = (data: ZoneCreateRequest) =>
  api.post('/zones', data).then(u)

export interface ZoneUpdateResult {
  camera_id?: string
  count?: number
  zones?: CameraZone[]
}

export const updateZones = (cameraId: string, zones: CameraZone[]) =>
  api.put<ApiResponse<ZoneUpdateResult>>(`/zones/${cameraId}`, zones).then(unwrap)

export const deleteZone = (cameraId: string, zoneId: string) =>
  api.delete(`/zones/${cameraId}/${zoneId}`).then(u)

export const getZoneSnapshotUrl = (cameraId: string) =>
  `/api/zones/snapshot/${cameraId}`
