import { api, u } from './client'

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

export const deleteZone = (cameraId: string, zoneId: string) =>
  api.delete(`/zones/${cameraId}/${zoneId}`).then(u)

export const getZoneSnapshotUrl = (cameraId: string) =>
  `/api/zones/snapshot/${cameraId}`
