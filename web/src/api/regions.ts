import type { NotificationTemplateMethod } from './system'
import { api, u } from './client'

export interface RegionNotificationTemplateItem {
  id: number
  name: string
  method: NotificationTemplateMethod
  enabled: boolean
}

export interface RegionItem {
  id: number
  name: string
  owner: string
  email: string
  phone: string
  notification_methods: string[]
  notification_template_ids: number[]
  notification_templates: RegionNotificationTemplateItem[]
  notification_methods_text?: string
  created_at?: string | null
  updated_at?: string | null
}

export interface RegionQueryParams {
  name?: string
  owner?: string
  phone?: string
  email?: string
}

export interface RegionPayload {
  name: string
  owner: string
  email?: string
  phone?: string
  notification_methods: string[]
  notification_template_ids: number[]
}

export const getRegions = (params?: RegionQueryParams) =>
  api.get('/regions/json', { params }).then(u) as Promise<{ regions: RegionItem[] }>

export const createRegion = (data: RegionPayload) =>
  api.post('/regions/json', data).then(u) as Promise<{ region: RegionItem }>

export const updateRegion = (regionId: number, data: RegionPayload) =>
  api.put(`/regions/${regionId}/json`, data).then(u) as Promise<{ region: RegionItem }>

export const deleteRegion = (regionId: number) =>
  api.delete(`/regions/${regionId}/json`).then(u) as Promise<{ id: number }>
