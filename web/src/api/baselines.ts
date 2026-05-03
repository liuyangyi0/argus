import type { ApiResponse, BaselineCollection } from '../types/api'
import { api, u, unwrap } from './client'

// ── Baselines ──
export const getBaselines = () => api.get('/baseline/list/json').then(u)

// 痛点 5: full collections inventory (incl. failed_* dirs and acceptance_rate)
export const getBaselineCollections = () =>
  api
    .get<ApiResponse<{ collections: BaselineCollection[] }>>('/baseline/collections/json')
    .then(unwrap)
export const activateCollection = (
  camera_id: string, zone_id: string, version: string,
) =>
  api.post(`/baseline/collections/${encodeURIComponent(camera_id)}/${encodeURIComponent(zone_id)}/${encodeURIComponent(version)}/activate`).then(u)
export const retireCollection = (
  camera_id: string, zone_id: string, version: string,
) =>
  api.post(`/baseline/collections/${encodeURIComponent(camera_id)}/${encodeURIComponent(zone_id)}/${encodeURIComponent(version)}/retire`).then(u)
export const deleteCollection = (
  camera_id: string, zone_id: string, version: string, opts?: { confirm?: boolean },
) =>
  api.delete(
    `/baseline/collections/${encodeURIComponent(camera_id)}/${encodeURIComponent(zone_id)}/${encodeURIComponent(version)}`,
    { params: opts?.confirm ? { confirm: true } : undefined },
  ).then(u)
export const optimizeBaseline = (data: { camera_id: string; zone_id?: string; target_ratio?: number }) =>
  api.post('/baseline/optimize/json', data).then(u)
export const previewOptimize = (params: { camera_id: string; zone_id?: string; target_ratio?: number }) =>
  api.get('/baseline/optimize/preview', { params }).then(u)

// ── Capture Jobs ──
export const startCaptureJob = (data: FormData) => api.post('/baseline/job', data).then(u)
/** @deprecated Use startCaptureJob instead */
export const startCapture = startCaptureJob
export const pauseCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/pause`).then(u)
export const resumeCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/resume`).then(u)
export const abortCaptureJob = (taskId: string) => api.post(`/baseline/job/${taskId}/abort`).then(u)

// ── False Positive Merge ──
export const mergeFalsePositives = (data: { camera_id: string; zone_id?: string; max_fp_images?: number }) =>
  api.post('/baseline/merge-fp', data).then(u)

// ── Camera Groups ──
export const getCameraGroups = () => api.get('/baseline/groups/json').then(u)
export const mergeGroupBaseline = (data: { group_id: string; zone_id?: string; target_count?: number }) =>
  api.post('/baseline/groups/merge', data).then(u)

// ── Baseline Lifecycle ──
export const getBaselineVersions = (params: { camera_id: string; zone_id?: string }) =>
  api.get('/baseline/versions/json', { params }).then(u)
export const verifyBaseline = (data: { camera_id: string; zone_id?: string; version: string; verified_by: string; verified_by_secondary?: string }) =>
  api.post('/baseline/verify', data).then(u)
export const activateBaseline = (data: { camera_id: string; zone_id?: string; version: string; user?: string }) =>
  api.post('/baseline/activate-baseline', data).then(u)
export const retireBaseline = (data: { camera_id: string; zone_id?: string; version: string; user?: string; reason?: string }) =>
  api.post('/baseline/retire', data).then(u)
export const deleteBaselineVersion = (data: { camera_id: string; zone_id?: string; version: string; user?: string }) =>
  api.post('/baseline/version/delete', data).then(u)

// ── Baseline Version Images (per-image CRUD) ──
export interface BaselineImageInfo {
  filename: string
  size_bytes: number
  created_at: string
}

export interface BaselineImageListResponse {
  camera_id: string
  zone_id: string
  version: string
  state: string | null
  is_active: boolean
  total: number
  total_bytes: number
  images: BaselineImageInfo[]
}

export const listBaselineImages = (
  cameraId: string,
  version: string,
  zoneId: string = 'default',
): Promise<BaselineImageListResponse> =>
  api.get(`/baseline/${encodeURIComponent(cameraId)}/${encodeURIComponent(version)}/images`, {
    params: { zone_id: zoneId },
  }).then(u)

export const deleteBaselineImage = (
  cameraId: string,
  version: string,
  filename: string,
  zoneId: string = 'default',
): Promise<{ filename: string }> =>
  api.delete(
    `/baseline/${encodeURIComponent(cameraId)}/${encodeURIComponent(version)}/images/${encodeURIComponent(filename)}`,
    { params: { zone_id: zoneId } },
  ).then(u)

export const uploadBaselineImage = (
  cameraId: string,
  version: string,
  file: File,
  zoneId: string = 'default',
): Promise<{ filename: string; size_bytes: number }> => {
  const form = new FormData()
  form.append('file', file)
  return api.post(
    `/baseline/${encodeURIComponent(cameraId)}/${encodeURIComponent(version)}/images`,
    form,
    { params: { zone_id: zoneId }, headers: { 'Content-Type': 'multipart/form-data' } },
  ).then(u)
}

/**
 * Compose the static URL used for thumbnail <img src> — the server serves the
 * raw image bytes here with a short cache-control.
 */
export const baselineImageContentUrl = (
  cameraId: string,
  version: string,
  filename: string,
  zoneId: string = 'default',
): string => `/api/baseline/${encodeURIComponent(cameraId)}/${encodeURIComponent(version)}/images/${encodeURIComponent(filename)}/content?zone_id=${encodeURIComponent(zoneId)}`
