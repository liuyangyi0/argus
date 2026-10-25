import { api, unwrap, u } from './client'
import { ApiError } from '../types/api'
import type { RollbackReadinessResponse } from '../types/generated-openapi'
import type {
  ApiResponse,
  ModelRegistryResponse, ShadowReportResponse,
  ABScoresResponse, ABDistributionResponse, ABLiveCompareResponse,
  BatchInferenceResponse, ModelRuntimeMutationResponse,
  ModelVersionEvent,
  ModelReleaseMutationResponse,
  ReleaseOperation,
  ReleaseOperationResponse,
  ReleaseReadiness,
} from '../types/api'

export interface ModelEventListResponse {
  events: ModelVersionEvent[]
}

export interface ModelRecalibrationResponse {
  n_samples: number
  sigmoid_recalibrated?: boolean
}

// ── Models ──
export const getModelRegistry = (cameraId?: string): Promise<ModelRegistryResponse> =>
  api
    .get<ApiResponse<ModelRegistryResponse>>('/models/json', {
      params: cameraId ? { camera_id: cameraId } : {},
    })
    .then(unwrap)
export const activateModel = (versionId: string): Promise<ModelRuntimeMutationResponse> =>
  api
    .post<ApiResponse<ModelRuntimeMutationResponse>>(`/models/${versionId}/activate`)
    .then(unwrap)
export const deleteModel = (versionId: string) => api.delete(`/models/${versionId}`).then(u)
export const reexportModel = (versionId: string, data: { export_format?: string; quantization?: string }) =>
  api.post(`/models/${versionId}/reexport`, data, { timeout: 300000 }).then(u)
export const recalibrateModel = (versionId: string) =>
  api
    .post<ApiResponse<ModelRecalibrationResponse>>(
      `/models/${versionId}/recalibrate`,
      {},
      { timeout: 120000 },
    )
    .then(unwrap)

// ── Release Pipeline ──
export interface ModelReleaseRequest {
  target_stage: string
  triggered_by?: string
  reason?: string
  canary_camera_id?: string
  expected_revision?: number
  idempotency_key?: string
}

export interface ModelRollbackRequest {
  source_model_version_id: string
  source_release_revision: number
  expected_revision: number
  reason: string
  idempotency_key?: string
}

export type RollbackReadiness = RollbackReadinessResponse
type ReleaseCommand = Partial<ModelReleaseRequest> & Partial<ModelRollbackRequest>

const RELEASE_KEYS_STORAGE = 'argus.release-idempotency.v1'
const releaseRequestKeys = new Map<string, string>()

function restoreReleaseKeys(): void {
  try {
    const stored: unknown = JSON.parse(sessionStorage.getItem(RELEASE_KEYS_STORAGE) || '{}')
    if (!stored || typeof stored !== 'object') return
    for (const [fingerprint, key] of Object.entries(stored)) {
      if (typeof key === 'string' && !releaseRequestKeys.has(fingerprint)) {
        releaseRequestKeys.set(fingerprint, key)
      }
    }
  } catch {
    // Storage may be disabled; in-memory keys still protect ordinary retries.
  }
}

function persistReleaseKeys(): void {
  try {
    sessionStorage.setItem(RELEASE_KEYS_STORAGE, JSON.stringify(Object.fromEntries(releaseRequestKeys)))
  } catch { /* In-memory fallback. */ }
}

function releaseFingerprint(versionId: string, action: string, data: ReleaseCommand): string {
  return JSON.stringify([
    versionId, action, data.target_stage ?? '', data.expected_revision ?? null,
    data.triggered_by ?? '', data.reason ?? '', data.canary_camera_id ?? '',
    data.source_model_version_id ?? '', data.source_release_revision ?? null,
  ])
}

function newReleaseKey(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  // getRandomValues also works on an authenticated LAN dashboard over HTTP.
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export function acknowledgeReleaseOperation(operation: ReleaseOperation): void {
  if (operation.status === 'pending' || !operation.idempotency_key) return
  restoreReleaseKeys()
  for (const [fingerprint, key] of releaseRequestKeys) {
    if (key === operation.idempotency_key) releaseRequestKeys.delete(fingerprint)
  }
  persistReleaseKeys()
}

async function submitRelease(
  versionId: string,
  action: 'promote' | 'retire' | 'rollback',
  data: ReleaseCommand,
): Promise<ModelReleaseMutationResponse> {
  restoreReleaseKeys()
  const fingerprint = releaseFingerprint(versionId, action, data)
  let key = data.idempotency_key || releaseRequestKeys.get(fingerprint)
  if (!key) {
    key = newReleaseKey()
    releaseRequestKeys.set(fingerprint, key)
    persistReleaseKeys()
  }
  try {
    const result = await api.post<ApiResponse<ModelReleaseMutationResponse>>(
      `/models/${versionId}/${action}`, { ...data, idempotency_key: key },
    ).then(unwrap)
    if (result.operation?.status !== 'pending') {
      releaseRequestKeys.delete(fingerprint)
      persistReleaseKeys()
    }
    return result
  } catch (error) {
    const operation = error instanceof ApiError && error.details && typeof error.details === 'object'
      ? (error.details as { operation?: ReleaseOperation }).operation : undefined
    // A timeout/5xx may occur after commit: retry must use the same key.
    const definitiveRejection = error instanceof ApiError
      && error.httpStatus !== null && error.httpStatus >= 400 && error.httpStatus < 500
    if ((operation && operation.status !== 'pending') || (definitiveRejection && !operation)) {
      releaseRequestKeys.delete(fingerprint)
      persistReleaseKeys()
    }
    throw error
  }
}

export const promoteModel = (versionId: string, data: ModelReleaseRequest): Promise<ModelReleaseMutationResponse> =>
  submitRelease(versionId, 'promote', data)
export const retireModel = (versionId: string, data?: Omit<ModelReleaseRequest, 'target_stage' | 'canary_camera_id'>): Promise<ModelReleaseMutationResponse> =>
  submitRelease(versionId, 'retire', data || {})
export const rollbackModel = (versionId: string, data: ModelRollbackRequest): Promise<ModelReleaseMutationResponse> =>
  submitRelease(versionId, 'rollback', data)
export const getRollbackReadiness = (versionId: string): Promise<RollbackReadiness> =>
  api.get<ApiResponse<RollbackReadiness>>(`/models/${versionId}/rollback-readiness`).then(unwrap)
export const getReleaseOperation = (operationId: string): Promise<ReleaseOperation> =>
  api
    .get<ApiResponse<ReleaseOperationResponse>>(`/models/release-operations/${operationId}`)
    .then(unwrap)
    .then(response => {
      acknowledgeReleaseOperation(response.operation)
      return response.operation
    })
export const getReleaseReadiness = (versionId: string): Promise<ReleaseReadiness> =>
  api
    .get<ApiResponse<ReleaseReadiness>>(`/models/${versionId}/release-readiness`)
    .then(unwrap)
export const getStageHistory = (versionId: string) =>
  api
    .get<ApiResponse<ModelEventListResponse>>(`/models/${versionId}/stage-history`)
    .then(unwrap)
export const getVersionEvents = (params?: { camera_id?: string; limit?: number }) =>
  api
    .get<ApiResponse<ModelEventListResponse>>('/models/events/list', { params })
    .then(unwrap)
export const getShadowReport = (versionId: string, params?: { camera_id?: string; days?: number }): Promise<ShadowReportResponse> =>
  api
    .get<ApiResponse<ShadowReportResponse>>(`/models/${versionId}/shadow-report`, { params })
    .then(unwrap)
export const getBackboneStatus = () => api.get('/models/backbone/status').then(u)

// ── Batch Inference ──
export const batchInference = (cameraId: string, imagePaths: string[]): Promise<BatchInferenceResponse> =>
  api
    .post<ApiResponse<BatchInferenceResponse>>(
      '/models/batch-inference',
      { camera_id: cameraId, image_paths: imagePaths },
      { timeout: 120000 },
    )
    .then(unwrap)

// ── Threshold Preview ──
export const getThresholdPreview = (params: { camera_id?: string; threshold?: number; days?: number }) =>
  api.get('/models/threshold-preview', { params }).then(u)

// ── A/B Comparison ──
export const getABScores = (versionId: string, params?: { camera_id?: string; days?: number; limit?: number }): Promise<ABScoresResponse> =>
  api
    .get<ApiResponse<ABScoresResponse>>(`/models/${versionId}/ab-scores`, { params })
    .then(unwrap)
export const getABDistribution = (versionId: string, params?: { camera_id?: string; days?: number; bins?: number }): Promise<ABDistributionResponse> =>
  api
    .get<ApiResponse<ABDistributionResponse>>(`/models/${versionId}/ab-distribution`, { params })
    .then(unwrap)
export const runABLiveCompare = (cameraId: string, shadowVersionId: string): Promise<ABLiveCompareResponse> =>
  api
    .post<ApiResponse<ABLiveCompareResponse>>(
      '/models/ab-live-compare',
      { camera_id: cameraId, shadow_version_id: shadowVersionId },
      { timeout: 60000 },
    )
    .then(unwrap)
