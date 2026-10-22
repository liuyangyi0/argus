import { api, unwrap } from './client'
import type { ApiResponse, StoryboardResponse } from '../types/api'
import type {
  ReplayClipListResponse,
  ReplayClipMutationResponse,
  ReplayMetadataResponse,
  ReplayPinFrameResponse,
  ReplaySignalsResponse,
} from '../types/generated-openapi'

export type ReplayMetadataPayload = Partial<ReplayMetadataResponse>

export type ReplaySignalsPayload = Omit<
  Partial<ReplaySignalsResponse>,
  'clips' | 'trajectory_points'
> & {
  clips?: Array<{
    start_index: number
    end_index: number
    label: string
    created_at?: string
  }>
  trajectory_points?: Record<string, Array<{
    frame_index: number
    x: number
    y: number
    t?: number
  }>>
}

export interface ReplayReferenceResponse {
  available: boolean
  frame_base64?: string | null
  source_date?: string
}

// ── Replay (FR-033) ──
export const getReplayMetadata = (alertId: string) =>
  api
    .get<ApiResponse<ReplayMetadataPayload>>(`/replay/${alertId}/metadata`)
    .then(unwrap)
export const getReplaySignals = (alertId: string) =>
  api
    .get<ApiResponse<ReplaySignalsPayload>>(`/replay/${alertId}/signals`)
    .then(unwrap)
export const getReplayVideoUrl = (alertId: string) =>
  `/api/replay/${alertId}/video`
export const getReplayFrameUrl = (alertId: string, index: number) =>
  `/api/replay/${alertId}/frame/${index}`
export const getReplayHeatmapUrl = (alertId: string, index: number) =>
  `/api/replay/${alertId}/heatmap/${index}`
export const getReplayReference = (
  alertId: string,
  params?: { date?: string; time?: string; frame_offset_seconds?: number },
) =>
  api
    .get<ApiResponse<ReplayReferenceResponse>>(`/replay/${alertId}/reference`, { params })
    .then(unwrap)
export const pinReplayFrame = (alertId: string, data: { index: number; label: string }) =>
  api
    .post<ApiResponse<ReplayPinFrameResponse>>(`/replay/${alertId}/pin-frame`, data)
    .then(unwrap)
export const addReplayClip = (
  alertId: string,
  data: { start_index: number; end_index: number; label?: string },
) => api
  .post<ApiResponse<ReplayClipMutationResponse>>(`/replay/${alertId}/clips`, data)
  .then(unwrap)
export const deleteReplayClip = (alertId: string, index: number) =>
  api
    .delete<ApiResponse<ReplayClipListResponse>>(`/replay/${alertId}/clips/${index}`)
    .then(unwrap)

// ── Multi-camera Storyboard ──
export const getStoryboard = (alertId: string): Promise<StoryboardResponse> =>
  api
    .get<ApiResponse<StoryboardResponse>>(`/replay/storyboard/${alertId}`)
    .then(unwrap)
