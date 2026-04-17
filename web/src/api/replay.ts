import { api, u } from './client'
import type { StoryboardResponse } from '../types/api'

// ── Replay (FR-033) ──
export const getReplayMetadata = (alertId: string) =>
  api.get(`/replay/${alertId}/metadata`).then(u)
export const getReplaySignals = (alertId: string) =>
  api.get(`/replay/${alertId}/signals`).then(u)
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
  api.get(`/replay/${alertId}/reference`, { params }).then(u)
export const pinReplayFrame = (alertId: string, data: { index: number; label: string }) =>
  api.post(`/replay/${alertId}/pin-frame`, data).then(u)
export const addReplayClip = (
  alertId: string,
  data: { start_index: number; end_index: number; label?: string },
) => api.post(`/replay/${alertId}/clips`, data).then(u)
export const deleteReplayClip = (alertId: string, index: number) =>
  api.delete(`/replay/${alertId}/clips/${index}`).then(u)

// ── Multi-camera Storyboard ──
export const getStoryboard = (alertId: string): Promise<StoryboardResponse> =>
  api.get(`/replay/storyboard/${alertId}`).then(u)
