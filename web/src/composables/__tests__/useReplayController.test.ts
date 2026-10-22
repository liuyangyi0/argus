import { defineComponent, nextTick } from 'vue'
import { mount, flushPromises } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useReplayController } from '../useReplayController'
import {
  getAlertTrajectory,
  getReplayMetadata,
  getReplayReference,
  getReplaySignals,
} from '../../api'
import type { TrajectoryResponse } from '../../api/physics'

vi.mock('ant-design-vue', () => ({
  message: {
    error: vi.fn(),
    success: vi.fn(),
    warning: vi.fn(),
  },
}))

vi.mock('../../api', () => ({
  getReplayMetadata: vi.fn(),
  getReplaySignals: vi.fn(),
  getReplayVideoUrl: vi.fn((alertId: string) => `/api/replay/${alertId}/video`),
  getReplayReference: vi.fn(),
  getAlertTrajectory: vi.fn(),
  pinReplayFrame: vi.fn(),
  addReplayClip: vi.fn(),
  deleteReplayClip: vi.fn(),
}))

function mountHarness() {
  const holder: { ctrl?: ReturnType<typeof useReplayController> } = {}
  const Harness = defineComponent({
    setup() {
      const ctrl = useReplayController('alert-1')
      holder.ctrl = ctrl
      return { loading: ctrl.loading }
    },
    template: '<div>{{ loading }}</div>',
  })
  const wrapper = mount(Harness)
  const ctrl = holder.ctrl
  if (!ctrl) throw new Error('Replay controller was not created')
  return { wrapper, ctrl }
}

describe('useReplayController', () => {
  beforeEach(() => {
    vi.mocked(getReplayMetadata).mockReset()
    vi.mocked(getReplaySignals).mockReset()
    vi.mocked(getReplayReference).mockReset()
    vi.mocked(getAlertTrajectory).mockReset()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('loads replay metadata, signals, trigger frame, clips, reference, and trajectory tracks', async () => {
    vi.mocked(getReplayMetadata).mockResolvedValue({
      alert_id: 'alert-1',
      camera_id: 'cam_01',
      fps: 10,
      frame_count: 4,
      trigger_frame_index: 2,
      trigger_timestamp: 1_777_777_000,
      status: 'complete',
      created_at: '2026-07-14T00:00:00Z',
      video_url: '/api/replay/alert-1/video',
    })
    vi.mocked(getReplaySignals).mockResolvedValue({
      timestamps: [1, 2, 3, 4],
      anomaly_scores: [0.1, 0.2, 0.9, 0.4],
      has_heatmaps: true,
      clips: [{ start_index: 1, end_index: 3, label: 'impact' }],
      trajectory_points: {
        track_a: [
          { frame_index: 0, x: 10, y: 20 },
          { frame_index: 2, x: 30, y: 40 },
          { frame_index: 9, x: 99, y: 99 },
        ],
      },
    })
    vi.mocked(getReplayReference).mockResolvedValue({
      available: true,
      frame_base64: 'abc123',
      source_date: '2026-05-24',
    })
    const trajectoryResponse = {
      alert_id: 'alert-1',
      primary: null,
      trajectories: [{
        track_id: 1,
        is_primary: true,
        model_type: 'ballistic',
        r_squared: 0.98,
        speed_ms: null,
        speed_px_per_sec: 120,
        origin: { x_mm: 0, y_mm: 0, z_mm: 0, x_px: 10, y_px: 20 },
        landing: { x_mm: 0, y_mm: 0, z_mm: 0, x_px: 30, y_px: 40 },
      }],
      classification: null,
    } satisfies TrajectoryResponse
    vi.mocked(getAlertTrajectory).mockResolvedValue(trajectoryResponse)

    const { wrapper, ctrl } = mountHarness()
    await ctrl.loadData()
    await flushPromises()
    await nextTick()

    expect(ctrl.loading.value).toBe(false)
    expect(ctrl.metadataState.value).toBe('ready')
    expect(ctrl.signalsState.value).toBe('ready')
    expect(ctrl.trajectoryState.value).toBe('ready')
    expect(ctrl.metadata.value?.camera_id).toBe('cam_01')
    expect(ctrl.fps.value).toBe(10)
    expect(ctrl.videoUrl.value).toBe(
      '/api/replay/alert-1/video?v=complete%3A4%3A2026-07-14T00%3A00%3A00Z',
    )
    expect(ctrl.hasHeatmaps.value).toBe(true)
    expect(ctrl.currentIndex.value).toBe(2)
    expect(ctrl.pendingSeekIndex.value).toBe(2)
    expect(ctrl.persistedClips.value).toEqual([
      { start_index: 1, end_index: 3, label: 'impact' },
    ])
    expect(ctrl.referenceFrame.value).toBe('data:image/jpeg;base64,abc123')
    expect(ctrl.referenceDate.value).toBe('2026-05-24')
    expect(ctrl.trajectoryFits.value).toHaveLength(1)
    expect(ctrl.signals.value?.trajectory_points_by_track?.track_a).toEqual([
      { x: 10, y: 20 },
      null,
      { x: 30, y: 40 },
      null,
    ])

    wrapper.unmount()
  })

  it('commits metadata when signals fail and exposes independent resource errors', async () => {
    vi.mocked(getReplayMetadata).mockResolvedValue({
      alert_id: 'alert-1',
      camera_id: 'cam_01',
      fps: 10,
      frame_count: 4,
      trigger_frame_index: 2,
      trigger_timestamp: 1_777_777_000,
      status: 'complete',
    })
    vi.mocked(getReplaySignals).mockRejectedValue(new Error('signals unavailable'))
    vi.mocked(getReplayReference).mockResolvedValue({ available: false })
    vi.mocked(getAlertTrajectory).mockRejectedValue(new Error('404 not found'))

    const { wrapper, ctrl } = mountHarness()
    await ctrl.loadData()
    await flushPromises()

    expect(ctrl.loading.value).toBe(false)
    expect(ctrl.metadata.value).toMatchObject({ camera_id: 'cam_01', frame_count: 4 })
    expect(ctrl.metadataState.value).toBe('ready')
    expect(ctrl.metadataError.value).toBe('')
    expect(ctrl.signals.value).toBe(null)
    expect(ctrl.signalsState.value).toBe('error')
    expect(ctrl.signalsError.value).toContain('signals unavailable')
    expect(ctrl.trajectoryState.value).toBe('unavailable')

    wrapper.unmount()
  })

  it('keeps successful signals when trajectory loading fails', async () => {
    vi.mocked(getReplayMetadata).mockResolvedValue({
      alert_id: 'alert-1',
      camera_id: 'cam_01',
      fps: 10,
      frame_count: 2,
      trigger_frame_index: 0,
      trigger_timestamp: 1_777_777_000,
      status: 'complete',
    })
    vi.mocked(getReplaySignals).mockResolvedValue({
      timestamps: [1, 2],
      anomaly_scores: [0.1, 0.2],
      clips: [],
    })
    vi.mocked(getReplayReference).mockResolvedValue({ available: false })
    vi.mocked(getAlertTrajectory).mockRejectedValue(new Error('physics service offline'))

    const { wrapper, ctrl } = mountHarness()
    await ctrl.loadData()
    await flushPromises()

    expect(ctrl.metadataState.value).toBe('ready')
    expect(ctrl.signalsState.value).toBe('ready')
    expect(ctrl.signals.value?.timestamps).toEqual([1, 2])
    expect(ctrl.trajectoryState.value).toBe('error')
    expect(ctrl.trajectoryError.value).toContain('physics service offline')

    wrapper.unmount()
  })

  it('tracks video loading, ready, failure, and retry independently', () => {
    const { wrapper, ctrl } = mountHarness()
    const load = vi.fn()
    ctrl.videoEl.value = { load } as unknown as HTMLVideoElement

    ctrl.markVideoLoading()
    expect(ctrl.videoState.value).toBe('loading')
    expect(ctrl.videoError.value).toBe('')

    ctrl.markVideoReady()
    expect(ctrl.videoState.value).toBe('ready')

    ctrl.markVideoError('decoder failed')
    expect(ctrl.videoState.value).toBe('error')
    expect(ctrl.videoError.value).toBe('decoder failed')

    ctrl.reloadVideo()
    expect(load).toHaveBeenCalledOnce()
    expect(ctrl.videoState.value).toBe('loading')
    expect(ctrl.videoError.value).toBe('')

    wrapper.unmount()
  })

  it('refreshes replay data while the recording is still being finalized', async () => {
    vi.useFakeTimers()
    vi.mocked(getReplayMetadata)
      .mockResolvedValueOnce({
        alert_id: 'alert-1',
        camera_id: 'cam_01',
        fps: 10,
        frame_count: 4,
        trigger_frame_index: 2,
        trigger_timestamp: 1_777_777_000,
        status: 'recording',
        created_at: '2026-07-14T00:00:00Z',
        video_url: '/api/replay/alert-1/video',
      })
      .mockResolvedValueOnce({
        alert_id: 'alert-1',
        camera_id: 'cam_01',
        fps: 10,
        frame_count: 6,
        trigger_frame_index: 2,
        trigger_timestamp: 1_777_777_000,
        status: 'recording',
        created_at: '2026-07-14T00:00:00Z',
        video_url: '/api/replay/alert-1/video',
      })
      .mockResolvedValueOnce({
        alert_id: 'alert-1',
        camera_id: 'cam_01',
        fps: 10,
        frame_count: 12,
        trigger_frame_index: 2,
        trigger_timestamp: 1_777_777_000,
        status: 'complete',
        created_at: '2026-07-14T00:00:00Z',
        video_url: '/api/replay/alert-1/video',
      })
    vi.mocked(getReplaySignals)
      .mockResolvedValueOnce({
        timestamps: [1, 2, 3, 4],
        anomaly_scores: [0.1, 0.2, 0.9, 0.4],
        clips: [],
      })
      .mockResolvedValueOnce({
        timestamps: [1, 2, 3, 4, 5, 6],
        anomaly_scores: [0.1, 0.2, 0.9, 0.4, 0.3, 0.2],
        clips: [],
      })
      .mockResolvedValueOnce({
        timestamps: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        anomaly_scores: [0.1, 0.2, 0.9, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        clips: [{ start_index: 2, end_index: 5, label: 'finalized' }],
      })
    vi.mocked(getReplayReference).mockResolvedValue({
      available: false,
      frame_base64: null,
      source_date: '',
    })
    vi.mocked(getAlertTrajectory).mockRejectedValue(new Error('404 not found'))

    const { wrapper, ctrl } = mountHarness()
    await ctrl.loadData()
    await flushPromises()

    expect(ctrl.metadata.value?.status).toBe('recording')
    expect(ctrl.metadata.value?.frame_count).toBe(4)
    const recordingVideoUrl = ctrl.videoUrl.value
    expect(recordingVideoUrl).toBe('/api/replay/alert-1/video')
    expect(getReplayMetadata).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(1999)
    expect(getReplayMetadata).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(1)
    await flushPromises()

    expect(getReplayMetadata).toHaveBeenCalledTimes(2)
    expect(ctrl.metadata.value?.status).toBe('recording')
    expect(ctrl.metadata.value?.frame_count).toBe(6)
    expect(ctrl.videoUrl.value).toBe(recordingVideoUrl)

    await vi.advanceTimersByTimeAsync(2_000)
    await flushPromises()

    expect(getReplayMetadata).toHaveBeenCalledTimes(3)
    expect(ctrl.metadata.value?.status).toBe('complete')
    expect(ctrl.metadata.value?.frame_count).toBe(12)
    expect(ctrl.videoUrl.value).toBe(
      '/api/replay/alert-1/video?v=complete%3A12%3A2026-07-14T00%3A00%3A00Z',
    )
    expect(ctrl.videoUrl.value).not.toBe(recordingVideoUrl)
    expect(ctrl.persistedClips.value).toEqual([
      { start_index: 2, end_index: 5, label: 'finalized' },
    ])

    await vi.advanceTimersByTimeAsync(2500)
    expect(getReplayMetadata).toHaveBeenCalledTimes(3)

    wrapper.unmount()
  })
})
