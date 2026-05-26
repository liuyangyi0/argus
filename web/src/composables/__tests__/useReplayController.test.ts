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
    vi.mocked(getAlertTrajectory).mockResolvedValue({
      alert_id: 'alert-1',
      primary: null,
      trajectories: [{ track_id: 'track_a', points: [] }],
      classification: null,
    } as any)

    const { wrapper, ctrl } = mountHarness()
    await ctrl.loadData()
    await flushPromises()
    await nextTick()

    expect(ctrl.loading.value).toBe(false)
    expect(ctrl.metadata.value.camera_id).toBe('cam_01')
    expect(ctrl.fps.value).toBe(10)
    expect(ctrl.videoUrl.value).toBe('/api/replay/alert-1/video')
    expect(ctrl.hasHeatmaps.value).toBe(true)
    expect(ctrl.currentIndex.value).toBe(2)
    expect(ctrl.pendingSeekIndex.value).toBe(2)
    expect(ctrl.persistedClips.value).toEqual([
      { start_index: 1, end_index: 3, label: 'impact' },
    ])
    expect(ctrl.referenceFrame.value).toBe('data:image/jpeg;base64,abc123')
    expect(ctrl.referenceDate.value).toBe('2026-05-24')
    expect(ctrl.trajectoryFits.value).toHaveLength(1)
    expect((ctrl.signals.value as any).trajectory_points_by_track.track_a).toEqual([
      { x: 10, y: 20 },
      null,
      { x: 30, y: 40 },
      null,
    ])

    wrapper.unmount()
  })

  it('keeps replay unavailable when metadata or signals fail to load', async () => {
    vi.mocked(getReplayMetadata).mockRejectedValue(new Error('missing recording'))
    vi.mocked(getReplaySignals).mockResolvedValue({ timestamps: [] })
    vi.mocked(getReplayReference).mockResolvedValue({ available: false })
    vi.mocked(getAlertTrajectory).mockResolvedValue(null as any)

    const { wrapper, ctrl } = mountHarness()
    await ctrl.loadData()
    await flushPromises()

    expect(ctrl.loading.value).toBe(false)
    expect(ctrl.metadata.value).toBe(null)

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
      })
      .mockResolvedValueOnce({
        alert_id: 'alert-1',
        camera_id: 'cam_01',
        fps: 10,
        frame_count: 12,
        trigger_frame_index: 2,
        trigger_timestamp: 1_777_777_000,
        status: 'complete',
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
        clips: [{ start_index: 2, end_index: 5, label: 'finalized' }],
      })
    vi.mocked(getReplayReference).mockResolvedValue({
      available: false,
      frame_base64: null,
      source_date: '',
    })
    vi.mocked(getAlertTrajectory).mockResolvedValue(null as any)

    const { wrapper, ctrl } = mountHarness()
    await ctrl.loadData()
    await flushPromises()

    expect(ctrl.metadata.value.status).toBe('recording')
    expect(ctrl.metadata.value.frame_count).toBe(4)
    expect(getReplayMetadata).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(1999)
    expect(getReplayMetadata).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(1)
    await flushPromises()

    expect(getReplayMetadata).toHaveBeenCalledTimes(2)
    expect(ctrl.metadata.value.status).toBe('complete')
    expect(ctrl.metadata.value.frame_count).toBe(12)
    expect(ctrl.persistedClips.value).toEqual([
      { start_index: 2, end_index: 5, label: 'finalized' },
    ])

    await vi.advanceTimersByTimeAsync(2500)
    expect(getReplayMetadata).toHaveBeenCalledTimes(2)

    wrapper.unmount()
  })
})
