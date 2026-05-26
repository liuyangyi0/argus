import { computed, defineComponent, ref, type Ref } from 'vue'
import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useStoryboardController } from '../useStoryboardController'
import { getReplayMetadata, getStoryboard } from '../../api'
import type { StoryboardResponse } from '../../types/api'

vi.mock('ant-design-vue', () => ({
  message: {
    error: vi.fn(),
  },
}))

vi.mock('../../api', () => ({
  getStoryboard: vi.fn(),
  getReplayMetadata: vi.fn(),
}))

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

function camera(alertId: string, cameraId = 'cam_01') {
  return {
    alert_id: alertId,
    camera_id: cameraId,
    trigger_timestamp: 0,
    metadata_url: `/api/replay/${alertId}/metadata`,
    video_url: `/api/replay/${alertId}/video`,
    signals_url: `/api/replay/${alertId}/signals`,
    trigger_offset_s: 0,
  }
}

function storyboard(alertId: string): StoryboardResponse {
  return {
    primary_alert_id: alertId,
    cameras: [camera(alertId)],
    count: 1,
  }
}

function mountHarness(initialAlertId = 'story-old') {
  const holder: {
    alertId?: Ref<string>
    ctrl?: ReturnType<typeof useStoryboardController>
  } = {}
  const Harness = defineComponent({
    setup() {
      const alertId = ref(initialAlertId)
      const ctrl = useStoryboardController(alertId)
      holder.alertId = alertId
      holder.ctrl = ctrl
      return {
        count: computed(() => ctrl.cameras.value.length),
        loading: ctrl.loading,
      }
    },
    template: '<div>{{ count }} {{ loading }}</div>',
  })
  const wrapper = mount(Harness)
  if (!holder.alertId || !holder.ctrl) throw new Error('Storyboard controller was not created')
  return { wrapper, alertId: holder.alertId, ctrl: holder.ctrl }
}

describe('useStoryboardController', () => {
  beforeEach(() => {
    vi.mocked(getStoryboard).mockReset()
    vi.mocked(getReplayMetadata).mockReset()
  })

  it('loads storyboard cameras and prefetches replay durations', async () => {
    vi.mocked(getStoryboard).mockResolvedValue({
      primary_alert_id: 'story-1',
      cameras: [camera('story-1', 'cam_01'), camera('story-2', 'cam_02')],
      count: 2,
    })
    vi.mocked(getReplayMetadata).mockResolvedValue({
      fps: 10,
      frame_count: 120,
    } as any)

    const { wrapper, ctrl } = mountHarness('story-1')
    await ctrl.load()
    await flushPromises()

    expect(getStoryboard).toHaveBeenCalledWith('story-1')
    expect(ctrl.loading.value).toBe(false)
    expect(ctrl.cameras.value.map(c => c.alert_id)).toEqual(['story-1', 'story-2'])
    expect(ctrl.durations.value).toEqual({
      'story-1': 12,
      'story-2': 12,
    })

    wrapper.unmount()
  })

  it('ignores stale storyboard responses after the route alert id changes', async () => {
    const oldLoad = deferred<StoryboardResponse>()
    const newLoad = deferred<StoryboardResponse>()
    vi.mocked(getStoryboard).mockImplementation((alertId: string) => {
      if (alertId === 'story-old') return oldLoad.promise
      if (alertId === 'story-new') return newLoad.promise
      throw new Error(`unexpected storyboard id ${alertId}`)
    })
    vi.mocked(getReplayMetadata).mockResolvedValue({
      fps: 5,
      frame_count: 50,
    } as any)

    const { wrapper, alertId, ctrl } = mountHarness('story-old')
    const oldPromise = ctrl.load()

    alertId.value = 'story-new'
    const newPromise = ctrl.load()
    newLoad.resolve(storyboard('story-new'))
    await newPromise
    await flushPromises()

    expect(ctrl.loading.value).toBe(false)
    expect(ctrl.cameras.value.map(c => c.alert_id)).toEqual(['story-new'])
    expect(ctrl.durations.value).toEqual({ 'story-new': 10 })

    oldLoad.resolve(storyboard('story-old'))
    await oldPromise
    await flushPromises()

    expect(ctrl.loading.value).toBe(false)
    expect(ctrl.cameras.value.map(c => c.alert_id)).toEqual(['story-new'])
    expect(ctrl.durations.value).toEqual({ 'story-new': 10 })

    wrapper.unmount()
  })
})
