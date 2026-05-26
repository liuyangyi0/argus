import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import OverviewView from '../OverviewView.vue'
import { getAnomalyDegradation } from '../../../api/system'
import { useWebSocket } from '../../../composables/useWebSocket'

vi.mock('../../../api/system', () => ({
  getAnomalyDegradation: vi.fn(),
}))

vi.mock('../../../composables/useWebSocket', () => ({
  useWebSocket: vi.fn(),
}))

vi.mock('ant-design-vue', () => ({
  Card: {
    props: ['title'],
    template: '<section class="ant-card"><h2>{{ title }}</h2><div data-test="card-extra"><slot name="extra" /></div><slot /></section>',
  },
  Tag: {
    props: ['color'],
    template: '<span class="ant-tag" :data-color="color"><slot /></span>',
  },
  Empty: {
    props: ['description'],
    template: '<div class="ant-empty">{{ description }}</div>',
  },
}))

function statusPayload(degraded: boolean, cameras: any[]) {
  return {
    anomaly: {
      degraded,
      reason: degraded ? 'model_load_failed' : null,
      since: degraded ? 1_716_600_000 : null,
      cameras,
    },
  }
}

function mountOverview() {
  return mount(OverviewView, {
    global: {
      stubs: {
        SystemOverviewPanel: { template: '<div data-test="system-overview" />' },
        ModelStatusPanel: { template: '<div data-test="model-status" />' },
      },
    },
  })
}

describe('System overview anomaly degradation card', () => {
  beforeEach(() => {
    vi.mocked(getAnomalyDegradation).mockReset()
    vi.mocked(useWebSocket).mockReset()
  })

  it('shows camera-level anomaly fallback state from the system API', async () => {
    vi.mocked(getAnomalyDegradation).mockResolvedValue(
      statusPayload(true, [
        { camera_id: 'cam_01', degraded: true, reason: 'model_load_failed', since: 1_716_600_000 },
        { camera_id: 'cam_02', degraded: false, reason: null, since: null },
      ]),
    )

    const wrapper = mountOverview()
    await flushPromises()

    expect(wrapper.text()).toContain('降级监控')
    expect(wrapper.text()).toContain('异常检测：降级')
    expect(wrapper.text()).toContain('cam_01')
    expect(wrapper.text()).toContain('原因：model_load_failed')
    expect(wrapper.text()).toContain('cam_02')
    expect(wrapper.text()).toContain('正常')
  })

  it('re-queries after system_degradation websocket events and configures fallback polling', async () => {
    let onMessage: ((topic: 'system_degradation', data: any) => void) | undefined
    vi.mocked(useWebSocket).mockImplementation((options: any) => {
      onMessage = options.onMessage
      return {
        connected: { value: true },
        reconnecting: { value: false },
        retryCount: { value: 0 },
        fallbackMode: { value: false },
        nextRetryIn: { value: 0 },
        audioMuted: { value: false },
        toggleAudioMute: vi.fn(),
        playAlertBeep: vi.fn(),
      } as any
    })
    vi.mocked(getAnomalyDegradation)
      .mockResolvedValueOnce(statusPayload(false, [
        { camera_id: 'cam_01', degraded: false, reason: null, since: null },
      ]))
      .mockResolvedValueOnce(statusPayload(true, [
        { camera_id: 'cam_01', degraded: true, reason: 'runtime_error', since: 1_716_600_100 },
      ]))

    const wrapper = mountOverview()
    await flushPromises()

    expect(useWebSocket).toHaveBeenCalledWith(expect.objectContaining({
      topics: ['system_degradation'],
      fallbackPoll: expect.any(Function),
      fallbackInterval: 15000,
    }))
    expect(wrapper.text()).toContain('异常检测：正常')

    onMessage?.('system_degradation', { type: 'anomaly.degradation_changed' })
    await flushPromises()

    expect(getAnomalyDegradation).toHaveBeenCalledTimes(2)
    expect(wrapper.text()).toContain('异常检测：降级')
    expect(wrapper.text()).toContain('原因：runtime_error')
  })
})
