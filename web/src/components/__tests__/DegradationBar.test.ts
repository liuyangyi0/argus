import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import DegradationBar from '../DegradationBar.vue'
import { getDegradationSummary } from '../../api'
import { getModelsStatus } from '../../api/system'
import { useWebSocket } from '../../composables/useWebSocket'
import type { ModelHealthStatus } from '../../types/api'

vi.mock('../../api', () => ({
  getDegradationSummary: vi.fn(),
}))

vi.mock('../../api/system', () => ({
  getModelsStatus: vi.fn(),
}))

vi.mock('../../composables/useWebSocket', () => ({
  useWebSocket: vi.fn(),
}))

vi.mock('ant-design-vue', () => ({
  Tag: { props: ['color'], template: '<span data-test="tag"><slot /></span>' },
  Typography: {
    Text: { props: ['strong', 'type'], template: '<span><slot /></span>' },
  },
}))

vi.mock('@ant-design/icons-vue', () => ({
  DownOutlined: { template: '<span />' },
}))

describe('DegradationBar', () => {
  const mismatch: ModelHealthStatus = {
    name: 'anomaly', camera_id: 'cam_gpu', loaded: true, backend: 'openvino',
    model_path: '/models/model.xml', image_size: [256, 256], last_error: null,
    last_error_ts: null, consecutive_failures: 0, total_inferences: 10,
    total_failures: 0, last_success_ts: null, extra: { expected_backend: 'torch-cuda' },
  }

  async function refresh() {
    const options = vi.mocked(useWebSocket).mock.calls[0]?.[0]
    await options?.fallbackPoll?.()
    await flushPromises()
  }

  beforeEach(() => {
    vi.mocked(getDegradationSummary).mockReset()
    vi.mocked(getModelsStatus).mockReset()
    vi.mocked(useWebSocket).mockClear()
    vi.mocked(getModelsStatus).mockResolvedValue([])
    vi.mocked(getDegradationSummary).mockResolvedValue({
      events: [],
      active_count: 0,
      max_level: null,
    })
  })

  it('surfaces model backend mismatches in the global degradation bar', async () => {
    vi.mocked(getModelsStatus).mockResolvedValue([
      {
        name: 'anomaly',
        camera_id: 'cam_gpu',
        loaded: true,
        backend: 'openvino',
        model_path: '/models/model.xml',
        image_size: [256, 256],
        last_error: null,
        last_error_ts: null,
        consecutive_failures: 0,
        total_inferences: 10,
        total_failures: 0,
        last_success_ts: null,
        extra: {
          expected_backend: 'torch-cuda',
        },
      },
    ])

    const wrapper = mount(DegradationBar)
    await flushPromises()

    expect(wrapper.text()).toContain('模型后端不符合要求')
    expect(wrapper.text()).toContain('cam_gpu')
    expect(wrapper.text()).toContain('当前 openvino，期望 torch-cuda')
    expect(wrapper.text()).toContain('检查模型发布')

    wrapper.unmount()
  })

  it('keeps a previous backend warning and marks its snapshot stale on query failure', async () => {
    vi.mocked(getModelsStatus).mockResolvedValue([mismatch])
    const wrapper = mount(DegradationBar)
    await flushPromises()

    vi.mocked(getModelsStatus).mockRejectedValue(new Error('network unavailable'))
    await refresh()

    expect(wrapper.text()).toContain('模型后端不符合要求')
    expect(wrapper.text()).toContain('模型状态未知')
    expect(wrapper.text()).toContain('数据可能已过期，不能视为恢复')
    expect(wrapper.attributes('role')).toBe('status')

    vi.mocked(getModelsStatus).mockResolvedValue([])
    await refresh()
    expect(wrapper.find('.degradation-bar').exists()).toBe(false)
    wrapper.unmount()
  })

  it('shows unknown on the first failed query instead of implying a healthy model', async () => {
    vi.mocked(getModelsStatus).mockRejectedValue(new Error('offline'))
    const wrapper = mount(DegradationBar)
    await flushPromises()

    expect(wrapper.text()).toContain('模型状态未知')
    expect(wrapper.text()).toContain('尚未取得模型状态')
    expect(wrapper.text()).not.toContain('模型后端不符合要求')
    wrapper.unmount()
  })

  it('shows unknown after a healthy snapshot becomes unavailable', async () => {
    const wrapper = mount(DegradationBar)
    await flushPromises()
    expect(wrapper.find('.degradation-bar').exists()).toBe(false)

    vi.mocked(getModelsStatus).mockRejectedValue(new Error('offline'))
    await refresh()
    expect(wrapper.text()).toContain('模型状态未知')
    expect(wrapper.text()).toContain('保留上次模型状态')
    wrapper.unmount()
  })

  it('retains system degradation events when their query fails too', async () => {
    vi.mocked(getDegradationSummary).mockResolvedValue({
      events: [{
        event_id: 'camera-offline', level: 'severe', category: 'camera', camera_id: 'cam_gpu',
        title: '摄像头断流', impact: '预览不可用', action: '检查源', started_at: 1,
      }],
      active_count: 1, max_level: 'severe',
    })
    const wrapper = mount(DegradationBar)
    await flushPromises()
    vi.mocked(getDegradationSummary).mockRejectedValue(new Error('offline'))
    await refresh()

    expect(wrapper.text()).toContain('摄像头断流')
    expect(wrapper.text()).toContain('系统降级状态未知')
    wrapper.unmount()
  })

  it.each(['success', 'failure'])('ignores a late older %s after a newer model snapshot', async (outcome) => {
    let resolve!: (value: ModelHealthStatus[]) => void
    let reject!: (reason: Error) => void
    const older = new Promise<ModelHealthStatus[]>((yes, no) => { resolve = yes; reject = no })
    vi.mocked(getModelsStatus).mockReturnValueOnce(older).mockResolvedValue([mismatch])
    const wrapper = mount(DegradationBar)
    await refresh()
    if (outcome === 'success') resolve([])
    else reject(new Error('old request failed'))
    await flushPromises()

    expect(wrapper.text()).toContain('模型后端不符合要求')
    expect(wrapper.text()).not.toContain('模型状态未知')
    wrapper.unmount()
  })
})
