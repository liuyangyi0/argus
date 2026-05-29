import { mount, flushPromises } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import ModelStatusPanel from '../ModelStatusPanel.vue'
import { getModelsStatus } from '../../../api/system'

vi.mock('../../../api/system', () => ({
  getModelsStatus: vi.fn(),
}))

const cardStub = {
  props: ['loading'],
  template: `
    <section data-test="model-card" :data-loading="String(Boolean(loading))">
      <slot name="extra" />
      <slot />
    </section>
  `,
}

const alertStub = {
  props: ['message', 'type'],
  template: '<div data-test="status-alert" :data-type="type">{{ message }}</div>',
}

const tableStub = {
  props: ['dataSource'],
  template: '<div data-test="status-table" :data-count="dataSource.length"><slot /></div>',
}

function mountPanel() {
  return mount(ModelStatusPanel, {
    global: {
      stubs: {
        'a-card': cardStub,
        'a-alert': alertStub,
        'a-table': tableStub,
        'a-table-column': true,
        'a-tag': { template: '<span><slot /></span>' },
        'a-tooltip': { template: '<span><slot /></span>' },
      },
    },
  })
}

describe('ModelStatusPanel', () => {
  beforeEach(() => {
    vi.mocked(getModelsStatus).mockReset()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('stops the loading state after a successful empty status response', async () => {
    vi.mocked(getModelsStatus).mockResolvedValue([])

    const wrapper = mountPanel()
    await flushPromises()

    expect(getModelsStatus).toHaveBeenCalledTimes(1)
    expect(wrapper.find('[data-test="model-card"]').attributes('data-loading')).toBe('false')
    expect(wrapper.find('[data-test="status-table"]').attributes('data-count')).toBe('0')

    wrapper.unmount()
  })

  it('surfaces low-light input limits before generic fallback degradation', async () => {
    vi.mocked(getModelsStatus).mockResolvedValue([
      {
        name: 'anomaly',
        camera_id: 'cam_01',
        loaded: true,
        backend: 'ssim-fallback',
        model_path: null,
        image_size: [256, 256],
        last_error: null,
        last_error_ts: null,
        consecutive_failures: 0,
        total_inferences: 0,
        total_failures: 0,
        last_success_ts: null,
        extra: {
          input_quality: {
            low_light: true,
            detection_limited: true,
            detection_limited_reason: 'low_light',
            ssim_calibration_blocked: true,
          },
        },
      },
    ])

    const wrapper = mountPanel()
    await flushPromises()

    const alert = wrapper.find('[data-test="status-alert"]')
    expect(alert.text()).toContain('输入处于低光或曝光恢复')
    expect(alert.text()).not.toContain('fallback 模式')

    wrapper.unmount()
  })
})
