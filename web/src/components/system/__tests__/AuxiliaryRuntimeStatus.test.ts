import { enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import AuxiliaryRuntimeStatus from '../AuxiliaryRuntimeStatus.vue'

enableAutoUnmount(afterEach)

vi.mock('ant-design-vue', () => ({
  Alert: {
    props: ['type', 'message'],
    template: '<p :data-type="type" role="status">{{ message }}</p>',
  },
}))

describe('AuxiliaryRuntimeStatus', () => {
  it('shows a per-camera failure reason, actual backend and an unfinished task', () => {
    const wrapper = mount(AuxiliaryRuntimeStatus, {
      props: {
        label: '分割器', enabled: true,
        runtime: {
          total_pipelines: 1, pipelines_attached: 1, pipelines_loaded: 1,
          pipelines_ready: 0, pipelines_failed: 1,
          details: [{
            camera_id: 'camera-1', state: 'failed', backend: 'sam2',
            loaded: true, ready: false, degraded: false, busy: true,
            reason: 'inference timed out',
          }],
        },
      },
    })
    expect(wrapper.text()).toContain('camera-1')
    expect(wrapper.text()).toContain('故障')
    expect(wrapper.text()).toContain('后端：sam2')
    expect(wrapper.text()).toContain('任务尚未结束')
    expect(wrapper.text()).toContain('inference timed out')
    expect(wrapper.find('[data-type]').attributes('data-type')).toBe('error')
  })
})
