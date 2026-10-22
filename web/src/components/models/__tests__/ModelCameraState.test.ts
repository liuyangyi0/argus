import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import ModelCameraState from '../ModelCameraState.vue'

describe('ModelCameraState', () => {
  it('reserves the first load and then keeps model history visible on camera refresh failure', async () => {
    const wrapper = mount(ModelCameraState, {
      props: { loading: true, loaded: false, cameraCount: 0 },
      slots: { default: '<div data-test="history">training history</div>' },
      global: {
        stubs: {
          ContentSkeleton: { template: '<div data-test="skeleton" />' },
        },
      },
    })

    expect(wrapper.find('[data-test="skeleton"]').exists()).toBe(true)
    expect(wrapper.find('[data-test="history"]').exists()).toBe(false)

    await wrapper.setProps({
      loading: false,
      loaded: true,
      error: 'camera registry unavailable',
    })

    expect(wrapper.text()).toContain('摄像头列表加载失败')
    expect(wrapper.text()).toContain('camera registry unavailable')
    expect(wrapper.find('[data-test="history"]').exists()).toBe(true)
    expect(wrapper.text()).not.toContain('尚未配置摄像头')
  })

  it('announces a successful zero-camera response without calling it an error', () => {
    const wrapper = mount(ModelCameraState, {
      props: { loading: false, loaded: true, cameraCount: 0 },
      slots: { default: '<div>existing records</div>' },
    })

    expect(wrapper.text()).toContain('尚未配置摄像头')
    expect(wrapper.text()).toContain('existing records')
    expect(wrapper.text()).not.toContain('加载失败')
  })
})
