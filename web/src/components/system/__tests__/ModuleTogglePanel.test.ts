import { enableAutoUnmount, mount, flushPromises } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { message } from 'ant-design-vue'

import ModuleTogglePanel from '../ModuleTogglePanel.vue'
import { getModuleStates, toggleModule } from '../../../api'

enableAutoUnmount(afterEach)

vi.mock('../../../api', () => ({
  getModuleStates: vi.fn(),
  toggleModule: vi.fn(),
}))

vi.mock('ant-design-vue', () => ({
  message: {
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
  },
}))

const cardStub = {
  props: ['title'],
  template: '<section class="ant-card"><h2>{{ title }}</h2><slot /></section>',
}

const switchStub = {
  props: ['checked', 'loading', 'disabled'],
  emits: ['update:checked', 'change'],
  template: `
    <button
      type="button"
      class="module-switch"
      :aria-pressed="checked ? 'true' : 'false'"
      :disabled="loading || disabled"
      @click="$emit('update:checked', !checked); $emit('change', !checked)"
    >
      {{ checked ? 'on' : 'off' }}
    </button>
  `,
}

describe('ModuleTogglePanel', () => {
  beforeEach(() => {
    vi.mocked(getModuleStates).mockReset()
    vi.mocked(toggleModule).mockReset()
    vi.mocked(message.success).mockReset()
    vi.mocked(message.warning).mockReset()
    vi.mocked(message.error).mockReset()
  })

  it('keeps unavailable switches disabled and surfaces configuration loading failures', async () => {
    vi.mocked(getModuleStates).mockRejectedValue(new Error('configuration unavailable'))
    const wrapper = mount(ModuleTogglePanel, {
      global: { stubs: { 'a-card': cardStub, 'a-switch': switchStub } },
    })
    await flushPromises()
    expect(wrapper.find('[role="alert"]').text()).toContain('configuration unavailable')
    expect(wrapper.findAll('.module-switch').every(button => button.attributes('disabled') !== undefined)).toBe(true)
    expect(toggleModule).not.toHaveBeenCalled()
  })

  it('keeps partial runtime application warnings visible after a successful HTTP response', async () => {
    vi.mocked(getModuleStates).mockResolvedValue({ 'segmenter.enabled': false })
    vi.mocked(toggleModule).mockResolvedValue({
      key: 'segmenter.enabled', value: true, restart_required: false,
      hot_reloaded: 0, hot_reloadable: true, hot_reload_failed: 1,
      pipelines_seen: 1, persisted: true, all_applied: false,
    })
    const wrapper = mount(ModuleTogglePanel, {
      global: { stubs: { 'a-card': cardStub, 'a-switch': switchStub } },
    })
    await flushPromises()
    const row = wrapper.findAll('.module-item').find(item => item.text().includes('SAM2 实例分割'))!
    await row.find('.module-switch').trigger('click')
    await flushPromises()
    expect(row.text()).toContain('运行时未全部生效')
    expect(row.find('.module-switch').attributes('aria-pressed')).toBe('true')
    expect(message.success).not.toHaveBeenCalled()
  })

  it('loads module state and reports hot-reload feedback after toggling classifier', async () => {
    vi.mocked(getModuleStates).mockResolvedValue({
      'classifier.enabled': false,
      'segmenter.enabled': true,
    })
    vi.mocked(toggleModule).mockResolvedValue({
      key: 'classifier.enabled',
      value: true,
      restart_required: false,
      hot_reloaded: 1,
      hot_reloadable: true,
      hot_reload_failed: 0,
      pipelines_seen: 1,
      persisted: false,
    })

    const wrapper = mount(ModuleTogglePanel, {
      global: {
        stubs: {
          'a-card': cardStub,
          'a-switch': switchStub,
        },
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('AI 异物分类')
    const classifierRow = wrapper
      .findAll('.module-item')
      .find(row => row.text().includes('AI 异物分类'))
    expect(classifierRow).toBeTruthy()
    expect(classifierRow!.find('.module-switch').attributes('aria-pressed')).toBe('false')

    await classifierRow!.find('.module-switch').trigger('click')
    await flushPromises()

    expect(toggleModule).toHaveBeenCalledWith('classifier.enabled', true)
    expect(message.warning).toHaveBeenCalledWith(
      'AI 异物分类 已在 1 个管线应用，但未写入 YAML，重启后可能丢失',
    )
  })

  it('rolls the switch back and shows an error when the backend rejects a toggle', async () => {
    vi.mocked(getModuleStates).mockResolvedValue({
      'cross_camera.enabled': false,
    })
    vi.mocked(toggleModule).mockRejectedValue(new Error('denied'))

    const wrapper = mount(ModuleTogglePanel, {
      global: {
        stubs: {
          'a-card': cardStub,
          'a-switch': switchStub,
        },
      },
    })
    await flushPromises()

    const crossCameraRow = wrapper
      .findAll('.module-item')
      .find(row => row.text().includes('跨相机关联'))
    expect(crossCameraRow).toBeTruthy()

    await crossCameraRow!.find('.module-switch').trigger('click')
    await flushPromises()

    expect(toggleModule).toHaveBeenCalledWith('cross_camera.enabled', true)
    expect(message.error).toHaveBeenCalledWith('denied')
    expect(crossCameraRow!.find('.module-switch').attributes('aria-pressed')).toBe('false')
  })
})
