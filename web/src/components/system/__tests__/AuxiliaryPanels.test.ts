import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { message } from 'ant-design-vue'
import { getClassifierConfig, getSegmenterConfig, toggleModule } from '../../../api'
import type { ClassifierConfigPayload, SegmenterConfigPayload } from '../../../api/system'
import ClassifierPanel from '../ClassifierPanel.vue'
import SegmenterPanel from '../SegmenterPanel.vue'

enableAutoUnmount(afterEach)

const stubs = vi.hoisted(() => ({
  container: { template: '<div><slot name="title" /><slot name="extra" /><slot /></div>' },
  text: { template: '<span><slot /></span>' },
  alert: { props: ['type', 'message'], template: '<p :data-type="type">{{ message }}</p>' },
  button: { props: ['disabled', 'loading'], emits: ['click'], template: '<button :disabled="disabled || loading" @click="$emit(\'click\')"><slot /></button>' },
  switch: {
    props: ['checked', 'disabled', 'loading'], emits: ['update:checked'],
    template: '<button data-switch :disabled="disabled || loading" :aria-pressed="checked" @click="$emit(\'update:checked\', !checked)">toggle</button>',
  },
}))

vi.mock('../../../api', () => ({
  getClassifierConfig: vi.fn(), getSegmenterConfig: vi.fn(), toggleModule: vi.fn(),
  updateClassifierVocabulary: vi.fn(), updateSegmenterParams: vi.fn(),
}))

vi.mock('ant-design-vue', () => ({
  Card: stubs.container, Tag: stubs.text, Switch: stubs.switch, Button: stubs.button,
  Tooltip: stubs.container, Space: stubs.container, Typography: { Text: stubs.text },
  Alert: stubs.alert, Empty: stubs.text, Input: stubs.text, Segmented: stubs.text,
  Popconfirm: stubs.container, InputNumber: stubs.text, Slider: stubs.text,
  message: { success: vi.fn(), warning: vi.fn(), error: vi.fn() },
}))

function classifierConfig(): ClassifierConfigPayload {
  return {
    enabled: false, model_name: 'fake-yolo-world', min_anomaly_score_to_classify: 0.5,
    vocabulary: [], high_risk_labels: [], low_risk_labels: [], suppress_labels: [], custom_vocabulary_path: null,
    runtime: { total_pipelines: 1, pipelines_attached: 0, pipelines_loaded: 0, pipelines_ready: 0 },
  }
}

function segmenterConfig(): SegmenterConfigPayload {
  return {
    enabled: true, model_size: 'small', max_points: 5, min_anomaly_score: 0.7,
    min_mask_area_px: 100, timeout_seconds: 10,
    runtime: { total_pipelines: 1, pipelines_attached: 1, pipelines_loaded: 1 },
  }
}

describe('auxiliary module control panels', () => {
  beforeEach(() => vi.clearAllMocks())

  it('does not show the SAM2 model as ready for a legacy loaded-only response', async () => {
    vi.mocked(getSegmenterConfig).mockResolvedValue(segmenterConfig())
    const wrapper = mount(SegmenterPanel)
    await flushPromises()
    expect(wrapper.text()).toContain('配置已启用')
    expect(wrapper.text()).toContain('无法确认模型可用')
    expect(wrapper.text()).not.toContain('已加载 SAM2 模型')
    expect(wrapper.text()).not.toContain('已在全部')
  })

  it('shows the fallback backend and the actual failure reason', async () => {
    const config = segmenterConfig()
    config.runtime.pipelines_degraded = 1
    config.runtime.details = [{
      camera_id: 'camera-1', state: 'degraded', backend: 'contour',
      loaded: true, ready: false, degraded: true, reason: 'SAM2 dependency missing',
    }]
    vi.mocked(getSegmenterConfig).mockResolvedValue(config)
    const wrapper = mount(SegmenterPanel)
    await flushPromises()
    expect(wrapper.text()).toContain('不能视为目标模型就绪')
    expect(wrapper.text()).toContain('后端：contour')
    expect(wrapper.text()).toContain('SAM2 dependency missing')
  })

  it('keeps a classifier hot-load failure visible even when the desired config is enabled', async () => {
    const initial = classifierConfig()
    const failed = classifierConfig()
    failed.enabled = true
    failed.runtime.pipelines_failed = 1
    failed.runtime.details = [{
      camera_id: 'camera-1', state: 'failed', backend: null,
      loaded: false, ready: false, degraded: false, reason: 'weights unavailable',
    }]
    vi.mocked(getClassifierConfig).mockResolvedValueOnce(initial).mockResolvedValueOnce(failed)
    vi.mocked(toggleModule).mockResolvedValue({
      key: 'classifier.enabled', value: true, restart_required: true,
      hot_reloadable: true, hot_reloaded: 0, hot_reload_failed: 1,
      pipelines_seen: 1, persisted: true, all_applied: false,
    })
    const wrapper = mount(ClassifierPanel)
    await flushPromises()
    await wrapper.find('[data-switch]').trigger('click')
    await flushPromises()
    expect(wrapper.text()).toContain('运行时未全部生效')
    expect(wrapper.text()).toContain('weights unavailable')
    expect(wrapper.text()).toContain('配置已启用')
    expect(message.success).not.toHaveBeenCalled()
  })

  it('rechecks authoritative configuration after a failed toggle request', async () => {
    vi.mocked(getClassifierConfig).mockResolvedValue(classifierConfig())
    vi.mocked(toggleModule).mockRejectedValue(new Error('request timed out'))
    const wrapper = mount(ClassifierPanel)
    await flushPromises()
    await wrapper.find('[data-switch]').trigger('click')
    await flushPromises()
    expect(getClassifierConfig).toHaveBeenCalledTimes(2)
    expect(wrapper.text()).toContain('运行时状态尚未确认')
    expect(wrapper.find('[data-switch]').attributes('aria-pressed')).toBe('false')
  })
})
