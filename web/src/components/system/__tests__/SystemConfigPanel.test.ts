import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { message } from 'ant-design-vue'

import SystemConfigPanel from '../SystemConfigPanel.vue'
import { useAuthStore } from '../../../stores/useAuthStore'
import {
  getAudioAlerts,
  listBackups,
  updateDetectionParams,
} from '../../../api'
import { getCameras } from '../../../api/cameras'

const antStubs = vi.hoisted(() => ({
  card: {
    props: ['title'],
    template: '<section class="ant-card"><h2>{{ title }}</h2><slot /></section>',
  },
  button: {
    props: ['type', 'loading', 'disabled'],
    emits: ['click'],
    template: '<button type="button" :disabled="disabled" @click="$emit(\'click\', $event)"><slot name="icon" /><slot /></button>',
  },
  inputNumber: {
    props: ['value', 'min', 'max', 'step'],
    emits: ['update:value'],
    template: '<input class="ant-input-number" type="number" :value="value" @input="$emit(\'update:value\', Number($event.target.value))" />',
  },
  slider: {
    props: ['value', 'min', 'max', 'step'],
    emits: ['update:value'],
    template: '<input data-test="anomaly-slider" type="range" :value="value" @input="$emit(\'update:value\', Number($event.target.value))" />',
  },
  table: {
    props: ['columns', 'dataSource'],
    template: '<div><slot v-for="record in dataSource" name="bodyCell" :record="record" :column="{ key: \'actions\' }" /></div>',
  },
  tag: {
    props: ['color'],
    template: '<span><slot /></span>',
  },
  typography: {
    Text: { template: '<span><slot /></span>' },
  },
  switch: {
    props: ['checked'],
    emits: ['update:checked'],
    template: '<button type="button" @click="$emit(\'update:checked\', !checked)">{{ checked ? "开" : "关" }}</button>',
  },
  select: {
    Option: {
      props: ['value'],
      template: '<option :value="value"><slot /></option>',
    },
    props: ['value'],
    emits: ['update:value'],
    template: '<select :value="value" @change="$emit(\'update:value\', $event.target.value)"><slot /></select>',
  },
  input: {
    props: ['value'],
    emits: ['update:value'],
    template: '<input :value="value" @input="$emit(\'update:value\', $event.target.value)" />',
  },
  space: { template: '<span><slot /></span>' },
  popconfirm: { template: '<span><slot /></span>' },
  tooltip: { template: '<span><slot /></span>' },
}))
let pinia: ReturnType<typeof createPinia>

vi.mock('../../../api', () => ({
  api: { get: vi.fn(), post: vi.fn() },
  reloadConfig: vi.fn(),
  saveConfig: vi.fn(),
  createBackup: vi.fn(),
  getAudioAlerts: vi.fn(),
  updateAudioAlerts: vi.fn(),
  updateDetectionParams: vi.fn(),
  updateNotifications: vi.fn(),
  testWebhook: vi.fn(),
  restartCamera: vi.fn(),
  clearLock: vi.fn(),
  listBackups: vi.fn(),
  restoreBackup: vi.fn(),
  deleteBackup: vi.fn(),
}))

vi.mock('../../../api/cameras', () => ({
  getCameras: vi.fn(),
}))

vi.mock('ant-design-vue', () => ({
  Card: antStubs.card,
  Space: antStubs.space,
  Button: antStubs.button,
  Typography: antStubs.typography,
  Switch: antStubs.switch,
  Select: antStubs.select,
  Input: antStubs.input,
  InputNumber: antStubs.inputNumber,
  Slider: antStubs.slider,
  Table: antStubs.table,
  Tag: antStubs.tag,
  Popconfirm: antStubs.popconfirm,
  Tooltip: antStubs.tooltip,
  message: {
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
  },
}))

vi.mock('@ant-design/icons-vue', () => ({
  ReloadOutlined: { template: '<span />' },
  SaveOutlined: { template: '<span />' },
  DeleteOutlined: { template: '<span />' },
  UndoOutlined: { template: '<span />' },
  SendOutlined: { template: '<span />' },
  LockOutlined: { template: '<span />' },
  PoweroffOutlined: { template: '<span />' },
}))

describe('SystemConfigPanel detection params', () => {
  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)
    useAuthStore().currentUser = { username: 'admin', role: 'admin' }
    vi.mocked(getAudioAlerts).mockReset()
    vi.mocked(getCameras).mockReset()
    vi.mocked(listBackups).mockReset()
    vi.mocked(updateDetectionParams).mockReset()
    vi.mocked(message.success).mockReset()
    vi.mocked(message.warning).mockReset()
    vi.mocked(message.error).mockReset()

    vi.mocked(getAudioAlerts).mockResolvedValue({ low: {}, medium: {}, high: {} })
    vi.mocked(getCameras).mockResolvedValue({ cameras: [{ camera_id: 'dev_cam', connected: true }] } as any)
    vi.mocked(listBackups).mockResolvedValue({ backups: [] })
  })

  it('submits changed detection thresholds and surfaces hot-reload feedback', async () => {
    vi.mocked(updateDetectionParams).mockResolvedValue({
      pipelines_updated: 1,
      anomaly_threshold: {
        changed: true,
        hot_reloaded: true,
        applied: 1,
        total: 1,
      },
      severity: {
        changed: true,
        hot_reloaded: false,
        applied: 0,
        total: 1,
      },
      temporal: { changed: false, hot_reloaded: false, applied: 0, total: 1 },
      suppression: { changed: false, hot_reloaded: false, applied: 0, total: 1 },
    } as any)

    const wrapper = mount(SystemConfigPanel, {
      global: { plugins: [pinia] },
    })
    await flushPromises()

    const inputs = wrapper.findAll('input.ant-input-number')
    await wrapper.find('[data-test="anomaly-slider"]').setValue(0.65)
    await inputs[0].setValue(0.25)
    await inputs[1].setValue(0.45)
    await inputs[2].setValue(0.75)
    await inputs[3].setValue(0.9)

    const saveButton = wrapper.findAll('button').find(button => button.text().includes('更新检测参数'))
    expect(saveButton).toBeTruthy()
    await saveButton!.trigger('click')
    await flushPromises()

    expect(updateDetectionParams).toHaveBeenCalledWith({
      anomaly_threshold: 0.65,
      sev_info: 0.25,
      sev_low: 0.45,
      sev_medium: 0.75,
      sev_high: 0.9,
    })
    expect(message.success).toHaveBeenCalledWith('异常阈值：✓ 已实时生效 (1/1)')
    expect(message.warning).toHaveBeenCalledWith('严重度阈值：⚠ 需要重启进程才生效')
  })

  it('allows engineers to edit config but keeps backup mutations admin-only', async () => {
    useAuthStore().currentUser = { username: 'engineer', role: 'engineer' }
    vi.mocked(listBackups).mockResolvedValue({
      backups: [
        {
          name: 'backup-20260525',
          created: '2026-05-25',
          size_mb: 1.2,
          has_db: true,
          has_configs: true,
          has_models: false,
        },
      ],
    })
    vi.mocked(updateDetectionParams).mockResolvedValue({
      pipelines_updated: 1,
      anomaly_threshold: { changed: false, hot_reloaded: false, applied: 0, total: 1 },
      severity: { changed: false, hot_reloaded: false, applied: 0, total: 1 },
      temporal: { changed: false, hot_reloaded: false, applied: 0, total: 1 },
      suppression: { changed: false, hot_reloaded: false, applied: 0, total: 1 },
    } as any)

    const wrapper = mount(SystemConfigPanel, {
      global: { plugins: [pinia] },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('重新加载配置')
    expect(wrapper.text()).toContain('保存当前配置')
    expect(wrapper.text()).toContain('更新检测参数')
    expect(wrapper.text()).toContain('保存音频配置')
    expect(wrapper.text()).not.toContain('立即备份')
    expect(wrapper.text()).not.toContain('恢复')

    const saveButton = wrapper.findAll('button').find(button => button.text().includes('更新检测参数'))
    expect(saveButton).toBeTruthy()
    await saveButton!.trigger('click')
    await flushPromises()

    expect(updateDetectionParams).toHaveBeenCalled()
  })
})
