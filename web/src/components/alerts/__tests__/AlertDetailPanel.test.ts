import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import AlertDetailPanel from '../AlertDetailPanel.vue'
import { useAlertStore } from '../../../stores/useAlertStore'
import { useAuthStore } from '../../../stores/useAuthStore'

const routerPush = vi.hoisted(() => vi.fn())
const downloadEvidencePackage = vi.hoisted(() => vi.fn())
let pinia: ReturnType<typeof createPinia>

vi.mock('vue-router', () => ({
  useRouter: () => ({
    push: routerPush,
  }),
}))

vi.mock('../../../api/alerts', () => ({
  downloadEvidencePackage,
}))

vi.mock('../../../api', () => ({
  api: { get: vi.fn() },
  getAlerts: vi.fn(),
  getCameras: vi.fn(),
  acknowledgeAlert: vi.fn(),
  markFalsePositive: vi.fn(),
  deleteAlert: vi.fn(),
  bulkDeleteAlerts: vi.fn(),
  bulkAcknowledge: vi.fn(),
  bulkFalsePositive: vi.fn(),
}))

vi.mock('ant-design-vue', () => ({
  Tag: {
    props: ['color'],
    template: '<span class="ant-tag" :data-color="color"><slot /></span>',
  },
  Button: {
    props: ['type', 'size', 'block', 'danger'],
    emits: ['click'],
    template: '<button type="button" @click="$emit(\'click\', $event)"><slot name="icon" /><slot /></button>',
  },
  Typography: {
    Text: { template: '<span><slot /></span>' },
  },
  Tooltip: {
    props: ['title'],
    template: '<span><slot /></span>',
  },
  Segmented: {
    props: ['options', 'value'],
    emits: ['update:value'],
    template: `
      <div class="ant-segmented">
        <button
          v-for="option in options"
          :key="option.value"
          type="button"
          @click="$emit('update:value', option.value)"
        >
          {{ option.label }}
        </button>
      </div>
    `,
  },
  message: {
    error: vi.fn(),
    success: vi.fn(),
  },
  Modal: {
    confirm: vi.fn(),
  },
}))

vi.mock('@ant-design/icons-vue', () => ({
  CloseOutlined: { template: '<span />' },
  CheckCircleOutlined: { template: '<span />' },
  StopOutlined: { template: '<span />' },
  ExportOutlined: { template: '<span />' },
  DeleteOutlined: { template: '<span />' },
  PlayCircleOutlined: { template: '<span />' },
  AppstoreOutlined: { template: '<span />' },
}))

vi.mock('../../ReplayPlayer.vue', () => ({
  default: {
    name: 'ReplayPlayer',
    props: ['alertId'],
    template: '<div data-test="replay-player" :data-alert-id="alertId">Replay {{ alertId }}</div>',
  },
}))

vi.mock('../../AnnotationOverlay.vue', () => ({
  default: {
    name: 'AnnotationOverlay',
    template: '<div data-test="annotation-overlay" />',
  },
}))

vi.mock('../../ImageCompareSlider.vue', () => ({
  default: {
    name: 'ImageCompareSlider',
    template: '<div data-test="image-compare-slider" />',
  },
}))

function alertWithRecording() {
  return {
    alert_id: 'live-with-evidence',
    timestamp: '2026-05-25T00:00:00Z',
    created_at: '2026-05-25T00:00:00Z',
    camera_id: 'cam_01',
    zone_id: 'DEFAULT',
    severity: 'medium',
    anomaly_score: 0.8421,
    acknowledged: false,
    false_positive: false,
    has_recording: true,
    recording_status: 'recording',
    workflow_status: 'new',
    notes: '',
    snapshot_path: 'data/alerts/live/snapshot.jpg',
    heatmap_path: 'data/alerts/live/heatmap.jpg',
  }
}

describe('AlertDetailPanel replay entry', () => {
  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)
    routerPush.mockReset()
    downloadEvidencePackage.mockReset()
  })

  it('keeps the replay entry visible for realtime alerts with recording evidence', async () => {
    const store = useAlertStore()
    store.selectedAlert = alertWithRecording()
    const auth = useAuthStore()
    auth.currentUser = { username: 'operator', role: 'operator' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('查看录像')
    expect(wrapper.text()).toContain('录像')
    expect(wrapper.text()).toContain('触发帧')
    expect(wrapper.find('[data-test="replay-player"]').attributes('data-alert-id')).toBe('live-with-evidence')

    const replayButton = wrapper
      .findAll('button')
      .find(button => button.text().includes('查看录像'))
    expect(replayButton).toBeTruthy()
    await replayButton!.trigger('click')

    expect(routerPush).toHaveBeenCalledWith('/replay/live-with-evidence')
  })

  it('links the triggering model version to the focused registry view for engineers', async () => {
    const store = useAlertStore()
    store.selectedAlert = {
      ...alertWithRecording(),
      model_version_id: 'target-version',
    }
    const auth = useAuthStore()
    auth.currentUser = { username: 'engineer', role: 'engineer' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    const modelButton = wrapper
      .findAll('button')
      .find(button => button.text().includes('target-version'))
    expect(modelButton).toBeTruthy()
    await modelButton!.trigger('click')

    expect(routerPush).toHaveBeenCalledWith({
      path: '/models/registry',
      query: { version_id: 'target-version' },
    })
  })

  it('shows the triggering model as read-only metadata for operators', async () => {
    const store = useAlertStore()
    store.selectedAlert = {
      ...alertWithRecording(),
      model_version_id: 'target-version',
    }
    const auth = useAuthStore()
    auth.currentUser = { username: 'operator', role: 'operator' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('触发模型')
    expect(wrapper.text()).toContain('target-version')
    const modelButton = wrapper
      .findAll('button')
      .find(button => button.text().includes('target-version'))
    expect(modelButton).toBeUndefined()
    expect(routerPush).not.toHaveBeenCalled()
  })

  it('allows engineers to mutate alert workflow state', () => {
    const store = useAlertStore()
    store.selectedAlert = alertWithRecording()
    const auth = useAuthStore()
    auth.currentUser = { username: 'engineer', role: 'engineer' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('确认真实')
    expect(wrapper.text()).toContain('标记误报')
    expect(wrapper.text()).toContain('删除告警')
  })
})
