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
  Popconfirm: {
    props: ['title', 'description'],
    emits: ['confirm'],
    template: '<span><slot /></span>',
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
    replay_ready: true,
    recording_status: 'recording',
    recording_unavailable_reason: null,
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

  it('keeps replay available but defaults to the trigger frame', async () => {
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
    expect(wrapper.find('[data-test="replay-player"]').exists()).toBe(false)

    const recordingTab = wrapper
      .findAll('button')
      .find(button => button.text() === '录像')
    expect(recordingTab).toBeTruthy()
    await recordingTab!.trigger('click')
    expect(wrapper.find('[data-test="replay-player"]').attributes('data-alert-id')).toBe('live-with-evidence')

    const replayButton = wrapper
      .findAll('button')
      .find(button => button.text().includes('查看录像'))
    expect(replayButton).toBeTruthy()
    await replayButton!.trigger('click')

    expect(routerPush).toHaveBeenCalledWith('/replay/live-with-evidence')
  })

  it('shows the trigger frame when replay artifacts are incomplete', () => {
    const store = useAlertStore()
    store.selectedAlert = {
      ...alertWithRecording(),
      has_recording: false,
      replay_ready: false,
      recording_status: 'incomplete',
      recording_unavailable_reason: 'missing_metadata',
    }
    const auth = useAuthStore()
    auth.currentUser = { username: 'operator', role: 'operator' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).not.toContain('查看录像')
    expect(wrapper.text()).toContain('录像索引缺失')
    expect(wrapper.text()).toContain('触发帧')
    expect(wrapper.find('[data-test="replay-player"]').exists()).toBe(false)
    expect(wrapper.find('.alert-detail-img').exists()).toBe(true)
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

  it('renders alert evidence metadata without requiring candidate evidence', async () => {
    const store = useAlertStore()
    store.selectedAlert = {
      ...alertWithRecording(),
      detection_type: 'new_object',
      evidence_summary: 'new_object, severity=medium, score=0.8421',
      evidence_json: {
        detection_type: 'new_object',
        frame_number: 17,
      },
      suppression_reason: 'duration_too_short',
      model_version_id: 'target-version',
    }
    const auth = useAuthStore()
    auth.currentUser = { username: 'operator', role: 'operator' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    const text = wrapper.text()
    expect(text).toContain('检测类型')
    expect(text).toContain('新增/遗留物')
    expect(text).toContain('触发帧')
    expect(text).toContain('#17')
    expect(text).toContain('证据摘要')
    expect(text).toContain('new_object, severity=medium, score=0.8421')
    expect(text).toContain('抑制原因')
    expect(text).toContain('duration_too_short')
    expect(text).toContain('证据回放')
    expect(text).toContain('打开回放')

    const replayLink = wrapper
      .findAll('button')
      .find(button => button.text().includes('打开回放'))
    expect(replayLink).toBeTruthy()
    await replayLink!.trigger('click')
    expect(routerPush).toHaveBeenCalledWith('/replay/live-with-evidence')
  })

  it('does not crash when optional evidence fields are missing', () => {
    const store = useAlertStore()
    store.selectedAlert = {
      alert_id: 'old-alert',
      timestamp: '2026-05-25T00:00:00Z',
      camera_id: 'cam_01',
      zone_id: 'DEFAULT',
      severity: 'low',
      anomaly_score: 0.71,
      acknowledged: false,
      false_positive: false,
      has_recording: false,
      recording_status: null,
      workflow_status: 'new',
      notes: '',
    }
    const auth = useAuthStore()
    auth.currentUser = { username: 'operator', role: 'operator' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('cam_01')
    expect(wrapper.text()).toContain('低')
  })

  it('renders candidate evidence from the alert detail payload', () => {
    const store = useAlertStore()
    store.selectedAlert = {
      ...alertWithRecording(),
      candidate_evidence: {
        event_count: 2,
        trigger_frame_index: 3,
        events: [
          {
            candidate_id: 'pc-near-trigger',
            candidate_type: 'persistent_change',
            decision: 'pending',
            stage: 'persistent_change',
            frame_index: 2,
            frame_number: 22,
            score: 0.87,
            suppression_reason: 'persistence_not_met',
            evidence: {
              changed_fraction: 0.0123,
              candidate_count: 1,
              candidates: [
                {
                  bbox: [10, 12, 20, 18],
                  persistence_seconds: 2.4,
                },
              ],
            },
          },
        ],
      },
    }
    const auth = useAuthStore()
    auth.currentUser = { username: 'operator', role: 'operator' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    const text = wrapper.text()
    expect(text).toContain('候选证据')
    expect(text).toContain('2 条')
    expect(text).toContain('触发帧 #3')
    expect(text).toContain('固定变化')
    expect(text).toContain('等待确认')
    expect(text).toContain('帧 #22')
    expect(text).toContain('分数 0.870')
    expect(text).toContain('变化面积 1.23%')
    expect(text).toContain('候选 1')
    expect(text).toContain('框 10, 12, 20, 18')
    expect(text).toContain('持续 2.4s')
    expect(text).toContain('抑制 persistence_not_met')
  })

  it('labels visible water leak alerts distinctly', () => {
    const store = useAlertStore()
    store.selectedAlert = {
      ...alertWithRecording(),
      category: 'water_leak',
    }
    const auth = useAuthStore()
    auth.currentUser = { username: 'operator', role: 'operator' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('可见漏水')
  })

  it('renders water leak candidate type and growth evidence', () => {
    const store = useAlertStore()
    store.selectedAlert = {
      ...alertWithRecording(),
      candidate_evidence: {
        event_count: 1,
        trigger_frame_index: 8,
        events: [
          {
            candidate_id: 'water-1',
            candidate_type: 'water_leak',
            decision: 'candidate',
            frame_index: 8,
            frame_number: 18,
            score: 0.91,
            evidence: {
              candidate_count: 1,
              candidates: [
                {
                  leak_type: 'pooling',
                  growth_area_px: 360,
                  bbox: [80, 90, 40, 30],
                  persistence_seconds: 1.5,
                },
              ],
            },
          },
        ],
      },
    }
    const auth = useAuthStore()
    auth.currentUser = { username: 'operator', role: 'operator' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    const text = wrapper.text()
    expect(text).toContain('可见漏水')
    expect(text).toContain('迹象 积水扩大')
    expect(text).toContain('增长 360 px')
    expect(text).toContain('持续 1.5s')
  })

  it('labels visible plume alerts distinctly', () => {
    const store = useAlertStore()
    store.selectedAlert = {
      ...alertWithRecording(),
      category: 'visible_plume',
    }
    const auth = useAuthStore()
    auth.currentUser = { username: 'operator', role: 'operator' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('可见烟雾/粉尘')
  })

  it('renders plume candidate type and diffuse evidence', () => {
    const store = useAlertStore()
    store.selectedAlert = {
      ...alertWithRecording(),
      candidate_evidence: {
        event_count: 1,
        trigger_frame_index: 9,
        events: [
          {
            candidate_id: 'plume-1',
            candidate_type: 'plume',
            decision: 'candidate',
            frame_index: 9,
            frame_number: 19,
            score: 0.92,
            evidence: {
              candidate_count: 1,
              candidates: [
                {
                  plume_type: 'smoke_or_dust',
                  growth_area_px: 420,
                  edge_density: 0.025,
                  bbox: [110, 70, 54, 36],
                  persistence_seconds: 1.4,
                },
              ],
            },
          },
        ],
      },
    }
    const auth = useAuthStore()
    auth.currentUser = { username: 'operator', role: 'operator' }

    const wrapper = mount(AlertDetailPanel, {
      global: {
        plugins: [pinia],
      },
    })

    const text = wrapper.text()
    expect(text).toContain('可见烟雾')
    expect(text).toContain('类型 烟雾/粉尘')
    expect(text).toContain('增长 420 px')
    expect(text).toContain('边缘密度 0.025')
    expect(text).toContain('持续 1.4s')
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
