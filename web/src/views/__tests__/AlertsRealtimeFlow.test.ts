import { mount, flushPromises, enableAutoUnmount } from '@vue/test-utils'
import { createPinia, setActivePinia, storeToRefs } from 'pinia'
import { computed, defineComponent, nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import Alerts from '../Alerts.vue'
import { getAlerts, getCameras } from '../../api'
import { getAlert } from '../../api/alerts'
import { useAlertStore } from '../../stores/useAlertStore'
import { ApiError } from '../../types/api'

enableAutoUnmount(afterEach)

const wsHarness = vi.hoisted(() => ({
  routerReplace: vi.fn(),
  route: null as { query: Record<string, string> } | null,
  options: undefined as undefined | {
    topics: string[]
    onMessage: (topic: string, data: any) => void
    fallbackPoll?: () => Promise<void>
    fallbackInterval?: number
  },
}))

let pinia: ReturnType<typeof createPinia>

vi.mock('vue-router', async () => {
  const vue = await vi.importActual<typeof import('vue')>('vue')
  wsHarness.route = vue.reactive({ query: {} as Record<string, string> })
  return {
    useRoute: () => wsHarness.route,
    useRouter: () => ({ replace: wsHarness.routerReplace }),
  }
})

vi.mock('../../composables/useWebSocket', () => ({
  useWebSocket: vi.fn((options) => {
    wsHarness.options = options
    return {}
  }),
}))

vi.mock('../../api', () => ({
  getAlerts: vi.fn(),
  getCameras: vi.fn(),
  acknowledgeAlert: vi.fn(),
  markFalsePositive: vi.fn(),
  deleteAlert: vi.fn(),
  bulkDeleteAlerts: vi.fn(),
  bulkAcknowledge: vi.fn(),
  bulkFalsePositive: vi.fn(),
}))

vi.mock('../../api/alerts', () => ({
  getAlert: vi.fn(),
}))

const AlertsTableStub = defineComponent({
  name: 'AlertsTable',
  emits: ['select'],
  setup() {
    const store = useAlertStore(pinia)
    const { alerts } = storeToRefs(store)
    return { alerts }
  },
  template: `
    <div data-test="alerts-table">
      <button
        v-for="alert in alerts"
        :key="alert.alert_id"
        class="alert-row"
        type="button"
        @click="$emit('select', alert)"
      >
        {{ alert.alert_id }} {{ alert.recording_status }}
      </button>
    </div>
  `,
})

const AlertDetailPanelStub = defineComponent({
  name: 'AlertDetailPanel',
  emits: ['close'],
  setup() {
    const store = useAlertStore(pinia)
    const { selectedAlert } = storeToRefs(store)
    const recordingState = computed(() => {
      if (!selectedAlert.value) return ''
      return `${selectedAlert.value.has_recording}:${selectedAlert.value.recording_status}`
    })
    return { selectedAlert, recordingState }
  },
  template: `
    <aside v-if="selectedAlert" data-test="alert-detail">
      <span>{{ selectedAlert.alert_id }}</span>
      <span>{{ recordingState }}</span>
    </aside>
  `,
})

function realtimeAlert(overrides: Record<string, any> = {}) {
  return {
    alert_id: 'live-alert-1',
    timestamp: '2026-05-25T00:00:00Z',
    created_at: '2026-05-25T00:00:00Z',
    camera_id: 'dev_cam',
    zone_id: 'DEFAULT',
    severity: 'medium',
    anomaly_score: 0.91,
    acknowledged: false,
    false_positive: false,
    has_recording: true,
    recording_status: 'recording',
    workflow_status: 'new',
    notes: '',
    snapshot_path: 'data/alerts/live/snapshot.jpg',
    heatmap_path: 'data/alerts/live/heatmap.jpg',
    ...overrides,
  }
}

describe('Alerts page realtime flow', () => {
  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)
    wsHarness.routerReplace.mockReset()
    wsHarness.route!.query = {}
    wsHarness.options = undefined
    vi.mocked(getAlerts).mockReset()
    vi.mocked(getCameras).mockReset()
    vi.mocked(getAlert).mockReset()
    vi.mocked(getAlerts).mockResolvedValue({ alerts: [], total: 0 } as any)
    vi.mocked(getCameras).mockResolvedValue({ cameras: [{ camera_id: 'dev_cam' }] } as any)
  })

  it('renders websocket alerts in the page list and opens their evidence detail', async () => {
    const wrapper = mount(Alerts, {
      global: {
        plugins: [pinia],
        stubs: {
          AlertsToolbar: { template: '<div data-test="alerts-toolbar" />' },
          AlertsTable: AlertsTableStub,
          AlertDetailPanel: AlertDetailPanelStub,
          ContentSkeleton: { template: '<div data-test="skeleton" />' },
        },
      },
    })
    await flushPromises()

    expect(wsHarness.options?.topics).toEqual(['alerts'])
    expect(wsHarness.options?.fallbackInterval).toBe(15000)
    expect(wrapper.find('[data-test="skeleton"]').exists()).toBe(false)

    wsHarness.options!.onMessage('alerts', realtimeAlert())
    await nextTick()

    const row = wrapper.find('.alert-row')
    expect(row.text()).toContain('live-alert-1')
    expect(row.text()).toContain('recording')

    await row.trigger('click')
    await nextTick()

    const detail = wrapper.find('[data-test="alert-detail"]')
    expect(detail.text()).toContain('live-alert-1')
    expect(detail.text()).toContain('true:recording')

    wsHarness.options!.onMessage('alerts', {
      alert_id: 'live-alert-1',
      has_recording: true,
      recording_status: 'complete',
    })
    await nextTick()

    expect(wrapper.find('[data-test="alert-detail"]').text()).toContain('true:complete')
  })

  it('refreshes a clicked row with the alert detail payload', async () => {
    vi.mocked(getAlerts).mockResolvedValue({
      alerts: [realtimeAlert({
        alert_id: 'row-alert',
        recording_status: 'recording',
      })],
      total: 1,
    } as any)
    vi.mocked(getAlert).mockResolvedValue(realtimeAlert({
      alert_id: 'row-alert',
      recording_status: 'complete',
    }) as any)

    const wrapper = mount(Alerts, {
      global: {
        plugins: [pinia],
        stubs: {
          AlertsToolbar: { template: '<div data-test="alerts-toolbar" />' },
          AlertsTable: AlertsTableStub,
          AlertDetailPanel: AlertDetailPanelStub,
          ContentSkeleton: { template: '<div data-test="skeleton" />' },
        },
      },
    })
    await flushPromises()

    await wrapper.find('.alert-row').trigger('click')
    await flushPromises()

    expect(getAlert).toHaveBeenCalledWith('row-alert')
    expect(wrapper.find('[data-test="alert-detail"]').text()).toContain('true:complete')
  })

  it('opens alert detail from query id even when the alert is outside the current list', async () => {
    wsHarness.route!.query = { id: 'deep-linked-alert' }
    vi.mocked(getAlert).mockResolvedValue(realtimeAlert({
      alert_id: 'deep-linked-alert',
      recording_status: 'complete',
    }) as any)

    const wrapper = mount(Alerts, {
      global: {
        plugins: [pinia],
        stubs: {
          AlertsToolbar: { template: '<div data-test="alerts-toolbar" />' },
          AlertsTable: AlertsTableStub,
          AlertDetailPanel: AlertDetailPanelStub,
          ContentSkeleton: { template: '<div data-test="skeleton" />' },
        },
      },
    })
    await flushPromises()

    expect(getAlert).toHaveBeenCalledWith('deep-linked-alert')
    const detail = wrapper.find('[data-test="alert-detail"]')
    expect(detail.text()).toContain('deep-linked-alert')
    expect(detail.text()).toContain('true:complete')
  })

  it('updates a deep-linked alert detail from websocket completion events', async () => {
    wsHarness.route!.query = { id: 'deep-linked-recording' }
    vi.mocked(getAlert).mockResolvedValue(realtimeAlert({
      alert_id: 'deep-linked-recording',
      recording_status: 'recording',
    }) as any)

    const wrapper = mount(Alerts, {
      global: {
        plugins: [pinia],
        stubs: {
          AlertsToolbar: { template: '<div data-test="alerts-toolbar" />' },
          AlertsTable: AlertsTableStub,
          AlertDetailPanel: AlertDetailPanelStub,
          ContentSkeleton: { template: '<div data-test="skeleton" />' },
        },
      },
    })
    await flushPromises()

    const store = useAlertStore(pinia)
    store.filters.severity = 'high'
    expect(wrapper.find('[data-test="alert-detail"]').text()).toContain('true:recording')

    wsHarness.options!.onMessage('alerts', {
      alert_id: 'deep-linked-recording',
      severity: 'medium',
      has_recording: true,
      recording_status: 'complete',
    })
    await nextTick()

    expect(wrapper.find('[data-test="alert-detail"]').text()).toContain('true:complete')
    expect(wrapper.findAll('.alert-row')).toHaveLength(0)
  })

  it('selects a websocket alert that matches a changed query id on the same page', async () => {
    const wrapper = mount(Alerts, {
      global: {
        plugins: [pinia],
        stubs: {
          AlertsToolbar: { template: '<div data-test="alerts-toolbar" />' },
          AlertsTable: AlertsTableStub,
          AlertDetailPanel: AlertDetailPanelStub,
          ContentSkeleton: { template: '<div data-test="skeleton" />' },
        },
      },
    })
    await flushPromises()

    wsHarness.route!.query = { id: 'ws-target' }
    await nextTick()
    wsHarness.options!.onMessage('alerts', realtimeAlert({
      alert_id: 'ws-target',
      recording_status: 'recording',
    }))
    await nextTick()

    const detail = wrapper.find('[data-test="alert-detail"]')
    expect(detail.text()).toContain('ws-target')
    expect(detail.text()).toContain('true:recording')
  })

  it('does not disguise an alert query failure as an empty result', async () => {
    vi.mocked(getAlerts).mockRejectedValue(new Error('alert database unavailable'))

    const wrapper = mount(Alerts, {
      global: {
        plugins: [pinia],
        stubs: {
          AlertsToolbar: { template: '<div data-test="alerts-toolbar" />' },
          AlertsTable: AlertsTableStub,
          AlertDetailPanel: AlertDetailPanelStub,
          ContentSkeleton: { template: '<div data-test="skeleton" />' },
        },
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('数据加载失败')
    expect(wrapper.text()).toContain('alert database unavailable')
    expect(wrapper.text()).not.toContain('没有符合条件的告警')
  })

  it('announces a missing deep-linked alert as not found', async () => {
    wsHarness.route!.query = { id: 'deleted-alert' }
    vi.mocked(getAlert).mockRejectedValue(new ApiError({
      httpStatus: 404,
      businessCode: 40400,
      message: 'alert not found',
    }))

    const wrapper = mount(Alerts, {
      global: {
        plugins: [pinia],
        stubs: {
          AlertsToolbar: { template: '<div data-test="alerts-toolbar" />' },
          AlertsTable: AlertsTableStub,
          AlertDetailPanel: AlertDetailPanelStub,
          ContentSkeleton: { template: '<div data-test="skeleton" />' },
        },
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('告警不存在或已被删除')
    expect(wrapper.text()).toContain('alert not found')
  })

  it('does not overwrite a newer websocket detail state with an older detail response', async () => {
    let finishDetail!: (value: any) => void
    vi.mocked(getAlert).mockReturnValue(new Promise(resolve => { finishDetail = resolve }))
    vi.mocked(getAlerts).mockResolvedValue({ alerts: [realtimeAlert()], total: 1 } as any)
    const wrapper = mount(Alerts, {
      global: {
        plugins: [pinia],
        stubs: {
          AlertsToolbar: true, AlertsTable: AlertsTableStub,
          AlertDetailPanel: AlertDetailPanelStub,
        },
      },
    })
    await flushPromises()
    await wrapper.find('.alert-row').trigger('click')

    wsHarness.options!.onMessage('alerts', {
      alert_id: 'live-alert-1', has_recording: true, recording_status: 'complete',
    })
    finishDetail(realtimeAlert({ recording_status: 'recording' }))
    await flushPromises()

    expect(wrapper.find('[data-test="alert-detail"]').text()).toContain('true:complete')
    wrapper.unmount()
  })

  it('reconciles a selected off-page detail when websocket requests a REST resync', async () => {
    wsHarness.route!.query = { id: 'off-page' }
    vi.mocked(getAlert)
      .mockResolvedValueOnce(realtimeAlert({ alert_id: 'off-page', recording_status: 'recording' }) as any)
      .mockResolvedValueOnce(realtimeAlert({ alert_id: 'off-page', recording_status: 'complete' }) as any)
    const wrapper = mount(Alerts, {
      global: {
        plugins: [pinia],
        stubs: {
          AlertsToolbar: true, AlertsTable: AlertsTableStub,
          AlertDetailPanel: AlertDetailPanelStub,
        },
      },
    })
    await flushPromises()

    await wsHarness.options!.fallbackPoll?.()
    await flushPromises()

    expect(getAlert).toHaveBeenCalledTimes(2)
    expect(wrapper.find('[data-test="alert-detail"]').text()).toContain('true:complete')
    wrapper.unmount()
  })

  it('removes a selected detail when REST reconciliation discovers a lost deletion event', async () => {
    wsHarness.route!.query = { id: 'deleted-off-page' }
    vi.mocked(getAlert)
      .mockResolvedValueOnce(realtimeAlert({ alert_id: 'deleted-off-page' }) as any)
      .mockRejectedValueOnce(new ApiError({
        httpStatus: 404, businessCode: 40400, message: 'alert was deleted',
      }))
    const wrapper = mount(Alerts, {
      global: {
        plugins: [pinia],
        stubs: {
          AlertsToolbar: true, AlertsTable: AlertsTableStub,
          AlertDetailPanel: AlertDetailPanelStub,
        },
      },
    })
    await flushPromises()
    expect(wrapper.find('[data-test="alert-detail"]').exists()).toBe(true)

    await wsHarness.options!.fallbackPoll?.()
    await flushPromises()

    expect(useAlertStore(pinia).selectedAlert).toBeNull()
    expect(wrapper.find('[data-test="alert-detail"]').exists()).toBe(false)
    expect(wrapper.text()).toContain('告警不存在或已被删除')
    wrapper.unmount()
  })
})
