import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia, storeToRefs } from 'pinia'
import { computed, defineComponent, nextTick } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import Alerts from '../Alerts.vue'
import { getAlerts, getCameras } from '../../api'
import { getAlert } from '../../api/alerts'
import { useAlertStore } from '../../stores/useAlertStore'

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
    const store = useAlertStore()
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
    const store = useAlertStore()
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
    setActivePinia(createPinia())
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
        plugins: [createPinia()],
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

  it('opens alert detail from query id even when the alert is outside the current list', async () => {
    wsHarness.route!.query = { id: 'deep-linked-alert' }
    vi.mocked(getAlert).mockResolvedValue(realtimeAlert({
      alert_id: 'deep-linked-alert',
      recording_status: 'complete',
    }) as any)

    const wrapper = mount(Alerts, {
      global: {
        plugins: [createPinia()],
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
        plugins: [createPinia()],
        stubs: {
          AlertsToolbar: { template: '<div data-test="alerts-toolbar" />' },
          AlertsTable: AlertsTableStub,
          AlertDetailPanel: AlertDetailPanelStub,
          ContentSkeleton: { template: '<div data-test="skeleton" />' },
        },
      },
    })
    await flushPromises()

    const store = useAlertStore()
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
        plugins: [createPinia()],
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
})
