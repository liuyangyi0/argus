import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import AlertsTable from '../AlertsTable.vue'
import { useAlertStore } from '../../../stores/useAlertStore'

let pinia: ReturnType<typeof createPinia>

vi.mock('../../../api', () => ({
  getAlertGroup: vi.fn(),
  getAlerts: vi.fn(),
  getCameras: vi.fn(),
  acknowledgeAlert: vi.fn(),
  markFalsePositive: vi.fn(),
  deleteAlert: vi.fn(),
  bulkDeleteAlerts: vi.fn(),
  bulkAcknowledge: vi.fn(),
  bulkFalsePositive: vi.fn(),
}))

vi.mock('../../../api/alerts', () => ({
  confirmRealAnomaly: vi.fn(),
}))

vi.mock('ant-design-vue', () => ({
  Tag: { template: '<span><slot /></span>' },
  Button: { template: '<button type="button"><slot name="icon" /><slot /></button>' },
  Space: { template: '<span><slot /></span>' },
  Typography: { Text: { template: '<span><slot /></span>' } },
  Tooltip: { template: '<span><slot /></span>' },
  Popover: { template: '<span><slot /><slot name="content" /></span>' },
  List: { Item: { template: '<span><slot /></span>' }, template: '<span><slot name="renderItem" /></span>' },
  Spin: { template: '<span />' },
  message: { error: vi.fn(), success: vi.fn() },
}))

vi.mock('@ant-design/icons-vue', () => ({
  CheckCircleOutlined: { template: '<span />' },
  StopOutlined: { template: '<span />' },
  DeleteOutlined: { template: '<span />' },
}))

vi.mock('../../common/ResponsiveTable.vue', () => ({
  default: {
    props: ['dataSource', 'customRow'],
    template: `
      <div>
        <div
          v-for="record in dataSource"
          :key="record.alert_id"
          data-test="alert-row"
          v-bind="customRow(record)"
        >
          <button type="button" class="nested-control">nested action</button>
        </div>
      </div>
    `,
  },
}))

vi.mock('../../common/DangerousActionConfirm.vue', () => ({
  default: { template: '<button type="button"><slot /></button>' },
}))

describe('AlertsTable keyboard interaction', () => {
  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)
  })

  it('opens a focused row with Enter or Space without hijacking nested controls', async () => {
    const record = {
      alert_id: 'alert-keyboard-1',
      camera_id: 'cam_01',
      severity: 'medium',
      workflow_status: 'new',
      anomaly_score: 0.8,
    }
    const store = useAlertStore()
    store.alerts = [record]
    store.totalAlerts = 1

    const wrapper = mount(AlertsTable, {
      global: { plugins: [pinia] },
    })
    const row = wrapper.get('[data-test="alert-row"]')

    expect(row.attributes('role')).toBe('button')
    expect(row.attributes('tabindex')).toBe('0')
    expect(row.attributes('aria-label')).toContain('alert-keyboard-1')

    await row.trigger('keydown', { key: 'Enter' })
    await row.trigger('keydown', { key: ' ' })

    expect(wrapper.emitted('select')).toEqual([[record], [record]])

    await wrapper.get('.nested-control').trigger('keydown', { key: 'Enter' })
    expect(wrapper.emitted('select')).toHaveLength(2)
  })
})
