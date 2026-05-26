import { mount, flushPromises } from '@vue/test-utils'
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { message } from 'ant-design-vue'

import Reports from '../Reports.vue'
import {
  downloadComplianceReport,
  getCameraDist,
  getDailyTrend,
  getFPTrend,
  getReportStats,
  getSeverityDist,
} from '../../api/reports'

vi.mock('echarts/core', () => ({ use: vi.fn() }))
vi.mock('echarts/charts', () => ({ BarChart: {}, LineChart: {}, PieChart: {} }))
vi.mock('echarts/renderers', () => ({ CanvasRenderer: {} }))
vi.mock('echarts/components', () => ({
  GridComponent: {},
  TooltipComponent: {},
  LegendComponent: {},
}))
vi.mock('vue-echarts', () => ({
  default: { name: 'VChart', template: '<div data-test="chart"></div>' },
}))
vi.mock('ant-design-vue', async () => {
  const actual = await vi.importActual<typeof import('ant-design-vue')>('ant-design-vue')
  return {
    ...actual,
    message: {
      success: vi.fn(),
      error: vi.fn(),
    },
  }
})
vi.mock('@ant-design/icons-vue', () => ({
  DownloadOutlined: { template: '<span />' },
}))

vi.mock('../../api/reports', () => ({
  getReportStats: vi.fn(),
  getDailyTrend: vi.fn(),
  getSeverityDist: vi.fn(),
  getCameraDist: vi.fn(),
  getFPTrend: vi.fn(),
  downloadComplianceReport: vi.fn(),
}))

const passthrough = { template: '<div><slot /></div>' }
const statisticStub = {
  props: ['title', 'value', 'suffix'],
  template: '<div class="stat"><span>{{ title }}</span><span>{{ value }}{{ suffix || "" }}</span></div>',
}
const buttonStub = {
  props: ['loading', 'type', 'size'],
  emits: ['click'],
  template: '<button type="button" :disabled="loading" @click="$emit(\'click\', $event)"><slot name="icon" /><slot /></button>',
}
const selectStub = {
  props: ['value'],
  emits: ['update:value', 'change'],
  template: '<select :value="value" @change="$emit(\'update:value\', Number($event.target.value)); $emit(\'change\', Number($event.target.value))"><slot /></select>',
  Option: {
    props: ['value'],
    template: '<option :value="value"><slot /></option>',
  },
}
const radioStub = {
  Group: {
    props: ['value'],
    emits: ['update:value'],
    template: `
      <div>
        <button type="button" data-radio-format="csv" @click="$emit('update:value', 'csv')">CSV</button>
        <button type="button" data-radio-format="pdf" @click="$emit('update:value', 'pdf')">PDF</button>
      </div>
    `,
  },
  Button: { template: '<button type="button"><slot /></button>' },
}
const typographyStub = {
  Title: { template: '<h1><slot /></h1>' },
  Text: { template: '<span><slot /></span>' },
}

describe('Reports evidence stats', () => {
  beforeAll(() => {
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    })
  })

  beforeEach(() => {
    vi.mocked(getReportStats).mockReset()
    vi.mocked(getDailyTrend).mockReset()
    vi.mocked(getSeverityDist).mockReset()
    vi.mocked(getCameraDist).mockReset()
    vi.mocked(getFPTrend).mockReset()
    vi.mocked(downloadComplianceReport).mockReset()
    vi.mocked(message.success).mockReset()
    vi.mocked(message.error).mockReset()
  })

  it('renders evidence coverage returned by reports API', async () => {
    vi.mocked(getReportStats).mockResolvedValue({
      total_alerts: 4,
      by_severity: { high: 1, medium: 2, low: 1, info: 0 },
      false_positive_count: 1,
      false_positive_rate: 25,
      acknowledged_count: 2,
      acknowledged_rate: 50,
      evidence: {
        total_alerts: 4,
        alerts_with_snapshot: 3,
        alerts_with_heatmap: 2,
        alerts_with_recording: 3,
        evidence_complete_count: 2,
        snapshot_rate: 75,
        heatmap_rate: 50,
        recording_rate: 75,
        evidence_complete_rate: 50,
      },
    })
    vi.mocked(getDailyTrend).mockResolvedValue({
      labels: ['2026-05-25'],
      high: [1],
      medium: [2],
      low: [1],
      info: [0],
    })
    vi.mocked(getSeverityDist).mockResolvedValue({ high: 1, medium: 2, low: 1, info: 0 })
    vi.mocked(getCameraDist).mockResolvedValue({ cameras: [{ camera_id: 'cam_01', count: 4 }] })
    vi.mocked(getFPTrend).mockResolvedValue({ labels: ['2026-05-25'], rates: [25] })

    const wrapper = mount(Reports, {
      global: {
        stubs: {
          Card: passthrough,
          Row: passthrough,
          Col: passthrough,
          Select: selectStub,
          'Select.Option': selectStub.Option,
          ASelect: selectStub,
          ASelectOption: selectStub.Option,
          Spin: passthrough,
          Button: buttonStub,
          Statistic: statisticStub,
          Empty: passthrough,
          Radio: radioStub,
          'Radio.Group': radioStub.Group,
          'Radio.Button': radioStub.Button,
          RadioGroup: radioStub.Group,
          RadioButton: radioStub.Button,
          ARadioGroup: radioStub.Group,
          ARadioButton: radioStub.Button,
          Typography: typographyStub,
          'Typography.Title': typographyStub.Title,
          'Typography.Text': typographyStub.Text,
          ContentSkeleton: true,
        },
      },
    })
    await flushPromises()

    const text = wrapper.text()
    expect(text).toContain('截图覆盖率')
    expect(text).toContain('75%')
    expect(text).toContain('3 / 4')
    expect(text).toContain('热力图覆盖率')
    expect(text).toContain('50%')
    expect(text).toContain('2 / 4')
    expect(text).toContain('Replay录像覆盖率')
    expect(text).toContain('完整证据率')
  })

  it('passes the selected report period to all scoped report APIs', async () => {
    vi.mocked(getReportStats).mockResolvedValue({
      total_alerts: 1,
      by_severity: { high: 1, medium: 0, low: 0, info: 0 },
      false_positive_count: 0,
      false_positive_rate: 0,
      acknowledged_count: 1,
      acknowledged_rate: 100,
      evidence: {
        total_alerts: 1,
        alerts_with_snapshot: 1,
        alerts_with_heatmap: 1,
        alerts_with_recording: 1,
        evidence_complete_count: 1,
        snapshot_rate: 100,
        heatmap_rate: 100,
        recording_rate: 100,
        evidence_complete_rate: 100,
      },
    })
    vi.mocked(getDailyTrend).mockResolvedValue({ labels: [], high: [], medium: [], low: [], info: [] })
    vi.mocked(getSeverityDist).mockResolvedValue({ high: 1, medium: 0, low: 0, info: 0 })
    vi.mocked(getCameraDist).mockResolvedValue({ cameras: [{ camera_id: 'cam_01', count: 1 }] })
    vi.mocked(getFPTrend).mockResolvedValue({ labels: [], rates: [] })

    const wrapper = mount(Reports, {
      global: {
        stubs: {
          Card: passthrough,
          Row: passthrough,
          Col: passthrough,
          Select: selectStub,
          'Select.Option': selectStub.Option,
          ASelect: selectStub,
          ASelectOption: selectStub.Option,
          Spin: passthrough,
          Button: buttonStub,
          Statistic: statisticStub,
          Empty: passthrough,
          Radio: radioStub,
          'Radio.Group': radioStub.Group,
          'Radio.Button': radioStub.Button,
          RadioGroup: radioStub.Group,
          RadioButton: radioStub.Button,
          ARadioGroup: radioStub.Group,
          ARadioButton: radioStub.Button,
          Typography: typographyStub,
          'Typography.Title': typographyStub.Title,
          'Typography.Text': typographyStub.Text,
          ContentSkeleton: true,
        },
      },
    })
    await flushPromises()

    expect(getReportStats).toHaveBeenCalledWith(30)
    expect(getDailyTrend).toHaveBeenCalledWith(30)
    expect(getSeverityDist).toHaveBeenCalledWith(30)
    expect(getCameraDist).toHaveBeenCalledWith(30)
    expect(getFPTrend).toHaveBeenCalledWith(30)

    await wrapper.find('select').setValue('7')
    await flushPromises()

    expect(getReportStats).toHaveBeenLastCalledWith(7)
    expect(getDailyTrend).toHaveBeenLastCalledWith(7)
    expect(getSeverityDist).toHaveBeenLastCalledWith(7)
    expect(getCameraDist).toHaveBeenLastCalledWith(7)
    expect(getFPTrend).toHaveBeenLastCalledWith(7)
  })

  it('downloads a compliance report using the selected compliance period and format', async () => {
    vi.mocked(getReportStats).mockResolvedValue({
      total_alerts: 0,
      by_severity: { high: 0, medium: 0, low: 0, info: 0 },
      false_positive_count: 0,
      false_positive_rate: 0,
      acknowledged_count: 0,
      acknowledged_rate: 0,
      evidence: {
        total_alerts: 0,
        alerts_with_snapshot: 0,
        alerts_with_heatmap: 0,
        alerts_with_recording: 0,
        evidence_complete_count: 0,
        snapshot_rate: 0,
        heatmap_rate: 0,
        recording_rate: 0,
        evidence_complete_rate: 0,
      },
    })
    vi.mocked(getDailyTrend).mockResolvedValue({ labels: [], high: [], medium: [], low: [], info: [] })
    vi.mocked(getSeverityDist).mockResolvedValue({ high: 0, medium: 0, low: 0, info: 0 })
    vi.mocked(getCameraDist).mockResolvedValue({ cameras: [] })
    vi.mocked(getFPTrend).mockResolvedValue({ labels: [], rates: [] })
    vi.mocked(downloadComplianceReport).mockResolvedValue()

    const wrapper = mount(Reports, {
      global: {
        stubs: {
          Card: passthrough,
          Row: passthrough,
          Col: passthrough,
          Select: selectStub,
          'Select.Option': selectStub.Option,
          ASelect: selectStub,
          ASelectOption: selectStub.Option,
          Spin: passthrough,
          Button: buttonStub,
          Statistic: statisticStub,
          Empty: passthrough,
          Radio: radioStub,
          'Radio.Group': radioStub.Group,
          'Radio.Button': radioStub.Button,
          RadioGroup: radioStub.Group,
          RadioButton: radioStub.Button,
          ARadioGroup: radioStub.Group,
          ARadioButton: radioStub.Button,
          Typography: typographyStub,
          'Typography.Title': typographyStub.Title,
          'Typography.Text': typographyStub.Text,
          ContentSkeleton: true,
        },
      },
    })
    await flushPromises()

    const selects = wrapper.findAll('select')
    expect(selects.length).toBeGreaterThanOrEqual(2)
    await selects[1].setValue('14')
    await wrapper.find('[data-radio-format="pdf"]').trigger('click')
    const downloadButton = wrapper.findAll('button').find(button => button.text().includes('下载报告'))
    expect(downloadButton).toBeTruthy()
    await downloadButton!.trigger('click')
    await flushPromises()

    expect(downloadComplianceReport).toHaveBeenCalledWith(14, 'pdf')
    expect(message.success).toHaveBeenCalledWith('报告下载已开始')
  })
})
