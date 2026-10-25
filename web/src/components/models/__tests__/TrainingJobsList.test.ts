import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { message } from 'ant-design-vue'

import TrainingJobsList from '../TrainingJobsList.vue'
import {
  confirmTrainingJob,
  getTrainingJob,
  getTrainingJobs,
  rejectTrainingJob,
} from '../../../api'

const antStubs = vi.hoisted(() => ({
  card: { template: '<section><slot /></section>' },
  alert: {
    props: ['message', 'description', 'type'],
    template: '<section class="alert" :data-type="type">{{ message }} {{ description }}</section>',
  },
  button: {
    props: ['type', 'size', 'danger'],
    emits: ['click'],
    template: '<button type="button" @click="$emit(\'click\', $event)"><slot name="icon" /><slot /></button>',
  },
  tag: {
    props: ['color'],
    template: '<span class="ant-tag" :data-color="color"><slot name="icon" /><slot /></span>',
  },
  space: { template: '<span><slot /></span>' },
  select: {
    Option: {
      props: ['value'],
      template: '<option :value="value"><slot /></option>',
    },
    props: ['value', 'placeholder'],
    emits: ['update:value', 'change'],
    template: '<select :value="value" @change="$emit(\'update:value\', $event.target.value); $emit(\'change\', $event.target.value)"><slot /></select>',
  },
  popconfirm: {
    emits: ['confirm'],
    template: '<span class="popconfirm" @click="$emit(\'confirm\')"><slot /></span>',
  },
  table: {
    props: ['columns', 'dataSource'],
    template: `
      <div>
        <div v-for="record in dataSource" :key="record.job_id" class="job-row" :data-job-id="record.job_id">
          <div v-for="column in columns" :key="column.key" :data-col="column.key">
            <slot name="bodyCell" :column="column" :record="record" />
            <template v-if="column.dataIndex">{{ record[column.dataIndex] }}</template>
          </div>
        </div>
      </div>
    `,
  },
  drawer: {
    props: ['open', 'title'],
    template: '<section v-if="open" data-test="job-detail"><h2>{{ title }}</h2><slot /></section>',
  },
  descriptions: {
    Item: {
      props: ['label'],
      template: '<div>{{ label }} <slot /></div>',
    },
    template: '<div><slot /></div>',
  },
  typography: {
    Title: {
      props: ['level'],
      template: '<h3><slot /></h3>',
    },
  },
}))

vi.mock('../../../api', () => ({
  getTrainingJobs: vi.fn(),
  getTrainingJob: vi.fn(),
  confirmTrainingJob: vi.fn(),
  rejectTrainingJob: vi.fn(),
}))

vi.mock('ant-design-vue', () => ({
  Card: antStubs.card,
  Alert: antStubs.alert,
  Table: antStubs.table,
  Button: antStubs.button,
  Tag: antStubs.tag,
  Space: antStubs.space,
  Select: antStubs.select,
  Popconfirm: antStubs.popconfirm,
  Drawer: antStubs.drawer,
  Descriptions: antStubs.descriptions,
  Typography: antStubs.typography,
  message: {
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
}))

vi.mock('@ant-design/icons-vue', () => ({
  CheckOutlined: { template: '<span />' },
  CloseOutlined: { template: '<span />' },
  ReloadOutlined: { template: '<span />' },
  ThunderboltOutlined: { template: '<span />' },
  ClockCircleOutlined: { template: '<span />' },
}))

vi.mock('../../common/ErrorDetailModal.vue', () => ({
  default: {
    name: 'ErrorDetailModal',
    template: '<div data-test="error-detail-modal" />',
  },
}))

function trainingJob(overrides: Record<string, any> = {}) {
  return {
    id: 1,
    job_id: 'job-pending-1',
    job_type: 'anomaly_head',
    camera_id: 'dev_cam',
    zone_id: 'default',
    model_type: 'patchcore',
    trigger_type: 'manual',
    triggered_by: 'operator',
    confirmation_required: true,
    confirmed_by: null,
    confirmed_at: null,
    status: 'pending_confirmation',
    base_model_version: null,
    dataset_version: null,
    hyperparameters: null,
    metrics: null,
    artifacts_path: null,
    validation_report: null,
    model_version_id: null,
    created_at: '2026-05-25T00:00:00Z',
    started_at: null,
    completed_at: null,
    duration_seconds: null,
    error: null,
    ...overrides,
  }
}

describe('TrainingJobsList', () => {
  beforeEach(() => {
    vi.mocked(getTrainingJobs).mockReset()
    vi.mocked(getTrainingJob).mockReset()
    vi.mocked(confirmTrainingJob).mockReset()
    vi.mocked(rejectTrainingJob).mockReset()
    vi.mocked(message.success).mockReset()
    vi.mocked(message.warning).mockReset()
    vi.mocked(message.error).mockReset()
    vi.mocked(message.info).mockReset()
  })

  it('shows pending jobs, emits pending count, and confirms a training job', async () => {
    vi.mocked(getTrainingJobs)
      .mockResolvedValueOnce({ jobs: [trainingJob()], pending_count: 1 })
      .mockResolvedValueOnce({
        jobs: [trainingJob({
          status: 'queued',
          confirmed_by: 'operator',
          confirmed_at: '2026-05-25T00:01:00Z',
        })],
        pending_count: 0,
      })
    vi.mocked(confirmTrainingJob).mockResolvedValue({ ok: true } as any)

    const wrapper = mount(TrainingJobsList, {
      props: {
        cameras: [{ camera_id: 'dev_cam', name: 'Dev camera' }] as any,
      },
    })
    await flushPromises()

    expect(wrapper.emitted('update:pendingCount')?.[0]).toEqual([1])
    const row = wrapper.find('[data-job-id="job-pending-1"]')
    expect(row.text()).toContain('异常检测头')
    expect(row.text()).toContain('待确认')

    const confirmButton = row.findAll('button').find(button => button.text().includes('确认'))
    expect(confirmButton).toBeTruthy()
    await confirmButton!.trigger('click')
    await flushPromises()

    expect(confirmTrainingJob).toHaveBeenCalledWith('job-pending-1')
    expect(message.success).toHaveBeenCalledWith('任务已确认，进入队列')
    expect(wrapper.emitted('update:pendingCount')?.at(-1)).toEqual([0])
    expect(wrapper.text()).toContain('排队中')
  })

  it('refreshes the list after a stale confirmation conflict', async () => {
    vi.mocked(getTrainingJobs)
      .mockResolvedValueOnce({ jobs: [trainingJob()], pending_count: 1 })
      .mockResolvedValueOnce({ jobs: [], pending_count: 0 })
    vi.mocked(confirmTrainingJob).mockRejectedValue({ code: 409 })

    const wrapper = mount(TrainingJobsList, {
      props: {
        cameras: [] as any,
      },
    })
    await flushPromises()

    const confirmButton = wrapper.findAll('button').find(button => button.text().includes('确认'))
    expect(confirmButton).toBeTruthy()
    await confirmButton!.trigger('click')
    await flushPromises()

    expect(message.warning).toHaveBeenCalledWith('任务已被其他操作员确认或状态已变更')
    expect(getTrainingJobs).toHaveBeenCalledTimes(2)
    expect(wrapper.emitted('update:pendingCount')?.at(-1)).toEqual([0])
  })

  async function openQualityDetail(metrics: unknown, status = 'complete') {
    const job = trainingJob({ status, metrics })
    vi.mocked(getTrainingJobs).mockResolvedValue({ jobs: [job], pending_count: 0 })
    vi.mocked(getTrainingJob).mockResolvedValue(job)
    const wrapper = mount(TrainingJobsList, { props: { cameras: [] } })
    await flushPromises()
    const detail = wrapper.findAll('button').find(button => button.text() === '详情')
    await detail!.trigger('click')
    await flushPromises()
    return wrapper
  }

  it('shows narrow fixed-scene suitability and warnings without presenting production approval', async () => {
    const wrapper = await openQualityDetail({ baseline_quality: {
      report_status: 'available', passed: true, integrity_passed: true,
      policy_version: 'argus.baseline-quality.scene-static.v1', scope: 'observed_fixed_scene_only',
      warnings: ['静态素材没有覆盖照明变化'], errors: [], production_readiness: 'not_evaluated',
    } })
    const state = wrapper.get('[data-test="baseline-quality-state"]')
    expect(state.text()).toBe('训练输入可用（有限覆盖）')
    expect(state.attributes('data-color')).toBe('orange')
    expect(wrapper.text()).toContain('仅覆盖已观察到的固定场景正常工况')
    expect(wrapper.text()).toContain('静态素材没有覆盖照明变化')
    expect(wrapper.text()).toContain('训练输入可用不等于生产验收通过')
  })

  it.each([null, {}, { baseline_quality: { passed: true } }, {
    baseline_quality: { report_status: 'invalid', passed: true, warnings: [], errors: [] },
  }, {
    baseline_quality: { report_status: 'available', passed: true, warnings: [], errors: ['完整性失败'] },
  }, {
    baseline_quality: { report_status: 'available', passed: true, warnings: [], errors: [], scope: 'observed_fixed_scene_only' },
  }])('never treats an absent or malformed quality report as passing: %j', async (metrics) => {
    const wrapper = await openQualityDetail(metrics)
    const state = wrapper.get('[data-test="baseline-quality-state"]')
    expect(state.text()).toBe('未评估或报告无效')
    expect(state.attributes('data-color')).toBe('default')
  })

  it('keeps failed input evidence and its warnings visible on failed training jobs', async () => {
    const wrapper = await openQualityDetail({ baseline_quality: {
      report_status: 'available', passed: false, scope: 'observed_fixed_scene_only',
      warnings: ['仅验证当前场景'], errors: ['图片身份已改变'],
    } }, 'failed')
    const state = wrapper.get('[data-test="baseline-quality-state"]')
    expect(state.text()).toBe('不满足训练输入要求')
    expect(state.attributes('data-color')).toBe('red')
    expect(wrapper.text()).toContain('图片身份已改变')
    expect(wrapper.text()).toContain('仅验证当前场景')
  })
})
