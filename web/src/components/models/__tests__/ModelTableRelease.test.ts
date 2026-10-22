import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { message } from 'ant-design-vue'

import ModelTable from '../ModelTable.vue'
import { useWebSocket } from '../../../composables/useWebSocket'
import type { ReleaseOperation } from '../../../types/api'
import {
  getShadowReport,
  getReleaseOperation,
  promoteModel,
  reexportModel,
  retireModel,
  rollbackModel,
} from '../../../api'

const routerPush = vi.hoisted(() => vi.fn())
const modalConfirm = vi.hoisted(() => vi.fn())
const antStubs = vi.hoisted(() => ({
  button: {
    props: ['type', 'size', 'danger', 'ghost', 'loading', 'disabled'],
    emits: ['click'],
    template: '<button type="button" :disabled="disabled" @click="$emit(\'click\', $event)"><slot name="icon" /><slot /></button>',
  },
  card: { template: '<section><slot name="title" /><slot /></section>' },
  tag: {
    props: ['color'],
    template: '<span class="ant-tag" :data-color="color"><slot /></span>',
  },
  space: { template: '<span><slot /></span>' },
  tooltip: { template: '<span><slot /></span>' },
  dropdown: { template: '<span><slot /><slot name="overlay" /></span>' },
  steps: {
    Step: {
      props: ['title', 'description'],
      template: '<span>{{ title }} {{ description }}</span>',
    },
    template: '<div><slot /></div>',
  },
  table: {
    props: ['columns', 'dataSource', 'rowClassName'],
    template: `
      <div>
        <div
          v-for="record in dataSource"
          :key="record.model_version_id"
          class="model-row"
          :class="rowClassName ? rowClassName(record) : ''"
          :data-vid="record.model_version_id"
        >
          <div v-for="column in columns" :key="column.key" :data-col="column.key">
            <slot name="bodyCell" :column="column" :record="record" />
          </div>
        </div>
      </div>
    `,
  },
  modal: {
    props: ['open', 'title', 'okText'],
    emits: ['ok'],
    template: `
      <section v-if="open" class="ant-modal">
        <h2>{{ title }}</h2>
        <slot />
        <button type="button" data-test="modal-ok" @click="$emit('ok')">{{ okText || 'OK' }}</button>
      </section>
    `,
  },
  form: {
    Item: {
      props: ['label'],
      template: '<label><span>{{ label }}</span><slot /></label>',
    },
    template: '<form><slot /></form>',
  },
  select: {
    Option: {
      props: ['value'],
      template: '<option :value="value"><slot /></option>',
    },
    props: ['value'],
    emits: ['update:value', 'change'],
    template: '<select :value="value" @change="$emit(\'update:value\', $event.target.value); $emit(\'change\', $event.target.value)"><slot /></select>',
  },
  input: {
    TextArea: {
      props: ['value'],
      emits: ['update:value'],
      template: '<textarea :value="value" @input="$emit(\'update:value\', $event.target.value)" />',
    },
    props: ['value'],
    emits: ['update:value'],
    template: '<input :value="value" @input="$emit(\'update:value\', $event.target.value)" />',
  },
  menu: {
    Item: {
      template: '<button type="button" class="menu-item"><slot /></button>',
    },
    Divider: { template: '<hr />' },
    emits: ['click'],
    methods: {
      menuKey(event: Event) {
        const text = ((event.target as HTMLElement).textContent || '').trim()
        if (text.includes('重新导出')) return 'reexport'
        if (text.includes('重新校准')) return 'recalibrate'
        if (text.includes('影子报告')) return 'shadow-report'
        if (text.includes('A/B')) return 'ab-compare'
        if (text.includes('阶段历史')) return 'stage-history'
        if (text.includes('退役')) return 'retire'
        if (text.includes('删除')) return 'delete'
        return undefined
      },
    },
    template: '<div @click="$emit(\'click\', { key: menuKey($event) })"><slot /></div>',
  },
  drawer: {
    props: ['open', 'title'],
    template: '<section v-if="open"><h2>{{ title }}</h2><slot /></section>',
  },
  descriptions: {
    Item: {
      props: ['label'],
      template: '<div>{{ label }} <slot /></div>',
    },
    template: '<div><slot /></div>',
  },
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush }),
}))

vi.mock('../../../composables/useWebSocket', () => ({
  useWebSocket: vi.fn(),
}))

vi.mock('../../../api', () => ({
  promoteModel: vi.fn(),
  reexportModel: vi.fn(),
  rollbackModel: vi.fn(),
  deleteModel: vi.fn(),
  retireModel: vi.fn(),
  getStageHistory: vi.fn().mockResolvedValue({ events: [] }),
  getShadowReport: vi.fn(),
  getReleaseOperation: vi.fn(),
  acknowledgeReleaseOperation: vi.fn(),
  recalibrateModel: vi.fn(),
  getCameras: vi.fn(),
  getTasks: vi.fn(),
  dismissTask: vi.fn(),
}))

vi.mock('ant-design-vue', () => ({
  Card: antStubs.card,
  Table: antStubs.table,
  Button: antStubs.button,
  Tag: antStubs.tag,
  Space: antStubs.space,
  Modal: Object.assign(antStubs.modal, { confirm: modalConfirm }),
  Form: antStubs.form,
  Select: antStubs.select,
  Input: antStubs.input,
  Descriptions: antStubs.descriptions,
  Drawer: antStubs.drawer,
  Steps: antStubs.steps,
  Tooltip: antStubs.tooltip,
  Dropdown: antStubs.dropdown,
  Menu: antStubs.menu,
  message: {
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
}))

vi.mock('@ant-design/icons-vue', () => ({
  ReloadOutlined: { template: '<span />' },
  RollbackOutlined: { template: '<span />' },
  DeleteOutlined: { template: '<span />' },
  ExperimentOutlined: { template: '<span />' },
  HistoryOutlined: { template: '<span />' },
  LoadingOutlined: { template: '<span />' },
  DownOutlined: { template: '<span />' },
  ExportOutlined: { template: '<span />' },
  AimOutlined: { template: '<span />' },
  SwapOutlined: { template: '<span />' },
}))

function models() {
  return [
    {
      model_version_id: 'candidate-1',
      camera_id: 'dev_cam',
      model_type: 'patchcore',
      stage: 'candidate',
      is_active: false,
      runtime_state: 'waiting',
      created_at: '2026-05-25T00:00:00Z',
    },
    {
      model_version_id: 'production-1',
      camera_id: 'dev_cam',
      model_type: 'patchcore',
      stage: 'production',
      is_active: true,
      runtime_state: 'applied',
      created_at: '2026-05-25T00:10:00Z',
    },
  ] as any[]
}

function mountTable(props: Record<string, any> = {}) {
  return mount(ModelTable, {
    props: {
      models: models(),
      cameras: [{ camera_id: 'dev_cam', name: 'Dev camera' }] as any,
      ...props,
    },
  })
}

function operation(overrides: Partial<ReleaseOperation> = {}): ReleaseOperation {
  return {
    operation_id: 'operation-new',
    model_version_id: 'candidate-1',
    camera_id: 'dev_cam',
    from_stage: 'candidate',
    target_stage: 'shadow',
    expected_revision: 4,
    applied_revision: null,
    status: 'pending',
    requested_by: 'tester',
    requested_at: '2026-10-21T10:00:00Z',
    ...overrides,
  }
}

function sendReleaseEvent(value: ReleaseOperation) {
  const callback = vi.mocked(useWebSocket).mock.calls.at(-1)?.[0].onMessage
  callback?.('model_release', { type: `release_operation_${value.status}`, operation: value })
}

describe('ModelTable release actions', () => {
  beforeEach(() => {
    vi.mocked(promoteModel).mockReset()
    vi.mocked(getShadowReport).mockReset()
    vi.mocked(getReleaseOperation).mockReset()
    vi.mocked(reexportModel).mockReset()
    vi.mocked(retireModel).mockReset()
    vi.mocked(rollbackModel).mockReset()
    routerPush.mockReset()
    modalConfirm.mockReset()
    vi.mocked(message.success).mockReset()
    vi.mocked(message.warning).mockReset()
    vi.mocked(message.error).mockReset()
    vi.mocked(message.info).mockReset()
    vi.mocked(useWebSocket).mockClear()
  })

  it('restores and polls a durable pending operation discovered on a fresh registry read', async () => {
    vi.useFakeTimers()
    const pending = operation()
    const applied = operation({ status: 'applied', applied_revision: 5 })
    vi.mocked(getReleaseOperation).mockResolvedValue(applied)
    const wrapper = mountTable({ models: [{
      ...models()[0], runtime_state: '', release_revision: 4,
      latest_release_operation: pending,
    }] })
    try {
      expect(wrapper.text()).toContain('发布处理中')
      await vi.advanceTimersByTimeAsync(2_000)
      await wrapper.vm.$nextTick()
      expect(getReleaseOperation).toHaveBeenCalledWith(pending.operation_id)
      expect(wrapper.emitted('changed')).toBeTruthy()
      expect(wrapper.text()).toContain('发布已应用')
    } finally {
      wrapper.unmount()
      vi.useRealTimers()
    }
  })

  it('ignores a pending registry operation older than the applied model revision', () => {
    const wrapper = mountTable({ models: [{
      ...models()[0], runtime_state: 'applied', release_revision: 6,
      latest_release_operation: operation({ expected_revision: 4 }),
    }] })
    expect(wrapper.text()).not.toContain('发布处理中')
    expect(wrapper.text()).toContain('已应用到运行时')
    wrapper.unmount()
  })

  it('promotes a candidate model to shadow with the operator identity', async () => {
    vi.mocked(promoteModel).mockResolvedValue({ runtime_synced: true } as any)
    const wrapper = mountTable()

    await wrapper.find('[data-vid="candidate-1"] button').trigger('click')
    await flushPromises()
    await wrapper.find('input').setValue('tester')
    await wrapper.find('[data-test="modal-ok"]').trigger('click')
    await flushPromises()

    expect(promoteModel).toHaveBeenCalledWith('candidate-1', {
      target_stage: 'shadow',
      triggered_by: 'tester',
      reason: undefined,
      canary_camera_id: undefined,
    })
    expect(wrapper.emitted('changed')).toBeTruthy()
  })

  it('can promote with the server-side current user when operator identity is blank', async () => {
    vi.mocked(promoteModel).mockResolvedValue({ runtime_synced: true } as any)
    const wrapper = mountTable()

    await wrapper.find('[data-vid="candidate-1"] button').trigger('click')
    await flushPromises()
    await wrapper.find('[data-test="modal-ok"]').trigger('click')
    await flushPromises()

    expect(promoteModel).toHaveBeenCalledWith('candidate-1', {
      target_stage: 'shadow',
    })
    expect(wrapper.emitted('changed')).toBeTruthy()
  })

  it('shows durable release history without hiding current runtime waiting', async () => {
    vi.mocked(promoteModel).mockResolvedValue({
      operation: {
        operation_id: 'release-operation-123456',
        model_version_id: 'candidate-1',
        camera_id: 'dev_cam',
        from_stage: 'candidate',
        target_stage: 'shadow',
        expected_revision: 4,
        applied_revision: 5,
        status: 'applied',
        requested_by: 'tester',
        runtime_state: { identity_confirmed: true },
      },
      model: { model_version_id: 'candidate-1', stage: 'shadow' },
      runtime_synced: true,
    } as any)
    const wrapper = mountTable()

    await wrapper.find('[data-vid="candidate-1"] button').trigger('click')
    await flushPromises()
    await wrapper.find('[data-test="modal-ok"]').trigger('click')
    await flushPromises()

    const row = wrapper.find('[data-vid="candidate-1"]')
    expect(row.text()).toContain('发布已应用')
    expect(row.find('.ant-tag[data-color="processing"]').text()).toBe('等待摄像头启动')
    expect(row.text()).toContain('release-oper')
    expect(row.text()).toContain('r5')
    expect(message.success).toHaveBeenCalledWith(expect.stringContaining('运行时已确认'))
  })

  it('keeps current waiting or failed runtime health above historical applied operations', async () => {
    const initial = models().map(model => ({ ...model, release_revision: 5 }))
    const wrapper = mountTable({ models: initial })
    sendReleaseEvent(operation({ status: 'applied', expected_revision: 4, applied_revision: 5 }))
    await flushPromises()
    const runtime = () => wrapper.find('[data-vid="candidate-1"] .runtime-state-cell')
    expect(runtime().find('.ant-tag').text()).toBe('等待摄像头启动')
    expect(runtime().find('small').text()).toContain('发布已应用')

    await wrapper.setProps({
      models: initial.map(model => model.model_version_id === 'candidate-1'
        ? { ...model, runtime_state: 'failed' } : model),
    })
    expect(runtime().find('.ant-tag').text()).toBe('应用失败')
    expect(runtime().find('.ant-tag').attributes('data-color')).toBe('error')
    wrapper.unmount()
  })

  it('rejects old release revisions and stops showing metadata superseded by REST', async () => {
    const initial = models().map(model => ({ ...model, release_revision: 5, runtime_state: 'applied' }))
    const wrapper = mountTable({ models: initial })
    sendReleaseEvent(operation({ expected_revision: 5, status: 'failed', error: 'latest load failure' }))
    await flushPromises()
    sendReleaseEvent(operation({
      operation_id: 'operation-old', expected_revision: 3, applied_revision: 4, status: 'applied',
    }))
    await flushPromises()
    let runtime = wrapper.find('[data-vid="candidate-1"] .runtime-state-cell')
    expect(runtime.text()).toContain('latest load failure')
    expect(runtime.text()).not.toContain('operation-ol')

    await wrapper.setProps({ models: initial.map(model => ({ ...model, release_revision: 6 })) })
    runtime = wrapper.find('[data-vid="candidate-1"] .runtime-state-cell')
    expect(runtime.find('small').exists()).toBe(false)
    wrapper.unmount()
  })

  it('handles pending and failed WS events without allowing pending to replace a terminal result', async () => {
    const wrapper = mountTable({
      models: models().map(model => ({ ...model, release_revision: 4, runtime_state: null })),
    })
    sendReleaseEvent(operation())
    await flushPromises()
    const runtime = () => wrapper.find('[data-vid="candidate-1"] .runtime-state-cell')
    expect(runtime().find('.ant-tag').text()).toBe('发布处理中')
    sendReleaseEvent(operation({ status: 'failed', error: 'CUDA load failed' }))
    await flushPromises()
    expect(runtime().find('.ant-tag').text()).toBe('发布失败')
    expect(runtime().text()).toContain('CUDA load failed')
    sendReleaseEvent(operation())
    await flushPromises()
    expect(runtime().find('.ant-tag').text()).toBe('发布失败')
    expect(message.error).toHaveBeenCalledWith('发布失败：CUDA load failed')
    wrapper.unmount()
  })

  it('keeps production promotion visibly disabled while release gates are NO-GO', () => {
    const wrapper = mountTable({
      models: [
        ...models(),
        {
          model_version_id: 'canary-1',
          camera_id: 'dev_cam',
          model_type: 'patchcore',
          stage: 'canary',
          is_active: true,
          runtime_state: 'applied',
          created_at: '2026-07-14T00:00:00Z',
        },
      ],
    })

    const productionButton = wrapper
      .find('[data-vid="canary-1"]')
      .findAll('button')
      .find(button => button.text().includes('生产 NO-GO'))
    expect(productionButton).toBeTruthy()
    expect(productionButton!.attributes('disabled')).toBeDefined()
    expect(promoteModel).not.toHaveBeenCalled()
  })

  it('shows the server-authoritative release gate reasons while staying fail closed', () => {
    const canary = {
      model_version_id: 'canary-1',
      camera_id: 'dev_cam',
      model_type: 'patchcore',
      stage: 'canary',
      is_active: true,
      runtime_state: 'applied',
      created_at: '2026-07-14T00:00:00Z',
    }
    const wrapper = mountTable({
      models: [...models(), canary],
      releaseReadiness: {
        'canary-1': {
          model_version_id: 'canary-1',
          camera_id: 'dev_cam',
          target_stage: 'production',
          status: 'no_go',
          production_ready: false,
          promotion_allowed: false,
          operation_id: null,
          desired_stage: null,
          desired_status: null,
          applied_stage: 'canary',
          applied_revision: 4,
          runtime_state: { primary_model_version: 'canary-1' },
          gates: [
            {
              key: 'trusted_soak',
              label: '可信 72 小时 soak',
              passed: false,
              blocking: true,
              reason: '可信 72 小时 soak 尚未验签',
            },
          ],
          reasons: ['可信 72 小时 soak 尚未验签'],
          checked_at: '2026-07-15T00:00:00Z',
        },
      },
    })

    const row = wrapper.find('[data-vid="canary-1"]')
    const productionButton = row
      .findAll('button')
      .find(button => button.text().includes('生产 NO-GO'))
    expect(productionButton).toBeTruthy()
    expect(productionButton!.attributes('disabled')).toBeDefined()
    expect(productionButton!.attributes('title')).toContain('可信 72 小时 soak 尚未验签')
    expect(row.text()).toContain('Readiness · NO-GO · 1 项未通过')
  })

  it('distinguishes an applied production model whose current safety gates failed', () => {
    const wrapper = mountTable({
      releaseReadiness: {
        'production-1': {
          model_version_id: 'production-1',
          camera_id: 'dev_cam',
          target_stage: 'production',
          status: 'applied',
          production_ready: false,
          promotion_allowed: false,
          operation_id: 'release-prod-1',
          desired_stage: 'production',
          desired_status: 'applied',
          applied_stage: 'production',
          applied_revision: 8,
          runtime_state: { primary_model_version: 'production-1' },
          gates: [
            {
              key: 'scene_guard_usable',
              label: 'SceneGuard scene match',
              passed: false,
              blocking: true,
              reason: '当前机位与生产 profile 不匹配',
            },
          ],
          reasons: ['当前机位与生产 profile 不匹配'],
          checked_at: '2026-07-15T00:00:00Z',
        },
      },
    })

    const productionRow = wrapper.find('[data-vid="production-1"]')
    expect(productionRow.text()).toContain('Readiness · 已应用但 NO-GO · 1 项未通过')
    expect(productionRow.find('small[title]').attributes('title')).toContain(
      '当前机位与生产 profile 不匹配',
    )
  })

  it('keeps legacy rollback disabled until the two-phase release path is ready', async () => {
    const wrapper = mountTable()

    const productionRow = wrapper.find('[data-vid="production-1"]')
    const rollbackButton = productionRow
      .findAll('button')
      .find(button => button.text().includes('回滚 NO-GO'))
    expect(rollbackButton).toBeTruthy()
    expect(rollbackButton!.attributes('disabled')).toBeDefined()
    expect(rollbackButton!.attributes('title')).toContain('两阶段 rollback 尚未实现')
    expect(rollbackButton!.attributes('title')).toContain('readiness 为 NO-GO')
    await rollbackButton!.trigger('click')
    await flushPromises()

    expect(rollbackModel).not.toHaveBeenCalled()
    expect(modalConfirm).not.toHaveBeenCalled()
    expect(wrapper.emitted('changed')).toBeFalsy()
  })

  it('retires a model without sending a hard-coded operator name', async () => {
    vi.mocked(retireModel).mockResolvedValue({
      model: { model_version_id: 'candidate-1', stage: 'retired' },
      runtime_synced: true,
    } as any)
    modalConfirm.mockImplementation(({ onOk }) => onOk())
    const wrapper = mountTable()

    const candidateRow = wrapper.find('[data-vid="candidate-1"]')
    const retireItem = candidateRow.findAll('.menu-item').find(item => item.text().includes('退役'))
    expect(retireItem).toBeTruthy()
    await retireItem!.trigger('click')
    await flushPromises()

    expect(retireModel).toHaveBeenCalledWith('candidate-1')
    expect(wrapper.emitted('changed')).toBeTruthy()
  })

  it('opens the re-export action from the row menu and submits default OpenVINO fp16 options', async () => {
    vi.mocked(reexportModel).mockResolvedValue({ status: 'ok' } as any)
    const wrapper = mountTable()

    const candidateRow = wrapper.find('[data-vid="candidate-1"]')
    const reexportItem = candidateRow.findAll('.menu-item').find(item => item.text().includes('重新导出'))
    expect(reexportItem).toBeTruthy()
    await reexportItem!.trigger('click')
    await flushPromises()
    await wrapper.find('[data-test="modal-ok"]').trigger('click')
    await flushPromises()

    expect(reexportModel).toHaveBeenCalledWith('candidate-1', {
      export_format: 'openvino',
      quantization: 'fp16',
    })
    expect(wrapper.emitted('changed')).toBeTruthy()
  })

  it('pins and highlights the focused model version from an alert deep link', () => {
    const wrapper = mountTable({ focusedVersionId: 'candidate-1' })

    const rows = wrapper.findAll('.model-row')
    expect(rows.map(row => row.attributes('data-vid'))).toEqual([
      'candidate-1',
      'production-1',
    ])
    expect(rows[0].classes()).toContain('row-focused')
  })

  it('opens A/B comparison on the current comparison route', async () => {
    const wrapper = mountTable({
      models: [
        ...models(),
        {
          model_version_id: 'shadow-1',
          camera_id: 'dev_cam',
          model_type: 'patchcore',
          stage: 'shadow',
          is_active: false,
          runtime_state: 'applied',
          created_at: '2026-05-25T00:20:00Z',
        },
      ],
    })

    const shadowRow = wrapper.find('[data-vid="shadow-1"]')
    const compareItem = shadowRow.findAll('.menu-item').find(item => item.text().includes('A/B'))
    expect(compareItem).toBeTruthy()
    await compareItem!.trigger('click')

    expect(routerPush).toHaveBeenCalledWith({
      path: '/models/comparison',
      query: { camera: 'dev_cam', shadow: 'shadow-1' },
    })
  })

  it('renders shadow report readiness checks', async () => {
    vi.mocked(getShadowReport).mockResolvedValue({
      total_samples: 2,
      production_alert_rate: 0.1,
      shadow_alert_rate: 0.2,
      false_positive_delta: 1,
      alert_rate_delta: 0.1,
      abs_alert_rate_delta: 0.1,
      avg_score_divergence: 0.03,
      avg_shadow_latency_ms: 90,
      gate_report: {
        ready_for_canary: false,
        checks: [
          {
            name: 'min_shadow_samples',
            label: 'Minimum shadow samples',
            value: 2,
            limit: 3,
            passed: false,
            enabled: true,
          },
        ],
      },
    } as any)
    const wrapper = mountTable({
      models: [
        ...models(),
        {
          model_version_id: 'shadow-1',
          camera_id: 'dev_cam',
          model_type: 'patchcore',
          stage: 'shadow',
          is_active: false,
          runtime_state: 'applied',
          created_at: '2026-05-25T00:20:00Z',
        },
      ],
    })

    const shadowRow = wrapper.find('[data-vid="shadow-1"]')
    const reportItem = shadowRow.findAll('.menu-item').find(item => item.text().includes('影子报告'))
    expect(reportItem).toBeTruthy()
    await reportItem!.trigger('click')
    await flushPromises()

    expect(getShadowReport).toHaveBeenCalledWith('shadow-1', { days: 7 })
    const text = wrapper.text()
    expect(text).toContain('推进到金丝雀检查')
    expect(text).toContain('未满足')
    expect(text).toContain('Minimum shadow samples')
    expect(text).toContain('未通过')
    expect(text).toContain('2 / 3')
  })
})
