import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ModelTable from '../ModelTable.vue'
import {
  promoteModel,
  reexportModel,
  retireModel,
  rollbackModel,
} from '../../../api'

const routerPush = vi.hoisted(() => vi.fn())
const modalConfirm = vi.hoisted(() => vi.fn())
const antStubs = vi.hoisted(() => ({
  button: {
    props: ['type', 'size', 'danger', 'ghost', 'loading'],
    emits: ['click'],
    template: '<button type="button" @click="$emit(\'click\', $event)"><slot name="icon" /><slot /></button>',
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

describe('ModelTable release actions', () => {
  beforeEach(() => {
    vi.mocked(promoteModel).mockReset()
    vi.mocked(reexportModel).mockReset()
    vi.mocked(retireModel).mockReset()
    vi.mocked(rollbackModel).mockReset()
    routerPush.mockReset()
    modalConfirm.mockReset()
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

  it('calls rollback for the active production model after confirmation', async () => {
    vi.mocked(rollbackModel).mockResolvedValue({
      activated: 'previous-1',
      runtime_synced: true,
    } as any)
    modalConfirm.mockImplementation(({ onOk }) => onOk())
    const wrapper = mountTable()

    const productionRow = wrapper.find('[data-vid="production-1"]')
    const rollbackButton = productionRow.findAll('button').find(button => button.text() === '')
    expect(rollbackButton).toBeTruthy()
    await rollbackButton!.trigger('click')
    await flushPromises()

    expect(rollbackModel).toHaveBeenCalledWith('production-1')
    expect(wrapper.emitted('changed')).toBeTruthy()
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
})
