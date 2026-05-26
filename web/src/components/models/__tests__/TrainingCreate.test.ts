import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { message } from 'ant-design-vue'

import TrainingCreate from '../TrainingCreate.vue'
import { createTrainingJob } from '../../../api'

const routeState = vi.hoisted(() => ({
  query: {} as Record<string, string>,
}))

const antStubs = vi.hoisted(() => ({
  card: { template: '<section><slot /></section>' },
  button: {
    props: ['type', 'loading', 'disabled'],
    emits: ['click'],
    template: '<button type="button" :disabled="disabled || loading" @click="$emit(\'click\', $event)"><slot name="icon" /><slot /></button>',
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
    props: ['value', 'disabled'],
    emits: ['update:value', 'change'],
    template: `
      <select
        :value="value || ''"
        :disabled="disabled"
        @change="$emit('update:value', $event.target.value); $emit('change', $event.target.value)"
      >
        <option value=""></option>
        <slot />
      </select>
    `,
  },
  input: {
    props: ['value', 'disabled'],
    emits: ['update:value'],
    template: '<input :value="value" :disabled="disabled" @input="$emit(\'update:value\', $event.target.value)" />',
  },
  modal: {
    props: ['open', 'confirmLoading'],
    emits: ['ok', 'update:open'],
    template: `
      <section v-if="open" data-test="training-modal">
        <slot />
        <button type="button" data-test="modal-ok" :disabled="confirmLoading" @click="$emit('ok')">创建</button>
      </section>
    `,
  },
  space: { template: '<span><slot /></span>' },
  checkbox: {
    props: ['checked', 'disabled'],
    emits: ['update:checked'],
    template: `
      <label>
        <input
          type="checkbox"
          :checked="checked"
          :disabled="disabled"
          @change="$emit('update:checked', $event.target.checked)"
        />
        <slot />
      </label>
    `,
  },
  badge: {
    props: ['count'],
    template: '<span data-test="pending-badge">{{ count }}</span>',
  },
}))

vi.mock('vue-router', () => ({
  useRoute: () => routeState,
}))

vi.mock('../../../api', () => ({
  createTrainingJob: vi.fn(),
}))

vi.mock('ant-design-vue', () => ({
  Card: antStubs.card,
  Button: antStubs.button,
  Form: antStubs.form,
  Select: antStubs.select,
  Input: antStubs.input,
  Modal: antStubs.modal,
  Space: antStubs.space,
  Checkbox: antStubs.checkbox,
  Badge: antStubs.badge,
  message: {
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
}))

vi.mock('@ant-design/icons-vue', () => ({
  PlusOutlined: { template: '<span />' },
}))

vi.mock('../../baseline/DatasetSelector.vue', () => ({
  default: {
    name: 'DatasetSelector',
    props: ['modelValue', 'cameraId'],
    emits: ['update:modelValue'],
    template: `
      <button
        type="button"
        data-test="dataset-selector"
        @click="$emit('update:modelValue', { items: [{ camera_id: cameraId, dataset_version: 'baseline-v1' }] })"
      >
        {{ cameraId || 'no-camera' }} {{ modelValue?.items?.[0]?.dataset_version || 'empty' }}
      </button>
    `,
  },
}))

function mountCreate() {
  return mount(TrainingCreate, {
    props: {
      cameras: [
        { camera_id: 'dev_cam', name: 'Dev camera' },
        { camera_id: 'deep_cam', name: 'Deep linked camera' },
      ],
      pendingCount: 2,
    },
  })
}

async function openCreateModal(wrapper: ReturnType<typeof mountCreate>) {
  const button = wrapper.findAll('button').find(item => item.text().includes('新建训练任务'))
  expect(button).toBeTruthy()
  await button!.trigger('click')
}

describe('TrainingCreate', () => {
  beforeEach(() => {
    routeState.query = {}
    vi.mocked(createTrainingJob).mockReset()
    vi.mocked(message.success).mockReset()
    vi.mocked(message.warning).mockReset()
    vi.mocked(message.error).mockReset()
    vi.mocked(message.info).mockReset()
  })

  it('requires a camera before creating an anomaly-head training job', async () => {
    const wrapper = mountCreate()
    await openCreateModal(wrapper)

    await wrapper.find('[data-test="modal-ok"]').trigger('click')
    await flushPromises()

    expect(createTrainingJob).not.toHaveBeenCalled()
    expect(message.warning).toHaveBeenCalledWith('请先选择摄像头')
  })

  it('creates an anomaly-head job with camera, dataset selection, and validation override', async () => {
    vi.mocked(createTrainingJob).mockResolvedValue({ job_id: 'job-1' } as any)
    const wrapper = mountCreate()
    await openCreateModal(wrapper)

    const selects = wrapper.findAll('select')
    expect(selects.length).toBeGreaterThanOrEqual(3)
    await selects[1].setValue('dev_cam')
    await wrapper.find('input:not([type="checkbox"])').setValue('zone-a')
    await wrapper.find('[data-test="dataset-selector"]').trigger('click')
    await wrapper.find('input[type="checkbox"]').setValue(true)
    await wrapper.find('[data-test="modal-ok"]').trigger('click')
    await flushPromises()

    expect(createTrainingJob).toHaveBeenCalledWith({
      job_type: 'anomaly_head',
      camera_id: 'dev_cam',
      model_type: 'patchcore',
      zone_id: 'zone-a',
      hyperparameters: { skip_baseline_validation: true },
      dataset_selection: {
        items: [{ camera_id: 'dev_cam', dataset_version: 'baseline-v1' }],
      },
    })
    expect(message.success).toHaveBeenCalledWith('训练任务已创建，等待确认')
    expect(wrapper.emitted('refresh')).toHaveLength(1)
  })

  it('keeps collection deep-link dataset selection after the camera prefill watcher runs', async () => {
    vi.mocked(createTrainingJob).mockResolvedValue({ job_id: 'job-deep' } as any)
    routeState.query = {
      preselect: btoa(JSON.stringify({
        items: [{ camera_id: 'deep_cam', dataset_version: 'collection-v7' }],
      })),
    }

    const wrapper = mountCreate()
    await flushPromises()

    expect(wrapper.find('[data-test="training-modal"]').exists()).toBe(true)
    expect(wrapper.find('[data-test="dataset-selector"]').text()).toContain('collection-v7')

    await wrapper.find('[data-test="modal-ok"]').trigger('click')
    await flushPromises()

    expect(createTrainingJob).toHaveBeenCalledWith(expect.objectContaining({
      camera_id: 'deep_cam',
      dataset_selection: {
        items: [{ camera_id: 'deep_cam', dataset_version: 'collection-v7' }],
      },
    }))
  })
})
