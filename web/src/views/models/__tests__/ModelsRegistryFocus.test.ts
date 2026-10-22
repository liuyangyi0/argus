import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import ModelsRegistryView from '../ModelsRegistryView.vue'
import ModelsTab from '../../../components/models/ModelsTab.vue'
import { getModelRegistry, getReleaseReadiness } from '../../../api'
import { ApiError } from '../../../types/api'
import { useWebSocket } from '../../../composables/useWebSocket'

vi.mock('vue-router', () => ({
  useRoute: () => ({ query: { version_id: 'target-version' } }),
}))

vi.mock('../../../composables/useModelState', () => ({
  useModelState: () => ({
    cameras: ref([{ camera_id: 'cam_01', name: 'Camera 01' }]),
    loadCameras: vi.fn().mockResolvedValue(undefined),
    camerasLoading: ref(false),
    camerasLoaded: ref(true),
    camerasError: ref(''),
  }),
}))

vi.mock('../../../composables/useWebSocket', () => ({
  useWebSocket: vi.fn(),
}))

vi.mock('../../../api', () => ({
  getModelRegistry: vi.fn(),
  getReleaseReadiness: vi.fn(),
}))

const cardStub = { template: '<div><slot name="title" /><slot /></div>' }
const statisticStub = {
  props: ['title', 'value'],
  template: '<div>{{ title }} {{ value }}</div>',
}
const alertStub = {
  name: 'Alert',
  props: ['type', 'message', 'description'],
  template: '<div data-test="focus-alert" :data-type="type">{{ message }} {{ description }}</div>',
}
const modelTableStub = {
  name: 'ModelTable',
  props: ['models', 'cameras', 'releaseReadiness', 'focusedVersionId'],
  template: '<div data-test="model-table">{{ focusedVersionId }}</div>',
}

function mountModelsTab(focusVersionId = 'target-version') {
  return mount(ModelsTab, {
    props: { cameras: [], focusVersionId },
    global: {
      stubs: {
        Alert: alertStub,
        AAlert: alertStub,
        Card: cardStub,
        ACard: cardStub,
        Statistic: statisticStub,
        AStatistic: statisticStub,
        ModelTable: modelTableStub,
        EventLog: true,
        BatchInference: true,
      },
    },
  })
}

describe('Models registry focus from alert lineage', () => {
  beforeEach(() => {
    vi.mocked(useWebSocket).mockClear()
    vi.mocked(getModelRegistry).mockReset()
    vi.mocked(getReleaseReadiness).mockReset()
    vi.mocked(getReleaseReadiness).mockResolvedValue({
      model_version_id: 'target-version',
      camera_id: 'cam_01',
      target_stage: 'production',
      status: 'applied',
      production_ready: true,
      promotion_allowed: false,
      operation_id: null,
      desired_stage: null,
      desired_status: null,
      applied_stage: 'production',
      applied_revision: 1,
      runtime_state: null,
      gates: [],
      reasons: [],
      checked_at: '2026-07-15T00:00:00Z',
    })
  })

  it('reconciles the registry when the model_release topic requests a REST resync', async () => {
    vi.mocked(getModelRegistry).mockResolvedValue({ models: [], total: 0 })
    const wrapper = mountModelsTab('')
    await flushPromises()
    const subscription = vi.mocked(useWebSocket).mock.calls.find(
      ([options]) => options.topics.includes('model_release'),
    )?.[0]

    expect(subscription).toBeTruthy()
    expect(subscription?.topics).toContain('models')
    await subscription?.fallbackPoll?.()
    expect(getModelRegistry).toHaveBeenCalledTimes(2)
    wrapper.unmount()
  })

  it('passes version_id query to the registry tab', () => {
    const wrapper = mount(ModelsRegistryView, {
      global: {
        stubs: {
          ModelsTab: {
            name: 'ModelsTab',
            props: ['cameras', 'focusVersionId'],
            template: '<div data-test="registry-tab">{{ focusVersionId }}</div>',
          },
        },
      },
    })

    expect(wrapper.find('[data-test="registry-tab"]').text()).toBe('target-version')
  })

  it('announces and highlights a matching model version', async () => {
    vi.mocked(getModelRegistry).mockResolvedValue({
      models: [
        {
          model_version_id: 'target-version',
          camera_id: 'cam_01',
          stage: 'production',
          is_active: true,
        },
      ],
    } as any)

    const wrapper = mountModelsTab()
    await flushPromises()

    const alert = wrapper.find('[data-test="focus-alert"]')
    expect(alert.exists()).toBe(true)
    expect(alert.attributes('data-type')).toBe('info')
    expect(alert.text()).toContain('已定位触发模型')
    expect(alert.text()).toContain('target-version')
    expect(wrapper.findComponent(modelTableStub).props('focusedVersionId')).toBe('target-version')
  })

  it('warns when the linked model version is no longer registered', async () => {
    vi.mocked(getModelRegistry).mockResolvedValue({ models: [] } as any)

    const wrapper = mountModelsTab()
    await flushPromises()

    const alert = wrapper.find('[data-test="focus-alert"]')
    expect(alert.exists()).toBe(true)
    expect(alert.attributes('data-type')).toBe('warning')
    expect(alert.text()).toContain('未找到模型版本')
    expect(alert.text()).toContain('target-version')
  })

  it('renders a registry request failure instead of the empty model state', async () => {
    vi.mocked(getModelRegistry).mockRejectedValue(new Error('registry offline'))

    const wrapper = mountModelsTab('')
    await flushPromises()

    expect(wrapper.text()).toContain('数据加载失败')
    expect(wrapper.text()).toContain('registry offline')
    expect(wrapper.text()).not.toContain('暂无注册的模型版本')
  })

  it('renders registry access denial as a permission state', async () => {
    vi.mocked(getModelRegistry).mockRejectedValue(new ApiError({
      httpStatus: 403,
      businessCode: 40300,
      message: 'registry access denied',
    }))

    const wrapper = mountModelsTab('')
    await flushPromises()

    expect(wrapper.text()).toContain('无权访问')
    expect(wrapper.text()).not.toContain('数据加载失败')
    expect(wrapper.text()).not.toContain('暂无注册的模型版本')
  })
})
