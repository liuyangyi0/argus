import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import ModelsRegistryView from '../ModelsRegistryView.vue'
import ModelsTab from '../../../components/models/ModelsTab.vue'
import { getModelRegistry } from '../../../api'

vi.mock('vue-router', () => ({
  useRoute: () => ({ query: { version_id: 'target-version' } }),
}))

vi.mock('../../../composables/useModelState', () => ({
  useModelState: () => ({
    cameras: ref([{ camera_id: 'cam_01', name: 'Camera 01' }]),
    loadCameras: vi.fn().mockResolvedValue(undefined),
  }),
}))

vi.mock('../../../composables/useWebSocket', () => ({
  useWebSocket: vi.fn(),
}))

vi.mock('../../../api', () => ({
  getModelRegistry: vi.fn(),
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
  props: ['models', 'cameras', 'focusedVersionId'],
  template: '<div data-test="model-table">{{ focusedVersionId }}</div>',
}

function mountModelsTab(focusVersionId = 'target-version') {
  return mount(ModelsTab, {
    props: { cameras: [], focusVersionId },
    global: {
      stubs: {
        Alert: alertStub,
        Card: cardStub,
        Statistic: statisticStub,
        ModelTable: modelTableStub,
        EventLog: true,
        BatchInference: true,
      },
    },
  })
}

describe('Models registry focus from alert lineage', () => {
  beforeEach(() => {
    vi.mocked(getModelRegistry).mockReset()
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

    const alert = wrapper.find('.ant-alert')
    expect(alert.exists()).toBe(true)
    expect(alert.classes()).toContain('ant-alert-info')
    expect(alert.text()).toContain('已定位触发模型')
    expect(alert.text()).toContain('target-version')
    expect(wrapper.findComponent(modelTableStub).props('focusedVersionId')).toBe('target-version')
  })

  it('warns when the linked model version is no longer registered', async () => {
    vi.mocked(getModelRegistry).mockResolvedValue({ models: [] } as any)

    const wrapper = mountModelsTab()
    await flushPromises()

    const alert = wrapper.find('.ant-alert')
    expect(alert.exists()).toBe(true)
    expect(alert.classes()).toContain('ant-alert-warning')
    expect(alert.text()).toContain('未找到模型版本')
    expect(alert.text()).toContain('target-version')
  })
})
