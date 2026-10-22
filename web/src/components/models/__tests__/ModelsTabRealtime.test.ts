import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { message } from 'ant-design-vue'

import ModelsTab from '../ModelsTab.vue'
import { getModelRegistry, getReleaseReadiness } from '../../../api'
import { useWebSocket } from '../../../composables/useWebSocket'
import type { ModelInfo, WsEvent } from '../../../types/api'

vi.mock('../../../api', () => ({ getModelRegistry: vi.fn(), getReleaseReadiness: vi.fn() }))
vi.mock('../../../composables/useWebSocket', () => ({ useWebSocket: vi.fn() }))
vi.mock('../ModelTable.vue', () => ({
  default: { props: ['models'], template: '<div data-test="model-ids">{{ models.map(model => model.model_version_id).join(",") }}</div>' },
}))
vi.mock('../EventLog.vue', () => ({ default: { template: '<div />' } }))
vi.mock('../BatchInference.vue', () => ({ default: { template: '<div />' } }))
vi.mock('../../common/AsyncState.vue', () => ({ default: { template: '<section><slot /></section>' } }))
vi.mock('../../common/StatusBanner.vue', () => ({ default: { template: '<div><slot /></div>' } }))
vi.mock('ant-design-vue', () => ({
  Statistic: { props: ['title', 'value'], template: '<div>{{ title }}:{{ value }}</div>' },
  Card: { template: '<section><slot /></section>' },
  Alert: { template: '<div />' }, Button: { template: '<button><slot /></button>' },
  message: { warning: vi.fn() },
}))

const model = { model_version_id: 'candidate', stage: 'candidate', camera_id: 'cam1', is_active: false } as ModelInfo

describe('ModelsTab realtime registry reconciliation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(getModelRegistry).mockResolvedValue({ models: [model], total: 1 })
  })

  it.each(['payload-type', 'legacy-event-key', 'envelope-type'])('reloads a model deletion immediately from its %s', async (shape) => {
    const wrapper = mount(ModelsTab, { props: { cameras: [] } })
    await flushPromises()
    expect(wrapper.get('[data-test="model-ids"]').text()).toBe('candidate')
    vi.mocked(getModelRegistry).mockResolvedValue({ models: [], total: 0 })
    const subscription = vi.mocked(useWebSocket).mock.calls[0]?.[0]
    const data = shape === 'payload-type'
      ? { type: 'model.deleted', model_version_id: 'candidate' }
      : shape === 'legacy-event-key' ? { event: 'model.deleted', model_version_id: 'candidate' } : {}
    const envelope: WsEvent<unknown> = {
      schema_version: 1, event_id: 'delete-event', topic: 'models', type: 'model.deleted',
      resource_id: 'candidate', revision: 2, occurred_at: '2026-10-22T00:00:00Z', data,
    }
    subscription?.onMessage?.('models', data, envelope)
    await flushPromises()

    expect(getModelRegistry).toHaveBeenCalledTimes(2)
    expect(wrapper.get('[data-test="model-ids"]').text()).toBe('')
    expect(getReleaseReadiness).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('refreshes release stage changes and preserves the explicit activation-failure warning', async () => {
    const wrapper = mount(ModelsTab, { props: { cameras: [] } })
    await flushPromises()
    const subscription = vi.mocked(useWebSocket).mock.calls[0]?.[0]
    subscription?.onMessage?.('model_release', { type: 'release_operation_applied' })
    await flushPromises()
    expect(getModelRegistry).toHaveBeenCalledTimes(2)

    subscription?.onMessage?.('models', {
      event: 'model.activation_failed', camera_id: 'cam1', attempted_version: 'v2', current_version: 'v1',
    })
    await flushPromises()
    expect(getModelRegistry).toHaveBeenCalledTimes(3)
    expect(message.warning).toHaveBeenCalledWith(expect.objectContaining({
      content: expect.stringContaining('激活 v2 失败，仍在使用 v1'),
    }))
    wrapper.unmount()
  })
})
