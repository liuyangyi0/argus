import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import CollectionsView from '../CollectionsView.vue'
import {
  activateCollection,
  deleteCollection,
  getBaselineCollections,
  retireCollection,
} from '../../api/baselines'
import { useAuthStore } from '../../stores/useAuthStore'

const routerPush = vi.hoisted(() => vi.fn())

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush }),
}))

vi.mock('../../api/baselines', () => ({
  getBaselineCollections: vi.fn(),
  activateCollection: vi.fn(),
  retireCollection: vi.fn(),
  deleteCollection: vi.fn(),
}))

vi.mock('../../components/common/ErrorDetailModal.vue', () => ({
  default: {
    name: 'ErrorDetailModal',
    template: '<div data-test="error-detail-modal" />',
  },
}))

vi.mock('@ant-design/icons-vue', () => ({
  ReloadOutlined: { template: '<span />' },
}))

const antStubs = vi.hoisted(() => ({
  button: {
    props: ['type', 'size', 'danger'],
    emits: ['click'],
    template: '<button type="button" @click="$emit(\'click\', $event)"><slot name="icon" /><slot /></button>',
  },
  card: { template: '<section><slot /></section>' },
  popconfirm: { template: '<span><slot /></span>' },
  space: { template: '<span><slot /></span>' },
  tag: {
    props: ['color'],
    template: '<span class="ant-tag" :data-color="color"><slot /></span>',
  },
  table: {
    props: ['columns', 'dataSource'],
    template: `
      <div>
        <div
          v-for="record in dataSource"
          :key="record.version"
          class="collection-row"
          :data-version="record.version"
        >
          <div v-for="column in columns" :key="column.key" :data-col="column.key">
            <slot name="bodyCell" :column="column" :record="record" />
            <template v-if="column.dataIndex">{{ record[column.dataIndex] }}</template>
          </div>
        </div>
      </div>
    `,
  },
  typography: {
    Title: {
      props: ['level'],
      template: '<h1><slot /></h1>',
    },
  },
}))

vi.mock('ant-design-vue', () => ({
  Button: antStubs.button,
  Card: antStubs.card,
  Popconfirm: antStubs.popconfirm,
  Space: antStubs.space,
  Table: antStubs.table,
  Tag: antStubs.tag,
  Typography: antStubs.typography,
  message: {
    success: vi.fn(),
    error: vi.fn(),
  },
}))

function collection(overrides: Record<string, unknown> = {}) {
  return {
    camera_id: 'dev_cam',
    zone_id: 'default',
    version: 'baseline-v1',
    session_label: 'session-a',
    status: 'ok',
    image_count: 24,
    acceptance_rate: 0.8,
    captured_at: '2026-05-25T00:00:00Z',
    state: 'active',
    is_current: false,
    error: null,
    ...overrides,
  }
}

function mountCollections(role: 'engineer' | 'operator') {
  const pinia = createPinia()
  setActivePinia(pinia)
  useAuthStore().currentUser = { username: role, role }
  return mount(CollectionsView, {
    global: {
      plugins: [pinia],
    },
  })
}

describe('CollectionsView role-gated actions', () => {
  beforeEach(() => {
    routerPush.mockReset()
    vi.mocked(getBaselineCollections).mockReset()
    vi.mocked(activateCollection).mockReset()
    vi.mocked(retireCollection).mockReset()
    vi.mocked(deleteCollection).mockReset()
    vi.mocked(getBaselineCollections).mockResolvedValue({
      collections: [collection()],
    } as any)
  })

  it('lets engineers deep-link a baseline collection into training', async () => {
    const wrapper = mountCollections('engineer')
    await flushPromises()

    const trainButton = wrapper.findAll('button').find(button => button.text().includes('用这批训练'))
    expect(trainButton).toBeTruthy()
    await trainButton!.trigger('click')

    expect(routerPush).toHaveBeenCalledWith({
      path: '/models/training',
      query: { preselect: expect.any(String) },
    })
  })

  it('hides baseline mutation and training actions from operators', async () => {
    const wrapper = mountCollections('operator')
    await flushPromises()

    const text = wrapper.text()
    expect(text).toContain('baseline-v1')
    expect(text).not.toContain('激活')
    expect(text).not.toContain('用这批训练')
    expect(text).not.toContain('Retire')
    expect(text).not.toContain('删除')

    expect(routerPush).not.toHaveBeenCalled()
    expect(activateCollection).not.toHaveBeenCalled()
    expect(retireCollection).not.toHaveBeenCalled()
    expect(deleteCollection).not.toHaveBeenCalled()
  })
})
