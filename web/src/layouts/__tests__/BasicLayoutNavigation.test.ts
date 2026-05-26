import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { markRaw, nextTick } from 'vue'

import BasicLayout from '../BasicLayout.vue'
import { useAuthStore } from '../../stores/useAuthStore'

const routerPush = vi.hoisted(() => vi.fn())
const routeState = vi.hoisted(() => ({ path: '/overview' }))

vi.mock('vue-router', () => ({
  useRoute: () => routeState,
  useRouter: () => ({ push: routerPush }),
}))

vi.mock('../../composables/useWebSocket', () => ({
  useWebSocket: vi.fn(() => ({
    connected: { value: true },
    reconnecting: { value: false },
    fallbackMode: { value: false },
    retryCount: { value: 0 },
    nextRetryIn: { value: 0 },
  })),
}))

vi.mock('../../composables/useSystemMode', () => ({
  useSystemMode: vi.fn(() => ({
    showBanner: { value: false },
    bannerLabel: { value: '' },
    bannerColor: { value: '#000' },
  })),
}))

vi.mock('../../components/DegradationBar.vue', () => ({
  default: { name: 'DegradationBar', template: '<div data-test="degradation-bar" />' },
}))

vi.mock('../../components/ErrorBoundary.vue', () => ({
  default: { name: 'ErrorBoundary', template: '<div><slot /></div>' },
}))

vi.mock('../../components/system/ErrorCenterDrawer.vue', () => ({
  default: { name: 'ErrorCenterDrawer', template: '<div data-test="error-center" />' },
}))

vi.mock('@ant-design/icons-vue', () => ({
  AlertOutlined: { template: '<span />' },
  BarChartOutlined: { template: '<span />' },
  BellOutlined: { template: '<span />' },
  DashboardOutlined: { template: '<span />' },
  DatabaseOutlined: { template: '<span />' },
  InboxOutlined: { template: '<span />' },
  RightOutlined: { template: '<span />' },
  SettingOutlined: { template: '<span />' },
  VideoCameraOutlined: { template: '<span />' },
}))

vi.mock('ant-design-vue', () => ({
  Button: { template: '<button type="button"><slot /></button>' },
  notification: { open: vi.fn() },
}))

const routerLinkStub = {
  props: ['to'],
  template: '<a class="router-link" :data-to="typeof to === \'string\' ? to : to.path"><slot /></a>',
}

const routerViewStub = {
  template: '<main data-test="router-view"><slot :Component="RouteComponent" /></main>',
  data() {
    return { RouteComponent: markRaw({ template: '<div />' }) }
  },
}

const mountedWrappers: ReturnType<typeof mount>[] = []

function mountLayout(role: 'admin' | 'engineer' | 'operator' | 'viewer' = 'engineer') {
  const pinia = createPinia()
  setActivePinia(pinia)
  const auth = useAuthStore()
  auth.currentUser = { username: role, role }

  const wrapper = mount(BasicLayout, {
    global: {
      plugins: [pinia],
      stubs: {
        RouterLink: routerLinkStub,
        RouterView: routerViewStub,
        KeepAlive: { template: '<div><slot /></div>' },
        'a-badge': { template: '<span><slot /></span>' },
        'a-button': {
          emits: ['click'],
          template: '<button type="button" @click="$emit(\'click\', $event)"><slot /></button>',
        },
        'a-dropdown': { template: '<span><slot /><slot name="overlay" /></span>' },
        'a-avatar': { template: '<span><slot /></span>' },
        'a-tag': { template: '<span><slot /></span>' },
        'a-menu': { template: '<span><slot /></span>' },
        'a-menu-item': { template: '<span><slot /></span>' },
        'a-menu-divider': { template: '<hr />' },
      },
    },
  })
  mountedWrappers.push(wrapper)
  return wrapper
}

describe('BasicLayout core-loop navigation', () => {
  beforeEach(() => {
    routeState.path = '/overview'
    routerPush.mockReset()
  })

  afterEach(() => {
    for (const wrapper of mountedWrappers.splice(0)) {
      wrapper.unmount()
    }
    vi.restoreAllMocks()
  })

  it('exposes the full Cameras-Alerts-Models-System-Reports loop for engineers', () => {
    const wrapper = mountLayout('engineer')
    const text = wrapper.text()

    expect(text).toContain('摄像头')
    expect(text).toContain('告警中心')
    expect(text).toContain('报表')
    expect(text).toContain('模型管理')
    expect(text).toContain('训练与评估')
    expect(text).toContain('模型与发布')
    expect(text).toContain('设置')
    expect(text).toContain('系统概览')
    expect(text).toContain('配置管理')
    expect(text).toContain('降级事件')

    const links = wrapper.findAll('[data-to]').map(link => link.attributes('data-to'))
    expect(links).toEqual(expect.arrayContaining([
      '/cameras',
      '/alerts',
      '/reports',
      '/models/training',
      '/models/registry',
      '/system/overview',
      '/system/config',
      '/system/degradation',
    ]))
  })

  it('keeps model publishing and config links hidden from operators', () => {
    const wrapper = mountLayout('operator')
    const text = wrapper.text()

    expect(text).toContain('摄像头')
    expect(text).toContain('告警中心')
    expect(text).toContain('报表')
    expect(text).toContain('模型管理')
    expect(text).toContain('设置')
    expect(text).not.toContain('训练与评估')
    expect(text).not.toContain('模型与发布')
    expect(text).not.toContain('配置管理')

    const links = wrapper.findAll('[data-to]').map(link => link.attributes('data-to'))
    expect(links).not.toContain('/models/registry')
    expect(links).not.toContain('/system/config')
  })

  it('routes keyboard shortcuts to stable core-loop entry points', async () => {
    mountLayout('engineer')

    for (const key of ['2', '3', '4', '6', '7']) {
      window.dispatchEvent(new KeyboardEvent('keydown', { key }))
      await nextTick()
    }

    expect(routerPush.mock.calls.map(call => call[0])).toEqual([
      '/cameras',
      '/alerts',
      '/reports',
      '/models',
      '/system',
    ])
  })
})
