import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { markRaw, nextTick } from 'vue'
import { Drawer } from 'ant-design-vue'

import BasicLayout from '../BasicLayout.vue'
import { useAuthStore } from '../../stores/useAuthStore'
import { useWebSocket } from '../../composables/useWebSocket'

const routerPush = vi.hoisted(() => vi.fn())
const routeState = vi.hoisted(() => ({ path: '/overview' }))

vi.mock('vue-router', () => ({
  useRoute: () => routeState,
  useRouter: () => ({ push: routerPush }),
}))

vi.mock('../../composables/useWebSocket', async () => {
  const { ref } = await import('vue')
  return {
    useWebSocket: vi.fn(() => ({
      connected: ref(true),
      reconnecting: ref(false),
      fallbackMode: ref(false),
      retryCount: ref(0),
      nextRetryIn: ref(0),
    })),
  }
})

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
  Drawer: {
    props: ['open', 'title'],
    emits: ['close', 'afterOpenChange'],
    template: '<div v-if="open"><slot /></div>',
  },
  notification: { open: vi.fn() },
}))

const routerLinkStub = {
  props: ['to'],
  template: '<a class="router-link" @click.prevent :href="typeof to === \'string\' ? to : to.path" :data-to="typeof to === \'string\' ? to : to.path"><slot /></a>',
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
    attachTo: document.body,
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
    vi.mocked(useWebSocket).mockClear()
    vi.stubGlobal('innerWidth', 1440)
  })

  afterEach(() => {
    for (const wrapper of mountedWrappers.splice(0)) {
      wrapper.unmount()
    }
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
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
    expect(text).not.toContain('3D仓库')

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

  it('opens every model subpage in a tablet drawer without changing the route', async () => {
    vi.stubGlobal('innerWidth', 1024)
    const wrapper = mountLayout('engineer')
    const trigger = wrapper.get('button[aria-label="模型管理"]')
    expect(trigger.attributes('aria-expanded')).toBe('false')
    await trigger.trigger('click')

    const dialog = wrapper.get('[role="dialog"]')
    expect(dialog.attributes('aria-label')).toBe('模型管理导航')
    expect(dialog.findAll('[data-to]').map(link => link.attributes('data-to'))).toEqual([
      '/models/baseline', '/models/collections', '/models/training', '/models/registry',
      '/models/comparison', '/models/labeling', '/models/threshold',
    ])
    expect(trigger.attributes('aria-expanded')).toBe('true')
    expect(routerPush).not.toHaveBeenCalled()
    window.dispatchEvent(new KeyboardEvent('keydown', { key: '7' }))
    expect(routerPush).not.toHaveBeenCalled()

    await dialog.get('[data-to="/models/labeling"]').trigger('click')
    expect(wrapper.find('[role="dialog"]').exists()).toBe(false)
    expect(trigger.attributes('aria-expanded')).toBe('false')
  })

  it('filters tablet navigation by the current role, including a role change while open', async () => {
    vi.stubGlobal('innerWidth', 1024)
    const wrapper = mountLayout('admin')
    await wrapper.get('button[aria-label="系统设置"]').trigger('click')
    expect(wrapper.get('[role="dialog"]').find('[data-to="/system/users"]').exists()).toBe(true)

    useAuthStore().currentUser = { username: 'viewer', role: 'viewer' }
    await nextTick()
    const dialog = wrapper.get('[role="dialog"]')
    const links = dialog.findAll('[data-to]').map(link => link.attributes('data-to'))
    expect(links).toContain('/system/degradation')
    expect(links).not.toContain('/system/config')
    expect(links).not.toContain('/system/audit')
    expect(links).not.toContain('/system/users')
  })

  it.each([1440, 390])('closes tablet navigation when resizing to %i px', async (width) => {
    vi.stubGlobal('innerWidth', 1024)
    const wrapper = mountLayout('engineer')
    await wrapper.get('button[aria-label="系统设置"]').trigger('click')
    vi.stubGlobal('innerWidth', width)
    window.dispatchEvent(new Event('resize'))
    await nextTick()
    expect(wrapper.find('[role="dialog"]').exists()).toBe(false)
  })

  it('keeps Tab focus inside the compact navigation and closes with Escape', async () => {
    vi.stubGlobal('innerWidth', 1024)
    const wrapper = mountLayout()
    await wrapper.get('button[aria-label="模型管理"]').trigger('click')
    const dialog = wrapper.get('[role="dialog"]')
    const close = dialog.get<HTMLButtonElement>('button[aria-label="关闭导航"]')
    const last = dialog.get<HTMLAnchorElement>('[data-to="/models/threshold"]')
    last.element.focus()
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', cancelable: true }))
    expect(document.activeElement).toBe(close.element)
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, cancelable: true }))
    expect(document.activeElement).toBe(last.element)
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', cancelable: true }))
    await nextTick()
    expect(wrapper.find('[role="dialog"]').exists()).toBe(false)
  })

  it('preserves reverse Tab focus when the opening animation finishes after keyboard navigation', async () => {
    vi.stubGlobal('innerWidth', 1024)
    const wrapper = mountLayout()
    await wrapper.get('button[aria-label="模型管理"]').trigger('click')
    const dialog = wrapper.get('[role="dialog"]')
    const close = dialog.get<HTMLButtonElement>('button[aria-label="关闭导航"]')
    const last = dialog.get<HTMLAnchorElement>('[data-to="/models/threshold"]')
    last.element.focus()
    last.element.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true }))
    expect(document.activeElement).toBe(close.element)
    close.element.dispatchEvent(new KeyboardEvent('keydown', {
      key: 'Tab', shiftKey: true, bubbles: true, cancelable: true,
    }))
    expect(document.activeElement).toBe(last.element)

    wrapper.findComponent(Drawer).vm.$emit('afterOpenChange', true)
    await nextTick()
    expect(document.activeElement).toBe(last.element)
  })

  it('initializes focus when opening finishes and restores the trigger after closing', async () => {
    vi.stubGlobal('innerWidth', 1024)
    const wrapper = mountLayout()
    const trigger = wrapper.get<HTMLButtonElement>('button[aria-label="模型管理"]')
    trigger.element.focus()
    await trigger.trigger('click')
    const drawer = wrapper.findComponent(Drawer)
    drawer.vm.$emit('afterOpenChange', true)
    await nextTick()
    expect(document.activeElement).toBe(wrapper.get('[role="dialog"] a').element)
    document.activeElement?.dispatchEvent(new KeyboardEvent('keydown', {
      key: 'Escape', bubbles: true, cancelable: true,
    }))
    await nextTick()
    drawer.vm.$emit('afterOpenChange', false)
    await nextTick()
    expect(document.activeElement).toBe(trigger.element)
  })

  it('does not restore old trigger focus when a close animation finishes after reopening', async () => {
    vi.stubGlobal('innerWidth', 1024)
    const wrapper = mountLayout()
    await wrapper.get('button[aria-label="模型管理"]').trigger('click')
    const drawer = wrapper.findComponent(Drawer)
    await wrapper.get('[role="dialog"] button[aria-label="关闭导航"]').trigger('click')
    await wrapper.get('button[aria-label="系统设置"]').trigger('click')
    const last = wrapper.get<HTMLAnchorElement>('[role="dialog"] [data-to="/system/regions"]')
    last.element.focus()
    drawer.vm.$emit('afterOpenChange', false)
    await nextTick()
    expect(document.activeElement).toBe(last.element)
    expect(wrapper.get('[role="dialog"]').attributes('aria-label')).toBe('系统设置导航')
  })

  it('reports realtime connection changes instead of a static Online label', async () => {
    const wrapper = mountLayout()
    const state = vi.mocked(useWebSocket).mock.results[0]!.value
    const status = wrapper.get('[aria-label="实时数据连接状态"]')
    expect(status.text()).toBe('实时已连接')

    state.connected.value = false
    state.reconnecting.value = true
    await nextTick()
    expect(status.text()).toBe('实时重连中')
    expect(wrapper.get('.connection-banner').text()).toContain('正在重新连接')

    state.fallbackMode.value = true
    await nextTick()
    expect(status.text()).toBe('实时数据受限')
    expect(wrapper.get('.connection-banner').text()).toContain('受限模式')
    state.fallbackMode.value = false
    state.reconnecting.value = false
    await nextTick()
    expect(status.text()).toBe('实时未连接')
    expect(wrapper.text()).not.toContain('Online')
  })
})
