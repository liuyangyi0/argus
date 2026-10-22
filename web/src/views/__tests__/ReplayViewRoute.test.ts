import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import ReplayView from '../ReplayView.vue'

const routeFixture = vi.hoisted(() => ({
  routerBack: vi.fn(),
  route: null as { params: { alertId: string } } | null,
}))

vi.mock('vue-router', async () => {
  const vue = await vi.importActual<typeof import('vue')>('vue')
  routeFixture.route = vue.reactive({ params: { alertId: 'ALT-live-001' } })
  return {
    useRoute: () => routeFixture.route,
    useRouter: () => ({ back: routeFixture.routerBack }),
  }
})

vi.mock('../../components/ReplayPlayer.vue', () => ({
  default: {
    name: 'ReplayPlayer',
    props: ['alertId'],
    template: '<div data-test="replay-player" :data-alert-id="alertId">Replay {{ alertId }}</div>',
  },
}))

describe('ReplayView route contract', () => {
  beforeEach(() => {
    routeFixture.routerBack.mockReset()
    routeFixture.route!.params.alertId = 'ALT-live-001'
  })

  it('passes the route alert id to ReplayPlayer immediately', async () => {
    const wrapper = mount(ReplayView)

    expect(wrapper.text()).toContain('录像回放')
    expect(wrapper.text()).toContain('ALT-live-001')

    const player = wrapper.find('[data-test="replay-player"]')
    expect(player.exists()).toBe(true)
    expect(player.attributes('data-alert-id')).toBe('ALT-live-001')
  })

  it('updates the replay player when the route alert id changes on the same page', async () => {
    const wrapper = mount(ReplayView)

    expect(wrapper.find('[data-test="replay-player"]').attributes('data-alert-id')).toBe('ALT-live-001')

    routeFixture.route!.params.alertId = 'ALT-live-002'
    await nextTick()

    expect(wrapper.text()).toContain('ALT-live-002')
    expect(wrapper.find('[data-test="replay-player"]').attributes('data-alert-id')).toBe('ALT-live-002')
  })

  it('uses router history for the back button', async () => {
    const wrapper = mount(ReplayView)

    await wrapper.find('button[aria-label="返回上一页"]').trigger('click')

    expect(routeFixture.routerBack).toHaveBeenCalledTimes(1)
  })
})
