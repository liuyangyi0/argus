import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
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

vi.mock('../../components/ContentSkeleton.vue', () => ({
  default: {
    name: 'ContentSkeleton',
    props: ['type', 'rows'],
    template: '<div data-test="replay-skeleton">{{ type }} {{ rows }}</div>',
  },
}))

describe('ReplayView route contract', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    routeFixture.routerBack.mockReset()
    routeFixture.route!.params.alertId = 'ALT-live-001'
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('passes the route alert id to ReplayPlayer after the route-entry skeleton', async () => {
    const wrapper = mount(ReplayView)

    expect(wrapper.text()).toContain('录像回放')
    expect(wrapper.text()).toContain('ALT-live-001')
    expect(wrapper.find('[data-test="replay-skeleton"]').exists()).toBe(true)
    expect(wrapper.find('[data-test="replay-player"]').exists()).toBe(false)

    vi.advanceTimersByTime(400)
    await nextTick()

    const player = wrapper.find('[data-test="replay-player"]')
    expect(player.exists()).toBe(true)
    expect(player.attributes('data-alert-id')).toBe('ALT-live-001')
    expect(wrapper.find('[data-test="replay-skeleton"]').exists()).toBe(false)
  })

  it('updates the replay player when the route alert id changes on the same page', async () => {
    const wrapper = mount(ReplayView)

    vi.advanceTimersByTime(400)
    await nextTick()

    expect(wrapper.find('[data-test="replay-player"]').attributes('data-alert-id')).toBe('ALT-live-001')

    routeFixture.route!.params.alertId = 'ALT-live-002'
    await nextTick()

    expect(wrapper.text()).toContain('ALT-live-002')
    expect(wrapper.find('[data-test="replay-player"]').attributes('data-alert-id')).toBe('ALT-live-002')
  })

  it('uses router history for the back button', async () => {
    const wrapper = mount(ReplayView)

    await wrapper.find('button[title="返回"]').trigger('click')

    expect(routeFixture.routerBack).toHaveBeenCalledTimes(1)
  })
})
