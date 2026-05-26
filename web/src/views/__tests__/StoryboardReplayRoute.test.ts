import { mount, flushPromises } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import StoryboardReplay from '../StoryboardReplay.vue'

const routeFixture = vi.hoisted(() => ({
  routerBack: vi.fn(),
  routerPush: vi.fn(),
  route: null as { params: { alertId: string } } | null,
}))

const storyboardHarness = vi.hoisted(() => ({
  loadIds: [] as string[],
}))

vi.mock('vue-router', async () => {
  const vue = await vi.importActual<typeof import('vue')>('vue')
  routeFixture.route = vue.reactive({ params: { alertId: 'ALT-story-001' } })
  return {
    useRoute: () => routeFixture.route,
    useRouter: () => ({
      back: routeFixture.routerBack,
      push: routeFixture.routerPush,
    }),
  }
})

vi.mock('../../composables/useStoryboardController', async () => {
  const vue = await vi.importActual<typeof import('vue')>('vue')
  return {
    useStoryboardController: vi.fn((alertId) => ({
      cameras: vue.ref([
        {
          alert_id: 'ALT-story-001',
          camera_id: 'cam_01',
          trigger_timestamp: 0,
          metadata_url: '/api/replay/ALT-story-001/metadata',
          video_url: '/api/replay/ALT-story-001/video',
          signals_url: '/api/replay/ALT-story-001/signals',
          trigger_offset_s: 0,
        },
      ]),
      loading: vue.ref(false),
      error: vue.ref(null),
      masterTime: vue.ref(0),
      playing: vue.ref(false),
      speed: vue.ref(1),
      timelineStart: vue.ref(0),
      timelineEnd: vue.ref(10),
      timelineDuration: vue.ref(10),
      load: vi.fn(async () => {
        storyboardHarness.loadIds.push(vue.unref(alertId))
      }),
      handleKeydown: vi.fn(),
      reportDuration: vi.fn(),
      seek: vi.fn(),
      togglePlay: vi.fn(),
      setSpeed: vi.fn(),
    })),
  }
})

vi.mock('../../components/replay/StoryboardPlayer.vue', () => ({
  default: {
    name: 'StoryboardPlayer',
    props: ['camera'],
    template: '<div data-test="storyboard-player">{{ camera.alert_id }}</div>',
  },
}))

vi.mock('@ant-design/icons-vue', () => ({
  CaretRightOutlined: { template: '<span />' },
  PauseOutlined: { template: '<span />' },
}))

describe('StoryboardReplay route contract', () => {
  beforeEach(() => {
    routeFixture.routerBack.mockReset()
    routeFixture.routerPush.mockReset()
    storyboardHarness.loadIds = []
    routeFixture.route!.params.alertId = 'ALT-story-001'
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('reloads storyboard data when the route alert id changes on the same page', async () => {
    const wrapper = mount(StoryboardReplay)
    await flushPromises()

    expect(wrapper.text()).toContain('多机位回放')
    expect(wrapper.text()).toContain('ALT-story-001')
    expect(storyboardHarness.loadIds).toEqual(['ALT-story-001'])
    expect(wrapper.find('[data-test="storyboard-player"]').exists()).toBe(true)

    routeFixture.route!.params.alertId = 'ALT-story-002'
    await nextTick()
    await flushPromises()

    expect(wrapper.text()).toContain('ALT-story-002')
    expect(storyboardHarness.loadIds).toEqual(['ALT-story-001', 'ALT-story-002'])

    await wrapper.findAll('button').find(button => button.text().includes('单机位'))!.trigger('click')
    expect(routeFixture.routerPush).toHaveBeenCalledWith('/replay/ALT-story-002')
  })

  it('uses router history for the back button', async () => {
    const wrapper = mount(StoryboardReplay)
    await wrapper.find('button[title="返回"]').trigger('click')

    expect(routeFixture.routerBack).toHaveBeenCalledTimes(1)
  })
})
