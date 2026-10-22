import { nextTick, ref } from 'vue'
import { mount, type VueWrapper } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import ReplayPlayer from '../ReplayPlayer.vue'

let controller: ReturnType<typeof makeController>
const mountedWrappers: VueWrapper[] = []

function makeController() {
  return {
    metadata: ref<Record<string, unknown> | null>(null),
    signals: ref<Record<string, unknown> | null>(null),
    hasHeatmaps: ref(false),
    videoUrl: ref('/api/replay/test/video?v=recording'),
    metadataState: ref<'idle' | 'loading' | 'ready' | 'error' | 'unavailable'>('idle'),
    signalsState: ref<'idle' | 'loading' | 'ready' | 'error' | 'unavailable'>('idle'),
    trajectoryState: ref<'idle' | 'loading' | 'ready' | 'error' | 'unavailable'>('idle'),
    videoState: ref<'idle' | 'loading' | 'ready' | 'error' | 'unavailable'>('idle'),
    metadataLoading: ref(false),
    signalsLoading: ref(false),
    metadataError: ref(''),
    metadataErrorStatus: ref<number | null>(null),
    signalsError: ref(''),
    trajectoryError: ref(''),
    videoError: ref(''),
    loading: ref(false),
    loadError: ref(''),
    loadErrorStatus: ref<number | null>(null),
    loadData: vi.fn(),
    loadMetadata: vi.fn(),
    retryMetadata: vi.fn(),
    loadSignals: vi.fn(),
    loadTrajectory: vi.fn(),
    reloadVideo: vi.fn(),
    handleKeydown: vi.fn(),
  }
}

vi.mock('../../composables/useReplayController', () => ({
  useReplayController: () => controller,
}))

vi.mock('../replay/ReplayCanvas.vue', () => ({ default: { template: '<div data-test="canvas" />' } }))
vi.mock('../replay/ReplayControls.vue', () => ({ default: { template: '<div />' } }))
vi.mock('../replay/ReplayReference.vue', () => ({ default: { template: '<div />' } }))
vi.mock('../replay/ReplaySignalTracks.vue', () => ({ default: { template: '<div />' } }))

const ResultStub = {
  props: ['status', 'title', 'subTitle'],
  template: '<section :data-status="status"><h2>{{ title }}</h2><p>{{ subTitle }}</p><slot name="extra" /></section>',
}

function mountReplay(alertId: string) {
  const wrapper = mount(ReplayPlayer, {
    props: { alertId },
    global: { stubs: { Result: ResultStub } },
  })
  mountedWrappers.push(wrapper)
  return wrapper
}

describe('ReplayPlayer async states', () => {
  beforeEach(() => {
    controller = makeController()
  })

  afterEach(() => {
    for (const wrapper of mountedWrappers.splice(0)) wrapper.unmount()
  })

  it('renders a true not-found state for a missing replay', () => {
    controller.metadataError.value = 'replay missing'
    controller.metadataErrorStatus.value = 404
    controller.metadataState.value = 'error'

    const wrapper = mountReplay('missing-alert')

    expect(wrapper.text()).toContain('内容不存在')
    expect(wrapper.text()).not.toContain('没有可回放的数据')
  })

  it('renders a permission state independently from not-found', () => {
    controller.metadataError.value = 'forbidden'
    controller.metadataErrorStatus.value = 403
    controller.metadataState.value = 'error'

    const wrapper = mountReplay('private-alert')

    expect(wrapper.text()).toContain('无权访问')
    expect(wrapper.text()).not.toContain('内容不存在')
  })

  it('keeps previously loaded replay content visible when refresh fails', () => {
    controller.metadata.value = { alert_id: 'stale-alert' }
    controller.metadataError.value = 'refresh timeout'
    controller.metadataErrorStatus.value = 503
    controller.metadataState.value = 'error'

    const wrapper = mountReplay('stale-alert')

    expect(wrapper.find('[data-test="canvas"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('当前显示上次成功加载的数据')
    expect(wrapper.text()).toContain('refresh timeout')
  })

  it('keeps video content visible when signals fail and retries only signals', async () => {
    controller.metadata.value = { alert_id: 'partial-replay' }
    controller.metadataState.value = 'ready'
    controller.signalsState.value = 'error'
    controller.signalsError.value = 'signals unavailable'

    const wrapper = mountReplay('partial-replay')

    expect(wrapper.find('[data-test="canvas"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('信号与时间轴加载失败')
    expect(wrapper.text()).toContain('signals unavailable')

    // The component performs one full initial load on mount. Isolate the
    // retry action so this assertion verifies it does not trigger another.
    controller.loadData.mockClear()
    const retry = wrapper.get('button[aria-label="重试信号与时间轴"]')
    await retry.trigger('click')
    expect(controller.loadSignals).toHaveBeenCalledOnce()
    expect(controller.loadData).not.toHaveBeenCalled()
  })

  it('keeps signals and trajectory UI available when video loading fails', async () => {
    controller.metadata.value = { alert_id: 'video-failed' }
    controller.signals.value = { timestamps: [1, 2] }
    controller.metadataState.value = 'ready'
    controller.signalsState.value = 'ready'
    controller.trajectoryState.value = 'ready'
    controller.videoState.value = 'error'
    controller.videoError.value = 'video decoder failed'

    const wrapper = mountReplay('video-failed')

    expect(wrapper.find('[data-test="canvas"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('录像视频加载失败，信号和轨迹仍可查看')
    expect(wrapper.text()).toContain('video decoder failed')

    const retry = wrapper.findAll('button').find(button => button.text() === '重新加载视频')
    expect(retry).toBeTruthy()
    await retry!.trigger('click')
    expect(controller.reloadVideo).toHaveBeenCalledOnce()
  })

  it('remounts the video surface when heatmaps arrive or recording completes', async () => {
    controller.metadata.value = {
      alert_id: 'mutable-replay',
      status: 'recording',
    }
    controller.metadataState.value = 'ready'
    const wrapper = mountReplay('mutable-replay')
    const recordingCanvas = wrapper.get('[data-test="canvas"]').element

    controller.hasHeatmaps.value = true
    await nextTick()
    const heatmapCanvas = wrapper.get('[data-test="canvas"]').element
    expect(heatmapCanvas).not.toBe(recordingCanvas)

    controller.metadata.value = {
      alert_id: 'mutable-replay',
      status: 'complete',
    }
    controller.videoUrl.value = '/api/replay/test/video?v=complete'
    await nextTick()
    expect(wrapper.get('[data-test="canvas"]').element).not.toBe(heatmapCanvas)
  })
})
