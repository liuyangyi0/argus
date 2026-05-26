import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { nextTick, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import CameraDetail from '../CameraDetail.vue'
import { useAuthStore } from '../../stores/useAuthStore'
import { getCameraDetail } from '../../api'
import { setCameraMode } from '../../api/cameras'
import { updateZones } from '../../api/zones'

const routerPush = vi.hoisted(() => vi.fn())
const streamStart = vi.hoisted(() => vi.fn())
const streamStop = vi.hoisted(() => vi.fn())
const routeFixture = vi.hoisted(() => ({
  route: null as { params: { id: string } } | null,
}))
const antStubs = vi.hoisted(() => ({
  button: {
    props: ['type', 'size', 'disabled'],
    emits: ['click'],
    template: '<button type="button" :disabled="disabled" @click="$emit(\'click\', $event)"><slot /></button>',
  },
  segmented: {
    props: ['value', 'options', 'disabled'],
    emits: ['change'],
    template: `
      <div>
        <button
          v-for="option in options"
          :key="option.value"
          type="button"
          :disabled="disabled"
          @click="$emit('change', option.value)"
        >
          {{ option.label }}
        </button>
      </div>
    `,
  },
  steps: {
    Step: { props: ['title'], template: '<span>{{ title }}</span>' },
    template: '<div><slot /></div>',
  },
}))
let pinia: ReturnType<typeof createPinia>

vi.mock('vue-router', async () => {
  const vue = await vi.importActual<typeof import('vue')>('vue')
  routeFixture.route = vue.reactive({ params: { id: 'dev_cam' } })
  return {
    useRoute: () => routeFixture.route,
    useRouter: () => ({ push: routerPush }),
  }
})

vi.mock('../../api', () => ({
  api: { get: vi.fn() },
  getCameraDetail: vi.fn(),
}))

vi.mock('../../api/cameras', () => ({
  setCameraMode: vi.fn(),
}))

vi.mock('../../api/zones', () => ({
  updateZones: vi.fn(),
}))

vi.mock('../../composables/useGo2RTC', () => ({
  useGo2RTC: (cameraId: string | { value: string }) => ({
    videoRef: ref(null),
    mjpegRef: ref(null),
    status: ref('fallback'),
    start: () => streamStart(typeof cameraId === 'string' ? cameraId : cameraId.value),
    stop: streamStop,
  }),
}))

vi.mock('@ant-design/icons-vue', () => ({
  ArrowLeftOutlined: { template: '<span />' },
}))

function cameraDetail(overrides: Record<string, any> = {}) {
  return {
    camera_id: 'dev_cam',
    name: 'Dev file camera',
    connected: true,
    running: true,
    stats: {
      frames_captured: 42,
      frames_analyzed: 40,
      alerts_emitted: 1,
      avg_latency_ms: 12.5,
    },
    runtime: {
      pipeline_mode: 'active',
    },
    stages: [
      { name: 'capture', status: 'completed' },
      { name: 'baseline', status: 'completed' },
      { name: 'training', status: 'completed' },
      { name: 'release', status: 'completed' },
      { name: 'inference', status: 'active' },
    ],
    zones: [{ zone_id: 'default', name: 'Default' }],
    ...overrides,
  }
}

const passthrough = { template: '<div><slot /></div>' }

vi.mock('ant-design-vue', () => ({
  Button: antStubs.button,
  Segmented: antStubs.segmented,
  Steps: antStubs.steps,
  message: {
    success: vi.fn(),
    error: vi.fn(),
  },
}))

describe('CameraDetail live stream fallback', () => {
  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)
    useAuthStore().currentUser = { username: 'operator', role: 'operator' }
    routeFixture.route!.params.id = 'dev_cam'
    routerPush.mockReset()
    streamStart.mockReset()
    streamStop.mockReset()
    vi.mocked(getCameraDetail).mockReset()
    vi.mocked(setCameraMode).mockReset()
    vi.mocked(updateZones).mockReset()
  })

  it('renders MJPEG stream when go2rtc falls back for a connected camera', async () => {
    vi.mocked(getCameraDetail).mockResolvedValue(cameraDetail())

    const wrapper = mount(CameraDetail, {
      global: {
        plugins: [pinia],
        stubs: {
          ZoneEditor: true,
          CalibrationWizard: true,
          Message: passthrough,
        },
      },
    })
    await flushPromises()

    expect(streamStart).toHaveBeenCalledTimes(1)
    expect(streamStart).toHaveBeenCalledWith('dev_cam')
    const stream = wrapper.find('img.player-vid')
    expect(stream.exists()).toBe(true)
    expect(stream.attributes('src')).toBe('/api/cameras/dev_cam/stream')
    expect(wrapper.text()).toContain('Dev file camera')
    expect(wrapper.text()).toContain('已采集帧')
    expect(wrapper.text()).toContain('42')

    wrapper.unmount()
    expect(streamStop).toHaveBeenCalled()
  })

  it('does not open a stream for an offline camera detail', async () => {
    vi.mocked(getCameraDetail).mockResolvedValue(cameraDetail({
      connected: false,
      running: false,
    }))

    const wrapper = mount(CameraDetail, {
      global: {
        plugins: [pinia],
        stubs: {
          ZoneEditor: true,
          CalibrationWizard: true,
          Message: passthrough,
        },
      },
    })
    await flushPromises()

    expect(streamStart).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('摄像头离线')
    expect(wrapper.find('img.player-vid').exists()).toBe(false)
  })

  it('stops an active stream when polling reports the camera offline', async () => {
    vi.useFakeTimers()
    vi.mocked(getCameraDetail)
      .mockResolvedValueOnce(cameraDetail())
      .mockResolvedValueOnce(cameraDetail({
        connected: false,
        running: false,
      }))

    try {
      const wrapper = mount(CameraDetail, {
        global: {
          plugins: [pinia],
          stubs: {
            ZoneEditor: true,
            CalibrationWizard: true,
            Message: passthrough,
          },
        },
      })
      await flushPromises()

      expect(streamStart).toHaveBeenCalledTimes(1)
      vi.advanceTimersByTime(5000)
      await flushPromises()

      expect(streamStop).toHaveBeenCalledTimes(1)
      expect(wrapper.text()).toContain('摄像头离线')
      wrapper.unmount()
    } finally {
      vi.useRealTimers()
    }
  })

  it('reloads camera detail and stream when the route camera id changes', async () => {
    vi.mocked(getCameraDetail).mockImplementation(async (id: string) => {
      if (id === 'cam_02') {
        return cameraDetail({
          camera_id: 'cam_02',
          name: 'Second camera',
          stats: {
            frames_captured: 7,
            frames_analyzed: 6,
            alerts_emitted: 0,
            avg_latency_ms: 18.2,
          },
          zones: [{ zone_id: 'zone_b', name: 'Zone B' }],
        })
      }
      return cameraDetail()
    })

    const wrapper = mount(CameraDetail, {
      global: {
        plugins: [pinia],
        stubs: {
          ZoneEditor: {
            props: ['modelValue', 'imageSrc'],
            template: '<div data-test="zone-editor">{{ imageSrc }} {{ modelValue?.[0]?.zone_id }}</div>',
          },
          CalibrationWizard: true,
          Message: passthrough,
        },
      },
    })
    await flushPromises()

    expect(getCameraDetail).toHaveBeenCalledWith('dev_cam')
    expect(wrapper.find('img.player-vid').attributes('src')).toBe('/api/cameras/dev_cam/stream')
    await wrapper.findAll('button').find(button => button.text().includes('区域编辑'))!.trigger('click')
    await nextTick()
    expect(wrapper.find('[data-test="zone-editor"]').text()).toContain('default')

    routeFixture.route!.params.id = 'cam_02'
    await nextTick()
    await flushPromises()

    expect(streamStop).toHaveBeenCalled()
    expect(streamStart).toHaveBeenLastCalledWith('cam_02')
    expect(getCameraDetail).toHaveBeenCalledWith('cam_02')
    expect(wrapper.text()).toContain('Second camera')
    expect(wrapper.find('[data-test="zone-editor"]').text()).toContain('zone_b')
    expect(wrapper.find('[data-test="zone-editor"]').text()).toContain('/api/cameras/cam_02/snapshot')

    await wrapper.findAll('button').find(button => button.text().includes('实时画面'))!.trigger('click')
    await nextTick()
    expect(wrapper.find('img.player-vid').attributes('src')).toBe('/api/cameras/cam_02/stream')
  })

  it('lets engineers operate camera mode and save zones', async () => {
    useAuthStore().currentUser = { username: 'engineer', role: 'engineer' }
    vi.mocked(getCameraDetail).mockResolvedValue(cameraDetail())
    vi.mocked(setCameraMode).mockResolvedValue({ pipeline_mode: 'maintenance' } as any)
    vi.mocked(updateZones).mockResolvedValue({ count: 1 } as any)

    const wrapper = mount(CameraDetail, {
      global: {
        plugins: [pinia],
        stubs: {
          ZoneEditor: {
            props: ['modelValue', 'imageSrc'],
            template: '<div data-test="zone-editor">{{ imageSrc }} {{ modelValue?.[0]?.zone_id }}</div>',
          },
          CalibrationWizard: true,
          Message: passthrough,
        },
      },
    })
    await flushPromises()

    await wrapper.findAll('button').find(button => button.text().includes('维护'))!.trigger('click')
    await flushPromises()
    expect(setCameraMode).toHaveBeenCalledWith('dev_cam', 'maintenance')

    await wrapper.findAll('button').find(button => button.text().includes('区域编辑'))!.trigger('click')
    await nextTick()
    expect(wrapper.text()).toContain('保存区域配置')

    await wrapper.findAll('button').find(button => button.text().includes('保存区域配置'))!.trigger('click')
    await flushPromises()
    expect(updateZones).toHaveBeenCalledWith('dev_cam', [{ zone_id: 'default', name: 'Default' }])
  })
})
