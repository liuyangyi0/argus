import { mount, flushPromises, enableAutoUnmount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { nextTick, ref } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { message } from 'ant-design-vue'

import CameraDetail from '../CameraDetail.vue'
import { useAuthStore } from '../../stores/useAuthStore'
import { getCameraDetail } from '../../api'
import { setCameraMode } from '../../api/cameras'
import { getAnalysisTasks } from '../../api/analysisTasks'
import { updateZones } from '../../api/zones'
import { ApiError, ErrorCode, type CameraDetailPayload } from '../../types/api'

enableAutoUnmount(afterEach)

const routerPush = vi.hoisted(() => vi.fn())
const streamStart = vi.hoisted(() => vi.fn())
const streamStop = vi.hoisted(() => vi.fn())
const streamVideoElementAtStart = vi.hoisted(() => vi.fn())
const routeFixture = vi.hoisted(() => ({
  route: null as { params: { id: string }; query: Record<string, string | string[]> } | null,
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
  routeFixture.route = vue.reactive({ params: { id: 'dev_cam' }, query: {} })
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

vi.mock('../../api/analysisTasks', () => ({
  getAnalysisTasks: vi.fn().mockResolvedValue({ tasks: [] }),
  getAnalysisTaskTemplates: vi.fn().mockResolvedValue({ templates: [] }),
}))

vi.mock('../../api/zones', () => ({
  updateZones: vi.fn(),
}))

vi.mock('../../composables/useGo2RTC', () => ({
  useGo2RTC: (cameraId: string | { value: string }) => {
    const videoRef = ref<HTMLVideoElement | null>(null)
    return {
      videoRef,
      mjpegRef: ref(null),
      status: ref('fallback'),
      fallbackReason: ref('webrtc_failed'),
      start: () => {
        streamVideoElementAtStart(videoRef.value)
        streamStart(typeof cameraId === 'string' ? cameraId : cameraId.value)
      },
      stop: streamStop,
    }
  },
  streamFallbackReasonLabel: (reason: string | null) => (
    reason === 'webrtc_failed' ? 'WebRTC 连接失败' : '实时流不可用'
  ),
}))

vi.mock('@ant-design/icons-vue', () => ({
  ArrowLeftOutlined: { template: '<span />' },
}))

function cameraDetail(
  overrides: Partial<CameraDetailPayload> = {},
): CameraDetailPayload {
  return {
    camera_id: 'dev_cam',
    name: 'Dev file camera',
    connected: true,
    running: true,
    stats: {
      frames_captured: 42,
      frames_analyzed: 40,
      anomalies_detected: 1,
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
    zones: [{ zone_id: 'default', name: 'Default', zone_type: 'include', vertices: [] }],
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
    routeFixture.route!.query = {}
    routerPush.mockReset()
    streamStart.mockReset()
    streamStop.mockReset()
    streamVideoElementAtStart.mockReset()
    vi.mocked(getCameraDetail).mockReset()
    // Polling can race a test's teardown. Keep a typed fallback response so a
    // later tick never resolves to undefined after one-shot fixtures are used.
    vi.mocked(getCameraDetail).mockResolvedValue(cameraDetail())
    vi.mocked(setCameraMode).mockReset()
    vi.mocked(getAnalysisTasks).mockClear()
    vi.mocked(updateZones).mockReset()
    vi.mocked(message.success).mockReset()
    vi.mocked(message.error).mockReset()
  })

  it.each<Record<string, string | string[]>>([
    { analysis_task: ['task-a', 'task-b'] },
    { analysis_task: 'task-a', task_section: ['deployment', 'models'] },
    { analysis_task: 'task-a', task_section: 'unsupported-section' },
  ])('rejects ambiguous task query %j instead of silently opening a new task form', async query => {
    useAuthStore().currentUser = { username: 'engineer', role: 'engineer' }
    routeFixture.route!.query = query
    const wrapper = mount(CameraDetail, { global: { plugins: [pinia], stubs: { ZoneEditor: true, CalibrationWizard: true } } })
    await flushPromises()
    expect(wrapper.text()).toContain('任务定位参数无效')
    expect(wrapper.find('[data-test="create-task"]').exists()).toBe(false)
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
    expect(getAnalysisTasks).toHaveBeenCalledWith('dev_cam')
    expect(streamVideoElementAtStart).toHaveBeenCalledWith(expect.any(HTMLVideoElement))
    const stream = wrapper.find('img.player-vid')
    expect(stream.exists()).toBe(true)
    expect(stream.attributes('src')).toBe('/api/cameras/dev_cam/stream')
    expect(wrapper.text()).toContain('MJPEG fallback')
    expect(wrapper.text()).toContain('非实时预览')
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

  it('shows low-light input quality on the live camera page', async () => {
    vi.mocked(getCameraDetail).mockResolvedValue(cameraDetail({
      detector: {
        low_light: true,
        last_brightness: 4.5,
        ssim_calibration_blocked: true,
      },
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

    const quality = wrapper.find('[data-test="input-quality-strip"]')
    expect(quality.text()).toContain('输入质量')
    expect(quality.text()).toContain('低光校准等待')
    expect(quality.text()).toContain('亮度 4.5')

    wrapper.unmount()
  })

  it('shows model backend mismatch on the live camera page', async () => {
    vi.mocked(getCameraDetail).mockResolvedValue(cameraDetail({
      model_backend: 'openvino',
      expected_model_backend: 'torch-cuda',
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

    const modelBackend = wrapper.find('[data-test="model-backend-strip"]')
    expect(modelBackend.text()).toContain('模型后端')
    expect(modelBackend.text()).toContain('模型后端不符合要求')
    expect(modelBackend.text()).toContain('当前 openvino')
    expect(modelBackend.text()).toContain('期望 torch-cuda')

    wrapper.unmount()
  })

  it('keeps preview available while SceneGuard blocks scoring and alerts', async () => {
    vi.mocked(getCameraDetail).mockResolvedValue(cameraDetail({
      runtime: {
        pipeline_mode: 'active',
        scene_guard: {
          enabled: true,
          usable: false,
          state: 'blocked',
          reason: 'camera pose moved beyond approved profile',
          profile_id: 'scene-main-v3',
          bound_model_version: 'model-v12',
          policy_recovery_count: 1,
        },
      },
      config: {
        scene_guard: { consecutive_matches: 3 },
      },
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

    const banner = wrapper.find('[data-test="scene-guard-blocked"]')
    expect(banner.text()).toContain('机位不匹配，检测已暂停')
    expect(banner.text()).toContain('视频预览保持可用')
    expect(banner.text()).toContain('异常评分与告警已暂停')
    expect(banner.text()).toContain('scene-main-v3')
    expect(banner.text()).toContain('model-v12')
    expect(banner.text()).toContain('1/3')
    expect(wrapper.find('img.player-vid').exists()).toBe(true)
    wrapper.unmount()
  })

  it('renders an explicit not-found state instead of an offline camera', async () => {
    vi.mocked(getCameraDetail).mockRejectedValue(new ApiError({
      httpStatus: 404,
      businessCode: ErrorCode.NOT_FOUND,
      message: '摄像头 dev_cam 不存在',
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

    expect(wrapper.find('[data-test="camera-not-found"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('未找到摄像头 dev_cam')
    expect(wrapper.text()).not.toContain('摄像头离线')
    expect(streamStart).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('deduplicates overlapping detail polls', async () => {
    vi.useFakeTimers()
    let resolveRequest!: (value: ReturnType<typeof cameraDetail>) => void
    const pending = new Promise<ReturnType<typeof cameraDetail>>((resolve) => {
      resolveRequest = resolve
    })
    vi.mocked(getCameraDetail).mockReturnValue(pending)

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

      await vi.advanceTimersByTimeAsync(15_000)
      expect(getCameraDetail).toHaveBeenCalledTimes(1)
      resolveRequest(cameraDetail())
      await flushPromises()
      expect(wrapper.find('[data-test="camera-loading"]').exists()).toBe(false)
      wrapper.unmount()
    } finally {
      vi.useRealTimers()
    }
  })

  it('renders runtime source diagnostics on the detail page', async () => {
    vi.mocked(getCameraDetail).mockResolvedValue(cameraDetail({
      source_diagnostics: {
        declared_source: '0',
        runtime_source: 'rtsp://127.0.0.1:8554/dev_cam',
        go2rtc_stream: 'dev_cam',
        fps_observed: 24.8,
        last_frame_index: 12345,
        last_snapshot_frame_index: 12340,
        last_preview_stream_frame_index: 12344,
        last_snapshot_ts: 1710000000.15,
        source_consistency: 'go2rtc_redirect',
      },
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

    const diagnostics = wrapper.find('[data-test="source-diagnostics"]')
    expect(diagnostics.text()).toContain('运行源诊断')
    expect(diagnostics.text()).toContain('声明源')
    expect(diagnostics.text()).toContain('0')
    expect(diagnostics.text()).toContain('实际检测源')
    expect(diagnostics.text()).toContain('rtsp://127.0.0.1:8554/dev_cam')
    expect(diagnostics.text()).toContain('go2rtc 重定向')
    expect(diagnostics.text()).toContain('24.8')
    expect(diagnostics.text()).toContain('12345')
    expect(diagnostics.text()).toContain('snapshot frame index')
    expect(diagnostics.text()).toContain('12340')
    expect(diagnostics.text()).toContain('后端预览 frame index')
    expect(diagnostics.text()).toContain('12344')

    wrapper.unmount()
  })

  it('renders candidate diagnostics evidence on the detail page', async () => {
    vi.mocked(getCameraDetail).mockResolvedValue(cameraDetail({
      candidate_diagnostics: [
        {
          frame_number: 88,
          timestamp: 1710000010.2,
          final_decision: 'info',
          candidate_records: [
            {
              candidate_id: 'dev_cam:88:1',
              candidate_type: 'persistent_change',
              decision: 'candidate',
              stage: 'persistent_change',
              suppression_reason: null,
              evidence: {
                ready: true,
                has_detection: true,
                score: 0.87,
                changed_fraction: 0.0021,
                candidate_count: 1,
                active_track_count: 1,
                candidates: [
                  {
                    track_id: 7,
                    bbox: [50, 40, 32, 24],
                    area_px: 768,
                    persistence_seconds: 3.2,
                    score: 0.87,
                  },
                ],
              },
            },
            {
              candidate_id: 'dev_cam:88:2',
              candidate_type: 'frame_anomaly',
              decision: 'info',
              stage: 'decision',
              suppression_reason: 'grader_no_alert',
              evidence: {
                anomaly_score: 0.87,
                detection_type: 'persistent_change',
                persistent_change: {
                  score: 0.87,
                  candidate_count: 1,
                  active_track_count: 1,
                },
              },
            },
          ],
        },
      ],
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

    const diagnostics = wrapper.find('[data-test="candidate-diagnostics"]')
    expect(diagnostics.text()).toContain('候选诊断')
    expect(diagnostics.text()).toContain('固定场景变化')
    expect(diagnostics.text()).toContain('候选')
    expect(diagnostics.text()).toContain('score0.870')
    expect(diagnostics.text()).toContain('变化占比0.002100')
    expect(diagnostics.text()).toContain('bbox 50, 40, 32, 24')
    expect(diagnostics.text()).toContain('768px')
    expect(diagnostics.text()).toContain('3.2s')
    expect(diagnostics.text()).toContain('帧异常')
    expect(diagnostics.text()).toContain('grader_no_alert')

    wrapper.unmount()
  })

  it('renders candidate diagnostics empty state', async () => {
    vi.mocked(getCameraDetail).mockResolvedValue(cameraDetail({
      candidate_diagnostics: [],
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

    const diagnostics = wrapper.find('[data-test="candidate-diagnostics"]')
    expect(diagnostics.text()).toContain('候选诊断')
    expect(diagnostics.text()).toContain('暂无候选记录')

    wrapper.unmount()
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
            anomalies_detected: 0,
            alerts_emitted: 0,
            avg_latency_ms: 18.2,
          },
          zones: [{ zone_id: 'zone_b', name: 'Zone B', zone_type: 'include', vertices: [] }],
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
    expect(getAnalysisTasks).toHaveBeenCalledWith('cam_02')
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
    expect(updateZones).toHaveBeenCalledWith('dev_cam', [{
      zone_id: 'default',
      name: 'Default',
      zone_type: 'include',
      vertices: [],
    }])
    expect(message.success).toHaveBeenCalledWith('区域配置已保存')
  })
})
