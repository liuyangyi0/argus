import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import Cameras from '../Cameras.vue'
import { useAuthStore } from '../../stores/useAuthStore'
import {
  addCamera,
  getCameras,
  getRegions,
  getUsbDevices,
} from '../../api'
import {
  testCameraConnection,
  testCameraConnectionDraft,
} from '../../api/cameras'
import { ApiError } from '../../types/api'

const routerPush = vi.hoisted(() => vi.fn())
const antStubs = vi.hoisted(() => ({
  button: {
    props: ['type', 'size', 'loading', 'disabled'],
    emits: ['click'],
    template: '<button type="button" :disabled="disabled" @click="$emit(\'click\', $event)"><slot name="icon" /><slot /></button>',
  },
  modal: {
    props: ['open', 'title', 'okText'],
    emits: ['ok'],
    template: `
      <section v-if="open" class="ant-modal">
        <h2>{{ title }}</h2>
        <slot />
        <button type="button" data-test="modal-ok" @click="$emit('ok')">{{ okText }}</button>
      </section>
    `,
  },
  form: {
    Item: {
      props: ['label'],
      template: '<label><span>{{ label }}</span><slot /></label>',
    },
    template: '<form><slot /></form>',
  },
  input: {
    props: ['value', 'placeholder', 'disabled'],
    emits: ['update:value'],
    template: '<input class="ant-input" :placeholder="placeholder" :disabled="disabled" :value="value" @input="$emit(\'update:value\', $event.target.value)" />',
  },
  inputNumber: {
    props: ['value', 'min', 'max', 'step'],
    emits: ['update:value'],
    template: '<input class="ant-input-number" type="number" :value="value" @input="$emit(\'update:value\', Number($event.target.value))" />',
  },
  select: {
    Option: {
      props: ['value'],
      template: '<option :value="value"><slot /></option>',
    },
    props: ['value', 'loading', 'placeholder'],
    emits: ['update:value'],
    template: '<select :value="value" @change="$emit(\'update:value\', $event.target.value)"><slot /></select>',
  },
  table: {
    props: ['columns', 'dataSource', 'customRow'],
    template: `
      <table>
        <tbody>
          <tr
            v-for="record in dataSource"
            :key="record.camera_id"
            class="camera-row"
            v-bind="customRow ? customRow(record) : {}"
          >
            <td v-for="column in columns" :key="column.key" :data-col="column.key">
              <slot name="bodyCell" :column="column" :record="record" />
              <template v-if="column.dataIndex">{{ record[column.dataIndex] }}</template>
            </td>
          </tr>
        </tbody>
      </table>
    `,
  },
  typography: {
    Title: {
      props: ['level'],
      template: '<h1><slot /></h1>',
    },
    Text: {
      template: '<span><slot /></span>',
    },
  },
  space: { template: '<span><slot /></span>' },
  badge: {
    props: ['status'],
    template: '<span class="ant-badge" :data-status="status" />',
  },
  result: {
    props: ['status', 'title', 'subTitle'],
    template: '<section :data-status="status"><h2>{{ title }}</h2><p>{{ subTitle }}</p><slot name="extra" /></section>',
  },
  popconfirm: {
    emits: ['confirm'],
    template: '<span><slot /></span>',
  },
  card: { template: '<div><slot /></div>' },
  skeleton: { template: '<div data-test="skeleton" />' },
}))
let pinia: ReturnType<typeof createPinia>

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush }),
}))

vi.mock('../../composables/useWebSocket', () => ({
  useWebSocket: vi.fn(() => ({})),
}))

vi.mock('../../api', () => ({
  api: { get: vi.fn() },
  addCamera: vi.fn(),
  deleteCamera: vi.fn(),
  getCameraConfig: vi.fn(),
  getCameras: vi.fn(),
  getRegions: vi.fn(),
  getUsbDevices: vi.fn(),
  startCamera: vi.fn(),
  stopCamera: vi.fn(),
  updateCamera: vi.fn(),
}))

vi.mock('../../api/cameras', () => ({
  testCameraConnection: vi.fn(),
  testCameraConnectionDraft: vi.fn(),
}))

vi.mock('ant-design-vue', () => ({
  Badge: antStubs.badge,
  Button: antStubs.button,
  Card: antStubs.card,
  Form: antStubs.form,
  Input: antStubs.input,
  InputNumber: antStubs.inputNumber,
  Modal: antStubs.modal,
  Popconfirm: antStubs.popconfirm,
  Result: antStubs.result,
  Select: antStubs.select,
  Skeleton: antStubs.skeleton,
  Space: antStubs.space,
  Table: antStubs.table,
  Typography: antStubs.typography,
  message: {
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
  },
}))

vi.mock('@ant-design/icons-vue', () => ({
  DeleteOutlined: { template: '<span />' },
  EditOutlined: { template: '<span />' },
  PlusOutlined: { template: '<span />' },
  CheckCircleOutlined: { template: '<span />' },
  CloseCircleOutlined: { template: '<span />' },
  ExclamationCircleOutlined: { template: '<span />' },
  InfoCircleOutlined: { template: '<span />' },
  LoadingOutlined: { template: '<span />' },
  DatabaseOutlined: { template: '<span />' },
  VideoCameraOutlined: { template: '<span />' },
}))

function formEntries(form: FormData): Record<string, string> {
  const entries: Record<string, string> = {}
  form.forEach((value, key) => {
    entries[key] = String(value)
  })
  return entries
}

describe('Cameras page add camera flow', () => {
  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)
    useAuthStore().currentUser = { username: 'engineer', role: 'engineer' }
    routerPush.mockReset()
    vi.mocked(addCamera).mockReset()
    vi.mocked(getCameras).mockReset()
    vi.mocked(getRegions).mockReset()
    vi.mocked(getUsbDevices).mockReset()
    vi.mocked(testCameraConnection).mockReset()
    vi.mocked(testCameraConnectionDraft).mockReset()
    vi.mocked(getCameras).mockResolvedValue({ cameras: [] } as any)
    vi.mocked(getRegions).mockResolvedValue({ regions: [] } as any)
  })

  it('keeps each source draft and ignores USB discovery that arrives after switching to network', async () => {
    let finish!: (result: Awaited<ReturnType<typeof getUsbDevices>>) => void
    vi.mocked(getUsbDevices).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const wrapper = mount(Cameras, { global: { plugins: [pinia], stubs: {
      ConnectionTestResult: true, HealthBadge: true, ModeBadge: true,
    } } })
    await flushPromises()
    await wrapper.findAll('button').find(button => button.text().includes('新增摄像头'))!.trigger('click')
    await flushPromises()
    await wrapper.findAll('input.ant-input')[2]!.setValue('rtsp://camera.example/stream')
    const type = wrapper.get('[data-test="camera-type"]')
    await type.setValue('usb'); await flushPromises()
    await type.setValue('rtsp'); await flushPromises()
    finish({ devices: [{ index: 1, name: 'Late USB camera', width: 640, height: 480 }] })
    await flushPromises()
    expect((wrapper.findAll('input.ant-input')[2]!.element as HTMLInputElement).value).toBe('rtsp://camera.example/stream')
    expect(wrapper.text()).toContain('网络视频流地址')
    expect(wrapper.text()).not.toContain('Late USB camera')
    wrapper.unmount()
  })

  it('allows an explicit USB index zero when device discovery cannot list an occupied device', async () => {
    vi.mocked(getUsbDevices).mockResolvedValueOnce({ devices: [] })
    vi.mocked(testCameraConnectionDraft).mockResolvedValueOnce({ ok: true } as never)
    const wrapper = mount(Cameras, { global: { plugins: [pinia], stubs: {
      ConnectionTestResult: true, HealthBadge: true, ModeBadge: true,
    } } })
    await flushPromises()
    await wrapper.findAll('button').find(button => button.text().includes('新增摄像头'))!.trigger('click')
    await flushPromises()
    await wrapper.get('[data-test="camera-type"]').setValue('usb'); await flushPromises()
    expect((wrapper.get('[data-test="usb-device-index"]').element as HTMLInputElement).value).toBe('0')
    expect(wrapper.text()).toContain('0 表示第一台 USB 相机')
    await wrapper.findAll('button').find(button => button.text().includes('测试连接'))!.trigger('click')
    await flushPromises()
    expect(testCameraConnectionDraft).toHaveBeenCalledWith({ source: '0', protocol: 'usb' })
    wrapper.unmount()
  })

  it('submits file camera settings and runs draft plus saved connection probes', async () => {
    vi.mocked(testCameraConnectionDraft).mockResolvedValue({
      ok: true,
      detail: { width: 640, height: 480, fps: 10 },
    } as any)
    vi.mocked(addCamera).mockResolvedValue({ camera_id: 'file_cam' } as any)
    vi.mocked(testCameraConnection).mockResolvedValue({
      ok: true,
      detail: { width: 640, height: 480, fps: 10 },
    } as any)

    const wrapper = mount(Cameras, {
      global: {
        plugins: [pinia],
        stubs: {
          ConnectionTestResult: {
            props: ['state', 'result'],
            template: '<div data-test="probe">{{ state }} {{ result?.ok }}</div>',
          },
          ContentSkeleton: true,
          HealthBadge: true,
          ModeBadge: true,
        },
      },
    })
    await flushPromises()

    await wrapper.findAll('button').find(button => button.text().includes('新增摄像头'))!.trigger('click')
    await flushPromises()

    const protocolSelect = wrapper.findAll('select').find(select => select.text().includes('文件'))
    expect(protocolSelect).toBeTruthy()
    await protocolSelect!.setValue('file')
    await flushPromises()

    const inputs = wrapper.findAll('input.ant-input')
    await inputs[0].setValue('file_cam')
    await inputs[1].setValue('File camera')
    await inputs[2].setValue('data/dev/dev_camera.avi')

    await wrapper.findAll('button').find(button => button.text().includes('测试连接'))!.trigger('click')
    await flushPromises()
    expect(testCameraConnectionDraft).toHaveBeenCalledWith({
      source: 'data/dev/dev_camera.avi',
      protocol: 'file',
    })
    expect(wrapper.find('[data-test="probe"]').text()).toContain('true')

    await wrapper.find('[data-test="modal-ok"]').trigger('click')
    await flushPromises()

    expect(addCamera).toHaveBeenCalledTimes(1)
    const submitted = formEntries(vi.mocked(addCamera).mock.calls[0][0] as FormData)
    expect(submitted).toMatchObject({
      camera_id: 'file_cam',
      name: 'File camera',
      source: 'data/dev/dev_camera.avi',
      protocol: 'file',
      fps_target: '5',
      resolution: '1920,1080',
    })
    expect(testCameraConnection).toHaveBeenCalledWith('file_cam')
    expect(wrapper.find('.ant-modal').exists()).toBe(false)
  })

  it('auto-selects a detected USB device and submits the USB source unchanged', async () => {
    vi.mocked(getUsbDevices).mockResolvedValue({
      devices: [
        { index: 0, name: 'USB Camera 0', width: 640, height: 480 },
      ],
    } as any)
    vi.mocked(testCameraConnectionDraft).mockResolvedValue({
      ok: true,
      detail: { width: 640, height: 480, fps: 15 },
    } as any)
    vi.mocked(addCamera).mockResolvedValue({ camera_id: 'usb_cam' } as any)
    vi.mocked(testCameraConnection).mockResolvedValue({
      ok: true,
      detail: { width: 640, height: 480, fps: 15 },
    } as any)

    const wrapper = mount(Cameras, {
      global: {
        plugins: [pinia],
        stubs: {
          ConnectionTestResult: {
            props: ['state', 'result'],
            template: '<div data-test="probe">{{ state }} {{ result?.ok }}</div>',
          },
          ContentSkeleton: true,
          HealthBadge: true,
          ModeBadge: true,
        },
      },
    })
    await flushPromises()

    await wrapper.findAll('button').find(button => button.text().includes('新增摄像头'))!.trigger('click')
    await flushPromises()

    const protocolSelect = wrapper.findAll('select').find(select => select.text().includes('USB'))
    expect(protocolSelect).toBeTruthy()
    await protocolSelect!.setValue('usb')
    await flushPromises()

    expect(getUsbDevices).toHaveBeenCalledTimes(1)
    expect(wrapper.findAll('select').some(select => select.text().includes('USB Camera 0'))).toBe(true)

    const inputs = wrapper.findAll('input.ant-input')
    await inputs[0].setValue('usb_cam')
    await inputs[1].setValue('USB smoke camera')

    await wrapper.findAll('button').find(button => button.text().includes('测试连接'))!.trigger('click')
    await flushPromises()
    expect(testCameraConnectionDraft).toHaveBeenCalledWith({
      source: '0',
      protocol: 'usb',
    })

    await wrapper.find('[data-test="modal-ok"]').trigger('click')
    await flushPromises()

    expect(addCamera).toHaveBeenCalledTimes(1)
    const submitted = formEntries(vi.mocked(addCamera).mock.calls[0][0] as FormData)
    expect(submitted).toMatchObject({
      camera_id: 'usb_cam',
      name: 'USB smoke camera',
      source: '0',
      protocol: 'usb',
      fps_target: '5',
      resolution: '1920,1080',
    })
    expect(testCameraConnection).toHaveBeenCalledWith('usb_cam')
    expect(wrapper.find('.ant-modal').exists()).toBe(false)
  })

  it('submits an RTSP camera with the default protocol unchanged', async () => {
    vi.mocked(testCameraConnectionDraft).mockResolvedValue({
      ok: true,
      detail: { width: 1920, height: 1080, fps: 25 },
    } as any)
    vi.mocked(addCamera).mockResolvedValue({ camera_id: 'rtsp_cam' } as any)
    vi.mocked(testCameraConnection).mockResolvedValue({
      ok: true,
      detail: { width: 1920, height: 1080, fps: 25 },
    } as any)

    const wrapper = mount(Cameras, {
      global: {
        plugins: [pinia],
        stubs: {
          ConnectionTestResult: {
            props: ['state', 'result'],
            template: '<div data-test="probe">{{ state }} {{ result?.ok }}</div>',
          },
          ContentSkeleton: true,
          HealthBadge: true,
          ModeBadge: true,
        },
      },
    })
    await flushPromises()

    await wrapper.findAll('button').find(button => button.text().includes('新增摄像头'))!.trigger('click')
    await flushPromises()

    const inputs = wrapper.findAll('input.ant-input')
    await inputs[0].setValue('rtsp_cam')
    await inputs[1].setValue('RTSP camera')
    await inputs[2].setValue('rtsp://admin:pass@192.0.2.100:554/stream')

    await wrapper.findAll('button').find(button => button.text().includes('测试连接'))!.trigger('click')
    await flushPromises()
    expect(testCameraConnectionDraft).toHaveBeenCalledWith({
      source: 'rtsp://admin:pass@192.0.2.100:554/stream',
      protocol: 'rtsp',
    })

    await wrapper.find('[data-test="modal-ok"]').trigger('click')
    await flushPromises()

    const submitted = formEntries(vi.mocked(addCamera).mock.calls[0][0] as FormData)
    expect(submitted).toMatchObject({
      camera_id: 'rtsp_cam',
      name: 'RTSP camera',
      source: 'rtsp://admin:pass@192.0.2.100:554/stream',
      protocol: 'rtsp',
    })
    expect(testCameraConnection).toHaveBeenCalledWith('rtsp_cam')
  })

  it('keeps camera setup controls off operator accounts while preserving run controls', async () => {
    useAuthStore().currentUser = { username: 'operator', role: 'operator' }
    vi.mocked(getCameras).mockResolvedValue({
      cameras: [
        {
          camera_id: 'cam_01',
          name: 'Line camera',
          connected: false,
          health: {},
          stats: {},
        },
      ],
    } as any)

    const wrapper = mount(Cameras, {
      global: {
        plugins: [pinia],
        stubs: {
          ConnectionTestResult: true,
          ContentSkeleton: true,
          HealthBadge: true,
          ModeBadge: true,
        },
      },
    })
    await flushPromises()

    expect(wrapper.text()).not.toContain('新增摄像头')
    expect(wrapper.text()).toContain('启动')
    expect(wrapper.text()).toContain('详情')
    const row = wrapper.get('tr.camera-row')
    expect(row.attributes('role')).toBe('link')
    expect(row.attributes('tabindex')).toBe('0')
    await row.trigger('keydown', { key: 'Enter' })
    expect(routerPush).toHaveBeenCalledWith('/cameras/cam_01')
  })

  it('renders a request failure instead of the camera empty state', async () => {
    vi.mocked(getCameras).mockRejectedValue(new Error('camera registry unavailable'))

    const wrapper = mount(Cameras, {
      global: {
        plugins: [pinia],
        stubs: {
          ConnectionTestResult: true,
          HealthBadge: true,
          ModeBadge: true,
        },
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('数据加载失败')
    expect(wrapper.text()).toContain('camera registry unavailable')
    expect(wrapper.text()).not.toContain('尚未配置摄像头')
  })

  it('renders camera access denial as a permission state', async () => {
    vi.mocked(getCameras).mockRejectedValue(new ApiError({
      httpStatus: 403,
      businessCode: 40300,
      message: 'camera access denied',
    }))

    const wrapper = mount(Cameras, {
      global: {
        plugins: [pinia],
        stubs: {
          ConnectionTestResult: true,
          HealthBadge: true,
          ModeBadge: true,
        },
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('无权访问')
    expect(wrapper.text()).not.toContain('数据加载失败')
    expect(wrapper.text()).not.toContain('刷新失败，当前显示上次成功获取的数据')
    expect(wrapper.text()).not.toContain('尚未配置摄像头')
  })
})
