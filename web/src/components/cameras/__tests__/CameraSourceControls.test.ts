import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../../../api/taskDeployments'
import CameraSourceControls from '../CameraSourceControls.vue'

vi.mock('../../../api/taskDeployments', () => ({ getCameraSource: vi.fn(), commandCameraSource: vi.fn() }))

describe('CameraSourceControls', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getCameraSource).mockResolvedValue({ camera_id: 'cam-a', state: 'stopped', connected: false })
    vi.mocked(api.commandCameraSource).mockResolvedValue({ camera_id: 'cam-a', state: 'running', connected: true })
  })

  it('keeps viewers read-only and does not interpret source running as task detection', async () => {
    const wrapper = mount(CameraSourceControls, { props: { cameraId: 'cam-a', canManage: false } })
    await flushPromises()
    expect(wrapper.find('[data-test="start-source"]').exists()).toBe(false)
    expect(wrapper.find('[data-test="stop-source"]').exists()).toBe(false)
    expect(wrapper.text()).toContain('启动源不加载异物模型')
    expect(api.commandCameraSource).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('ignores source A after changing cameras and preserves operation errors against pre-command reads', async () => {
    let finish!: (value: { camera_id: string; state: string }) => void
    vi.mocked(api.getCameraSource).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockResolvedValue({ camera_id: 'cam-b', state: 'running', connected: true })
    const wrapper = mount(CameraSourceControls, { props: { cameraId: 'cam-a', canManage: true } })
    await wrapper.setProps({ cameraId: 'cam-b' }); await flushPromises()
    finish({ camera_id: 'cam-a', state: 'old-camera-failure' }); await flushPromises()
    expect(wrapper.text()).not.toContain('old-camera-failure')
    vi.mocked(api.commandCameraSource).mockRejectedValueOnce(new Error('cleanup not confirmed'))
    await wrapper.get('[data-test="stop-source"]').trigger('click'); await flushPromises()
    expect(api.commandCameraSource).toHaveBeenCalledWith('cam-b', 'stop')
    expect(wrapper.get('[role="alert"]').text()).toContain('cleanup not confirmed')
    expect(wrapper.text()).toContain('running')
    wrapper.unmount()
  })

  it('uses only the dedicated source command and explains whole-source stop impact', async () => {
    vi.mocked(api.getCameraSource).mockResolvedValue({ camera_id: 'cam-a', state: 'running', connected: true })
    const wrapper = mount(CameraSourceControls, { props: { cameraId: 'cam-a', canManage: true } })
    await flushPromises()
    await wrapper.get('[data-test="start-source"]').trigger('click'); await flushPromises()
    expect(api.commandCameraSource).toHaveBeenCalledWith('cam-a', 'start')
    expect(wrapper.text()).toContain('所有任务、预览和录像')
    wrapper.unmount()
  })

  it('can cancel a pending source start and ignores its late response', async () => {
    let finish!: (value: { camera_id: string; state: string; connected: boolean }) => void
    vi.mocked(api.commandCameraSource).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockResolvedValueOnce({ camera_id: 'cam-a', state: 'stopped', connected: false })
    const wrapper = mount(CameraSourceControls, { props: { cameraId: 'cam-a', canManage: true } })
    await flushPromises()
    await wrapper.get('[data-test="start-source"]').trigger('click')
    expect(wrapper.get('[data-test="stop-source"]').attributes('disabled')).toBeUndefined()
    await wrapper.get('[data-test="stop-source"]').trigger('click'); await flushPromises()
    expect(api.commandCameraSource).toHaveBeenNthCalledWith(2, 'cam-a', 'stop')
    finish({ camera_id: 'cam-a', state: 'late-start-running', connected: true }); await flushPromises()
    expect(wrapper.text()).not.toContain('late-start-running')
    expect(wrapper.text()).toContain('stopped')
    wrapper.unmount()
  })
})
