import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { getAnalysisTasks, type AnalysisTask } from '../../../api/analysisTasks'
import AnalysisTaskPanel from '../AnalysisTaskPanel.vue'
import { createPinia, setActivePinia } from 'pinia'
import { useAuthStore } from '../../../stores/useAuthStore'

vi.mock('../../../api/analysisTasks', () => ({ getAnalysisTasks: vi.fn() }))
vi.mock('../CameraSourceControls.vue', () => ({ default: { template: '<div />' } }))

function task(): AnalysisTask {
  return {
    task_id: 'task-1', camera_id: 'cam1', template_id: 'legacy_anomaly.v1',
    authority: 'legacy', current_revision: 1, config_sha256: 'a'.repeat(64),
    created_at: null,
    deployment: {
      authority: 'legacy', desired_artifact_id: 'artifact-1', applied_artifact_id: null,
      runtime_status: 'unknown', registry_release_revision: 1, projection_status: 'authorized',
    },
    runtime: { state: 'stopped', running: false, generation: 1 },
    source: { state: 'running', running: true, connected: true, generation: 2 },
  }
}

describe('AnalysisTaskPanel', () => {
  beforeEach(() => { setActivePinia(undefined); vi.mocked(getAnalysisTasks).mockReset(); vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers() })

  it('separates source from task and does not infer applied model from desired', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [task()] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1' } })
    await flushPromises()
    expect(wrapper.text()).toContain('已停止')
    expect(wrapper.text()).toContain('运行中')
    expect(wrapper.text()).toContain('未确认（不能由注册状态推断）')
    expect(wrapper.text()).toContain('新任务仅支持旁路运行')
    expect(wrapper.findAll('button')).toHaveLength(1)
    wrapper.unmount()
  })

  it('shows errors rather than an empty-success state', async () => {
    vi.mocked(getAnalysisTasks).mockRejectedValue(new Error('无权读取分析任务'))
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1' } })
    await flushPromises()
    expect(wrapper.get('[data-test="task-error"]').text()).toContain('无权读取分析任务')
    expect(wrapper.find('[data-test="task-empty"]').exists()).toBe(false)
    wrapper.unmount()
  })

  it('rejects another camera task list without exposing a wrong task action', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [{ ...task(), camera_id: 'different-camera' }] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1' } })
    await flushPromises()
    expect(wrapper.get('[data-test="task-error"]').text()).toContain('不匹配')
    expect(wrapper.find('[data-test="analysis-task"]').exists()).toBe(false)
    wrapper.unmount()
  })

  it('does not silently replace a missing requested task with another task', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [task()] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1', requestedTaskId: 'missing' } })
    await flushPromises()
    expect(wrapper.text()).toContain('指定任务不存在或不属于当前摄像头')
    wrapper.unmount()
  })

  it('restores a readonly legacy task focus after clearing and readding its query', async () => {
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [task()] })
    const wrapper = mount(AnalysisTaskPanel, { attachTo: document.body, props: { cameraId: 'cam1', requestedTaskId: 'task-1', requestedSection: 'legacy_runtime' } })
    await flushPromises()
    expect(document.activeElement).toBe(wrapper.get('[data-test="analysis-task"]').element)
    await wrapper.setProps({ requestedTaskId: undefined }); await flushPromises()
    const refresh = wrapper.get('header button').element as HTMLButtonElement
    refresh.focus()
    await wrapper.setProps({ requestedTaskId: 'task-1' }); await flushPromises()
    expect(document.activeElement).toBe(wrapper.get('[data-test="analysis-task"]').element)
    wrapper.unmount()
  })

  it.each([
    ['legacy', 'active', true], ['managed', 'archived', true], ['managed', 'active', false],
  ] as const)('tracks same-camera readonly navigation for %s/%s (manage=%s)', async (authority, lifecycle, canManage) => {
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [
      { ...task(), authority, lifecycle }, { ...task(), task_id: 'task-2', authority, lifecycle },
    ] })
    const wrapper = mount(AnalysisTaskPanel, { attachTo: document.body,
      props: { cameraId: 'cam1', canManage, requestedTaskId: 'task-1', requestedSection: 'configuration' },
      global: { stubs: { TaskWorkbench: true } } })
    await flushPromises()
    expect(document.activeElement).toBe(wrapper.get('[data-task-id="task-1"]').element)
    await wrapper.setProps({ requestedTaskId: 'task-2' }); await flushPromises()
    expect(document.activeElement).toBe(wrapper.get('[data-task-id="task-2"]').element)
    await wrapper.setProps({ requestedSection: 'source' }); await flushPromises()
    expect(document.activeElement).toBe(wrapper.get('[data-test="task-section-source"]').element)
    const refresh = wrapper.get('header button').element as HTMLButtonElement
    refresh.focus()
    await wrapper.setProps({ requestedTaskId: 'deleted-task' }); await flushPromises()
    expect(wrapper.text()).toContain('指定任务不存在或不属于当前摄像头')
    expect(document.activeElement).toBe(refresh)
    wrapper.unmount()
  })

  it('marks retained state stale after a failed refresh and reconciles every minute', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValueOnce({ tasks: [task()] })
      .mockRejectedValueOnce(new Error('暂不可用'))
      .mockResolvedValueOnce({ tasks: [] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1' } })
    await flushPromises()
    await vi.advanceTimersByTimeAsync(60_000)
    await flushPromises()
    expect(wrapper.text()).toContain('上次成功读取的状态')
    expect(wrapper.find('[data-test="analysis-task"]').exists()).toBe(true)
    await vi.advanceTimersByTimeAsync(60_000)
    await flushPromises()
    expect(wrapper.find('[data-test="task-error"]').exists()).toBe(false)
    expect(wrapper.find('[data-test="task-empty"]').exists()).toBe(true)
    wrapper.unmount()
    await vi.advanceTimersByTimeAsync(120_000)
    expect(getAnalysisTasks).toHaveBeenCalledTimes(3)
  })

  it.each([
    [401, undefined], [403, undefined], [200, 40100], [200, 40300],
  ])('clears task state and management after denial %s/%s', async (status, code) => {
    vi.mocked(getAnalysisTasks).mockResolvedValueOnce({ tasks: [task()] })
      .mockRejectedValueOnce({ response: { status, data: { code, msg: '无权读取分析任务' } } })
      .mockResolvedValueOnce({ tasks: [task()] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1', canManage: true },
      global: { stubs: { TaskWorkbench: true } } })
    await flushPromises()
    expect(wrapper.find('[data-test="analysis-task"]').exists()).toBe(true)
    expect(wrapper.findComponent({ name: 'TaskWorkbench' }).exists()).toBe(true)
    await wrapper.get('header button').trigger('click'); await flushPromises()
    expect(wrapper.get('[data-test="task-error"]').text()).toContain('无权读取分析任务')
    expect(wrapper.get('[data-test="task-error"]').text()).not.toContain('上次成功读取')
    expect(wrapper.find('[data-test="analysis-task"]').exists()).toBe(false)
    expect(wrapper.findComponent({ name: 'TaskWorkbench' }).exists()).toBe(false)
    expect(wrapper.find('[data-test="task-empty"]').exists()).toBe(false)
    await wrapper.get('header button').trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="task-error"]').exists()).toBe(false)
    expect(wrapper.find('[data-test="analysis-task"]').exists()).toBe(true)
    expect(wrapper.findComponent({ name: 'TaskWorkbench' }).exists()).toBe(true)
    wrapper.unmount()
  })

  it('discards an earlier identity response and stops reading without a personal identity', async () => {
    const pinia = createPinia(); setActivePinia(pinia)
    const auth = useAuthStore(pinia)
    auth.currentUser = { username: 'first-operator', role: 'operator' }
    let finish!: (response: { tasks: AnalysisTask[] }) => void
    vi.mocked(getAnalysisTasks).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockResolvedValueOnce({ tasks: [{ ...task(), task_id: 'new-user-task' }] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1' }, global: { plugins: [pinia] } })
    auth.currentUser = null
    finish({ tasks: [task()] }); await flushPromises()
    expect(wrapper.find('[data-test="analysis-task"]').exists()).toBe(false)
    expect(wrapper.find('[data-test="task-empty"]').exists()).toBe(false)
    await vi.advanceTimersByTimeAsync(120_000); await flushPromises()
    expect(getAnalysisTasks).toHaveBeenCalledTimes(1)
    auth.currentUser = { username: 'second-operator', role: 'operator' }; await flushPromises()
    expect(getAnalysisTasks).toHaveBeenCalledTimes(2)
    expect(wrapper.text()).toContain('new-user-task')
    expect(wrapper.text()).not.toContain('task-1')
    wrapper.unmount()
  })

  it('requires the next operator to reopen an earlier personal event-review panel', async () => {
    const pinia = createPinia(); setActivePinia(pinia)
    const auth = useAuthStore(pinia)
    auth.currentUser = { username: 'first-operator', role: 'operator' }
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [{ ...task(), template_id: 'object_detection.v1', authority: 'managed' }] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1' },
      global: { plugins: [pinia], stubs: { TaskLifecyclePanel: true } } })
    await flushPromises()
    await wrapper.get('[data-test="open-task-event-review"]').trigger('click')
    expect(wrapper.findComponent({ name: 'TaskLifecyclePanel' }).exists()).toBe(true)
    auth.currentUser = { username: 'second-operator', role: 'operator' }; await flushPromises()
    expect(wrapper.findComponent({ name: 'TaskLifecyclePanel' }).exists()).toBe(false)
    expect(getAnalysisTasks).toHaveBeenCalledTimes(2)
    wrapper.unmount()
  })

  it('does not load task state or mount management before the personal identity is ready', async () => {
    const pinia = createPinia(); setActivePinia(pinia)
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1', canManage: true },
      global: { plugins: [pinia], stubs: { TaskWorkbench: true } } })
    await flushPromises()
    expect(getAnalysisTasks).not.toHaveBeenCalled()
    expect(wrapper.get('[data-test="task-error"]').text()).toContain('个人身份尚未就绪')
    expect(wrapper.findComponent({ name: 'TaskWorkbench' }).exists()).toBe(false)
    wrapper.unmount()
  })

  it('remounts a same-camera workbench when the personal identity changes', async () => {
    const pinia = createPinia(); setActivePinia(pinia)
    const auth = useAuthStore(pinia)
    auth.currentUser = { username: 'first-engineer', role: 'engineer' }
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [task()] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1', canManage: true },
      global: { plugins: [pinia], stubs: { TaskWorkbench: true } } })
    await flushPromises()
    const oldWorkbench = wrapper.findComponent({ name: 'TaskWorkbench' }).vm
    auth.currentUser = { username: 'second-engineer', role: 'engineer' }; await flushPromises()
    expect(wrapper.findComponent({ name: 'TaskWorkbench' }).vm).not.toBe(oldWorkbench)
    expect(getAnalysisTasks).toHaveBeenCalledTimes(2)
    wrapper.unmount()
  })

  it('rejects a late response from a previous camera', async () => {
    let finishOld!: (value: { tasks: AnalysisTask[] }) => void
    vi.mocked(getAnalysisTasks).mockReturnValueOnce(new Promise((resolve) => { finishOld = resolve }))
      .mockResolvedValueOnce({ tasks: [{ ...task(), task_id: 'task-2', camera_id: 'cam2' }] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1' } })
    await wrapper.setProps({ cameraId: 'cam2' })
    await flushPromises()
    finishOld({ tasks: [task()] })
    await flushPromises()
    expect(wrapper.text()).toContain('task-2')
    expect(wrapper.text()).not.toContain('task-1')
    wrapper.unmount()
  })

  it('does not restart reconciliation after an in-flight request resolves on unmount', async () => {
    let finish!: (value: { tasks: AnalysisTask[] }) => void
    vi.mocked(getAnalysisTasks).mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1' } })
    wrapper.unmount()
    finish({ tasks: [task()] })
    await flushPromises()
    await vi.advanceTimersByTimeAsync(120_000)
    expect(getAnalysisTasks).toHaveBeenCalledTimes(1)
  })

  it('shows sampling and cleanup failures instead of flattening them to unknown', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [{
      ...task(),
      runtime: { state: 'sampling_insufficient', reason_code: 'SAMPLING_INSUFFICIENT' },
      source: { state: 'cleanup_pending', reason_code: 'CLEANUP_PENDING', connected: false },
    }] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1' } })
    await flushPromises()
    expect(wrapper.text()).toContain('采样不足')
    expect(wrapper.text()).toContain('有序帧窗口丢帧')
    expect(wrapper.text()).toContain('等待清理，重启受阻')
    expect(wrapper.text()).toContain('资源尚未确认释放')
    wrapper.unmount()
  })

  it('labels imported config and stale registry snapshots without implying runtime application', async () => {
    const fixture = task()
    fixture.deployment = { ...fixture.deployment!, projection_status: 'stale' }
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [fixture] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1' } })
    await flushPromises()
    expect(wrapper.text()).toContain('导入快照版本')
    expect(wrapper.text()).toContain('导入快照不是当前生效配置')
    expect(wrapper.text()).toContain('注册记录已变化，快照过期')
    expect(wrapper.text()).not.toContain('注册授权已核验')
    wrapper.unmount()
  })

  it('distinguishes an event evidence failure from independently running inference and preview', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [{ ...task(),
      runtime: { state: 'running', running: true, event_error: 'EVENT_QUEUE_UNAVAILABLE' },
    }] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1' } })
    await flushPromises()
    expect(wrapper.get('[data-test="task-event-error"]').text()).toContain('部分事件可能缺失')
    expect(wrapper.text()).toContain('运行中')
    expect(wrapper.text()).toContain('已连接')
    wrapper.unmount()
  })

  it('provides an on-demand event review entry for operators without exposing the model workbench', async () => {
    const pinia = createPinia(); setActivePinia(pinia)
    useAuthStore(pinia).currentUser = { username: 'event-operator', role: 'operator' }
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [{ ...task(), template_id: 'object_detection.v1', authority: 'managed' }] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1', canManage: false },
      global: { plugins: [pinia], stubs: { TaskLifecyclePanel: true, TaskWorkbench: true } } })
    await flushPromises()
    expect(wrapper.findComponent({ name: 'TaskWorkbench' }).exists()).toBe(false)
    expect(wrapper.findComponent({ name: 'TaskLifecyclePanel' }).exists()).toBe(false)
    await wrapper.get('[data-test="open-task-event-review"]').trigger('click')
    expect(wrapper.findComponent({ name: 'TaskLifecyclePanel' }).props('task').task_id).toBe('task-1')
    await wrapper.setProps({ cameraId: 'cam2' }); await flushPromises()
    expect(wrapper.findComponent({ name: 'TaskLifecyclePanel' }).exists()).toBe(false)
    wrapper.unmount()
  })

  it('does not provide a historical review entry to a viewer or an unready identity', async () => {
    const pinia = createPinia(); setActivePinia(pinia)
    const auth = useAuthStore(pinia)
    auth.currentUser = { username: 'viewer', role: 'viewer' }
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [{ ...task(), template_id: 'object_detection.v1', authority: 'managed' }] })
    const wrapper = mount(AnalysisTaskPanel, { props: { cameraId: 'cam1', canManage: false }, global: { plugins: [pinia] } })
    await flushPromises()
    expect(wrapper.find('[data-test="open-task-event-review"]').exists()).toBe(false)
    auth.currentUser = null; await flushPromises()
    expect(wrapper.find('[data-test="open-task-event-review"]').exists()).toBe(false)
    wrapper.unmount()
  })
})
