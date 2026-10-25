import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { getAnalysisTasks, type AnalysisTask } from '../../../api/analysisTasks'
import AnalysisTaskPanel from '../AnalysisTaskPanel.vue'

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
  beforeEach(() => { vi.mocked(getAnalysisTasks).mockReset(); vi.useFakeTimers() })
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
})
