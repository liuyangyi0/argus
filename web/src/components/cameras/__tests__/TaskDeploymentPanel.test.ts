import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../../../api/taskDeployments'
import type { AnalysisTask } from '../../../api/analysisTasks'
import { ApiError } from '../../../types/api'
import TaskDeploymentPanel from '../TaskDeploymentPanel.vue'

vi.mock('../../../api/taskDeployments', () => ({
  getTaskDeployment: vi.fn(), getTaskDeploymentOperations: vi.fn(),
  requestTaskDeployment: vi.fn(), reconcileTaskDeployment: vi.fn(),
}))
const task = (id = 'task-a'): AnalysisTask => ({
  task_id: id, camera_id: 'cam', template_id: 'fast_motion.v1', authority: 'managed',
  current_revision: 1, config_sha256: 'a'.repeat(64), runtime: {}, source: {},
})
const bundleId = 'bundle-' + 'a'.repeat(32)
const props = () => ({ task: task(), bundleId, bundles: [{ bundle_id: bundleId, task_revision: 1 }] })
const view = (revision = 0, state = 'stopped', applied = 'old-bundle', taskId = 'task-a') => ({
  document: { task_id: taskId, deployment_revision: revision, state, desired_state: 'running', applied_bundle_id: applied },
})

describe('TaskDeploymentPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getTaskDeployment).mockResolvedValue(view())
    vi.mocked(api.getTaskDeploymentOperations).mockResolvedValue({ documents: [] })
    vi.mocked(api.requestTaskDeployment).mockResolvedValue({ document: { task_id: 'task-a', operation_id: 'accepted-operation', status: 'pending' } })
  })

  it('submits a CAS intent and does not claim 202 means the candidate is applied', async () => {
    vi.mocked(api.getTaskDeployment).mockResolvedValue(view(3, 'preparing'))
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await flushPromises()
    await wrapper.get('[data-test="deploy-task"]').trigger('click'); await flushPromises()
    expect(api.requestTaskDeployment).toHaveBeenCalledWith('task-a', {
      action: 'deploy', bundle_id: bundleId, expected_deployment_revision: 3,
    }, expect.any(String))
    expect(wrapper.text()).toContain('尚不代表已应用')
    expect(wrapper.text()).toContain('old-bundle')
    wrapper.unmount()
  })

  it('retries an unknown request using its original body and key even after a newer GET', async () => {
    vi.mocked(api.requestTaskDeployment).mockRejectedValueOnce(new Error('acknowledgement lost'))
      .mockResolvedValueOnce({ document: { task_id: 'task-a', operation_id: 'same-operation' } })
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await flushPromises()
    await wrapper.get('[data-test="deploy-task"]').trigger('click'); await flushPromises()
    vi.mocked(api.getTaskDeployment).mockResolvedValue(view(1, 'preparing'))
    await wrapper.findAll('button').find(button => button.text() === '刷新部署')!.trigger('click'); await flushPromises()
    await wrapper.get('[data-test="retry-deployment"]').trigger('click'); await flushPromises()
    const calls = vi.mocked(api.requestTaskDeployment).mock.calls
    expect(calls).toHaveLength(2)
    expect(calls[1]).toEqual(calls[0])
    expect(calls[1]?.[1].expected_deployment_revision).toBe(0)
    wrapper.unmount()
  })

  it('treats a confirmed conflict as rejection, refreshes CAS and does not lock the form into unknown retry', async () => {
    vi.mocked(api.requestTaskDeployment).mockRejectedValueOnce(new ApiError({ httpStatus: 409, message: 'revision changed' }))
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await flushPromises()
    vi.mocked(api.getTaskDeployment).mockResolvedValue(view(2))
    await wrapper.get('[data-test="deploy-task"]').trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="retry-deployment"]').exists()).toBe(false)
    expect(wrapper.get('[data-test="deploy-task"]').attributes('disabled')).toBeUndefined()
    await wrapper.get('[data-test="deploy-task"]').trigger('click'); await flushPromises()
    expect(vi.mocked(api.requestTaskDeployment).mock.calls[1]?.[1].expected_deployment_revision).toBe(2)
    wrapper.unmount()
  })

  it('allows stop while an older HTTP response is late and ignores that response', async () => {
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(api.requestTaskDeployment).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockResolvedValueOnce({ document: { task_id: 'task-a', operation_id: 'stop-operation' } })
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await flushPromises()
    await wrapper.get('[data-test="deploy-task"]').trigger('click')
    vi.mocked(api.getTaskDeployment).mockResolvedValue(view(1, 'pending'))
    await wrapper.get('[data-test="stop-task"]').trigger('click'); await flushPromises()
    expect(vi.mocked(api.requestTaskDeployment).mock.calls[1]?.[1]).toEqual({
      action: 'stop', bundle_id: null, expected_deployment_revision: 1,
    })
    finish({ document: { task_id: 'task-a', operation_id: 'late-deploy-operation' } }); await flushPromises()
    expect(wrapper.text()).toContain('stop-operation')
    expect(wrapper.text()).not.toContain('late-deploy-operation')
    wrapper.unmount()
  })

  it('rejects late reads after switching A to B to A and blocks a stale configuration bundle', async () => {
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(api.getTaskDeployment).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockResolvedValueOnce(view(2, 'stopped', 'b-current', 'task-b'))
      .mockResolvedValueOnce(view(4, 'running', 'a-current'))
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await wrapper.setProps({ task: task('task-b') }); await flushPromises()
    await wrapper.setProps({ task: { ...task(), current_revision: 2 } }); await flushPromises()
    finish(view(0, 'failed', 'a-stale')); await flushPromises()
    expect(wrapper.text()).toContain('a-current')
    expect(wrapper.text()).not.toContain('a-stale')
    expect(wrapper.get('[data-test="deploy-task"]').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-test="rollback-task"]').attributes('disabled')).toBeDefined()
    wrapper.unmount()
  })
})
