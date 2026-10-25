import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../../../api/taskDeployments'
import type { AnalysisTask } from '../../../api/analysisTasks'
import { ApiError } from '../../../types/api'
import TaskDeploymentPanel from '../TaskDeploymentPanel.vue'

vi.mock('../../../api/taskDeployments', () => ({
  getTaskDeployment: vi.fn(), getTaskDeploymentOperations: vi.fn(),
  requestTaskDeployment: vi.fn(), reconcileTaskDeployment: vi.fn(), restoreTaskConfiguration: vi.fn(),
}))
const historical = 'bundle-' + 'a'.repeat(32)
const restored = 'bundle-' + 'b'.repeat(32)
const task = (revision = 2, taskId = 'task-a'): AnalysisTask => ({
  task_id: taskId, camera_id: 'cam', template_id: 'fast_motion.v1', authority: 'managed',
  current_revision: revision, config_sha256: 'a'.repeat(64), runtime: {}, source: {},
})
const stopped = (changes: Record<string, unknown> = {}) => ({ document: {
  task_id: 'task-a', deployment_revision: 2, applied_deployment_revision: 2,
  state: 'stopped', desired_state: 'stopped', applied_bundle_id: historical,
  operation: { task_id: 'task-a', operation_id: 'stop-2', deployment_revision: 2, action: 'stop', status: 'applied' },
  ...changes,
} })
const result = (prior = 2) => ({ document: {
  task: { task_id: 'task-a', current_revision: prior + 1 },
  bundle: { task_id: 'task-a', bundle_id: restored, task_revision: prior + 1 },
  restore: { restored_from_bundle_id: historical, restored_from_task_revision: 1,
    prior_task_revision: prior, new_task_revision: prior + 1, restored_by: 'human', production_allowed: false },
} })
const props = () => ({ task: task(), bundleId: historical,
  bundles: [{ task_id: 'task-a', bundle_id: historical, task_revision: 1 }] })

describe('Task configuration restoration', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    vi.mocked(api.getTaskDeployment).mockResolvedValue(stopped())
    vi.mocked(api.getTaskDeploymentOperations).mockResolvedValue({ documents: [
      { task_id: 'task-a', operation_id: 'previous-apply', action: 'deploy', status: 'applied', bundle_id: historical },
    ] })
    vi.mocked(api.restoreTaskConfiguration).mockResolvedValue(result())
  })

  it.each([
    { desired_state: 'running' }, { state: 'cleanup_pending' }, { applied_deployment_revision: 1 },
    { operation: { task_id: 'task-a', action: 'stop', status: 'pending', deployment_revision: 2 } },
  ])('requires a complete durable stopped receipt (%j)', async changes => {
    vi.mocked(api.getTaskDeployment).mockResolvedValue(stopped(changes))
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await flushPromises()
    expect(wrapper.get('[data-test="restore-configuration"]').attributes('disabled')).toBeDefined()
    expect(api.restoreTaskConfiguration).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('does not offer restoration for a bundle without a known historical apply identity', async () => {
    vi.mocked(api.getTaskDeployment).mockResolvedValue(stopped({ applied_bundle_id: 'another-bundle' }))
    vi.mocked(api.getTaskDeploymentOperations).mockResolvedValue({ documents: [] })
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await flushPromises()
    expect(wrapper.get('[data-test="restore-configuration"]').attributes('disabled')).toBeDefined()
    wrapper.unmount()
  })

  it('uses the authoritative full-history annotation for targets outside the recent history window', async () => {
    vi.mocked(api.getTaskDeployment).mockResolvedValue(stopped({ applied_bundle_id: 'another-bundle' }))
    vi.mocked(api.getTaskDeploymentOperations).mockResolvedValue({ documents: [] })
    const wrapper = mount(TaskDeploymentPanel, { props: { ...props(), bundles: [
      { task_id: 'task-a', bundle_id: historical, task_revision: 1, historically_applied: true },
    ] } })
    await flushPromises()
    expect(wrapper.get('[data-test="restore-configuration"]').attributes('disabled')).toBeUndefined()
    wrapper.unmount()
  })

  it('does not override an authoritative false with a stale applied identity', async () => {
    const wrapper = mount(TaskDeploymentPanel, { props: { ...props(), bundles: [
      { task_id: 'task-a', bundle_id: historical, task_revision: 1, historically_applied: false },
    ] } })
    await flushPromises()
    expect(wrapper.get('[data-test="restore-configuration"]').attributes('disabled')).toBeDefined()
    wrapper.unmount()
  })

  it('restores an old configuration into a new revision/bundle without deploying it', async () => {
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await flushPromises()
    expect(wrapper.get('[data-test="deploy-task"]').attributes('disabled')).toBeDefined()
    await wrapper.get('[data-test="restore-configuration"]').trigger('click'); await flushPromises()
    expect(api.restoreTaskConfiguration).toHaveBeenCalledWith('task-a', { bundle_id: historical, expected_revision: 2 }, expect.any(String))
    expect(wrapper.text()).toContain('新版本 r3')
    expect(wrapper.text()).toContain(restored)
    expect(wrapper.text()).toContain('未启动、未部署')
    expect(wrapper.emitted('refresh')).toHaveLength(1)
    expect(api.requestTaskDeployment).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it.each([null, 408, 429, 500])('keeps the original restoration body/key after unknown status %s', async status => {
    const failure = status === null ? new Error('lost restoration acknowledgement')
      : new ApiError({ httpStatus: status, message: 'restore result unknown' })
    vi.mocked(api.restoreTaskConfiguration).mockRejectedValueOnce(failure).mockResolvedValueOnce(result())
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await flushPromises()
    await wrapper.get('[data-test="restore-configuration"]').trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="retry-configuration-restore"]').exists()).toBe(true)
    expect(wrapper.get('[data-test="deploy-task"]').attributes('disabled')).toBeDefined()
    await wrapper.setProps({ task: task(3) })
    await wrapper.get('[data-test="retry-configuration-restore"]').trigger('click'); await flushPromises()
    const calls = vi.mocked(api.restoreTaskConfiguration).mock.calls
    expect(calls).toHaveLength(2)
    expect(calls[1]).toEqual(calls[0])
    expect(calls[1]?.[1].expected_revision).toBe(2)
    expect(api.requestTaskDeployment).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('clears a definite conflict and requests current task data before a new explicit restore', async () => {
    vi.mocked(api.restoreTaskConfiguration).mockRejectedValueOnce(new ApiError({ httpStatus: 409, message: 'task revision changed' }))
      .mockResolvedValueOnce(result(3))
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await flushPromises()
    await wrapper.get('[data-test="restore-configuration"]').trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="retry-configuration-restore"]').exists()).toBe(false)
    expect(wrapper.emitted('refresh')).toHaveLength(1)
    await wrapper.setProps({ task: task(3) })
    await wrapper.get('[data-test="restore-configuration"]').trigger('click'); await flushPromises()
    const calls = vi.mocked(api.restoreTaskConfiguration).mock.calls
    expect(calls[1]?.[1].expected_revision).toBe(3)
    expect(calls[1]?.[2]).not.toBe(calls[0]?.[2])
    wrapper.unmount()
  })

  it('rejects an invalid restore receipt rather than claiming a new configuration is confirmed', async () => {
    const response = result()
    response.document.bundle.task_id = 'another-task'
    vi.mocked(api.restoreTaskConfiguration).mockResolvedValue(response)
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await flushPromises()
    await wrapper.get('[data-test="restore-configuration"]').trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="retry-configuration-restore"]').exists()).toBe(true)
    expect(wrapper.text()).not.toContain('配置恢复已确认：')
    expect(wrapper.emitted('refresh')).toBeUndefined()
    wrapper.unmount()
  })

  it('does not apply a late restoration receipt after selecting a different task', async () => {
    let finish!: (value: ReturnType<typeof result>) => void
    vi.mocked(api.restoreTaskConfiguration).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const wrapper = mount(TaskDeploymentPanel, { props: props() })
    await flushPromises()
    await wrapper.get('[data-test="restore-configuration"]').trigger('click')
    vi.mocked(api.getTaskDeployment).mockResolvedValue(stopped({ task_id: 'task-b' }))
    vi.mocked(api.getTaskDeploymentOperations).mockResolvedValue({ documents: [] })
    await wrapper.setProps({ task: task(2, 'task-b'), bundleId: '', bundles: [] }); await flushPromises()
    finish(result()); await flushPromises()
    expect(wrapper.text()).not.toContain('配置恢复已确认：')
    expect(wrapper.find('[data-test="retry-configuration-restore"]').exists()).toBe(false)
    expect(wrapper.emitted('refresh')).toBeUndefined()
    wrapper.unmount()
  })
})
