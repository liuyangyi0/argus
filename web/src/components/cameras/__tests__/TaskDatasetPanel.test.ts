import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, getActivePinia, setActivePinia } from 'pinia'
import * as api from '../../../api/analysisTasks'
import * as lifecycleApi from '../../../api/taskLifecycle'
import type { AnalysisTask } from '../../../api/analysisTasks'
import { useAuthStore } from '../../../stores/useAuthStore'
import TaskDatasetPanel from '../TaskDatasetPanel.vue'
import TaskLifecyclePanel from '../TaskLifecyclePanel.vue'

vi.mock('../../../api/analysisTasks', async load => ({
  ...await load<typeof import('../../../api/analysisTasks')>(),
  getTaskWorkflows: vi.fn(), getTaskTrainingRecipes: vi.fn(),
  reviewTaskSample: vi.fn(), createTaskWorkflow: vi.fn(), importTaskCandidate: vi.fn(),
  exportTaskReview: vi.fn(), exportTaskTraining: vi.fn(),
}))
vi.mock('../../../composables/useWebSocket', () => ({ useWebSocket: () => ({}) }))
vi.mock('../../../api/taskLifecycle', async load => ({
  ...await load<typeof import('../../../api/taskLifecycle')>(), getTaskEvents: vi.fn(), getTaskTrainingExecutions: vi.fn(),
  getTaskReleaseDossiers: vi.fn(), getTaskLifecycleReadiness: vi.fn(),
}))
enableAutoUnmount(afterEach)
const task: AnalysisTask = {
  task_id: 'task-1', camera_id: 'cam1', template_id: 'object_detection.v1',
  authority: 'managed', current_revision: 1, config_sha256: 'a'.repeat(64),
  runtime: { state: 'stopped' }, source: { state: 'running' },
}
const annotation = { schema: 'argus.task-annotation.v1', coordinate_space: 'normalized', kind: 'detection', objects: [] }
const row = (revision = 1, notes = 'Original review') => ({
  workflow_id: 'workflow-1', task_revision: 1, revision, state: 'review',
  manifest: {
    samples: [{ sample_id: 'sample-1', image_sha256: 'b'.repeat(64), identity: { frame_index: 7 } }],
    reviews: { 'sample-1': { decision: 'needs_review', notes, annotation } },
  },
})
const candidate = () => ({ relative_path: 'external/model.onnx', descriptor: { name: 'Reviewed external model',
  adapter_id: 'onnx_detection.v1', output_schema: 'yolov8_detection.v1', labels: ['person'] } })
const binding = (value = task) => ({ task_id: value.task_id, camera_id: value.camera_id,
  template_id: value.template_id, task_revision: value.current_revision, task_config_sha256: value.config_sha256 })
function candidateRow(value = task): Record<string, unknown> {
  return { ...row(3), task_id: value.task_id, state: 'training_export_ready', manifest_sha256: 'd'.repeat(64),
    manifest: { ...row().manifest, binding: binding(value), dataset_id: 'dataset-1', training: { approved_by: 'human-data-reviewer' } } }
}
function registered(value = task, raw: Record<string, unknown> = candidate()): Record<string, unknown> {
  const receipt = { operation_id: '7'.repeat(64), requested_revision: 3, stage: 'candidate', artifact_id: 'external-artifact',
    request_identity: { workflow_id: 'workflow-1', package_sha256: 'c'.repeat(64), candidate: raw } }
  const current = candidateRow(value)
  return { ...current, state: 'candidate_registered', revision: 5, candidate_import_receipt: receipt,
    manifest: { ...(current.manifest as Record<string, unknown>), training: { approved_by: 'human-data-reviewer', candidate: { ...receipt } } } }
}
function pending(value = task): Record<string, unknown> {
  return { ...candidateRow(value), state: 'candidate_importing', revision: 4, operation: { kind: 'candidate_import',
    requested_by: 'engineer', requested_revision: 3, package_sha256: 'c'.repeat(64),
    approval_manifest_sha256: 'd'.repeat(64), candidate: { ...candidate(), artifact_sha256: 'e'.repeat(64) },
    request_identity: { workflow_id: 'workflow-1', package_sha256: 'c'.repeat(64), candidate: candidate() },
    training_manifest: { package_sha256: 'c'.repeat(64) } } }
}
function personalPanel(value = task) {
  useAuthStore().currentUser = { username: 'engineer', role: 'engineer' }
  return mount(TaskDatasetPanel, { props: { task: value }, global: { plugins: [getActivePinia()!] } })
}
async function fillManual(wrapper: ReturnType<typeof personalPanel>, raw: Record<string, unknown> = candidate()) {
  await wrapper.get('[data-test="manual-package-sha"]').setValue('c'.repeat(64))
  await wrapper.get('[data-test="manual-candidate-json"]').setValue(JSON.stringify(raw))
  await wrapper.get('[data-test="prepare-manual-candidate"]').trigger('click'); await flushPromises()
}

describe('TaskDatasetPanel', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    setActivePinia(createPinia())
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [row()] })
    vi.mocked(api.getTaskTrainingRecipes).mockResolvedValue({ documents: [] })
    vi.mocked(lifecycleApi.getTaskEvents).mockResolvedValue({ documents: [] })
    vi.mocked(lifecycleApi.getTaskTrainingExecutions).mockResolvedValue({ documents: [] })
    vi.mocked(lifecycleApi.getTaskReleaseDossiers).mockResolvedValue({ documents: [] })
    vi.mocked(lifecycleApi.getTaskLifecycleReadiness).mockResolvedValue({ document: { production_ready: false } })
  })

  it('keeps drafts and requires explicit reload after review revision changes', async () => {
    const wrapper = mount(TaskDatasetPanel, { props: { task } })
    await flushPromises()
    await wrapper.get('[data-test="sample-select"]').setValue('sample-1')
    await wrapper.get('img').trigger('load')
    const notes = wrapper.findAll('textarea')[0]!
    await notes.setValue('Unsaved draft')
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [row(2, 'New human review')] })
    await wrapper.findAll('button').find(button => button.text() === '刷新流程')!.trigger('click')
    await flushPromises()
    expect((notes.element as HTMLTextAreaElement).value).toBe('Unsaved draft')
    const approve = () => wrapper.findAll('button').find(button => button.text() === '批准此标注')!
    expect(approve().attributes('disabled')).toBeDefined()
    expect(api.reviewTaskSample).not.toHaveBeenCalled()
    await wrapper.findAll('button').find(button => button.text() === '重新载入当前审核')!.trigger('click')
    expect((notes.element as HTMLTextAreaElement).value).toBe('New human review')
    vi.mocked(api.reviewTaskSample).mockResolvedValue({ document: row(3) })
    await approve().trigger('click')
    await flushPromises()
    expect(api.reviewTaskSample).toHaveBeenCalledWith('task-1', 'workflow-1', expect.objectContaining({
      expected_revision: 2, expected_image_sha256: 'b'.repeat(64), notes: 'New human review',
    }))
  })

  it('ignores an older workflow fetch that completes after a command refresh', async () => {
    let finish!: (value: { documents: Record<string, unknown>[] }) => void
    vi.mocked(api.getTaskWorkflows).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockResolvedValueOnce({ documents: [row(3)] })
    vi.mocked(api.createTaskWorkflow).mockResolvedValue({ document: row(3) })
    const wrapper = mount(TaskDatasetPanel, { props: { task } })
    await wrapper.findAll('button').find(button => button.text() === '新建数据流程')!.trigger('click')
    await flushPromises()
    expect(wrapper.text()).toContain('流程版本 3')
    finish({ documents: [row(1)] })
    await flushPromises()
    expect(wrapper.text()).toContain('流程版本 3')
    expect(wrapper.text()).not.toContain('流程版本 1')
  })

  it('never unlocks approval for a failed image or a late image from another sample', async () => {
    const workflow = row()
    workflow.manifest.samples.push({ sample_id: 'sample-2', image_sha256: 'c'.repeat(64), identity: { frame_index: 8 } })
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [workflow] })
    const wrapper = mount(TaskDatasetPanel, { props: { task } })
    await flushPromises()
    const select = wrapper.get('[data-test="sample-select"]')
    await select.setValue('sample-1')
    const oldImage = wrapper.get('img')
    const approve = () => wrapper.findAll('button').find(button => button.text() === '批准此标注')!
    expect(approve().attributes('disabled')).toBeDefined()
    await oldImage.trigger('error')
    expect(wrapper.text()).toContain('审核图片加载失败')
    expect(approve().attributes('disabled')).toBeDefined()
    await select.setValue('sample-2')
    await oldImage.trigger('load')
    expect(approve().attributes('disabled')).toBeDefined()
    expect(api.reviewTaskSample).not.toHaveBeenCalled()
    await wrapper.get('img').trigger('load')
    expect(approve().attributes('disabled')).toBeUndefined()
    expect(wrapper.text()).toContain('待人工复核')
  })

  it('keeps pose external candidates usable without inventing a training job and requires personal confirmation', async () => {
    const pose = { ...task, template_id: 'pose_estimation.v1' }
    const raw = { relative_path: 'external/pose.onnx', descriptor: { name: 'External pose', adapter_id: 'onnx_pose.v1' } }
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow(pose)] })
    vi.mocked(api.importTaskCandidate).mockResolvedValue({ document: registered(pose, raw) })
    const wrapper = personalPanel(pose); await flushPromises()
    await wrapper.get('[data-test="sample-select"]').setValue('sample-1')
    const notes = wrapper.findAll('textarea')[0]!
    await notes.setValue('Keep this human review draft')
    await fillManual(wrapper, raw)
    expect(wrapper.get('[data-test="register-manual-candidate"]').attributes('disabled')).toBeDefined()
    await wrapper.get('[data-test="confirm-manual-candidate"]').setValue(true)
    await wrapper.get('[data-test="register-manual-candidate"]').trigger('click'); await flushPromises()
    expect(api.importTaskCandidate).toHaveBeenCalledWith('task-1', 'workflow-1', {
      expected_revision: 3, expected_package_sha256: 'c'.repeat(64), candidate: raw,
    })
    expect(wrapper.get('[data-test="manual-candidate-registered"]').text()).toContain('external-artifact')
    expect((notes.element as HTMLTextAreaElement).value).toBe('Keep this human review draft')
    expect(lifecycleApi.getTaskTrainingExecutions).not.toHaveBeenCalled()
  })

  it('freezes an unknown manual request across refresh and input edits instead of rebasing revision', async () => {
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow()] })
    vi.mocked(api.importTaskCandidate).mockRejectedValueOnce(new Error('unknown commit')).mockResolvedValue({ document: registered() })
    const wrapper = personalPanel(); await flushPromises(); await fillManual(wrapper)
    await wrapper.get('[data-test="confirm-manual-candidate"]').setValue(true)
    await wrapper.get('[data-test="register-manual-candidate"]').trigger('click'); await flushPromises()
    const original = vi.mocked(api.importTaskCandidate).mock.calls[0]
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [pending()] })
    await wrapper.findAll('button').find(button => button.text() === '刷新流程')!.trigger('click'); await flushPromises()
    await wrapper.get('[data-test="manual-package-sha"]').setValue('0'.repeat(64))
    await wrapper.get('[data-test="manual-candidate-json"]').setValue('{"relative_path":"wrong.onnx","descriptor":{}}')
    expect(wrapper.find('[data-test="manual-candidate-blocked"]').exists()).toBe(false)
    await wrapper.get('[data-test="confirm-manual-candidate"]').setValue(true)
    await wrapper.get('[data-test="register-manual-candidate"]').trigger('click'); await flushPromises()
    expect(vi.mocked(api.importTaskCandidate).mock.calls[1]).toEqual(original)
    expect(wrapper.get('[data-test="manual-candidate-registered"]').text()).toContain('external-artifact')
  })

  it('retains manual unknown requests across real unmount while resetting human confirmation', async () => {
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow()] })
    vi.mocked(api.importTaskCandidate).mockRejectedValueOnce(new Error('response lost')).mockResolvedValue({ document: registered() })
    const first = personalPanel(); await flushPromises(); await fillManual(first)
    await first.get('[data-test="confirm-manual-candidate"]').setValue(true)
    await first.get('[data-test="register-manual-candidate"]').trigger('click'); await flushPromises()
    first.unmount()
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [pending()] })
    const reopened = personalPanel(); await flushPromises()
    expect(reopened.get('[data-test="manual-candidate-draft"]').text()).toContain('r3')
    expect((reopened.get('[data-test="confirm-manual-candidate"]').element as HTMLInputElement).checked).toBe(false)
    await reopened.get('[data-test="register-manual-candidate"]').trigger('click'); await flushPromises()
    expect(api.importTaskCandidate).toHaveBeenCalledTimes(1)
    await reopened.get('[data-test="confirm-manual-candidate"]').setValue(true)
    await reopened.get('[data-test="register-manual-candidate"]').trigger('click'); await flushPromises()
    expect(vi.mocked(api.importTaskCandidate).mock.calls[1]).toEqual(vi.mocked(api.importTaskCandidate).mock.calls[0])
  })

  it('does not let the training panel adopt or submit a manual candidate intent', async () => {
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow()] })
    const manual = personalPanel(); await flushPromises(); await fillManual(manual)
    vi.mocked(lifecycleApi.getTaskTrainingExecutions).mockResolvedValue({ documents: [{
      job_id: 'training-job', task_id: 'task-1', workflow_id: 'workflow-1', revision: 7, status: 'complete',
      request_sha256: 'f'.repeat(64), package_sha256: 'c'.repeat(64), binding: binding(), dataset_id: 'dataset-1',
      production_allowed: false, approval_scope: 'training_only', result: {
        schema: 'argus.task-training-result.v1', job_id: 'training-job', execution_id: 'execution-1',
        request_sha256: 'f'.repeat(64), package_sha256: 'c'.repeat(64), provider: 'torch-cuda',
        production_allowed: false, stage: 'candidate', candidate: { ...candidate(), artifact_sha256: 'e'.repeat(64) },
      },
    }] })
    const training = mount(TaskLifecyclePanel, { props: { task }, global: { plugins: [getActivePinia()!] } })
    await flushPromises()
    expect(training.find('[data-test="training-candidate-draft"]').exists()).toBe(false)
    expect(training.get('[data-test="prepare-training-candidate"]').attributes('disabled')).toBeDefined()
    expect(training.text()).toContain('数据流程入口已保留手工候选请求')
    await (training.vm as unknown as { prepareCandidate: (job: Record<string, unknown>) => void }).prepareCandidate({ status: 'complete' })
    await (training.vm as unknown as { registerCandidate: () => Promise<void> }).registerCandidate()
    expect(api.importTaskCandidate).not.toHaveBeenCalled()
  })

  it('keeps the shared manual transport claim when its panel closes before the response', async () => {
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow()] })
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(api.importTaskCandidate).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const first = personalPanel(); await flushPromises(); await fillManual(first)
    await first.get('[data-test="confirm-manual-candidate"]').setValue(true)
    await first.get('[data-test="register-manual-candidate"]').trigger('click'); await flushPromises()
    first.unmount()
    const reopened = personalPanel(); await flushPromises()
    expect(reopened.get('[data-test="register-manual-candidate"]').attributes('disabled')).toBeDefined()
    await (reopened.vm as unknown as { registerManualCandidate: () => Promise<void> }).registerManualCandidate()
    expect(api.importTaskCandidate).toHaveBeenCalledTimes(1)
    finish({ document: registered() }); await flushPromises()
    expect(reopened.find('[data-test="manual-candidate-registered"]').exists()).toBe(false)
    expect((reopened.get('[data-test="confirm-manual-candidate"]').element as HTMLInputElement).checked).toBe(false)
    await reopened.get('[data-test="confirm-manual-candidate"]').setValue(true)
    await reopened.get('[data-test="register-manual-candidate"]').trigger('click'); await flushPromises()
    expect(api.importTaskCandidate).toHaveBeenCalledTimes(2)
  })

  it.each(['stage', 'request'])('retains unknown state for a mismatched persisted manual %s receipt', async field => {
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow()] })
    const response = registered(), manifest = response.manifest as { training: { candidate: Record<string, unknown> } }
    if (field === 'stage') manifest.training.candidate.stage = 'production'
    else manifest.training.candidate.request_identity = { workflow_id: 'different-workflow' }
    vi.mocked(api.importTaskCandidate).mockResolvedValue({ document: response })
    const wrapper = personalPanel(); await flushPromises(); await fillManual(wrapper)
    await wrapper.get('[data-test="confirm-manual-candidate"]').setValue(true)
    await wrapper.get('[data-test="register-manual-candidate"]').trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="manual-candidate-registered"]').exists()).toBe(false)
    expect(wrapper.get('[data-test="manual-candidate-error"]').text()).toContain('回执身份尚未确认')
  })

  it('blocks an unverified or unauthenticated manual candidate while keeping human review drafts', async () => {
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow()] })
    const wrapper = personalPanel(); await flushPromises(); await fillManual(wrapper)
    vi.mocked(api.getTaskWorkflows).mockRejectedValueOnce(new Error('workflow read failed'))
    await wrapper.findAll('button').find(button => button.text() === '刷新流程')!.trigger('click'); await flushPromises()
    expect(wrapper.get('[data-test="manual-candidate-blocked"]').text()).toContain('状态未核实')
    useAuthStore().clear(); await flushPromises()
    expect(wrapper.find('[data-test="manual-candidate-draft"]').exists()).toBe(false)
    await (wrapper.vm as unknown as { registerManualCandidate: () => Promise<void> }).registerManualCandidate()
    expect(api.importTaskCandidate).not.toHaveBeenCalled()
  })

  it.each(['training', 'cvat', 'fiftyone'] as const)('discards a late %s export after switching camera and task', async target => {
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow()] })
    let finish!: (value: { document: Record<string, unknown> }) => void
    const pendingExport = new Promise<{ document: Record<string, unknown> }>(resolve => { finish = resolve })
    if (target === 'training') vi.mocked(api.exportTaskTraining).mockReturnValueOnce(pendingExport)
    else vi.mocked(api.exportTaskReview).mockReturnValueOnce(pendingExport)
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    try {
      const wrapper = personalPanel(); await flushPromises()
      const label = target === 'training' ? '导出训练交接清单' : `导出 ${target} 清单`
      await wrapper.findAll('button').find(button => button.text() === label)!.trigger('click'); await flushPromises()
      const other = { ...task, task_id: 'task-2', camera_id: 'cam2' }
      vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow(other)] })
      await wrapper.setProps({ task: other }); await flushPromises()
      await wrapper.get('[data-test="manual-package-sha"]').setValue('9'.repeat(64))
      finish({ document: { workflow_id: 'workflow-1', binding: binding(), package_sha256: 'c'.repeat(64),
        target, workflow_revision: 3, report_sha256: 'd'.repeat(64) } }); await flushPromises()
      expect((wrapper.get('[data-test="manual-package-sha"]').element as HTMLInputElement).value).toBe('9'.repeat(64))
      expect(click).not.toHaveBeenCalled()
      expect(wrapper.text()).not.toContain('清单已导出')
    } finally { click.mockRestore() }
  })

  it.each(['training', 'cvat', 'fiftyone'] as const)('downloads a verified current %s manifest and preserves human review notes', async target => {
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow()] })
    const document = { workflow_id: 'workflow-1', binding: binding(), package_sha256: 'c'.repeat(64),
      target, workflow_revision: 3, report_sha256: 'd'.repeat(64) }
    if (target === 'training') vi.mocked(api.exportTaskTraining).mockResolvedValue({ document })
    else vi.mocked(api.exportTaskReview).mockResolvedValue({ document })
    const createUrl = vi.fn(() => 'blob:local-review'), revokeUrl = vi.fn()
    vi.stubGlobal('URL', { createObjectURL: createUrl, revokeObjectURL: revokeUrl })
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    try {
      const wrapper = personalPanel(); await flushPromises()
      await wrapper.get('[data-test="sample-select"]').setValue('sample-1')
      await wrapper.findAll('textarea')[0]!.setValue('Keep these unsaved human notes')
      const label = target === 'training' ? '导出训练交接清单' : `导出 ${target} 清单`
      await wrapper.findAll('button').find(button => button.text() === label)!.trigger('click'); await flushPromises()
      expect(createUrl).toHaveBeenCalledTimes(1); expect(click).toHaveBeenCalledTimes(1)
      expect((wrapper.findAll('textarea')[0]!.element as HTMLTextAreaElement).value).toBe('Keep these unsaved human notes')
      expect((wrapper.get('[data-test="manual-package-sha"]').element as HTMLInputElement).value)
        .toBe(target === 'training' ? 'c'.repeat(64) : '')
      await new Promise(resolve => setTimeout(resolve, 0))
      expect(revokeUrl).toHaveBeenCalledWith('blob:local-review')
    } finally { click.mockRestore(); vi.unstubAllGlobals() }
  })

  it.each(['logout', 'workflow_roundtrip'])('ignores a late training export after %s', async change => {
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow(), { ...candidateRow(), workflow_id: 'workflow-2' }] })
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(api.exportTaskTraining).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    try {
      const wrapper = personalPanel(); await flushPromises()
      await wrapper.findAll('button').find(button => button.text() === '导出训练交接清单')!.trigger('click'); await flushPromises()
      if (change === 'logout') useAuthStore().clear()
      else {
        // User navigation is disabled during an operation. Exercise an actual
        // programmatic owner change, not a change event on a disabled select.
        expect(wrapper.get('[data-test="workflow-select"]').attributes('disabled')).toBeDefined()
        const state = wrapper.vm as unknown as { workflowId: string }
        state.workflowId = 'workflow-2'; await flushPromises()
        expect(wrapper.get('[data-test="workflow-select"]').element).toHaveProperty('value', 'workflow-2')
        state.workflowId = 'workflow-1'; await flushPromises()
      }
      await flushPromises()
      finish({ document: { workflow_id: 'workflow-1', binding: binding(), package_sha256: 'c'.repeat(64) } }); await flushPromises()
      expect((wrapper.get('[data-test="manual-package-sha"]').element as HTMLInputElement).value).toBe('')
      expect(click).not.toHaveBeenCalled()
    } finally { click.mockRestore() }
  })

  it('rejects an export response for a different camera without replacing the human package draft', async () => {
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow()] })
    vi.mocked(api.exportTaskTraining).mockResolvedValue({ document: {
      workflow_id: 'workflow-1', binding: { ...binding(), camera_id: 'other-camera' }, package_sha256: 'c'.repeat(64),
    } })
    const wrapper = personalPanel(); await flushPromises()
    await wrapper.get('[data-test="manual-package-sha"]').setValue('9'.repeat(64))
    await wrapper.findAll('button').find(button => button.text() === '导出训练交接清单')!.trigger('click'); await flushPromises()
    expect((wrapper.get('[data-test="manual-package-sha"]').element as HTMLInputElement).value).toBe('9'.repeat(64))
    expect(wrapper.text()).toContain('导出清单身份与当前任务或流程版本不一致')
  })

  it('releases its busy state but discards an export when the same task configuration changes', async () => {
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [candidateRow()] })
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(api.exportTaskTraining).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const wrapper = personalPanel(); await flushPromises()
    const exportButton = () => wrapper.findAll('button').find(button => button.text() === '导出训练交接清单')!
    await exportButton().trigger('click'); await flushPromises()
    await wrapper.setProps({ task: { ...task, current_revision: 2, config_sha256: 'f'.repeat(64) } })
    finish({ document: { workflow_id: 'workflow-1', binding: binding(), package_sha256: 'c'.repeat(64) } }); await flushPromises()
    expect((wrapper.get('[data-test="manual-package-sha"]').element as HTMLInputElement).value).toBe('')
    expect(exportButton().attributes('disabled')).toBeUndefined()
    expect(wrapper.get('section').attributes('aria-busy')).toBe('false')
    expect(wrapper.text()).not.toContain('清单已导出')
  })
})
