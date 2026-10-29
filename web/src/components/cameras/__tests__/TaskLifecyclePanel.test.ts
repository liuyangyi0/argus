import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import { createPinia, getActivePinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useAuthStore } from '../../../stores/useAuthStore'
import type { AnalysisTask } from '../../../api/analysisTasks'
import * as dataApi from '../../../api/analysisTasks'
import * as api from '../../../api/taskLifecycle'
import TaskLifecyclePanel from '../TaskLifecyclePanel.vue'
import TaskDatasetPanel from '../TaskDatasetPanel.vue'

const socket = vi.hoisted(() => ({ options: null as {
  topics: string[]
  onMessage: (topic: string, data: unknown) => void
  fallbackPoll: () => unknown | Promise<unknown>
  reconcileInterval: number
  fallbackInterval: number
} | null }))
vi.mock('../../../composables/useWebSocket', () => ({
  useWebSocket: (options: NonNullable<typeof socket.options>) => { socket.options = options; return {} },
}))

vi.mock('../../../api/analysisTasks', async load => ({
  ...await load<typeof import('../../../api/analysisTasks')>(), getTaskWorkflows: vi.fn(), importTaskCandidate: vi.fn(),
  getTaskTrainingRecipes: vi.fn(),
}))
vi.mock('../../../api/taskLifecycle', async load => ({
  ...await load<typeof import('../../../api/taskLifecycle')>(),
  getTaskEvents: vi.fn(), reviewTaskEvent: vi.fn(), getTaskEventReplay: vi.fn(),
  getTaskReleaseDossiers: vi.fn(), getTaskLifecycleReadiness: vi.fn(), requestTaskRelease: vi.fn(),
  reviewTaskRelease: vi.fn(), reviseTaskReleaseEvidence: vi.fn(), revokeTaskRelease: vi.fn(),
  getTaskTrainingExecutions: vi.fn(), getTaskTrainingCapabilities: vi.fn(), submitTaskTrainingExecution: vi.fn(),
  cancelTaskTrainingExecution: vi.fn(), exportTaskAnnotationsV2: vi.fn(), importTaskAnnotationsV2: vi.fn(),
}))

const task = (id = 'task-a'): AnalysisTask => ({
  task_id: id, camera_id: 'cam-a', template_id: 'object_detection.v1', name: 'Objects',
  authority: 'managed', lifecycle: 'active', current_revision: 1, config_sha256: 'a'.repeat(64),
  config: {}, deployment: { authority: 'managed', deployment_revision: 0 }, runtime: { state: 'stopped' }, source: { state: 'running' },
})
const event = (id = 'event-a') => ({ event_id: id, task_id: 'task-a', revision: 1, status: 'new',
  mode: 'validation', notes: 'earlier event note', frame_index: 42, deployment_revision: 1 })
const release = (revision = 1, createdBy = 'requester') => ({
  release_id: 'release-a', task_id: 'task-a', revision, subject_sha256: 'f'.repeat(64),
  state: 'requested', created_by: createdBy, evidence_authors: [createdBy],
  review: { notes: 'earlier review note' }, review_history: [],
})
const workflow = () => ({ workflow_id: 'workflow-a', task_id: 'task-a', revision: 3, state: 'training_export_ready' })
const candidateBinding = () => ({ task_id: 'task-a', camera_id: 'cam-a', task_revision: 1,
  task_config_sha256: 'a'.repeat(64), template_id: 'object_detection.v1' })
const trainedCandidate = () => ({ relative_path: 'task-training/job-a/execution-a/model.onnx', artifact_sha256: 'e'.repeat(64),
  descriptor: { name: 'Trained detector', adapter_id: 'onnx_detection.v1', output_schema: 'yolov8_detection.v1',
    labels: ['box'], input_size: [640, 640], input_layout: 'NCHW', input_dtype: 'float32', color_order: 'RGB' } })
const completedTraining = (): Record<string, unknown> => ({ job_id: 'job-complete', task_id: 'task-a', workflow_id: 'workflow-a',
  revision: 7, status: 'complete', request_sha256: 'f'.repeat(64), binding: candidateBinding(), dataset_id: 'dataset-a',
  package_sha256: 'c'.repeat(64), production_allowed: false, approval_scope: 'training_only',
  result: { schema: 'argus.task-training-result.v1', job_id: 'job-complete', execution_id: 'execution-a',
    request_sha256: 'f'.repeat(64), package_sha256: 'c'.repeat(64), provider: 'torch-cuda',
    candidate: trainedCandidate(), production_allowed: false, stage: 'candidate' } })
const candidateWorkflow = (overrides: Record<string, unknown> = {}): Record<string, unknown> => ({ ...workflow(),
  task_revision: 1, task_config_sha256: 'a'.repeat(64), template_id: 'object_detection.v1', manifest_sha256: 'd'.repeat(64),
  manifest: { binding: candidateBinding(), dataset_id: 'dataset-a', training: { approved_by: 'data-approver' } },
  operation: null, ...overrides })
const pendingCandidateWorkflow = () => candidateWorkflow({ state: 'candidate_importing', revision: 4,
  manifest_sha256: '9'.repeat(64), operation: { kind: 'candidate_import', requested_by: 'reviewer', requested_revision: 3,
    package_sha256: 'c'.repeat(64), approval_manifest_sha256: 'd'.repeat(64), candidate: trainedCandidate(),
    training_manifest: { package_sha256: 'c'.repeat(64) } } })
const candidateReceipt = () => ({ artifact_id: 'artifact-trained', stage: 'candidate', requested_revision: 3,
  import_revision: 4, operation_id: '7'.repeat(64), request_identity: { workflow_id: 'workflow-a',
    package_sha256: 'c'.repeat(64), candidate: trainedCandidate() } })
const registeredCandidateWorkflow = () => candidateWorkflow({ state: 'candidate_registered', revision: 5,
  manifest: { binding: candidateBinding(), dataset_id: 'dataset-a', training: { approved_by: 'data-approver',
    candidate: candidateReceipt() } }, candidate_import_receipt: candidateReceipt() })
function availableTrainingCandidate() {
  vi.mocked(api.getTaskTrainingExecutions).mockResolvedValue({ documents: [completedTraining()] })
  vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [candidateWorkflow()] })
  vi.mocked(dataApi.importTaskCandidate).mockResolvedValue({ document: registeredCandidateWorkflow() })
}
async function prepareCandidate(wrapper: VueWrapper) {
  await wrapper.get('[data-test="prepare-training-candidate"]').trigger('click'); await flushPromises()
  await wrapper.get('[data-test="confirm-training-candidate"]').setValue(true)
}
const capabilities = (overrides: Partial<api.TaskTrainingCapabilitiesResponse> = {}): api.TaskTrainingCapabilitiesResponse => ({
  schema: 'argus.task-training-capabilities.v1', recipe_id: 'yolov8_detection.v1',
  configured_enabled: false, maintenance_approved: false, resource_mode: 'exclusive_maintenance', device: 'cuda:0',
  minimum_free_memory_mb: 4096, poll_interval_seconds: 5,
  trusted_base_models: [{ model_id: 'base-a', name: 'Approved base A', relative_path: 'trusted/base-a.pt',
    artifact_sha256: 'b'.repeat(64), configured_trust: true }],
  runtime: { state: 'disabled', enabled: false, error_code: null },
  resource: { state: 'waiting_maintenance', reason_code: 'TRAINING_RESOURCE_APPROVAL_REQUIRED',
    device: 'cuda:0', free_memory_mb: null, memory_limit_enforced: false },
  validation_only: true, production_allowed: false,
  ...overrides,
})
async function prepareTraining(wrapper: VueWrapper) {
  await wrapper.get('[data-test="lifecycle-workflow"]').setValue('workflow-a'); await flushPromises()
  await wrapper.get('[data-test="read-training-capabilities"]').trigger('click'); await flushPromises()
  await wrapper.get('[data-test="training-package-sha"]').setValue('c'.repeat(64))
  await wrapper.get('[data-test="training-base-model"]').setValue('base-a')
  await wrapper.get('[data-test="confirm-training"]').setValue(true)
}
const wrappers: VueWrapper[] = []
function mountCurrentPanel() {
  const wrapper = mount(TaskLifecyclePanel, { props: { task: task(), bundleId: 'bundle-' + 'a'.repeat(32) },
    global: { plugins: [getActivePinia()!] } })
  wrappers.push(wrapper)
  return wrapper
}
function closePanel(wrapper: VueWrapper) {
  const index = wrappers.indexOf(wrapper)
  if (index >= 0) wrappers.splice(index, 1)
  wrapper.unmount()
}
function setup(role: 'admin' | 'engineer' | 'viewer' | 'operator' = 'engineer', username = 'reviewer') {
  const pinia = createPinia(); setActivePinia(pinia)
  useAuthStore(pinia).currentUser = { username, role }
  return mountCurrentPanel()
}

describe('TaskLifecyclePanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    socket.options = null
    vi.mocked(api.getTaskEvents).mockResolvedValue({ documents: [event()] })
    vi.mocked(api.getTaskReleaseDossiers).mockResolvedValue({ documents: [release()] })
    vi.mocked(api.getTaskTrainingExecutions).mockResolvedValue({ documents: [] })
    vi.mocked(api.getTaskTrainingCapabilities).mockResolvedValue(capabilities())
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [] })
    vi.mocked(dataApi.getTaskTrainingRecipes).mockResolvedValue({ documents: [] })
    vi.mocked(api.getTaskLifecycleReadiness).mockResolvedValue({ document: { production_ready: false,
      reasons: ['TEMPLATE_POLICY_NOT_APPROVED', 'PRODUCTION_APPLY_UNSUPPORTED'] } })
    vi.mocked(api.getTaskEventReplay).mockResolvedValue({ document: { kind: 'keyframe_only', ready: true,
      video_available: false, image_sha256: 'a'.repeat(64), source: { source_frame_index: null, frame_index: 42, pts: null } } })
    vi.mocked(api.reviewTaskEvent).mockResolvedValue({ document: event() })
    vi.mocked(api.reviewTaskRelease).mockResolvedValue({ document: release(2) })
    vi.mocked(api.requestTaskRelease).mockResolvedValue({ document: release() })
  })
  afterEach(() => { wrappers.splice(0).forEach(wrapper => wrapper.unmount()); vi.useRealTimers() })

  it('shows keyframe-only evidence without claiming a video or production alert', async () => {
    const wrapper = setup()
    await flushPromises()
    await wrapper.get('[data-test="lifecycle-event"]').setValue('event-a'); await flushPromises()
    expect(wrapper.text()).toContain('仅关键帧，未提供连续视频回放')
    expect(wrapper.find('video').exists()).toBe(false)
    expect(wrapper.get('img').attributes('src')).toContain('/task-a/events/event-a/evidence')
    expect(wrapper.text()).toContain('生产发布：NO-GO')
    expect(wrapper.text()).toContain('任务模板的正式生产安全契约尚未批准')
    expect(wrapper.text()).toContain('原始源帧号：未知（源未提供）')
    expect(api.submitTaskTrainingExecution).not.toHaveBeenCalled()
    expect(api.getTaskTrainingCapabilities).not.toHaveBeenCalled()
  })

  it('preserves event notes on a conflict and requires explicit refresh of the revision', async () => {
    const wrapper = setup()
    await flushPromises()
    await wrapper.get('[data-test="lifecycle-event"]').setValue('event-a'); await flushPromises()
    await wrapper.get('textarea').setValue('my event draft')
    vi.mocked(api.reviewTaskEvent).mockRejectedValueOnce(new Error('409 event version changed'))
    await wrapper.findAll('button').find(button => button.text() === '已确认')!.trigger('click'); await flushPromises()
    expect((wrapper.get('textarea').element as HTMLTextAreaElement).value).toBe('my event draft')
    vi.mocked(api.getTaskEvents).mockResolvedValue({ documents: [{ ...event(), revision: 2, notes: 'other person' }] })
    await wrapper.findAll('button').find(button => button.text() === '刷新闭环状态')!.trigger('click'); await flushPromises()
    expect(wrapper.text()).toContain('事件版本已更新')
    expect((wrapper.get('textarea').element as HTMLTextAreaElement).value).toBe('my event draft')
    expect(wrapper.findAll('button').find(button => button.text() === '已确认')!.attributes('disabled')).toBeDefined()
  })

  it('reuses the exact release review key after an uncertain response and keeps the draft', async () => {
    const wrapper = setup()
    await flushPromises()
    await wrapper.get('[data-test="lifecycle-release"]').setValue('release-a'); await flushPromises()
    await wrapper.get('textarea[aria-label="发布复核备注"]').setValue('keep release draft')
    await wrapper.get('[data-test="confirm-release-review"]').setValue(true)
    vi.mocked(api.reviewTaskRelease).mockRejectedValue(new Error('response unknown'))
    await wrapper.get('[data-test="approve-release-review"]').trigger('click'); await flushPromises()
    await wrapper.get('[data-test="approve-release-review"]').trigger('click'); await flushPromises()
    const calls = vi.mocked(api.reviewTaskRelease).mock.calls
    expect(calls).toHaveLength(2)
    expect(calls[0]?.[3]).toEqual(calls[1]?.[3])
    expect(calls[0]?.[2]).toMatchObject({ expected_revision: 1, notes: 'keep release draft', expected_subject_sha256: 'f'.repeat(64) })
    expect((wrapper.get('textarea[aria-label="发布复核备注"]').element as HTMLTextAreaElement).value).toBe('keep release draft')
  })

  it('disables self review and does not issue training or approvals on mount', async () => {
    const wrapper = setup('engineer', 'requester')
    await flushPromises()
    await wrapper.get('[data-test="lifecycle-release"]').setValue('release-a'); await flushPromises()
    expect(wrapper.get('[data-test="approve-release-review"]').attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('申请人不能复核自己的申请')
    expect(api.reviewTaskRelease).not.toHaveBeenCalled()
    expect(api.submitTaskTrainingExecution).not.toHaveBeenCalled()
    expect(api.requestTaskRelease).not.toHaveBeenCalled()
  })

  it('blocks the person who revised evidence from approving that same dossier', async () => {
    vi.mocked(api.getTaskReleaseDossiers).mockResolvedValue({ documents: [{ ...release(),
      evidence_authors: ['requester', 'reviewer'],
    }] })
    const wrapper = setup('engineer', 'reviewer'); await flushPromises()
    await wrapper.get('[data-test="lifecycle-release"]').setValue('release-a'); await flushPromises()
    await wrapper.get('[data-test="confirm-release-review"]').setValue(true)
    expect(wrapper.get('[data-test="approve-release-review"]').attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('证据提交者不能复核')
    expect(api.reviewTaskRelease).not.toHaveBeenCalled()
  })

  it.each([{ authors: undefined }, { authors: [] }, { authors: ['requester', 1] }])('requires a known evidence author list before independent review: $authors', async ({ authors }) => {
    vi.mocked(api.getTaskReleaseDossiers).mockResolvedValue({ documents: [{ ...release(), evidence_authors: authors }] })
    const wrapper = setup(); await flushPromises()
    await wrapper.get('[data-test="lifecycle-release"]').setValue('release-a'); await flushPromises()
    await wrapper.get('[data-test="confirm-release-review"]').setValue(true)
    expect(wrapper.get('[data-test="approve-release-review"]').attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('证据提交身份未知')
  })

  it('does not expose model/training mutation controls to a viewer', async () => {
    const wrapper = setup('viewer')
    await flushPromises()
    await wrapper.get('[data-test="lifecycle-event"]').setValue('event-a'); await flushPromises()
    expect(wrapper.find('[data-test="request-release"]').exists()).toBe(false)
    expect(api.getTaskReleaseDossiers).not.toHaveBeenCalled()
    expect(api.getTaskTrainingExecutions).not.toHaveBeenCalled()
    expect(api.getTaskTrainingCapabilities).not.toHaveBeenCalled()
    expect(wrapper.find('[data-test="read-training-capabilities"]').exists()).toBe(false)
    expect(wrapper.findAll('button').find(button => button.text() === '已确认')!.attributes('disabled')).toBeDefined()
  })

  it('fences an older event response across task A → B → A', async () => {
    let finish!: (value: { documents: Record<string, unknown>[] }) => void
    vi.mocked(api.getTaskEvents).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockResolvedValueOnce({ documents: [] }).mockResolvedValueOnce({ documents: [event('fresh-event')] })
    const wrapper = setup()
    await wrapper.setProps({ task: task('task-b') }); await flushPromises()
    await wrapper.setProps({ task: task() }); await flushPromises()
    finish({ documents: [event('old-event')] }); await flushPromises()
    expect(wrapper.find('option[value="fresh-event"]').exists()).toBe(true)
    expect(wrapper.find('option[value="old-event"]').exists()).toBe(false)
  })

  it('ignores a late release command receipt after switching task', async () => {
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(api.requestTaskRelease).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const wrapper = setup()
    await flushPromises()
    await wrapper.get('[data-test="request-release"]').trigger('click')
    await wrapper.setProps({ task: task('task-b') }); await flushPromises()
    finish({ document: release() }); await flushPromises()
    expect(wrapper.text()).not.toContain('发布档案已登记')
    expect(wrapper.emitted('refresh')).toBeUndefined()
  })

  it('does not clear a human draft when the same task object is refreshed', async () => {
    const wrapper = setup()
    await flushPromises()
    await wrapper.get('[data-test="lifecycle-release"]').setValue('release-a'); await flushPromises()
    await wrapper.get('textarea[aria-label="发布复核备注"]').setValue('human draft remains')
    await wrapper.setProps({ task: { ...task(), current_revision: 2 } }); await flushPromises()
    expect((wrapper.get('textarea[aria-label="发布复核备注"]').element as HTMLTextAreaElement).value).toBe('human draft remains')
    expect(api.getTaskEvents).toHaveBeenCalledTimes(1)
  })

  it('invalidates the review form after an evidence update without clearing the human draft', async () => {
    const wrapper = setup()
    await flushPromises()
    await wrapper.get('[data-test="lifecycle-release"]').setValue('release-a'); await flushPromises()
    await wrapper.get('textarea[aria-label="发布复核备注"]').setValue('preserve unsaved comments')
    const changed = { ...release(2), subject_sha256: 'b'.repeat(64), review: null }
    vi.mocked(api.getTaskReleaseDossiers).mockResolvedValue({ documents: [changed] })
    vi.mocked(api.reviseTaskReleaseEvidence).mockResolvedValue({ document: changed })
    await wrapper.findAll('button').find(button => button.text() === '更新证据并使旧结论失效')!.trigger('click')
    await flushPromises()
    expect((wrapper.get('textarea[aria-label="发布复核备注"]').element as HTMLTextAreaElement).value).toBe('preserve unsaved comments')
    expect(wrapper.text()).toContain('档案已更新，备注草稿保留')
    expect(wrapper.get('[data-test="approve-release-review"]').attributes('disabled')).toBeDefined()
  })

  it('refreshes only for the current task websocket topic and coalesces a burst', async () => {
    vi.useFakeTimers()
    setup(); await flushPromises()
    expect(socket.options?.topics).toEqual(['task_events'])
    socket.options!.onMessage('task_events', { task_id: 'other-task', camera_id: 'cam-a' })
    socket.options!.onMessage('task_events', { task_id: 'task-a', camera_id: 'other-camera' })
    await vi.advanceTimersByTimeAsync(200); await flushPromises()
    expect(api.getTaskEvents).toHaveBeenCalledTimes(1)
    for (let index = 0; index < 5; index++) socket.options!.onMessage('task_events', { task_id: 'task-a', camera_id: 'cam-a' })
    await vi.advanceTimersByTimeAsync(100); await flushPromises()
    expect(api.getTaskEvents).toHaveBeenCalledTimes(2)
  })

  it('reconciles a missed websocket event through REST without rebasing review notes', async () => {
    const wrapper = setup(); await flushPromises()
    await wrapper.get('[data-test="lifecycle-event"]').setValue('event-a'); await flushPromises()
    await wrapper.get('textarea').setValue('keep across missed websocket')
    expect(socket.options?.reconcileInterval).toBe(60_000)
    expect(socket.options?.fallbackInterval).toBe(60_000)
    vi.mocked(api.getTaskEvents).mockResolvedValue({ documents: [{ ...event(), revision: 2, notes: 'remote user' }] })
    await socket.options!.fallbackPoll(); await flushPromises()
    expect((wrapper.get('textarea').element as HTMLTextAreaElement).value).toBe('keep across missed websocket')
    expect(wrapper.text()).toContain('事件版本已更新')
    expect(wrapper.findAll('button').find(button => button.text() === '已确认')!.attributes('disabled')).toBeDefined()
  })

  it.each(['REST', 'WebSocket'])('revalidates the same selected keyframe on %s refresh without replacing notes', async source => {
    vi.useFakeTimers()
    const wrapper = setup(); await flushPromises()
    await wrapper.get('[data-test="lifecycle-event"]').setValue('event-a'); await flushPromises()
    expect(wrapper.find('img').exists()).toBe(true)
    await wrapper.get('textarea').setValue('keep notes after evidence changed')
    vi.mocked(api.getTaskEventReplay).mockResolvedValue({ document: { kind: 'keyframe_only', ready: false,
      video_available: false, image_sha256: 'a'.repeat(64), reason: 'EVIDENCE_SOURCE_IDENTITY_MISMATCH' } })
    if (source === 'REST') await socket.options!.fallbackPoll()
    else {
      socket.options!.onMessage('task_events', { task_id: 'task-a', camera_id: 'cam-a' })
      await vi.advanceTimersByTimeAsync(100)
    }
    await flushPromises()
    expect(api.getTaskEventReplay).toHaveBeenCalledTimes(2)
    expect(vi.mocked(api.getTaskEventReplay).mock.calls).toEqual([['task-a', 'event-a'], ['task-a', 'event-a']])
    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.text()).toContain('关键帧不可用或摘要核验失败')
    expect((wrapper.get('textarea').element as HTMLTextAreaElement).value).toBe('keep notes after evidence changed')
  })

  it('does not let a late replay response restore evidence invalidated by a newer refresh', async () => {
    let finish!: (value: { document: Record<string, unknown> }) => void
    const wrapper = setup(); await flushPromises()
    vi.mocked(api.getTaskEventReplay).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    await wrapper.get('[data-test="lifecycle-event"]').setValue('event-a'); await flushPromises()
    await wrapper.get('textarea').setValue('review draft stays intact')
    vi.mocked(api.getTaskEventReplay).mockResolvedValue({ document: { kind: 'keyframe_only', ready: false,
      video_available: false, image_sha256: 'a'.repeat(64) } })
    await socket.options!.fallbackPoll(); await flushPromises()
    expect(wrapper.text()).toContain('关键帧不可用或摘要核验失败')
    finish({ document: { kind: 'keyframe_only', ready: true, image_sha256: 'a'.repeat(64) } }); await flushPromises()
    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.text()).toContain('关键帧不可用或摘要核验失败')
    expect((wrapper.get('textarea').element as HTMLTextAreaElement).value).toBe('review draft stays intact')
  })

  it('fences late replay evidence across task A → B → A with the same event identity', async () => {
    let finish!: (value: { document: Record<string, unknown> }) => void
    const wrapper = setup(); await flushPromises()
    vi.mocked(api.getTaskEventReplay).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    await wrapper.get('[data-test="lifecycle-event"]').setValue('event-a'); await flushPromises()
    await wrapper.setProps({ task: task('task-b') }); await flushPromises()
    await wrapper.setProps({ task: task() }); await flushPromises()
    vi.mocked(api.getTaskEventReplay).mockResolvedValue({ document: { kind: 'keyframe_only', ready: false,
      video_available: false, image_sha256: 'a'.repeat(64) } })
    await wrapper.get('[data-test="lifecycle-event"]').setValue('event-a'); await flushPromises()
    finish({ document: { kind: 'keyframe_only', ready: true, image_sha256: 'a'.repeat(64) } }); await flushPromises()
    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.text()).toContain('关键帧不可用或摘要核验失败')
  })

  it('cancels an old scheduled websocket refresh when the task changes', async () => {
    vi.useFakeTimers()
    const wrapper = setup(); await flushPromises()
    socket.options!.onMessage('task_events', { task_id: 'task-a', camera_id: 'cam-a' })
    await wrapper.setProps({ task: task('task-b') }); await flushPromises()
    await vi.advanceTimersByTimeAsync(1000); await flushPromises()
    expect(api.getTaskEvents).toHaveBeenCalledTimes(2)
    expect(vi.mocked(api.getTaskEvents).mock.calls.map(call => call[0])).toEqual(['task-a', 'task-b'])
  })

  it('uses only an explicitly selected server-trusted base while a disabled scheduler leaves work queued', async () => {
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [workflow()] })
    vi.mocked(api.submitTaskTrainingExecution).mockResolvedValue({ document: { job_id: 'job-a', status: 'queued' } })
    const wrapper = setup(); await flushPromises()
    expect(api.getTaskTrainingCapabilities).not.toHaveBeenCalled()
    await wrapper.get('[data-test="lifecycle-workflow"]').setValue('workflow-a'); await flushPromises()
    expect(wrapper.get('[data-test="submit-training"]').attributes('disabled')).toBeDefined()
    expect(wrapper.find('input[aria-label="基础模型 SHA256"]').exists()).toBe(false)
    await prepareTraining(wrapper)
    expect(api.getTaskTrainingCapabilities).toHaveBeenCalledExactlyOnceWith('task-a')
    expect(wrapper.text()).toContain('已配置禁用')
    expect(wrapper.text()).toContain('等待维护窗口批准')
    expect(wrapper.text()).toContain('不是 GPU 预留或启动授权')
    expect(wrapper.get('[data-test="training-base-identity"]').text()).toContain('trusted/base-a.pt')
    expect(wrapper.get('[data-test="submit-training"]').attributes('disabled')).toBeUndefined()
    await wrapper.get('[data-test="submit-training"]').trigger('click'); await flushPromises()
    expect(api.submitTaskTrainingExecution).toHaveBeenCalledWith('task-a', 'workflow-a', {
      expected_revision: 3, expected_package_sha256: 'c'.repeat(64),
      base_model_path: 'trusted/base-a.pt', base_model_sha256: 'b'.repeat(64),
      params: { epochs: 10, imgsz: 640, batch: 4, seed: 0 },
    }, expect.any(String))
    expect(wrapper.text()).toContain('排队不等于已执行')
    expect(api.requestTaskRelease).not.toHaveBeenCalled()
  })

  it('fails closed for an unreadable capability snapshot instead of claiming an empty trusted registry', async () => {
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [workflow()] })
    const wrapper = setup(); await flushPromises(); await prepareTraining(wrapper)
    vi.mocked(api.getTaskTrainingCapabilities).mockRejectedValueOnce(new Error('403 training config unavailable'))
    await wrapper.get('[data-test="read-training-capabilities"]').trigger('click'); await flushPromises()
    expect(wrapper.get('[data-test="training-capabilities-error"]').text()).toContain('403 training config unavailable')
    expect(wrapper.text()).not.toContain('未配置可信基础模型')
    expect(wrapper.find('[data-test="training-capabilities"]').exists()).toBe(false)
    expect((wrapper.get('[data-test="training-base-model"]').element as HTMLSelectElement).value).toBe('')
    expect((wrapper.get('[data-test="confirm-training"]').element as HTMLInputElement).checked).toBe(false)
    expect(wrapper.get('[data-test="submit-training"]').attributes('disabled')).toBeDefined()
    expect((wrapper.get('[data-test="training-package-sha"]').element as HTMLInputElement).value).toBe('c'.repeat(64))
    expect(api.submitTaskTrainingExecution).not.toHaveBeenCalled()
  })

  it('closes both training gates while a global refresh is pending and starts trust revalidation immediately', async () => {
    let finish!: (value: { documents: Record<string, unknown>[] }) => void
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [workflow()] })
    const wrapper = setup(); await flushPromises(); await prepareTraining(wrapper)
    vi.mocked(api.getTaskEvents).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    await wrapper.findAll('button').find(button => button.text() === '刷新闭环状态')!.trigger('click'); await flushPromises()
    expect(api.getTaskTrainingCapabilities).toHaveBeenCalledTimes(2)
    expect(wrapper.get('[data-test="confirm-training"]').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-test="submit-training"]').attributes('disabled')).toBeDefined()
    await wrapper.get('[data-test="submit-training"]').trigger('click')
    await (wrapper.vm as unknown as { submitTraining: () => Promise<void> }).submitTraining()
    expect(api.submitTaskTrainingExecution).not.toHaveBeenCalled()
    finish({ documents: [event()] }); await flushPromises()
    expect((wrapper.get('[data-test="confirm-training"]').element as HTMLInputElement).checked).toBe(true)
    expect(wrapper.get('[data-test="confirm-training"]').attributes('disabled')).toBeUndefined()
    expect(wrapper.get('[data-test="submit-training"]').attributes('disabled')).toBeUndefined()
  })

  it('still invalidates changed trust when another global refresh read fails first', async () => {
    let finish!: (value: api.TaskTrainingCapabilitiesResponse) => void
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [workflow()] })
    const wrapper = setup(); await flushPromises(); await prepareTraining(wrapper)
    vi.mocked(api.getTaskEvents).mockRejectedValueOnce(new Error('event read failed'))
    vi.mocked(api.getTaskTrainingCapabilities).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    await wrapper.findAll('button').find(button => button.text() === '刷新闭环状态')!.trigger('click'); await flushPromises()
    expect(api.getTaskTrainingCapabilities).toHaveBeenCalledTimes(2)
    expect(wrapper.get('[data-test="submit-training"]').attributes('disabled')).toBeDefined()
    finish(capabilities({ trusted_base_models: [] })); await flushPromises()
    expect((wrapper.get('[data-test="training-base-model"]').element as HTMLSelectElement).value).toBe('')
    expect((wrapper.get('[data-test="confirm-training"]').element as HTMLInputElement).checked).toBe(false)
    expect(wrapper.get('[data-test="submit-training"]').attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('未配置可信基础模型')
  })

  it.each([
    { name: 'an untrusted model', changes: { trusted_base_models: [{ ...capabilities().trusted_base_models[0]!, configured_trust: false }] } },
    { name: 'a production-capable response', changes: { production_allowed: true } },
    { name: 'an unsupported device', changes: { device: 'cpu' } },
  ])('rejects $name without accepting user-supplied provenance', async ({ changes }) => {
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [workflow()] })
    vi.mocked(api.getTaskTrainingCapabilities).mockResolvedValue(capabilities(changes as unknown as Partial<api.TaskTrainingCapabilitiesResponse>))
    const wrapper = setup(); await flushPromises()
    await wrapper.get('[data-test="lifecycle-workflow"]').setValue('workflow-a'); await flushPromises()
    await wrapper.get('[data-test="read-training-capabilities"]').trigger('click'); await flushPromises()
    expect(wrapper.get('[data-test="training-capabilities-error"]').text()).toContain('训练配置响应无效')
    expect(wrapper.get('[data-test="submit-training"]').attributes('disabled')).toBeDefined()
    expect(api.submitTaskTrainingExecution).not.toHaveBeenCalled()
  })

  it('preserves drafts and confirmation for unchanged trust but invalidates selection when the trusted SHA changes', async () => {
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [workflow()] })
    const wrapper = setup(); await flushPromises(); await prepareTraining(wrapper)
    await wrapper.get('[data-test="lifecycle-release"]').setValue('release-a'); await flushPromises()
    await wrapper.get('textarea[aria-label="发布复核备注"]').setValue('release draft remains')
    vi.mocked(api.getTaskTrainingCapabilities).mockResolvedValue(capabilities({
      runtime: { state: 'running', enabled: true, error_code: null },
    }))
    await wrapper.findAll('button').find(button => button.text() === '刷新闭环状态')!.trigger('click'); await flushPromises()
    expect((wrapper.get('[data-test="training-base-model"]').element as HTMLSelectElement).value).toBe('base-a')
    expect((wrapper.get('[data-test="confirm-training"]').element as HTMLInputElement).checked).toBe(true)
    vi.mocked(api.getTaskTrainingCapabilities).mockResolvedValue(capabilities({
      trusted_base_models: [{ ...capabilities().trusted_base_models[0]!, artifact_sha256: 'd'.repeat(64) }],
    }))
    await wrapper.get('[data-test="read-training-capabilities"]').trigger('click'); await flushPromises()
    expect((wrapper.get('[data-test="training-base-model"]').element as HTMLSelectElement).value).toBe('')
    expect((wrapper.get('[data-test="confirm-training"]').element as HTMLInputElement).checked).toBe(false)
    expect((wrapper.get('[data-test="training-package-sha"]').element as HTMLInputElement).value).toBe('c'.repeat(64))
    expect((wrapper.get('textarea[aria-label="发布复核备注"]').element as HTMLTextAreaElement).value).toBe('release draft remains')
  })

  it('does not let a late capability response restore trust cleared by a newer failed read', async () => {
    let finish!: (value: api.TaskTrainingCapabilitiesResponse) => void
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [workflow()] })
    const wrapper = setup(); await flushPromises(); await prepareTraining(wrapper)
    vi.mocked(api.getTaskTrainingCapabilities).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockRejectedValueOnce(new Error('new trust read unavailable'))
    await wrapper.get('[data-test="read-training-capabilities"]').trigger('click'); await flushPromises()
    expect(wrapper.get('[data-test="submit-training"]').attributes('disabled')).toBeDefined()
    await socket.options!.fallbackPoll(); await flushPromises()
    expect(wrapper.get('[data-test="training-capabilities-error"]').text()).toContain('new trust read unavailable')
    finish(capabilities()); await flushPromises()
    expect(wrapper.find('[data-test="training-capabilities"]').exists()).toBe(false)
    expect((wrapper.get('[data-test="confirm-training"]').element as HTMLInputElement).checked).toBe(false)
    expect(wrapper.get('[data-test="submit-training"]').attributes('disabled')).toBeDefined()
  })

  it('fences late capability trust across task A → B → A', async () => {
    let finish!: (value: api.TaskTrainingCapabilitiesResponse) => void
    vi.mocked(api.getTaskTrainingCapabilities).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const wrapper = setup(); await flushPromises()
    await wrapper.get('[data-test="read-training-capabilities"]').trigger('click'); await flushPromises()
    await wrapper.setProps({ task: task('task-b') }); await flushPromises()
    await wrapper.setProps({ task: task() }); await flushPromises()
    finish(capabilities()); await flushPromises()
    expect(wrapper.find('[data-test="training-capabilities"]').exists()).toBe(false)
    expect(api.getTaskTrainingCapabilities).toHaveBeenCalledTimes(1)
  })

  it('fences late capability trust across an identity change and requires the new engineer to read again', async () => {
    let finish!: (value: api.TaskTrainingCapabilitiesResponse) => void
    vi.mocked(api.getTaskTrainingCapabilities).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const wrapper = setup(); await flushPromises()
    await wrapper.get('[data-test="read-training-capabilities"]').trigger('click'); await flushPromises()
    const auth = useAuthStore()
    auth.currentUser = { username: 'other-engineer', role: 'engineer' }; await flushPromises()
    finish(capabilities()); await flushPromises()
    expect(wrapper.find('[data-test="training-capabilities"]').exists()).toBe(false)
    expect(api.getTaskTrainingCapabilities).toHaveBeenCalledTimes(1)
    auth.currentUser = { username: 'other-engineer', role: 'viewer' }; await flushPromises()
    expect(wrapper.find('[data-test="read-training-capabilities"]').exists()).toBe(false)
  })

  it('registers only an explicitly confirmed completed candidate without changing other drafts or deploying', async () => {
    availableTrainingCandidate()
    const wrapper = setup(); await flushPromises()
    await wrapper.get('[data-test="lifecycle-workflow"]').setValue('workflow-a'); await flushPromises()
    await wrapper.get('textarea[aria-label="标注回传 v2 JSON"]').setValue('{"draft":"keep annotation"}')
    await wrapper.get('[data-test="training-package-sha"]').setValue('keep training draft')
    await wrapper.get('[data-test="lifecycle-event"]').setValue('event-a'); await flushPromises()
    await wrapper.get('textarea').setValue('keep event review draft')
    await wrapper.get('[data-test="prepare-training-candidate"]').trigger('click'); await flushPromises()
    expect(dataApi.importTaskCandidate).not.toHaveBeenCalled()
    expect(wrapper.get('[data-test="register-training-candidate"]').attributes('disabled')).toBeDefined()
    await wrapper.get('[data-test="confirm-training-candidate"]').setValue(true)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(dataApi.importTaskCandidate).toHaveBeenCalledExactlyOnceWith('task-a', 'workflow-a', {
      expected_revision: 3, expected_package_sha256: 'c'.repeat(64), candidate: trainedCandidate(),
    })
    expect(wrapper.get('[data-test="candidate-registered"]').text()).toContain('artifact-trained')
    expect(wrapper.text()).toContain('后续仍需旁路评估和独立审批')
    expect((wrapper.get('textarea').element as HTMLTextAreaElement).value).toBe('keep event review draft')
    expect((wrapper.get('textarea[aria-label="标注回传 v2 JSON"]').element as HTMLTextAreaElement).value).toBe('{"draft":"keep annotation"}')
    expect((wrapper.get('[data-test="training-package-sha"]').element as HTMLInputElement).value).toBe('keep training draft')
    expect(api.submitTaskTrainingExecution).not.toHaveBeenCalled()
    expect(api.requestTaskRelease).not.toHaveBeenCalled()
  })

  it('reconciles an uncertain candidate submission with the original body after the workflow revision advances', async () => {
    availableTrainingCandidate()
    vi.mocked(dataApi.importTaskCandidate).mockRejectedValueOnce(new Error('500 commit outcome unknown'))
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(wrapper.get('[data-test="candidate-error"]').text()).toContain('500 commit outcome unknown')
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [pendingCandidateWorkflow()] })
    await wrapper.findAll('button').find(button => button.text() === '刷新闭环状态')!.trigger('click'); await flushPromises()
    expect(wrapper.get('[data-test="training-candidate-draft"]').text()).toContain('原流程 r3')
    expect(wrapper.find('[data-test="candidate-blocked"]').exists()).toBe(false)
    await wrapper.get('[data-test="confirm-training-candidate"]').setValue(true)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    const calls = vi.mocked(dataApi.importTaskCandidate).mock.calls
    expect(calls).toHaveLength(2)
    expect(calls[1]).toEqual(calls[0])
    expect(calls[1]![2].expected_revision).toBe(3)
    expect(wrapper.get('[data-test="candidate-registered"]').text()).toContain('artifact-trained')
    expect(api.submitTaskTrainingExecution).not.toHaveBeenCalled()
    expect(api.requestTaskRelease).not.toHaveBeenCalled()
  })

  it('restores only the exact persisted pending intent for the same candidate and actor', async () => {
    availableTrainingCandidate()
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [pendingCandidateWorkflow()] })
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    expect(wrapper.get('[data-test="register-training-candidate"]').text()).toBe('使用原请求核对候选登记')
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(dataApi.importTaskCandidate).toHaveBeenCalledExactlyOnceWith('task-a', 'workflow-a', {
      expected_revision: 3, expected_package_sha256: 'c'.repeat(64), candidate: trainedCandidate(),
    })
  })

  it.each(['workflow', 'candidate', 'package'])('rejects a conflicting pending raw request %s before restoring a draft', async field => {
    availableTrainingCandidate()
    const row = pendingCandidateWorkflow(), operation = row.operation as Record<string, unknown>
    const request: Record<string, unknown> = { workflow_id: 'workflow-a', package_sha256: 'c'.repeat(64), candidate: trainedCandidate() }
    if (field === 'workflow') request.workflow_id = 'other-workflow'
    if (field === 'candidate') request.candidate = { ...trainedCandidate(), artifact_sha256: '0'.repeat(64) }
    if (field === 'package') request.package_sha256 = '0'.repeat(64)
    operation.request_identity = request
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [row] })
    const wrapper = setup(); await flushPromises()
    expect(wrapper.get('[data-test="prepare-training-candidate"]').attributes('disabled')).toBeDefined()
    expect(wrapper.find('[data-test="training-candidate-draft"]').exists()).toBe(false)
    expect(dataApi.importTaskCandidate).not.toHaveBeenCalled()
  })

  it.each(['request', 'approval'])('preserves an unknown request but blocks reconciliation after pending %s identity changes', async field => {
    availableTrainingCandidate()
    vi.mocked(dataApi.importTaskCandidate).mockRejectedValueOnce(new Error('lost acknowledgement'))
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    const original = vi.mocked(dataApi.importTaskCandidate).mock.calls[0]
    const row = pendingCandidateWorkflow(), operation = row.operation as Record<string, unknown>
    operation.request_identity = { workflow_id: 'workflow-a', package_sha256: 'c'.repeat(64), candidate: trainedCandidate() }
    if (field === 'request') operation.request_identity = { workflow_id: 'other-workflow' }
    if (field === 'approval') operation.approval_manifest_sha256 = '0'.repeat(64)
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [row] })
    await wrapper.findAll('button').find(button => button.text() === '刷新闭环状态')!.trigger('click'); await flushPromises()
    expect(wrapper.get('[data-test="candidate-blocked"]').text()).toContain('不会用新版本替换原请求')
    expect(wrapper.get('[data-test="register-training-candidate"]').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-test="training-candidate-draft"]').text()).toContain('原流程 r3')
    expect(vi.mocked(dataApi.importTaskCandidate).mock.calls).toEqual([original])
  })

  it('uses the exact persisted raw request instead of reconstructing it from normalized pending metadata', async () => {
    availableTrainingCandidate()
    const rawCandidate: Record<string, unknown> = trainedCandidate()
    const descriptor = { ...trainedCandidate().descriptor } as Record<string, unknown>
    delete descriptor.name; delete descriptor.input_layout; delete descriptor.input_dtype; delete descriptor.color_order
    rawCandidate.descriptor = descriptor
    const job = completedTraining(), result = job.result as Record<string, unknown>
    result.candidate = rawCandidate
    vi.mocked(api.getTaskTrainingExecutions).mockResolvedValue({ documents: [job] })
    const row = pendingCandidateWorkflow(), operation = row.operation as Record<string, unknown>
    operation.request_identity = { workflow_id: 'workflow-a', package_sha256: 'c'.repeat(64), candidate: rawCandidate }
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [row] })
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(dataApi.importTaskCandidate).toHaveBeenCalledExactlyOnceWith('task-a', 'workflow-a', {
      expected_revision: 3, expected_package_sha256: 'c'.repeat(64), candidate: rawCandidate,
    })
    // The mock receipt describes the normalized request, so it cannot confirm
    // this distinct original body even though its model bytes match.
    expect(wrapper.find('[data-test="candidate-registered"]').exists()).toBe(false)
    expect(wrapper.get('[data-test="candidate-error"]').text()).toContain('回执身份尚未确认')
  })

  it('does not fabricate a legacy raw request from different normalized pending metadata', async () => {
    availableTrainingCandidate()
    const row = pendingCandidateWorkflow(), operation = row.operation as Record<string, unknown>
    operation.candidate = { ...trainedCandidate(), descriptor: { ...trainedCandidate().descriptor, name: 'Normalized name' } }
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [row] })
    const wrapper = setup(); await flushPromises()
    expect(wrapper.get('[data-test="prepare-training-candidate"]').attributes('disabled')).toBeDefined()
    expect(wrapper.find('[data-test="training-candidate-draft"]').exists()).toBe(false)
    expect(dataApi.importTaskCandidate).not.toHaveBeenCalled()
  })

  it.each(['actor', 'candidate', 'package'])('does not resume a pending candidate intent with a mismatched %s', async field => {
    availableTrainingCandidate()
    const row = pendingCandidateWorkflow(), operation = row.operation as Record<string, unknown>
    if (field === 'actor') operation.requested_by = 'another-engineer'
    if (field === 'candidate') operation.candidate = { ...trainedCandidate(), artifact_sha256: '0'.repeat(64) }
    if (field === 'package') operation.package_sha256 = '0'.repeat(64)
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [row] })
    const wrapper = setup(); await flushPromises()
    expect(wrapper.get('[data-test="prepare-training-candidate"]').attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('待对账意图不属于本次产物与个人身份')
    expect(dataApi.importTaskCandidate).not.toHaveBeenCalled()
  })

  it.each(['artifact', 'request', 'task', 'binding', 'execution', 'production'])('rejects a completed result with mismatched %s identity', async field => {
    availableTrainingCandidate()
    const job = completedTraining(), result = job.result as Record<string, unknown>
    if (field === 'artifact') result.candidate = { ...trainedCandidate(), artifact_sha256: 'not-a-sha' }
    if (field === 'request') result.request_sha256 = '0'.repeat(64)
    if (field === 'task') job.task_id = 'other-task'
    if (field === 'binding') job.binding = { ...candidateBinding(), task_config_sha256: '0'.repeat(64) }
    if (field === 'execution') result.execution_id = ''
    if (field === 'production') result.production_allowed = true
    vi.mocked(api.getTaskTrainingExecutions).mockResolvedValue({ documents: [job] })
    const wrapper = setup(); await flushPromises()
    expect(wrapper.get('[data-test="prepare-training-candidate"]').attributes('disabled')).toBeDefined()
    await wrapper.get('[data-test="prepare-training-candidate"]').trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="training-candidate-draft"]').exists()).toBe(false)
    expect(dataApi.importTaskCandidate).not.toHaveBeenCalled()
  })

  it.each(['job', 'workflow', 'revision', 'task-config'])('keeps the frozen candidate but blocks submission after %s changes', async field => {
    availableTrainingCandidate()
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    if (field === 'job') vi.mocked(api.getTaskTrainingExecutions).mockResolvedValue({ documents: [] })
    if (field === 'workflow') vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [] })
    if (field === 'revision') vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [candidateWorkflow({ revision: 4 })] })
    if (field === 'task-config') await wrapper.setProps({ task: { ...task(), current_revision: 2, config_sha256: '0'.repeat(64) } })
    await wrapper.findAll('button').find(button => button.text() === '刷新闭环状态')!.trigger('click'); await flushPromises()
    expect(wrapper.get('[data-test="training-candidate-draft"]').text()).toContain(trainedCandidate().relative_path)
    expect(wrapper.get('[data-test="training-candidate-draft"]').text()).toContain('原流程 r3')
    expect(wrapper.find('[data-test="candidate-blocked"]').exists()).toBe(true)
    expect(wrapper.get('[data-test="register-training-candidate"]').attributes('disabled')).toBeDefined()
    expect((wrapper.get('[data-test="confirm-training-candidate"]').element as HTMLInputElement).checked).toBe(false)
    await (wrapper.vm as unknown as { registerCandidate: () => Promise<void> }).registerCandidate()
    expect(dataApi.importTaskCandidate).not.toHaveBeenCalled()
  })

  it('blocks candidate confirmation during refresh and after a failed read without deleting its request', async () => {
    availableTrainingCandidate()
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    let reject!: (error: Error) => void
    vi.mocked(api.getTaskEvents).mockReturnValueOnce(new Promise((_resolve, fail) => { reject = fail }))
    await wrapper.findAll('button').find(button => button.text() === '刷新闭环状态')!.trigger('click'); await flushPromises()
    expect(wrapper.get('[data-test="confirm-training-candidate"]').attributes('disabled')).toBeDefined()
    await (wrapper.vm as unknown as { registerCandidate: () => Promise<void> }).registerCandidate()
    expect(dataApi.importTaskCandidate).not.toHaveBeenCalled()
    reject(new Error('offline lifecycle read')); await flushPromises()
    expect(wrapper.get('[data-test="candidate-blocked"]').text()).toContain('闭环读取失败')
    expect(wrapper.get('[data-test="training-candidate-draft"]').text()).toContain('原流程 r3')
    expect(wrapper.get('[data-test="register-training-candidate"]').attributes('disabled')).toBeDefined()
  })

  it('fences late registration across task A → B → A while preserving the original unknown request', async () => {
    availableTrainingCandidate()
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(dataApi.importTaskCandidate).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    const original = vi.mocked(dataApi.importTaskCandidate).mock.calls[0]
    await wrapper.setProps({ task: task('task-b') }); await flushPromises()
    expect(wrapper.find('[data-test="training-candidate-draft"]').exists()).toBe(false)
    await wrapper.setProps({ task: task() }); await flushPromises()
    expect(wrapper.get('[data-test="training-candidate-draft"]').text()).toContain('原流程 r3')
    expect((wrapper.get('[data-test="confirm-training-candidate"]').element as HTMLInputElement).checked).toBe(false)
    finish({ document: registeredCandidateWorkflow() }); await flushPromises()
    expect(wrapper.find('[data-test="candidate-registered"]').exists()).toBe(false)
    expect(wrapper.get('[data-test="register-training-candidate"]').text()).toBe('使用原请求核对候选登记')
    await wrapper.get('[data-test="confirm-training-candidate"]').setValue(true)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(vi.mocked(dataApi.importTaskCandidate).mock.calls[1]).toEqual(original)
  })

  it('rejects malformed registration receipts and retains the original candidate as unknown', async () => {
    availableTrainingCandidate()
    vi.mocked(dataApi.importTaskCandidate).mockResolvedValue({ document: { ...registeredCandidateWorkflow(), task_id: 'other-task' } })
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="candidate-registered"]').exists()).toBe(false)
    expect(wrapper.get('[data-test="candidate-error"]').text()).toContain('回执身份尚未确认')
    expect(wrapper.get('[data-test="register-training-candidate"]').text()).toBe('使用原请求核对候选登记')
  })

  it('does not let an unrelated form change permit duplicate in-flight candidate registration', async () => {
    availableTrainingCandidate()
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(dataApi.importTaskCandidate).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    await wrapper.get('[data-test="lifecycle-event"]').setValue('event-a'); await flushPromises()
    await (wrapper.vm as unknown as { registerCandidate: () => Promise<void> }).registerCandidate()
    expect(dataApi.importTaskCandidate).toHaveBeenCalledTimes(1)
    finish({ document: registeredCandidateWorkflow() }); await flushPromises()
  })

  it('shares the candidate claim with the real external-training data panel', async () => {
    availableTrainingCandidate()
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(dataApi.importTaskCandidate).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const lifecycle = setup(); await flushPromises(); await prepareCandidate(lifecycle)
    const dataset = mount(TaskDatasetPanel, { props: { task: task() }, global: { plugins: [getActivePinia()!] } })
    wrappers.push(dataset); await flushPromises()
    expect(dataset.get('[data-test="prepare-manual-candidate"]').attributes('disabled')).toBeDefined()
    expect(dataset.text()).toContain('训练闭环入口已保留此任务的候选请求')
    await lifecycle.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    await (dataset.vm as unknown as { prepareManualCandidate: () => void }).prepareManualCandidate()
    await (dataset.vm as unknown as { registerManualCandidate: () => Promise<void> }).registerManualCandidate()
    expect(dataApi.importTaskCandidate).toHaveBeenCalledTimes(1)
    finish({ document: registeredCandidateWorkflow() }); await flushPromises()
    expect(lifecycle.get('[data-test="candidate-registered"]').text()).toContain('artifact-trained')
    expect(dataset.find('[data-test="manual-candidate-draft"]').exists()).toBe(false)
  })

  it.each(['stage', 'request'])('does not accept a contradictory persisted %s receipt under a matching top-level projection', async field => {
    availableTrainingCandidate()
    const response = registeredCandidateWorkflow(), manifest = response.manifest as { training: { candidate: Record<string, unknown> } }
    if (field === 'stage') manifest.training.candidate.stage = 'production'
    else manifest.training.candidate.request_identity = { workflow_id: 'other-workflow' }
    vi.mocked(dataApi.importTaskCandidate).mockResolvedValue({ document: response })
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="candidate-registered"]').exists()).toBe(false)
    expect(wrapper.get('[data-test="candidate-error"]').text()).toContain('回执身份尚未确认')
  })

  it('requires a personal model-manager identity and does not reuse another person’s draft confirmation', async () => {
    availableTrainingCandidate()
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    const auth = useAuthStore()
    auth.currentUser = { username: 'other-engineer', role: 'engineer' }; await flushPromises()
    expect(wrapper.find('[data-test="training-candidate-draft"]').exists()).toBe(false)
    auth.currentUser = { username: 'reviewer', role: 'engineer' }; await flushPromises()
    expect(wrapper.find('[data-test="training-candidate-draft"]').exists()).toBe(false)
    await wrapper.get('[data-test="prepare-training-candidate"]').trigger('click'); await flushPromises()
    expect((wrapper.get('[data-test="confirm-training-candidate"]').element as HTMLInputElement).checked).toBe(false)
    auth.currentUser = { username: 'reviewer', role: 'operator' }; await flushPromises()
    expect(wrapper.find('[data-test="register-training-candidate"]').exists()).toBe(false)
    await (wrapper.vm as unknown as { registerCandidate: () => Promise<void> }).registerCandidate()
    expect(dataApi.importTaskCandidate).not.toHaveBeenCalled()
  })

  it('retains the exact unknown candidate through a real unmount/remount but never carries confirmation', async () => {
    availableTrainingCandidate()
    vi.mocked(dataApi.importTaskCandidate).mockRejectedValueOnce(new Error('response lost after intent'))
    const first = setup(); await flushPromises(); await prepareCandidate(first)
    await first.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    const original = vi.mocked(dataApi.importTaskCandidate).mock.calls[0]
    closePanel(first)
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [pendingCandidateWorkflow()] })
    const reopened = mountCurrentPanel(); await flushPromises()
    expect(reopened.get('[data-test="training-candidate-draft"]').text()).toContain('原流程 r3')
    expect(reopened.get('[data-test="register-training-candidate"]').text()).toBe('使用原请求核对候选登记')
    expect((reopened.get('[data-test="confirm-training-candidate"]').element as HTMLInputElement).checked).toBe(false)
    await reopened.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(dataApi.importTaskCandidate).toHaveBeenCalledTimes(1)
    await reopened.get('[data-test="confirm-training-candidate"]').setValue(true)
    await reopened.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(vi.mocked(dataApi.importTaskCandidate).mock.calls[1]).toEqual(original)
    expect(reopened.get('[data-test="candidate-registered"]').text()).toContain('artifact-trained')
  })

  it('shares a frozen request across simultaneous task views but isolates local confirmations and the POST claim', async () => {
    availableTrainingCandidate()
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(dataApi.importTaskCandidate).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const first = setup(); await flushPromises(); await prepareCandidate(first)
    const second = mountCurrentPanel(); await flushPromises()
    expect(second.get('[data-test="training-candidate-draft"]').text()).toContain('原流程 r3')
    expect((second.get('[data-test="confirm-training-candidate"]').element as HTMLInputElement).checked).toBe(false)
    await second.get('[data-test="confirm-training-candidate"]').setValue(true)
    await first.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(second.get('[data-test="register-training-candidate"]').attributes('disabled')).toBeDefined()
    await (second.vm as unknown as { registerCandidate: () => Promise<void> }).registerCandidate()
    expect(dataApi.importTaskCandidate).toHaveBeenCalledTimes(1)
    finish({ document: registeredCandidateWorkflow() }); await flushPromises()
    expect(first.get('[data-test="candidate-registered"]').text()).toContain('artifact-trained')
    expect(second.get('[data-test="candidate-registered"]').text()).toContain('artifact-trained')
  })

  it('does not drop an in-flight claim when the submitting component unmounts', async () => {
    availableTrainingCandidate()
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(dataApi.importTaskCandidate).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const first = setup(); await flushPromises(); await prepareCandidate(first)
    await first.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    closePanel(first)
    const reopened = mountCurrentPanel(); await flushPromises()
    expect(reopened.get('[data-test="register-training-candidate"]').attributes('disabled')).toBeDefined()
    await (reopened.vm as unknown as { registerCandidate: () => Promise<void> }).registerCandidate()
    expect(dataApi.importTaskCandidate).toHaveBeenCalledTimes(1)
    finish({ document: registeredCandidateWorkflow() }); await flushPromises()
    expect(reopened.find('[data-test="candidate-registered"]').exists()).toBe(false)
    expect((reopened.get('[data-test="confirm-training-candidate"]').element as HTMLInputElement).checked).toBe(false)
    expect(reopened.get('[data-test="confirm-training-candidate"]').attributes('disabled')).toBeUndefined()
    await reopened.get('[data-test="confirm-training-candidate"]').setValue(true)
    await reopened.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(dataApi.importTaskCandidate).toHaveBeenCalledTimes(2)
    expect(vi.mocked(dataApi.importTaskCandidate).mock.calls[1]).toEqual(vi.mocked(dataApi.importTaskCandidate).mock.calls[0])
  })

  it('clears an invalidated personal session even while the task panel is closed and fences its late result', async () => {
    availableTrainingCandidate()
    let finish!: (value: { document: Record<string, unknown> }) => void
    vi.mocked(dataApi.importTaskCandidate).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const first = setup(); await flushPromises(); await prepareCandidate(first)
    await first.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    closePanel(first)
    const auth = useAuthStore()
    auth.clear()
    auth.currentUser = { username: 'reviewer', role: 'engineer' }
    const reopened = mountCurrentPanel(); await flushPromises()
    expect(reopened.find('[data-test="training-candidate-draft"]').exists()).toBe(false)
    expect(reopened.get('[data-test="prepare-training-candidate"]').attributes('disabled')).toBeDefined()
    finish({ document: registeredCandidateWorkflow() }); await flushPromises()
    expect(reopened.find('[data-test="candidate-registered"]').exists()).toBe(false)
    expect(reopened.find('[data-test="training-candidate-draft"]').exists()).toBe(false)
    await reopened.get('[data-test="prepare-training-candidate"]').trigger('click'); await flushPromises()
    expect((reopened.get('[data-test="confirm-training-candidate"]').element as HTMLInputElement).checked).toBe(false)
    expect(dataApi.importTaskCandidate).toHaveBeenCalledTimes(1)
  })

  it.each(['missing-operation', 'wrong-operation', 'missing-request', 'wrong-candidate', 'wrong-package'])('does not claim success for a %s receipt', async mismatch => {
    availableTrainingCandidate()
    const response = registeredCandidateWorkflow(), receipt = response.candidate_import_receipt as Record<string, unknown>
    if (mismatch === 'missing-operation') delete receipt.operation_id
    if (mismatch === 'wrong-operation') receipt.operation_id = '8'.repeat(64)
    if (mismatch === 'missing-request') delete receipt.request_identity
    if (mismatch === 'wrong-candidate') (receipt.request_identity as Record<string, unknown>).candidate = { ...trainedCandidate(), artifact_sha256: '0'.repeat(64) }
    if (mismatch === 'wrong-package') (receipt.request_identity as Record<string, unknown>).package_sha256 = '0'.repeat(64)
    vi.mocked(dataApi.importTaskCandidate).mockResolvedValue({ document: response })
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="candidate-registered"]').exists()).toBe(false)
    expect(wrapper.get('[data-test="candidate-error"]').text()).toContain('回执身份尚未确认')
    expect(wrapper.get('[data-test="training-candidate-draft"]').text()).toContain('原流程 r3')
    expect(wrapper.get('[data-test="register-training-candidate"]').text()).toBe('使用原请求核对候选登记')
  })

  it.each(['candidate', 'package'])('does not authorize reconciliation from a registered GET with a different %s projection', async field => {
    availableTrainingCandidate()
    vi.mocked(dataApi.importTaskCandidate).mockRejectedValueOnce(new Error('unknown write'))
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    const row = registeredCandidateWorkflow()
    const manifest = row.manifest as { training: { candidate: { request_identity: Record<string, unknown> } } }
    if (field === 'candidate') manifest.training.candidate.request_identity.candidate = { ...trainedCandidate(), artifact_sha256: '0'.repeat(64) }
    if (field === 'package') manifest.training.candidate.request_identity.package_sha256 = '0'.repeat(64)
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [row] })
    await wrapper.findAll('button').find(button => button.text() === '刷新闭环状态')!.trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="candidate-registered"]').exists()).toBe(false)
    expect(wrapper.get('[data-test="candidate-blocked"]').text()).toContain('不会用新版本替换原请求')
    expect(wrapper.get('[data-test="register-training-candidate"]').attributes('disabled')).toBeDefined()
    expect(dataApi.importTaskCandidate).toHaveBeenCalledTimes(1)
  })

  it('keeps a legacy registered GET unconfirmed and verifies it only through the frozen original POST', async () => {
    availableTrainingCandidate()
    vi.mocked(dataApi.importTaskCandidate).mockRejectedValueOnce(new Error('response lost'))
    const wrapper = setup(); await flushPromises(); await prepareCandidate(wrapper)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    const original = vi.mocked(dataApi.importTaskCandidate).mock.calls[0], row = registeredCandidateWorkflow()
    const manifest = row.manifest as { training: { candidate: { request_identity?: unknown } } }
    delete manifest.training.candidate.request_identity
    delete row.candidate_import_receipt
    vi.mocked(dataApi.getTaskWorkflows).mockResolvedValue({ documents: [row] })
    await wrapper.findAll('button').find(button => button.text() === '刷新闭环状态')!.trigger('click'); await flushPromises()
    expect(wrapper.find('[data-test="candidate-registered"]').exists()).toBe(false)
    expect(wrapper.find('[data-test="candidate-blocked"]').exists()).toBe(false)
    await wrapper.get('[data-test="confirm-training-candidate"]').setValue(true)
    await wrapper.get('[data-test="register-training-candidate"]').trigger('click'); await flushPromises()
    expect(vi.mocked(dataApi.importTaskCandidate).mock.calls[1]).toEqual(original)
    expect(wrapper.get('[data-test="candidate-registered"]').text()).toContain('artifact-trained')
  })

  it('shows queue positions, explicit execution and cleanup states, and CAS cancellation for reconciling jobs', async () => {
    vi.mocked(api.getTaskTrainingExecutions).mockResolvedValue({ documents: [
      { job_id: 'job-queued', status: 'queued', revision: 1, queue_position: 2 },
      { job_id: 'job-waiting', status: 'waiting_resources', revision: 2, result: { error_code: 'TRAINING_RESOURCE_APPROVAL_REQUIRED' } },
      { job_id: 'job-running', status: 'running', revision: 3 },
      { job_id: 'job-cancelling', status: 'cancelling', revision: 4 },
      { job_id: 'job-reconciling', status: 'reconciling', revision: 5, result: { error_code: 'TRAINING_OWNER_LIVENESS_UNKNOWN' } },
    ] })
    vi.mocked(api.cancelTaskTrainingExecution).mockResolvedValue({ document: { job_id: 'job-reconciling', status: 'cancelling', revision: 6 } })
    const wrapper = setup(); await flushPromises()
    expect(wrapper.text()).toContain('队列位置 2')
    expect(wrapper.text()).toContain('资源阻塞，尚未开始')
    expect(wrapper.text()).toContain('执行中')
    expect(wrapper.text()).toContain('取消中，等待进程清理确认')
    expect(wrapper.text()).toContain('等待对账，尚未确认旧进程退出')
    expect(wrapper.text()).toContain('旧训练进程的存活状态尚未核实')
    expect(wrapper.find('[role="progressbar"]').exists()).toBe(false)
    const reconciling = wrapper.findAll('[data-test="training-job"]').find(row => row.text().includes('等待对账'))!
    await reconciling.get('button').trigger('click'); await flushPromises()
    expect(api.cancelTaskTrainingExecution).toHaveBeenCalledExactlyOnceWith('task-a', 'job-reconciling', { expected_revision: 5 })
    expect(api.submitTaskTrainingExecution).not.toHaveBeenCalled()
  })
})
