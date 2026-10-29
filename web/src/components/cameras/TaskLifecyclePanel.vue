<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { getActivePinia } from 'pinia'
import { useAuthStore } from '../../stores/useAuthStore'
import { getTaskWorkflows, importTaskCandidate, type AnalysisTask } from '../../api/analysisTasks'
import {
  getTaskEvents, reviewTaskEvent, getTaskEventReplay, taskEventEvidenceUrl,
  getTaskReleaseDossiers, getTaskLifecycleReadiness, requestTaskRelease, reviewTaskRelease,
  reviseTaskReleaseEvidence, revokeTaskRelease, getTaskTrainingExecutions,
  getTaskTrainingCapabilities, type TaskTrainingCapabilitiesResponse,
  submitTaskTrainingExecution, cancelTaskTrainingExecution, exportTaskAnnotationsV2, importTaskAnnotationsV2,
} from '../../api/taskLifecycle'
import type { TaskCandidateImportRequest, TaskReleaseRequest } from '../../types/generated-openapi'
import { extractErrorMessage } from '../../utils/error'
import { useWebSocket } from '../../composables/useWebSocket'
import { useTrainingCandidateLedger, type TrainingCandidateDraft } from '../../composables/useTrainingCandidateLedger'

const props = defineProps<{ task: AnalysisTask; bundleId?: string }>()
const emit = defineEmits<{ refresh: [] }>()
const pinia = getActivePinia()
const auth = pinia ? useAuthStore(pinia) : null
const username = computed(() => auth?.currentUser?.username ?? '')
const role = computed(() => auth?.currentUser?.role ?? '')
const canManage = computed(() => !!username.value && ['admin', 'engineer'].includes(auth?.currentUser?.role ?? ''))
const canReviewEvents = computed(() => !!username.value && ['admin', 'engineer', 'operator'].includes(auth?.currentUser?.role ?? ''))
type Document = Record<string, unknown>
const events = ref<Document[]>([]), releases = ref<Document[]>([]), jobs = ref<Document[]>([]), workflows = ref<Document[]>([])
const loading = ref(false), busy = ref(false), loaded = ref(false), error = ref(''), notice = ref('')
const lifecycleReadFailed = ref(false), candidateConfirmed = ref(false)
const candidateLedger = useTrainingCandidateLedger(auth)
const sharedCandidateDraft = computed(() => candidateLedger.get(props.task.camera_id, props.task.task_id))
const candidateDraft = computed(() => {
  const draft = sharedCandidateDraft.value
  return draft?.origin === 'training' ? draft : null
})
const candidateSubmitting = computed(() => candidateLedger.submitting(props.task.camera_id, props.task.task_id))
const eventId = ref(''), releaseId = ref(''), workflowId = ref('')
const eventNotes = ref(''), eventRevision = ref<number | null>(null)
const releaseNotes = ref(''), releaseRevision = ref<number | null>(null), releaseSubject = ref('')
const releaseConfirmed = ref(false)
const evidenceText = ref('[]'), readiness = ref<Document>({}), replay = ref<Document | null>(null), imageFailed = ref(false)
const workflowRevision = ref<number | null>(null), annotationText = ref('{}')
const packageSha = ref(''), baseModelId = ref(''), trainingConfirmed = ref(false)
const trainingCapabilities = ref<TaskTrainingCapabilitiesResponse | null>(null)
const capabilitiesRequested = ref(false), capabilitiesLoading = ref(false), capabilitiesError = ref('')
const epochs = ref(10), imageSize = ref(640), batch = ref(4), seed = ref(0)
let generation = 0, reads = 0, previewReads = 0, readinessReads = 0, capabilitiesReads = 0, operations = 0, formGeneration = 0, disposed = false
let eventRefreshTimer: ReturnType<typeof setTimeout> | null = null
const keys = new Map<string, string>()
const record = (value: unknown): Document => value && typeof value === 'object' && !Array.isArray(value) ? value as Document : {}
const text = (value: unknown): string => typeof value === 'string' ? value : ''
const integer = (value: unknown): number | null => typeof value === 'number' && Number.isInteger(value) ? value : null
const sha256 = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value)
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
  if (value && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([name, item]) => `${JSON.stringify(name)}:${canonical(item)}`).join(',')}}`
  return JSON.stringify(value) ?? 'null'
}
function taskBinding() {
  return { task_id: props.task.task_id, camera_id: props.task.camera_id, task_revision: props.task.current_revision,
    task_config_sha256: props.task.config_sha256, template_id: props.task.template_id }
}
function candidateJobIdentity(job: Document) {
  return canonical([job.job_id, job.task_id, job.workflow_id, job.revision, job.status, job.request_sha256,
    job.binding, job.dataset_id, job.package_sha256, job.result, job.production_allowed, job.approval_scope])
}
function candidateJobError(job: Document): string {
  const result = record(job.result), candidate = record(result.candidate), descriptor = record(candidate.descriptor)
  if (job.status !== 'complete') return '训练尚未确认完成，不能登记候选'
  if (job.task_id !== props.task.task_id || props.task.template_id !== 'object_detection.v1'
    || canonical(job.binding) !== canonical(taskBinding())) return '训练结果不属于当前任务、场景配置或版本'
  if (!text(job.job_id) || !text(job.workflow_id) || !sha256(job.request_sha256) || !sha256(job.package_sha256)
    || integer(job.revision) === null || job.production_allowed !== false || job.approval_scope !== 'training_only'
    || result.schema !== 'argus.task-training-result.v1' || result.job_id !== job.job_id
    || !text(result.execution_id) || result.request_sha256 !== job.request_sha256
    || result.package_sha256 !== job.package_sha256 || result.provider !== 'torch-cuda'
    || result.production_allowed !== false || result.stage !== 'candidate'
    || Object.keys(candidate).sort().join(',') !== 'artifact_sha256,descriptor,relative_path'
    || !sha256(candidate.artifact_sha256) || !text(candidate.relative_path)
    || descriptor.adapter_id !== 'onnx_detection.v1' || descriptor.output_schema !== 'yolov8_detection.v1') {
    return '完成结果的执行、训练交接或候选制品身份不完整'
  }
  const row = workflows.value.find(item => item.workflow_id === job.workflow_id), manifest = record(row?.manifest)
  if (!row || row.task_id !== props.task.task_id || canonical(manifest.binding) !== canonical(job.binding)
    || manifest.dataset_id !== job.dataset_id || !text(job.dataset_id)
    || !text(record(manifest.training).approved_by) || integer(row.revision) === null || !sha256(row.manifest_sha256)) {
    return '原数据流程、冻结数据集或人工训练批准尚未核实'
  }
  return ''
}
function candidateRequestIdentity(draft: TrainingCandidateDraft) {
  return canonical({ workflow_id: draft.workflowId, package_sha256: draft.body.expected_package_sha256,
    candidate: draft.body.candidate })
}
function candidateReceiptMatches(receipt: Document, draft: TrainingCandidateDraft) {
  return sha256(receipt.operation_id) && receipt.stage === 'candidate' && text(receipt.artifact_id)
    && receipt.requested_revision === draft.body.expected_revision
    && canonical(receipt.request_identity) === candidateRequestIdentity(draft)
}
function pendingRequestMatches(operation: Document, workflowId: string, packageSha: string, candidate: Document) {
  // New intents persist the raw request independently of validator-normalized
  // model metadata. Legacy intents may only reuse an already known exact body.
  const frozen = record(operation.candidate)
  if (frozen.relative_path !== candidate.relative_path || frozen.artifact_sha256 !== candidate.artifact_sha256) return false
  return operation.request_identity !== undefined
    ? canonical(operation.request_identity) === canonical({ workflow_id: workflowId, package_sha256: packageSha, candidate })
    : canonical(operation.candidate) === canonical(candidate)
}
function pendingCandidateMatches(row: Document, draft: TrainingCandidateDraft) {
  const operation = record(row.operation)
  return row.state === 'candidate_importing' && operation.kind === 'candidate_import'
    && operation.requested_by === draft.actor && operation.requested_revision === draft.body.expected_revision
    && operation.package_sha256 === draft.body.expected_package_sha256
    && operation.approval_manifest_sha256 === draft.manifestSha
    && pendingRequestMatches(operation, draft.workflowId, draft.body.expected_package_sha256, draft.body.candidate)
    && record(operation.training_manifest).package_sha256 === draft.body.expected_package_sha256
}
const candidateBlocked = computed(() => {
  const draft = candidateDraft.value
  if (!draft) return ''
  if (!canManage.value || username.value !== draft.actor) return '登记需要原个人工程师／管理员身份重新核对'
  if (draft.taskId !== props.task.task_id || draft.cameraId !== props.task.camera_id
    || draft.binding !== canonical(taskBinding())) return '任务或场景配置已变化；保留原请求，不能改绑登记'
  if (loading.value) return '正在重新读取闭环状态，暂不能登记'
  if (lifecycleReadFailed.value) return '最近一次闭环读取失败，不能使用可能过期的训练结果'
  const job = jobs.value.find(item => item.job_id === draft.jobId)
  if (!job || candidateJobIdentity(job) !== draft.jobIdentity) return '训练记录缺席或身份已变化，原候选草稿保留'
  const problem = candidateJobError(job)
  if (problem) return problem
  const row = workflows.value.find(item => item.workflow_id === draft.workflowId)!
  if (row.state === 'training_export_ready' && row.revision === draft.body.expected_revision
    && row.manifest_sha256 === draft.manifestSha) return ''
  if (draft.phase === 'unknown' && pendingCandidateMatches(row, draft)) return ''
  const receipt = record(record(record(row.manifest).training).candidate)
  if (draft.phase !== 'prepared' && row.state === 'candidate_registered') {
    if (candidateReceiptMatches(receipt, draft)) return ''
    // A legacy GET cannot prove the candidate request. It may only enable
    // the frozen original POST, whose response must include the exact receipt.
    if (draft.phase === 'unknown' && receipt.request_identity === undefined && sha256(receipt.operation_id)
      && receipt.requested_revision === draft.body.expected_revision && receipt.stage === 'candidate'
      && text(receipt.artifact_id)) return ''
  }
  return '数据流程或批准版本已变化；不会用新版本替换原请求，请先核对登记状态'
})
function candidatePreparationReason(job: Document): string {
  if (loading.value || lifecycleReadFailed.value) return '请先成功刷新闭环状态，再核对训练候选'
  if (sharedCandidateDraft.value?.origin === 'manual') return '数据流程入口已保留手工候选请求，请在原入口核对；不能改绑为训练产物'
  if (candidateSubmitting.value) return '其它入口或上次会话仍在提交此任务的候选；先等待原请求结束，不重复提交'
  if (candidateDraft.value?.phase === 'unknown') return '原候选登记尚未确认，请先使用保留请求对账'
  if (!candidateDraft.value && candidateLedger.full()) return '本会话候选草稿已达上限，请先清除已确认结果的草稿；未知登记不会自动丢弃'
  const problem = candidateJobError(job)
  if (problem) return problem
  const row = workflows.value.find(item => item.workflow_id === job.workflow_id)!, operation = record(row.operation)
  if (row.state === 'training_export_ready') return ''
  if (row.state === 'candidate_importing' && operation.kind === 'candidate_import'
    && operation.requested_by === username.value && (integer(operation.requested_revision) ?? 0) > 0
    && operation.package_sha256 === job.package_sha256 && sha256(operation.approval_manifest_sha256)
    && record(operation.training_manifest).package_sha256 === job.package_sha256
    && pendingRequestMatches(operation, text(job.workflow_id), text(job.package_sha256), record(record(job.result).candidate))) return ''
  return '原流程不再等待候选登记，或待对账意图不属于本次产物与个人身份；不会覆盖其它候选'
}
function prepareCandidate(job: Document) {
  if (!canManage.value || busy.value || candidateSubmitting.value || loading.value || lifecycleReadFailed.value) return
  if (candidateDraft.value?.phase === 'unknown') {
    candidateDraft.value.error = '上次登记结果未知，请先使用保留的原请求对账，不能换候选重试'
    return
  }
  const problem = candidatePreparationReason(job)
  if (problem) { error.value = problem; return }
  const row = workflows.value.find(item => item.workflow_id === job.workflow_id)!
  const operation = record(row.operation), reconciling = row.state === 'candidate_importing'
  const prepared = candidateLedger.prepare({ taskId: props.task.task_id, cameraId: props.task.camera_id, actor: username.value,
    origin: 'training',
    jobId: text(job.job_id), workflowId: text(job.workflow_id), jobIdentity: candidateJobIdentity(job),
    binding: canonical(job.binding), manifestSha: text(reconciling ? operation.approval_manifest_sha256 : row.manifest_sha256),
    body: { expected_revision: integer(reconciling ? operation.requested_revision : row.revision)!, expected_package_sha256: text(job.package_sha256),
      candidate: JSON.parse(JSON.stringify(record(job.result).candidate)) as Document },
    phase: reconciling ? 'unknown' : 'prepared', error: '', artifactId: '' })
  if (!prepared) error.value = '此任务已有其它入口保留的候选请求或提交，请先核对原请求，不能覆盖'
  candidateConfirmed.value = false
}
async function registerCandidate() {
  const draft = candidateDraft.value
  if (!draft || !candidateConfirmed.value || draft.phase === 'registered' || candidateBlocked.value || busy.value || candidateSubmitting.value) return
  const lease = candidateLedger.acquire(draft)
  if (!lease) return
  const token = generation, id = props.task.task_id, operation = ++operations
  busy.value = true; candidateConfirmed.value = false; draft.phase = 'unknown'; draft.error = ''; notice.value = ''
  // The existing endpoint derives its stable operation ID from this exact
  // workflow/package/candidate body. Never rebase revision after an unknown POST.
  const body = JSON.parse(JSON.stringify(draft.body)) as TaskCandidateImportRequest
  try {
    const result = await importTaskCandidate(draft.taskId, draft.workflowId, body)
    if (!lease.current() || !current(token, id) || operation !== operations || candidateDraft.value !== draft
      || draft.binding !== canonical(taskBinding()) || username.value !== draft.actor) return
    const row = result.document, receipt = record(row.candidate_import_receipt)
    const persisted = record(record(record(row.manifest).training).candidate)
    if (row.task_id !== draft.taskId || row.workflow_id !== draft.workflowId || row.state !== 'candidate_registered'
      || canonical(record(row.manifest).binding) !== draft.binding || !candidateReceiptMatches(receipt, draft)
      || persisted.operation_id !== receipt.operation_id || persisted.artifact_id !== receipt.artifact_id
      || persisted.stage !== 'candidate' || persisted.requested_revision !== draft.body.expected_revision
      || (persisted.request_identity !== undefined && canonical(persisted.request_identity) !== candidateRequestIdentity(draft))) {
      throw new Error('候选登记回执身份尚未确认，请使用原请求对账')
    }
    draft.phase = 'registered'; draft.artifactId = text(receipt.artifact_id)
    notice.value = '训练候选已登记；未冻结执行包、未部署、未发布，后续仍需旁路评估和独立审批'
    emit('refresh'); await refresh()
  } catch (cause: unknown) {
    if (lease.current() && current(token, id) && operation === operations && candidateDraft.value === draft) {
      draft.error = extractErrorMessage(cause, '候选登记结果未确认')
    }
  } finally {
    lease.release()
    if (current(token, id)) {
      if (operation === operations) busy.value = false
    }
  }
}
function clearCandidateDraft() {
  if (busy.value || candidateSubmitting.value || candidateDraft.value?.phase === 'unknown') return
  if (candidateDraft.value) candidateLedger.clear(candidateDraft.value)
  candidateConfirmed.value = false
}
const event = computed(() => events.value.find(row => text(row.event_id) === eventId.value))
const eventSource = computed(() => record(replay.value?.source ?? record(event.value?.payload).source))
const release = computed(() => releases.value.find(row => text(row.release_id) === releaseId.value))
const workflow = computed(() => workflows.value.find(row => text(row.workflow_id) === workflowId.value))
const trustedBaseModels = computed(() => trainingCapabilities.value?.trusted_base_models ?? [])
const baseModel = computed(() => trustedBaseModels.value.find(row => row.model_id === baseModelId.value))
const trustedTrainingReady = computed(() => !!trainingCapabilities.value && !loading.value && !capabilitiesLoading.value
  && !capabilitiesError.value && !!baseModel.value)
const eventStale = computed(() => !!event.value && eventRevision.value !== integer(event.value.revision))
const releaseStale = computed(() => !!release.value && (releaseRevision.value !== integer(release.value.revision)
  || releaseSubject.value !== text(release.value.subject_sha256)))
const workflowStale = computed(() => !!workflow.value && workflowRevision.value !== integer(workflow.value.revision))
const evidenceAuthorsKnown = computed(() => {
  const authors = release.value?.evidence_authors
  return Array.isArray(authors) && authors.length > 0
    && authors.every(author => typeof author === 'string' && !!author.trim())
    && new Set(authors).size === authors.length && authors.includes(text(release.value?.created_by))
})
const selfReview = computed(() => !!release.value && (text(release.value.created_by) === username.value
  || (Array.isArray(release.value.evidence_authors) && release.value.evidence_authors.includes(username.value))))
const reviewable = computed(() => !!release.value && ['requested', 'rejected'].includes(text(release.value.state)))
const imageKey = computed(() => JSON.stringify([props.task.task_id, eventId.value, replay.value?.image_sha256]))
const states: Record<string, string> = { new: '待复核', acknowledged: '已确认', false_positive: '误报', resolved: '已解决',
  requested: '待档案复核', review_approved: '档案复核通过（非生产批准）', rejected: '已拒绝', revoked: '已撤销', stale: '已失效，需重新核对',
  queued: '已排队，尚未开始', waiting_resources: '资源阻塞，尚未开始', running: '执行中', cancelling: '取消中，等待进程清理确认',
  cancelled: '已取消', failed: '失败', complete: '完成', completed: '完成', reconciling: '等待对账，尚未确认旧进程退出',
  disabled: '未启用', ready: '就绪', waiting_maintenance: '等待维护窗口批准', blocked: '资源阻塞', leased: '资源已占用',
  starting: '启动中', stopping: '停止中', stopped: '已停止', scheduled: '调度器已启动', idle: '空闲', error: '故障' }
const runtimeStates: Record<string, string> = { disabled: '未启用', not_started: '尚未启动',
  running: '调度器运行中（不等于训练已开始）', waiting_resources: '等待资源，尚未开始训练',
  reconciling: '等待旧进程对账，队列阻塞', owner_unavailable: '调度器所有权不可用',
  stopping: '调度器停止中', stopped: '调度器已停止', cleanup_unconfirmed: '进程清理未确认，资源未释放', error: '调度器故障' }
const reasonLabels: Record<string, string> = {
  TEMPLATE_POLICY_NOT_APPROVED: '任务模板的正式生产安全契约尚未批准',
  PRODUCTION_APPLY_UNSUPPORTED: '任务生产切换协议尚未启用',
  TRUSTED_TASK_EVIDENCE_VERIFIER_UNAVAILABLE: '尚无可授权生产的可信任务证据验证器',
  RELEASE_DOSSIER_REQUIRED: '请先选择并复核发布档案', INDEPENDENT_RELEASE_REVIEW_REQUIRED: '需要独立个人身份复核',
  RELEASE_EVIDENCE_INCOMPLETE: '评估、校准、基线、语料及可信长稳证据尚不完整',
  RELEASE_BINDING_CHANGED: '任务、模型、配置或部署绑定已变化',
  EVIDENCE_SOURCE_IDENTITY_MISMATCH: '报告或证据文件摘要已变化', RELEASE_REVOKED: '发布申请已撤销',
  EVIDENCE_AUTHOR_IDENTITY_UNAVAILABLE: '历史证据提交身份未知，须重新提交当前证据后由独立个人复核',
  TRAINING_RESOURCE_APPROVAL_REQUIRED: '缺少维护窗口或独占 GPU 资源批准，尚未开始执行',
  TRAINING_MAINTENANCE_APPROVAL_REQUIRED: '等待维护窗口批准；需要修改配置并重启，页面不会启用调度器',
  TRAINING_RUNTIME_DISABLED: '训练调度器未启用',
  TRAINING_RESOURCE_UNAVAILABLE: '授权 GPU 资源不可用',
  TRAINING_RESOURCE_BUSY: '授权 GPU 资源正在使用',
  TRAINING_MEMORY_INSUFFICIENT: 'GPU 可用显存不足',
  TRAINING_RUNTIME_OWNER_UNAVAILABLE: '训练调度器所有权不可用',
  TRAINING_OWNER_LIVENESS_UNKNOWN: '旧训练进程的存活状态尚未核实',
  TRAINING_RECONCILIATION_REQUIRED: '必须先确认旧训练进程退出，当前队列不会继续启动',
  TRAINING_DISABLED: '启动配置未启用训练，队列不会执行',
  TRAINING_MAINTENANCE_REQUIRED: '等待显式维护批准；需修改启动配置并重启',
  TRAINING_RESOURCE_IN_USE: '当前训练资源仍被占用，等待正向清理确认',
  TRAINING_ONLINE_COMPUTE_ACTIVE: '在线模型尚未退出或另一 Argus 实例持有资源；不会自动停止检测',
  TRAINING_LEGACY_JOB_ACTIVE: '旧训练作业仍在执行，不能并行准入',
  TRAINING_EXTERNAL_COMPUTE_ACTIVE: '检测到其他 GPU 计算进程，任务继续排队',
  TRAINING_INSUFFICIENT_GPU_MEMORY: '最近观测的空闲显存低于准入阈值',
  TRAINING_RESOURCE_OBSERVATION_UNAVAILABLE: '无法核实 GPU 设备、进程或显存，不能启动训练',
  TRAINING_RESOURCE_ADMISSION_UNAVAILABLE: '资源准入检查失败，未启动训练',
  TRAINING_OWNER_RECONCILIATION_REQUIRED: '旧执行尚未对账，队列继续阻塞',
  TRAINING_RUNTIME_CLEANUP_UNCONFIRMED: '清理结果未确认，保留所有权与资源锁',
}
function reason(value: unknown) { const code = text(value); return reasonLabels[code] ?? code }
function capabilityTrust(value: TaskTrainingCapabilitiesResponse) {
  return JSON.stringify([value.schema, value.recipe_id, value.validation_only, value.production_allowed,
    [...value.trusted_base_models].sort((left, right) => left.model_id.localeCompare(right.model_id))])
}
function validCapabilities(value: TaskTrainingCapabilitiesResponse): boolean {
  if (value?.schema !== 'argus.task-training-capabilities.v1' || value.recipe_id !== 'yolov8_detection.v1'
    || value.validation_only !== true || value.production_allowed !== false
    || typeof value.configured_enabled !== 'boolean' || typeof value.maintenance_approved !== 'boolean'
    || value.resource_mode !== 'exclusive_maintenance' || value.device !== 'cuda:0'
    || !Array.isArray(value.trusted_base_models) || !value.runtime || !value.resource
    || typeof value.runtime.state !== 'string' || typeof value.runtime.enabled !== 'boolean'
    || !['disabled', 'waiting_maintenance', 'blocked', 'ready', 'leased'].includes(value.resource.state)) return false
  return value.trusted_base_models.every(row => row.configured_trust === true
    && typeof row.model_id === 'string' && !!row.model_id && typeof row.name === 'string'
    && typeof row.relative_path === 'string' && !!row.relative_path
    && typeof row.artifact_sha256 === 'string' && /^[a-f0-9]{64}$/.test(row.artifact_sha256))
    && new Set(value.trusted_base_models.map(row => row.model_id)).size === value.trusted_base_models.length
}
function invalidateTrainingTrust() { baseModelId.value = ''; trainingConfirmed.value = false }
async function readTrainingCapabilities() {
  if (!canManage.value) return
  capabilitiesRequested.value = true
  const token = generation, id = props.task.task_id, read = ++capabilitiesReads
  capabilitiesLoading.value = true; capabilitiesError.value = ''
  try {
    const result = await getTaskTrainingCapabilities(id)
    if (!current(token, id) || read !== capabilitiesReads) return
    if (!validCapabilities(result)) throw new Error('训练配置响应无效；不能确认基础模型的服务端可信来源')
    if (!trainingCapabilities.value || capabilityTrust(trainingCapabilities.value) !== capabilityTrust(result)) invalidateTrainingTrust()
    trainingCapabilities.value = result
  } catch (cause: unknown) {
    if (current(token, id) && read === capabilitiesReads) {
      trainingCapabilities.value = null; invalidateTrainingTrust()
      capabilitiesError.value = extractErrorMessage(cause, '无法读取可信训练配置；不能提交训练申请')
    }
  } finally {
    if (current(token, id) && read === capabilitiesReads) capabilitiesLoading.value = false
  }
}
function key(action: string, payload: unknown): string {
  const identity = JSON.stringify([props.task.task_id, action, payload])
  if (!keys.has(identity)) keys.set(identity, crypto.randomUUID())
  return keys.get(identity)!
}
function current(token: number, id: string) { return !disposed && token === generation && id === props.task.task_id }
function stopEventRefresh() { if (eventRefreshTimer) clearTimeout(eventRefreshTimer); eventRefreshTimer = null }
function scheduleEventRefresh() {
  if (disposed || eventRefreshTimer !== null) return
  eventRefreshTimer = setTimeout(() => {
    eventRefreshTimer = null
    if (disposed) return
    if (busy.value || loading.value) { scheduleEventRefresh(); return }
    void refresh()
  }, 100)
}
async function readReadiness() {
  if (!canManage.value) return
  const token = generation, id = props.task.task_id, selected = releaseId.value, read = ++readinessReads
  try {
    const result = await getTaskLifecycleReadiness(id, selected || undefined)
    if (current(token, id) && read === readinessReads && selected === releaseId.value) readiness.value = result.document
  } catch (cause: unknown) {
    if (current(token, id) && read === readinessReads && selected === releaseId.value) {
      readiness.value = {}; error.value = extractErrorMessage(cause, '无法读取发布安全门')
    }
  }
}
async function refresh() {
  const token = generation, id = props.task.task_id, read = ++reads, operation = operations
  loading.value = true
  // Revalidate trust as soon as a refresh starts, independently of other reads.
  // A failed or superseded lifecycle read must not skip this requested check.
  const capabilitiesRead = capabilitiesRequested.value ? readTrainingCapabilities() : Promise.resolve()
  try {
    const [eventRows, releaseRows, jobRows, workflowRows] = await Promise.all([
      getTaskEvents(id),
      canManage.value ? getTaskReleaseDossiers(id) : Promise.resolve({ documents: [] }),
      canManage.value ? getTaskTrainingExecutions(id) : Promise.resolve({ documents: [] }),
      canManage.value ? getTaskWorkflows(id) : Promise.resolve({ documents: [] }),
    ])
    if (!current(token, id) || read !== reads || operation !== operations) return
    events.value = eventRows.documents; releases.value = releaseRows.documents
    jobs.value = jobRows.documents; workflows.value = workflowRows.documents
    loaded.value = true; error.value = ''; lifecycleReadFailed.value = false
    // A background refresh must not rebase an old human form or notes.
    // Recheck evidence even when the selected event and its image hash are unchanged.
    // Start both reads before awaiting so a late refresh cannot target a new selection.
    await Promise.all([readReadiness(), readReplay(), capabilitiesRead])
  } catch (cause: unknown) {
    if (current(token, id) && read === reads && operation === operations) {
      lifecycleReadFailed.value = true
      error.value = extractErrorMessage(cause, '无法读取任务闭环状态')
    }
  } finally {
    if (current(token, id) && read === reads) {
      loading.value = false
    }
  }
}
async function act(action: () => Promise<unknown>, message: string, reload: 'event' | 'release' | null = null) {
  if (busy.value) return
  const token = generation, id = props.task.task_id, operation = ++operations, form = formGeneration
  busy.value = true; error.value = ''; notice.value = ''
  try {
    await action()
    if (!current(token, id) || operation !== operations || form !== formGeneration) return
    notice.value = message; emit('refresh')
    await refresh()
    if (!current(token, id) || operation !== operations || form !== formGeneration) return
    if (reload === 'event') loadEventForm()
    if (reload === 'release') loadReleaseForm()
  } catch (cause: unknown) {
    if (current(token, id) && operation === operations && form === formGeneration) error.value = extractErrorMessage(cause, '操作失败；草稿和请求身份已保留，请先核对当前版本')
  } finally { if (current(token, id) && operation === operations && form === formGeneration) busy.value = false }
}
function loadEventForm() { eventRevision.value = integer(event.value?.revision); eventNotes.value = text(event.value?.notes) }
function loadReleaseForm() {
  releaseConfirmed.value = false
  releaseRevision.value = integer(release.value?.revision); releaseSubject.value = text(release.value?.subject_sha256)
  releaseNotes.value = text(record(release.value?.review).notes)
}
async function readReplay() {
  const token = generation, id = props.task.task_id, selected = eventId.value, read = ++previewReads
  replay.value = null; imageFailed.value = false
  if (!selected) return
  try {
    const result = await getTaskEventReplay(id, selected)
    if (current(token, id) && read === previewReads && selected === eventId.value) replay.value = result.document
  } catch (cause: unknown) {
    if (current(token, id) && read === previewReads && selected === eventId.value) error.value = extractErrorMessage(cause, '事件关键帧不可用')
  }
}
function evidenceFailed(domEvent: Event) {
  if ((domEvent.target as HTMLImageElement).dataset.evidenceKey === imageKey.value) imageFailed.value = true
}
async function reviewEvent(status: 'acknowledged' | 'false_positive' | 'resolved') {
  if (!canReviewEvents.value || !event.value || eventStale.value || eventRevision.value === null) return
  const selected = eventId.value
  await act(() => reviewTaskEvent(props.task.task_id, selected,
    { expected_revision: eventRevision.value!, status, notes: eventNotes.value }), '事件复核已保存', 'event')
}
function evidenceSources(): NonNullable<TaskReleaseRequest['evidence_sources']> {
  const data: unknown = JSON.parse(evidenceText.value)
  if (!Array.isArray(data) || data.length > 6) throw new Error('证据应为最多六项的 JSON 数组')
  const roles = ['evaluation', 'calibration', 'baseline', 'corpus', 'soak', 'soak_telemetry'] as const
  return data.map((value: unknown) => {
    const item = record(value), role = roles.find(candidate => candidate === item.role)
    if (!role || typeof item.relative_path !== 'string' || typeof item.sha256 !== 'string'
      || !/^[a-f0-9]{64}$/.test(item.sha256) || Object.keys(item).some(name => !['role', 'relative_path', 'sha256'].includes(name))) {
      throw new Error('每项证据需要声明的 role、受控相对路径及 SHA256；不能提交生产放行标记')
    }
    return { role, relative_path: item.relative_path, sha256: item.sha256 }
  }).sort((left, right) => left.role.localeCompare(right.role))
}
async function createRelease() {
  if (!canManage.value || !props.bundleId) return
  await act(() => {
    const revision = integer(props.task.deployment?.deployment_revision)
    if (revision === null) throw new Error('当前部署版本尚未核实')
    const body = { bundle_id: props.bundleId!, expected_task_revision: props.task.current_revision,
      expected_deployment_revision: revision, evidence_sources: evidenceSources() }
    return requestTaskRelease(props.task.task_id, body, key('release-create', body))
  }, '发布档案已登记；不代表生产批准或已应用')
}
async function reviewRelease(decision: 'approve' | 'reject') {
  if (!canManage.value || !release.value || !reviewable.value || !evidenceAuthorsKnown.value
    || releaseStale.value || selfReview.value || releaseRevision.value === null) return
  if (decision === 'approve' && !releaseConfirmed.value) return
  const selected = releaseId.value
  await act(() => {
    const body = { expected_revision: releaseRevision.value!, expected_subject_sha256: releaseSubject.value,
      decision, notes: releaseNotes.value }
    return reviewTaskRelease(props.task.task_id, selected, body, key(`release-review:${selected}`, body))
  }, '档案复核已记录；生产仍须独立可信契约和全部安全门', 'release')
}
async function updateEvidence() {
  if (!canManage.value || !release.value || release.value.state === 'revoked' || releaseStale.value || releaseRevision.value === null) return
  await act(() => {
    const body = { expected_revision: releaseRevision.value!, evidence_sources: evidenceSources() }
    return reviseTaskReleaseEvidence(props.task.task_id, releaseId.value, body, key(`release-evidence:${releaseId.value}`, body))
  }, '证据已更新；旧复核结论失效，历史备注和当前草稿保留，请重新核对复核表单')
}
async function revoke() {
  if (!canManage.value || !release.value || release.value.state === 'revoked' || releaseStale.value || releaseRevision.value === null) return
  await act(() => {
    if (!releaseNotes.value.trim()) throw new Error('撤销需要填写原因')
    const body = { expected_revision: releaseRevision.value!, notes: releaseNotes.value }
    return revokeTaskRelease(props.task.task_id, releaseId.value, body, key(`release-revoke:${releaseId.value}`, body))
  }, '发布申请已撤销；未变更当前旁路部署', 'release')
}
async function submitTraining() {
  if (!canManage.value || !workflow.value || workflowStale.value || !trainingConfirmed.value
    || workflowRevision.value === null || !trustedTrainingReady.value || !baseModel.value) return
  const selectedBase = baseModel.value
  await act(() => {
    const body = { expected_revision: workflowRevision.value!, expected_package_sha256: packageSha.value.trim(),
      base_model_path: selectedBase.relative_path, base_model_sha256: selectedBase.artifact_sha256,
      params: { epochs: epochs.value, imgsz: imageSize.value, batch: batch.value, seed: seed.value } }
    return submitTaskTrainingExecution(props.task.task_id, workflowId.value, body, key(`training:${workflowId.value}`, body))
  }, '训练申请已登记到队列；排队不等于已执行，未发布候选模型')
}
async function exportAnnotations(target: 'cvat' | 'fiftyone') {
  if (!canManage.value || !workflowId.value) return
  await act(async () => {
    const token = generation, id = props.task.task_id, selected = workflowId.value
    const result = await exportTaskAnnotationsV2(id, selected, target)
    if (current(token, id) && selected === workflowId.value) annotationText.value = JSON.stringify(result.document, null, 2)
  }, 'v2 标注交换清单已生成；未上传外部服务')
}
async function importAnnotations() {
  if (!canManage.value || !workflow.value || workflowStale.value || workflowRevision.value === null) return
  await act(() => {
    const payload: unknown = JSON.parse(annotationText.value)
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('标注回传必须是 JSON 对象')
    return importTaskAnnotationsV2(props.task.task_id, workflowId.value,
      { expected_revision: workflowRevision.value!, payload: payload as Document })
  }, '标注回传已核验；仍需逐图复核，旧数据集和训练批准不会自动沿用')
}
function changeForm() { formGeneration += 1; busy.value = false; error.value = ''; notice.value = '' }
watch(eventId, () => { changeForm(); loadEventForm(); void readReplay() })
watch(releaseId, () => { changeForm(); loadReleaseForm(); void readReadiness() })
watch(workflowId, () => {
  changeForm()
  workflowRevision.value = integer(workflow.value?.revision); annotationText.value = '{}'
  packageSha.value = ''; trainingConfirmed.value = false
})
watch(baseModelId, () => { trainingConfirmed.value = false })
watch([packageSha, epochs, imageSize, batch, seed], () => { trainingConfirmed.value = false })
watch(() => integer(workflow.value?.revision), () => { trainingConfirmed.value = false })
watch(() => canonical([taskBinding(), candidateDraft.value?.phase, candidateDraft.value?.body,
  jobs.value.find(row => row.job_id === candidateDraft.value?.jobId),
  workflows.value.find(row => row.workflow_id === candidateDraft.value?.workflowId)]), () => {
  candidateConfirmed.value = false
})
useWebSocket({
  topics: ['task_events'],
  onMessage(topic, data: unknown) {
    const payload = record(data)
    if (topic === 'task_events' && payload.task_id === props.task.task_id
      && (payload.camera_id === undefined || payload.camera_id === props.task.camera_id)) scheduleEventRefresh()
  },
  fallbackPoll: () => {
    if (busy.value || loading.value) { scheduleEventRefresh(); return }
    return refresh()
  },
  fallbackInterval: 60_000,
  reconcileInterval: 60_000,
})
watch([() => props.task.task_id, () => props.task.camera_id, username, role], () => {
  candidateConfirmed.value = false
  generation += 1; operations += 1; reads += 1; previewReads += 1; readinessReads += 1; capabilitiesReads += 1
  stopEventRefresh(); keys.clear(); eventId.value = ''; releaseId.value = ''; workflowId.value = ''
  events.value = []; releases.value = []; jobs.value = []; workflows.value = []; readiness.value = {}; replay.value = null
  eventNotes.value = ''; releaseNotes.value = ''; evidenceText.value = '[]'; annotationText.value = '{}'
  eventRevision.value = null; releaseRevision.value = null; releaseSubject.value = ''; workflowRevision.value = null
  packageSha.value = ''; invalidateTrainingTrust()
  trainingCapabilities.value = null; capabilitiesRequested.value = false; capabilitiesLoading.value = false; capabilitiesError.value = ''
  loading.value = false; busy.value = false; loaded.value = false; lifecycleReadFailed.value = false; error.value = ''; notice.value = ''
  void refresh()
}, { immediate: true, flush: 'sync' })
onUnmounted(() => { disposed = true; generation += 1; stopEventRefresh() })
</script>

<template>
  <section aria-label="目标检测事件与发布闭环" class="task-lifecycle" :aria-busy="busy || loading || capabilitiesLoading || candidateSubmitting">
    <h4>目标检测事件 → 复核 → 发布档案</h4>
    <p class="note">当前保存的是独立验证事件，不混入生产告警。档案复核不是生产许可；生产契约与可信证据未完成时始终 NO-GO。</p>
    <p v-if="error" role="alert" class="error">{{ error }}<span v-if="loaded">；保留的上次读取结果可能过期，草稿未被覆盖。</span></p><p v-if="notice" role="status">{{ notice }}</p>
    <button :disabled="busy || loading" @click="refresh">刷新闭环状态</button>
    <p v-if="loading && !loaded" role="status">正在读取事件、申请与执行队列…</p>
    <p v-else-if="loaded && !error && !events.length">暂无验证事件；这不代表模型已经识别正常或生产告警已启用。</p>
    <label>验证事件<select v-model="eventId" data-test="lifecycle-event" :disabled="busy"><option value="">请选择事件</option><option v-for="row in events" :key="text(row.event_id)" :value="text(row.event_id)">{{ states[text(row.status)] || row.status }} · {{ row.occurred_at }} · {{ text(row.event_id).slice(0, 12) }}</option></select></label>
    <div v-if="event">
      <p>事件模式：{{ event.mode }} · 部署 r{{ event.deployment_revision }} · 采集序号 {{ event.frame_index ?? '未知' }} · 复核 r{{ event.revision }}</p>
      <p>原始源帧号：{{ eventSource.source_frame_index ?? '未知（源未提供）' }} · PTS：{{ eventSource.pts ?? '未知' }}</p>
      <p v-if="eventStale" role="status">事件版本已更新，备注草稿保留；请先重新载入，不能用旧版本提交。</p>
      <button v-if="eventStale" :disabled="busy" @click="loadEventForm">重新载入事件复核</button>
      <p v-if="replay">证据：仅关键帧，未提供连续视频回放。</p>
      <img v-if="replay?.ready === true" :key="imageKey" :data-evidence-key="imageKey" :src="taskEventEvidenceUrl(task.task_id, eventId)" alt="事件原始身份对应的验证关键帧" @error="evidenceFailed" />
      <p v-if="replay?.ready === false || imageFailed" role="alert">关键帧不可用或摘要核验失败；不能把缺失证据当成已验证。</p>
      <details><summary>源帧与事件身份</summary><pre>{{ JSON.stringify(replay?.source ?? event.payload, null, 2) }}</pre></details>
      <label>事件复核备注<textarea v-model="eventNotes" maxlength="4000" :disabled="busy || !canReviewEvents" /></label>
      <button v-for="status in (['acknowledged', 'false_positive', 'resolved'] as const)" :key="status" :disabled="busy || !canReviewEvents || eventStale" @click="reviewEvent(status)">{{ states[status] }}</button>
    </div>
    <template v-if="canManage">
      <h4>发布申请与独立人审</h4>
      <p role="status">生产发布：NO-GO</p>
      <ul v-if="Array.isArray(readiness.reasons)"><li v-for="item in readiness.reasons" :key="text(item)">{{ reason(item) }}</li></ul>
      <label>受控证据清单 JSON（role、relative_path、sha256）<textarea v-model="evidenceText" rows="5" :disabled="busy" aria-label="发布证据清单" /></label>
      <button data-test="request-release" :disabled="busy || !bundleId" @click="createRelease">为选中执行包登记发布档案</button>
      <label>发布档案<select v-model="releaseId" data-test="lifecycle-release" :disabled="busy"><option value="">请选择需要复核的档案</option><option v-for="row in releases" :key="text(row.release_id)" :value="text(row.release_id)">{{ states[text(row.state)] || row.state }} · r{{ row.revision }} · {{ text(row.release_id).slice(0, 12) }}</option></select></label>
      <div v-if="release">
        <p>申请人：{{ release.created_by }} · 当前身份：{{ username }} · 复核范围：发布档案，不是生产放行。</p>
        <p v-if="record(release.revocation).notes">撤销原因：{{ record(release.revocation).notes }}</p>
        <p v-if="text(release.created_by) === username" role="status">申请人不能复核自己的申请，请另一位个人账号复核。</p>
        <p v-else-if="selfReview" role="status">证据提交者不能复核自己参与的档案，请未参与证据提交的另一位个人账号复核。</p>
        <p v-if="!evidenceAuthorsKnown" role="status">证据提交身份未知；请重新提交当前证据，再由独立个人核对，旧备注保留。</p>
        <p v-if="releaseStale" role="status">档案已更新，备注草稿保留；请重新载入后再复核。</p>
        <button v-if="releaseStale" :disabled="busy" @click="loadReleaseForm">重新载入发布复核</button>
        <details><summary>待复核的冻结配置、模型与证据摘要</summary><pre>{{ JSON.stringify({ binding: release.binding, evidence_sources: release.evidence_sources, subject_sha256: release.subject_sha256 }, null, 2) }}</pre></details>
        <label>发布复核 / 撤销原因<textarea v-model="releaseNotes" maxlength="4000" :disabled="busy" aria-label="发布复核备注" /></label>
        <label><input v-model="releaseConfirmed" data-test="confirm-release-review" type="checkbox" :disabled="busy || !reviewable || !evidenceAuthorsKnown || releaseStale || selfReview" />我已核对当前冻结配置、模型和证据摘要；这不是生产批准</label>
        <button data-test="approve-release-review" :disabled="busy || !reviewable || !evidenceAuthorsKnown || releaseStale || selfReview || !releaseConfirmed" @click="reviewRelease('approve')">独立复核档案</button>
        <button :disabled="busy || !reviewable || !evidenceAuthorsKnown || releaseStale || selfReview" @click="reviewRelease('reject')">拒绝申请</button>
        <button :disabled="busy || release.state === 'revoked' || releaseStale" @click="updateEvidence">更新证据并使旧结论失效</button>
        <button :disabled="busy || release.state === 'revoked' || releaseStale || !releaseNotes.trim()" @click="revoke">确认撤销此发布申请</button>
        <details><summary>保留的历史复核与备注</summary><pre>{{ JSON.stringify(release.review_history, null, 2) }}</pre></details>
      </div>
      <h4>标注回传与受控训练执行</h4>
      <p class="note">先在数据流程完成人审、数据集冻结和训练批准。队列登记不等于训练开始；只有授权 GPU 测试机显式配置调度器后才能执行，不自动下载模型或回退 CPU。</p>
      <button data-test="read-training-capabilities" :disabled="busy || capabilitiesLoading" @click="readTrainingCapabilities">{{ capabilitiesRequested ? '重新读取可信训练配置与调度器状态' : '读取可信训练配置与调度器状态' }}</button>
      <p v-if="capabilitiesLoading" role="status">正在核对服务端可信基础模型与调度器只读状态；核对期间不能提交。</p>
      <p v-if="capabilitiesError" data-test="training-capabilities-error" role="alert" class="error">{{ capabilitiesError }}；可信选择和训练确认已失效，请重新读取。</p>
      <div v-if="trainingCapabilities" data-test="training-capabilities">
        <p>执行范围：仅验证环境 · 禁止生产发布。服务端可信目录只表示已配置来源，不表示模型质量已通过或生产已批准。</p>
        <p>调度器配置：{{ trainingCapabilities.configured_enabled ? '已配置启用' : '已配置禁用' }} · 运行状态：{{ runtimeStates[trainingCapabilities.runtime.state] || trainingCapabilities.runtime.state }}{{ trainingCapabilities.runtime.enabled ? '' : '（未启用）' }}</p>
        <p v-if="!trainingCapabilities.maintenance_approved" role="status">等待维护窗口批准；需要修改配置并重启，页面不会启用调度器。</p>
        <p>资源状态：{{ states[trainingCapabilities.resource.state] || trainingCapabilities.resource.state }} · 设备 {{ trainingCapabilities.device }}<span v-if="typeof trainingCapabilities.resource.free_memory_mb === 'number'"> · 最近观测可用显存 {{ trainingCapabilities.resource.free_memory_mb }} MB</span></p>
        <p v-if="trainingCapabilities.runtime.error_code">调度器原因：{{ reason(trainingCapabilities.runtime.error_code) }}</p>
        <p v-if="trainingCapabilities.resource.reason_code">资源原因：{{ reason(trainingCapabilities.resource.reason_code) }}</p>
        <p class="note">这是缓存的只读观测，不是 GPU 预留或启动授权。配置禁用、维护未批准或资源阻塞时仍可登记已批准申请，但不会因此开始执行。</p>
        <p v-if="!trustedBaseModels.length" role="status">未配置可信基础模型；不能用手工路径或自填 SHA256 代替来源批准。</p>
      </div>
      <label>已建立的数据流程<select v-model="workflowId" data-test="lifecycle-workflow" :disabled="busy"><option value="">请选择流程</option><option v-for="row in workflows" :key="text(row.workflow_id)" :value="text(row.workflow_id)">{{ row.state }} · r{{ row.revision }} · {{ text(row.workflow_id).slice(0, 12) }}</option></select></label>
      <template v-if="workflow">
        <p v-if="workflowStale" role="status">流程版本变化，回传和训练草稿保留；先核对当前版本。</p>
        <button v-if="workflowStale" :disabled="busy" @click="workflowRevision = integer(workflow.revision)">核对后载入当前流程版本</button>
        <details><summary>CVAT / FiftyOne 标注交换 v2</summary>
          <button v-for="target in (['cvat', 'fiftyone'] as const)" :key="target" :disabled="busy" @click="exportAnnotations(target)">导出 {{ target }} v2</button>
          <label>标注回传 v2 JSON<textarea v-model="annotationText" rows="6" aria-label="标注回传 v2 JSON" :disabled="busy" /></label>
          <button :disabled="busy || workflowStale" @click="importAnnotations">核验并导入 v2 标注</button>
        </details>
        <details><summary>提交检测训练到受控队列</summary>
          <label>已批准训练交接 SHA256<input v-model="packageSha" data-test="training-package-sha" :disabled="busy" /></label>
          <p v-if="!capabilitiesRequested" role="status">请先读取服务端可信训练配置，再选择基础模型；自填摘要不能证明模型来源。</p>
          <label>服务端配置的可信基础模型<select v-model="baseModelId" data-test="training-base-model" :disabled="busy || capabilitiesLoading || !trainingCapabilities"><option value="">请选择已配置的基础模型</option><option v-for="model in trustedBaseModels" :key="model.model_id" :value="model.model_id">{{ model.name }} · {{ model.model_id }}</option></select></label>
          <p v-if="baseModel" data-test="training-base-identity">受控相对路径：{{ baseModel.relative_path }} · 服务端配置 SHA256：{{ baseModel.artifact_sha256 }}</p>
          <div class="grid"><label>训练 epochs<input v-model.number="epochs" :disabled="busy" type="number" min="1" max="1000" /></label><label>图像尺寸<input v-model.number="imageSize" :disabled="busy" type="number" min="32" max="2048" step="32" /></label><label>batch<input v-model.number="batch" :disabled="busy" type="number" min="1" max="64" /></label><label>随机种子<input v-model.number="seed" :disabled="busy" type="number" min="0" /></label></div>
          <label><input v-model="trainingConfirmed" data-test="confirm-training" type="checkbox" :disabled="busy || !trustedTrainingReady || workflowStale" />确认仅提交已批准的数据与当前服务端可信模型到授权测试环境队列，不代表生产批准</label>
          <button data-test="submit-training" :disabled="busy || workflowStale || !trainingConfirmed || !packageSha || !trustedTrainingReady" @click="submitTraining">确认提交训练申请</button>
        </details>
      </template>
      <p v-if="loaded && !error && !jobs.length">暂无训练执行记录；不会因为打开页面而启动训练。</p>
      <ul><li v-for="job in jobs" :key="text(job.job_id)" data-test="training-job">{{ states[text(job.status)] || job.status }} · {{ text(job.job_id).slice(0, 14) }}
        <span v-if="integer(job.queue_position) !== null && (integer(job.queue_position) ?? 0) > 0"> · 队列位置 {{ job.queue_position }}</span>
        <p v-if="record(job.result).error_code">原因：{{ reason(record(job.result).error_code) }}</p>
        <p v-if="job.status === 'reconciling'" class="note">旧进程退出或清理尚未确认，队列保持阻塞；可登记取消并重新读取状态，页面不会绕过对账或释放资源。</p>
        <button v-if="['queued', 'waiting_resources', 'running', 'cancelling', 'reconciling'].includes(text(job.status))" :disabled="busy || integer(job.revision) === null" @click="act(() => cancelTaskTrainingExecution(task.task_id, text(job.job_id), { expected_revision: integer(job.revision) ?? 0 }), '已登记取消，仍需等待清理确认')">{{ job.status === 'cancelling' ? '重新登记取消意图' : '取消此训练申请' }}</button>
        <template v-if="job.status === 'complete'">
          <p v-if="candidatePreparationReason(job)" class="note" data-test="candidate-preparation-reason">{{ candidatePreparationReason(job) }}</p>
          <button data-test="prepare-training-candidate" :disabled="busy || candidateSubmitting || !!candidatePreparationReason(job)" @click="prepareCandidate(job)">核对这次训练产物并登记候选</button>
        </template>
      </li></ul>
      <section v-if="candidateDraft" data-test="training-candidate-draft" aria-label="训练候选登记核对">
        <h4>训练完成 → 候选模型登记</h4>
        <p>任务 {{ candidateDraft.taskId }} · 数据流程 {{ candidateDraft.workflowId }} · 原流程 r{{ candidateDraft.body.expected_revision }} · 训练 {{ candidateDraft.jobId }}</p>
        <p>冻结交接 SHA256：{{ candidateDraft.body.expected_package_sha256 }}</p>
        <p>候选制品：{{ candidateDraft.body.candidate.relative_path }} · SHA256：{{ candidateDraft.body.candidate.artifact_sha256 }}</p>
        <details><summary>只读冻结请求与模型描述</summary><pre>{{ JSON.stringify(candidateDraft.body, null, 2) }}</pre></details>
        <p class="note">只登记 candidate，不重新训练、不冻结执行包、不部署或发布；后续旁路评估与独立审批不变。原标注、训练与复核草稿不会被此操作覆盖。</p>
        <p v-if="candidateDraft.phase === 'unknown'" role="status">登记提交中或结果未知；保留原流程版本、交接和制品身份，只能使用同一请求对账，不换候选重试。</p>
        <p v-if="candidateBlocked" data-test="candidate-blocked" role="status">{{ candidateBlocked }}</p>
        <p v-if="candidateDraft.error" data-test="candidate-error" role="alert">{{ candidateDraft.error }}；原登记请求已保留。</p>
        <p v-if="candidateDraft.phase === 'registered'" data-test="candidate-registered" role="status">候选已登记：{{ candidateDraft.artifactId }}；尚未部署或生产发布。</p>
        <template v-else>
          <label><input v-model="candidateConfirmed" data-test="confirm-training-candidate" type="checkbox" :disabled="busy || candidateSubmitting || !!candidateBlocked" />我已核对这次训练、原数据批准、交接及制品摘要，仅批准登记为候选</label>
          <button data-test="register-training-candidate" :disabled="busy || candidateSubmitting || !!candidateBlocked || !candidateConfirmed" @click="registerCandidate">{{ candidateDraft.phase === 'unknown' ? '使用原请求核对候选登记' : '确认登记此训练候选' }}</button>
        </template>
        <button :disabled="busy || candidateSubmitting || candidateDraft.phase === 'unknown'" @click="clearCandidateDraft">清除此核对草稿</button>
      </section>
    </template>
    <p v-else class="note">模型审批与训练管理需要个人工程师 / 管理员身份；服务端会独立校验权限。</p>
  </section>
</template>

<style scoped>
.task-lifecycle { margin: 12px 0; border-top: 1px solid var(--border-default); padding-top: 12px; }
h4 { margin: 12px 0 8px; } .note { color: var(--text-secondary); font-size: 12px; line-height: 1.6; }
label { display: grid; gap: 5px; margin: 10px 0; font-size: 12px; }
input, select, textarea, button { padding: 8px; color: var(--text-primary); background: var(--bg); border: 1px solid var(--border-default); border-radius: 6px; }
textarea { width: 100%; box-sizing: border-box; } button { margin: 4px 6px 4px 0; cursor: pointer; }
button:disabled { opacity: .6; cursor: not-allowed; } .error { color: var(--status-danger); }
img { width: 100%; max-height: 380px; object-fit: contain; } pre { max-height: 220px; overflow: auto; font-size: 11px; }
details { margin: 10px 0; } summary { cursor: pointer; } .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
:is(input, select, textarea, button, summary):focus-visible { outline: 2px solid var(--status-info); outline-offset: 2px; }
</style>
