<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { getActivePinia } from 'pinia'
import { useAuthStore } from '../../stores/useAuthStore'
import { useTrainingCandidateLedger, type ManualCandidateDraft } from '../../composables/useTrainingCandidateLedger'
import type { TaskCandidateImportRequest } from '../../types/generated-openapi'
import {
  getTaskWorkflows, createTaskWorkflow, collectTaskData, stopTaskCollection, reviewTaskSample,
  freezeTaskDataset, exportTaskReview, importTaskReview, getTaskTrainingRecipes,
  proposeTaskTraining, approveTaskTraining, exportTaskTraining, importTaskCandidate,
  taskSampleImageUrl, suggestTaskAnnotations, type AnalysisTask,
} from '../../api/analysisTasks'
import { extractErrorMessage } from '../../utils/error'

const props = defineProps<{ task: AnalysisTask }>()
const pinia = getActivePinia()
const auth = pinia ? useAuthStore(pinia) : null
const username = computed(() => auth?.currentUser?.username ?? '')
const role = computed(() => auth?.currentUser?.role ?? '')
const canRegister = computed(() => !!username.value && ['admin', 'engineer'].includes(role.value))
const candidateLedger = useTrainingCandidateLedger(auth)
const sharedCandidateDraft = computed(() => candidateLedger.get(props.task.camera_id, props.task.task_id))
const manualCandidate = computed(() => {
  const draft = sharedCandidateDraft.value
  return draft?.origin === 'manual' ? draft : null
})
const candidateSubmitting = computed(() => candidateLedger.submitting(props.task.camera_id, props.task.task_id))
const candidateConfirmed = ref(false)
const readFailed = ref(false)
const workflows = ref<Record<string, unknown>[]>([])
const workflowId = ref('')
const sampleId = ref('')
const recipes = ref<Record<string, unknown>[]>([])
const recipeId = ref('')
const notes = ref('')
const annotation = ref('{}')
const importText = ref('{}')
const candidateText = ref('{"relative_path":"","descriptor":{}}')
const packageSha = ref('')
const error = ref('')
const notice = ref('')
const busy = ref(false)
const loading = ref(false)
const reviewRevision = ref<number | null>(null)
const reviewImageSha = ref('')
const imageElement = ref<HTMLImageElement | null>(null)
const imageState = ref<'loading' | 'ready' | 'failed'>('loading')
const imageRetry = ref(0)
let generation = 0
let actionSequence = 0
let refreshSequence = 0
let disposed = false
let createKey = crypto.randomUUID()
let timer: ReturnType<typeof setTimeout> | undefined
const workflow = computed(() => workflows.value.find(row => row.workflow_id === workflowId.value))
const manifest = computed(() => record(workflow.value?.manifest))
const samples = computed(() => Array.isArray(manifest.value.samples) ? manifest.value.samples.map(record) : [])
const sample = computed(() => samples.value.find(row => row.sample_id === sampleId.value))
const revision = computed(() => Number(workflow.value?.revision ?? 0))
const reviewStale = computed(() => !!sample.value && reviewRevision.value !== revision.value)
const reviewImageKey = computed(() => JSON.stringify([workflowId.value, sampleId.value, sample.value?.image_sha256, imageRetry.value]))
const reviewDecision = computed(() => text(record(record(manifest.value.reviews)[sampleId.value]).decision))
const labels: Record<string, string> = {
  created: '待采集', collectable: '待采集', collecting: '采集中', review: '待复核', collection_failed: '采集失败',
  dataset_frozen: '数据集已冻结', training_proposed: '等待人工批准外部训练',
  training_export_ready: '可导出外部训练清单', training_blocked: '训练受阻',
  candidate_importing: '候选登记待对账', candidate_registered: '候选模型已登记',
}
function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}
function text(value: unknown): string { return typeof value === 'string' ? value : '' }
const sha256 = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value)
const integer = (value: unknown): value is number => typeof value === 'number' && Number.isInteger(value) && value > 0
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
  if (value && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([name, item]) => `${JSON.stringify(name)}:${canonical(item)}`).join(',')}}`
  return JSON.stringify(value) ?? 'null'
}
function taskBinding() {
  return { task_id: props.task.task_id, camera_id: props.task.camera_id, template_id: props.task.template_id,
    task_revision: props.task.current_revision, task_config_sha256: props.task.config_sha256 }
}
function candidateRequestIdentity(draft: ManualCandidateDraft) {
  return canonical({ workflow_id: draft.workflowId, package_sha256: draft.body.expected_package_sha256,
    candidate: draft.body.candidate })
}
function receiptMatches(receipt: Record<string, unknown>, draft: ManualCandidateDraft) {
  return sha256(receipt.operation_id) && receipt.stage === 'candidate' && !!text(receipt.artifact_id)
    && receipt.requested_revision === draft.body.expected_revision
    && canonical(receipt.request_identity) === candidateRequestIdentity(draft)
}
function pendingMatches(row: Record<string, unknown>, draft: ManualCandidateDraft) {
  const operation = record(row.operation)
  return row.state === 'candidate_importing' && operation.kind === 'candidate_import'
    && operation.requested_by === draft.actor && operation.requested_revision === draft.body.expected_revision
    && operation.package_sha256 === draft.body.expected_package_sha256
    && operation.approval_manifest_sha256 === draft.manifestSha
    && record(operation.training_manifest).package_sha256 === draft.body.expected_package_sha256
    && (operation.request_identity === undefined
      || canonical(operation.request_identity) === candidateRequestIdentity(draft))
}
const manualCandidateBlocked = computed(() => {
  const draft = manualCandidate.value
  if (!draft) return ''
  if (!canRegister.value || username.value !== draft.actor) return '需要原个人工程师／管理员身份重新核对'
  if (props.task.lifecycle === 'archived' || draft.binding !== canonical(taskBinding())) return '任务配置已变化；原候选请求保留，不会改绑'
  if (loading.value || readFailed.value) return '当前数据流程状态未核实，先成功刷新再对账'
  const row = workflows.value.find(item => item.workflow_id === draft.workflowId)
  const manifest = record(row?.manifest)
  if (!row || row.task_id !== draft.taskId || canonical(manifest.binding) !== draft.binding
    || !text(manifest.dataset_id) || !text(record(manifest.training).approved_by)) return '原流程、数据集或人工训练批准已变化，不能使用旧请求继续登记'
  if (row.state === 'training_export_ready' && row.revision === draft.body.expected_revision
    && row.manifest_sha256 === draft.manifestSha) return ''
  if (draft.phase === 'unknown' && pendingMatches(row, draft)) return ''
  const receipt = record(record(manifest.training).candidate)
  if (draft.phase !== 'prepared' && row.state === 'candidate_registered') {
    if (receiptMatches(receipt, draft)) return ''
    if (draft.phase === 'unknown' && receipt.request_identity === undefined && sha256(receipt.operation_id)
      && receipt.requested_revision === draft.body.expected_revision && receipt.stage === 'candidate'
      && text(receipt.artifact_id)) return ''
  }
  return '原候选登记条件或版本已变化；不会用刷新后的版本替换原请求'
})
function prepareManualCandidate() {
  if (!canRegister.value || busy.value || loading.value || readFailed.value || candidateSubmitting.value) return
  if (sharedCandidateDraft.value) { error.value = '此任务已有候选原请求，请先在原入口对账或清除已确认草稿'; return }
  try {
    const row = workflow.value, manifest = record(row?.manifest)
    if (!row || row.task_id !== props.task.task_id || canonical(manifest.binding) !== canonical(taskBinding())
      || !text(manifest.dataset_id) || !text(record(manifest.training).approved_by) || !sha256(row.manifest_sha256)) {
      throw new Error('任务、冻结数据集或人工训练批准尚未核实')
    }
    if (!sha256(packageSha.value)) throw new Error('训练交接摘要必须是完整 SHA256')
    const candidate = parse(candidateText.value)
    if (!text(candidate.relative_path) || !Object.keys(record(candidate.descriptor)).length) throw new Error('需要候选受控相对路径和模型描述')
    const operation = record(row.operation), reconciling = row.state === 'candidate_importing'
    const expectedRevision = reconciling ? operation.requested_revision : row.revision
    if (!integer(expectedRevision)) throw new Error('原流程版本尚未核实')
    const body: TaskCandidateImportRequest = { expected_revision: expectedRevision,
      expected_package_sha256: packageSha.value, candidate }
    const draft: ManualCandidateDraft = { origin: 'manual', taskId: props.task.task_id, cameraId: props.task.camera_id,
      actor: username.value, workflowId: text(row.workflow_id), binding: canonical(taskBinding()),
      manifestSha: text(reconciling ? operation.approval_manifest_sha256 : row.manifest_sha256), body,
      phase: reconciling ? 'unknown' : 'prepared', error: '', artifactId: '' }
    if (reconciling) {
      // A fresh panel cannot reconstruct a raw request from validator-normalized
      // model data. Only the durable original request projection can restore it.
      if (!pendingMatches(row, draft) || canonical(operation.request_identity) !== candidateRequestIdentity(draft)) {
        throw new Error('待对账意图不匹配原个人身份或候选请求；不能按当前版本重建')
      }
    } else if (row.state !== 'training_export_ready') throw new Error('流程尚未获准登记外部训练候选')
    if (!candidateLedger.prepare(draft)) throw new Error('会话已有候选请求或草稿已达上限，不会覆盖原请求')
    candidateConfirmed.value = false; error.value = ''
  } catch (cause: unknown) { error.value = extractErrorMessage(cause, '无法冻结候选登记请求') }
}
async function registerManualCandidate() {
  const draft = manualCandidate.value
  if (!draft || !candidateConfirmed.value || manualCandidateBlocked.value || busy.value
    || candidateSubmitting.value || draft.phase === 'registered') return
  const lease = candidateLedger.acquire(draft)
  if (!lease) return
  const token = generation
  busy.value = true; candidateConfirmed.value = false; draft.phase = 'unknown'; draft.error = ''
  try {
    const body = JSON.parse(JSON.stringify(draft.body)) as TaskCandidateImportRequest
    const result = await importTaskCandidate(draft.taskId, draft.workflowId, body)
    if (!lease.current() || disposed || token !== generation || manualCandidate.value !== draft
      || draft.binding !== canonical(taskBinding()) || username.value !== draft.actor) return
    const row = result.document, receipt = record(row.candidate_import_receipt)
    const persisted = record(record(record(row.manifest).training).candidate)
    if (row.task_id !== draft.taskId || row.workflow_id !== draft.workflowId || row.state !== 'candidate_registered'
      || canonical(record(row.manifest).binding) !== draft.binding || !receiptMatches(receipt, draft)
      || persisted.operation_id !== receipt.operation_id || persisted.artifact_id !== receipt.artifact_id
      || persisted.stage !== 'candidate' || persisted.requested_revision !== draft.body.expected_revision
      || (persisted.request_identity !== undefined && canonical(persisted.request_identity) !== candidateRequestIdentity(draft))) {
      throw new Error('候选登记回执身份尚未确认，请使用原请求对账')
    }
    draft.phase = 'registered'; draft.artifactId = text(receipt.artifact_id)
    notice.value = '候选已登记；未部署或发布，原人审与标注草稿保留'
    await refresh()
  } catch (cause: unknown) {
    if (lease.current() && !disposed && token === generation && manualCandidate.value === draft) {
      draft.error = extractErrorMessage(cause, '候选登记结果未知，原请求已保留')
    }
  } finally {
    lease.release()
    if (!disposed && token === generation) busy.value = false
  }
}
function clearManualCandidate() {
  if (manualCandidate.value && !candidateSubmitting.value && !busy.value) candidateLedger.clear(manualCandidate.value)
  candidateConfirmed.value = false
}
function imageFinished(event: Event, state: 'ready' | 'failed') {
  // A detached image can finish after another sample is selected. Only the
  // currently rendered, SHA-bound image may unlock human approval.
  if (event.currentTarget === imageElement.value) imageState.value = state
}
function parse(value: string): Record<string, unknown> {
  const result: unknown = JSON.parse(value)
  if (!result || typeof result !== 'object' || Array.isArray(result)) throw new Error('请输入 JSON 对象')
  return result as Record<string, unknown>
}
async function refresh() {
  if (timer) clearTimeout(timer)
  const token = generation; const taskId = props.task.task_id
  const sequence = ++refreshSequence
  loading.value = true
  try {
    const [rows, recipeResponse] = await Promise.all([getTaskWorkflows(taskId), getTaskTrainingRecipes(taskId)])
    if (disposed || token !== generation || sequence !== refreshSequence) return
    workflows.value = rows.documents; recipes.value = recipeResponse.documents
    if (!workflowId.value && rows.documents.length) workflowId.value = text(rows.documents[0]?.workflow_id)
    if (!recipeId.value && recipes.value.length) recipeId.value = text(recipes.value[0]?.recipe_id)
    error.value = ''; readFailed.value = false
  } catch (cause: unknown) {
    if (!disposed && token === generation && sequence === refreshSequence) {
      readFailed.value = true
      error.value = extractErrorMessage(cause, '无法读取任务数据流程；下列状态可能过期')
    }
  } finally {
    if (!disposed && token === generation && sequence === refreshSequence) {
      loading.value = false
      timer = setTimeout(() => { void refresh() }, workflow.value?.state === 'collecting' ? 5000 : 60_000)
    }
  }
}
async function act(action: (current: () => boolean) => Promise<unknown>, message: string, reloadReview = false) {
  if (busy.value || candidateSubmitting.value) return
  const token = generation; const id = workflowId.value
  const sequence = ++actionSequence; const binding = canonical(taskBinding())
  const actor = username.value; const actorRole = role.value
  const ownsAction = () => !disposed && token === generation && sequence === actionSequence && id === workflowId.value
  const current = () => ownsAction() && binding === canonical(taskBinding())
    && actor === username.value && actorRole === role.value
  busy.value = true; error.value = ''; notice.value = ''
  try {
    await action(current)
    if (!current()) return
    notice.value = message
    await refresh()
    if (current() && reloadReview) loadReviewForm()
  } catch (cause: unknown) {
    if (current()) error.value = extractErrorMessage(cause, '数据流程操作失败，请刷新后检查版本和前置条件')
  } finally { if (ownsAction()) busy.value = false }
}
function download(payload: Record<string, unknown>, name: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
  const anchor = document.createElement('a')
  // Documents stay local: downloading a manifest performs no external upload.
  anchor.href = url; anchor.download = name; anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}
async function create() {
  await act(async current => { await createTaskWorkflow(props.task.task_id, createKey); if (current()) createKey = crypto.randomUUID() }, '已建立绑定当前配置的数据流程')
}
async function exportDocument(target: 'cvat' | 'fiftyone' | 'training') {
  const taskId = props.task.task_id, id = workflowId.value, binding = canonical(taskBinding())
  const frozenRevision = workflow.value?.revision, manifestSha = workflow.value?.manifest_sha256
  if (!workflow.value || loading.value || readFailed.value) return
  await act(async current => {
    const response = target === 'training'
      ? await exportTaskTraining(taskId, id) : await exportTaskReview(taskId, id, target)
    // Check before either a form mutation or a browser download. act() cannot
    // undo those side effects when a previous task's request completes late.
    if (!current()) return
    if (workflow.value?.revision !== frozenRevision || workflow.value?.manifest_sha256 !== manifestSha) {
      throw new Error('流程已更新，旧导出清单不再应用，请重新核对后导出')
    }
    const document = response.document
    if (document.workflow_id !== id || canonical(document.binding) !== binding
      || (target === 'training' ? !sha256(document.package_sha256)
        : document.target !== target || document.workflow_revision !== frozenRevision || document.report_sha256 !== manifestSha)) {
      throw new Error('导出清单身份与当前任务或流程版本不一致，请刷新后重试')
    }
    if (target === 'training') packageSha.value = text(document.package_sha256)
    download(document, target === 'training' ? 'task-training-manifest.json' : `${target}-task-review.json`)
  }, target === 'training' ? '训练交接清单已导出；尚未执行训练' : '离线清单已导出；未上传外部服务')
}
async function review(decision: 'approved' | 'rejected' | 'needs_review') {
  const selected = sample.value
  if (!selected) return
  if (decision === 'approved' && imageState.value !== 'ready') {
    error.value = '图片尚未成功加载，不能批准此标注'
    return
  }
  await act(() => reviewTaskSample(props.task.task_id, workflowId.value, {
    expected_revision: reviewRevision.value ?? 0, sample_id: text(selected.sample_id),
    expected_image_sha256: reviewImageSha.value, decision, notes: notes.value,
    annotation: decision === 'approved' ? parse(annotation.value) : {},
  }), '审核已保存；旧数据集和训练批准已失效，需要重新冻结', true)
}
function loadReviewForm() {
  const previous = record(record(manifest.value.reviews)[sampleId.value])
  reviewRevision.value = sample.value ? revision.value : null
  reviewImageSha.value = text(sample.value?.image_sha256)
  notes.value = text(previous.notes)
  annotation.value = JSON.stringify(previous.annotation ?? record(sample.value?.auto_annotation).annotation ?? {
    schema: 'argus.task-annotation.v1', coordinate_space: 'normalized',
    kind: props.task.template_id === 'pose_estimation.v1' ? 'pose' : 'detection', objects: [],
  }, null, 2)
}
watch(sampleId, loadReviewForm)
watch(reviewImageKey, () => { imageState.value = 'loading' }, { flush: 'sync' })
watch(workflowId, () => {
  actionSequence += 1
  sampleId.value = ''; notice.value = ''; busy.value = false
  notes.value = ''; annotation.value = '{}'; importText.value = '{}'; packageSha.value = ''
  candidateText.value = '{"relative_path":"","descriptor":{}}'
  reviewRevision.value = null; reviewImageSha.value = ''
})
watch(() => canonical([taskBinding(), manualCandidate.value?.phase, manualCandidate.value?.body,
  workflows.value.find(row => row.workflow_id === manualCandidate.value?.workflowId)]), () => { candidateConfirmed.value = false })
watch([() => props.task.task_id, () => props.task.camera_id, username, role], () => {
  generation += 1; createKey = crypto.randomUUID(); workflowId.value = ''; sampleId.value = ''; recipeId.value = ''
  workflows.value = []; recipes.value = []; error.value = ''; notice.value = ''; busy.value = false
  candidateConfirmed.value = false; readFailed.value = false
  void refresh()
}, { immediate: true })
onUnmounted(() => { disposed = true; generation += 1; if (timer) clearTimeout(timer) })
</script>

<template>
  <section class="task-data" aria-label="任务采集与审核" :aria-busy="busy || loading || candidateSubmitting">
    <h4>{{ task.template_id === 'fast_motion.v1' ? '采集 → 复核 → 数据留存' : '采集 → 复核 → 冻结 → 外部训练 → 候选登记' }}</h4>
    <p v-if="task.template_id === 'fast_motion.v1'" class="note">固定算法没有训练权重或训练配方，本机不会执行训练。可使用现有视频源定时采集复核图片；采集不代表已经通过采样和效果验收。</p>
    <p v-else class="note">支持定时抽样和目标检测事件关键帧采集。事件策略只接收采集启动后、同一执行与源会话中新提交的事件，不补抓未来帧；没有事件不等于正常。CVAT / FiftyOne 导出仍是离线图片清单；模型候选登记后仍需旁路验证。</p>
    <p v-if="error" role="alert" class="error">{{ error }}</p><p v-if="notice" role="status">{{ notice }}</p>
    <button :disabled="busy" @click="create">新建数据流程</button><button :disabled="busy || loading" @click="refresh">刷新流程</button>
    <label>数据流程<select v-model="workflowId" data-test="workflow-select" :disabled="busy"><option value="">请选择</option><option v-for="row in workflows" :key="text(row.workflow_id)" :value="text(row.workflow_id)">{{ labels[text(row.state)] || row.state }} · r{{ row.task_revision }} · {{ text(row.workflow_id).slice(0, 8) }}</option></select></label>
    <template v-if="workflow">
      <p>状态：{{ labels[text(workflow.state)] || workflow.state }} · 流程版本 {{ revision }} · 样本 {{ samples.length }}</p>
      <p v-if="record(workflow.collection_runtime).state" role="status">采集运行观测：{{ record(workflow.collection_runtime).state }}<span v-if="record(workflow.collection_runtime).error_code"> · {{ record(workflow.collection_runtime).error_code }}</span></p>
      <p v-if="record(workflow.collection_runtime).policy === 'events'" class="note">事件关键帧采集 · 已读取游标 {{ record(workflow.collection_runtime).last_cursor }} · 按间隔过滤 {{ record(workflow.collection_runtime).throttled_events }} 个事件。重连或任务执行身份变化后需重新启动采集；零图片不能冻结训练集。</p>
      <div class="actions">
        <button :disabled="busy || workflow.state === 'collecting'" @click="act(() => collectTaskData(task.task_id, workflowId, { expected_revision: revision, params: {} }), '采集已启动')">按任务配置采集</button>
        <button :disabled="busy" @click="act(() => stopTaskCollection(task.task_id, workflowId, { expected_revision: revision }), '停止和采集清单对账已处理')">停止 / 重试对账</button>
        <button :disabled="busy || workflow.state === 'collecting'" @click="act(() => freezeTaskDataset(task.task_id, workflowId, { expected_revision: revision }), '数据集已冻结')">冻结已审数据</button>
        <button :disabled="busy || workflow.state === 'collecting'" @click="act(() => suggestTaskAnnotations(task.task_id, workflowId, { expected_revision: revision }), '精确同帧的预标注建议已保存，仍需逐帧人工批准')">从旁路结果生成预标注</button>
      </div>
      <p class="note">冻结要求所有样本完成人工复核，且已批准样本至少来自两个独立源会话，用于隔离训练集和验证集。</p>
      <label>复核图片<select v-model="sampleId" data-test="sample-select" :disabled="busy"><option value="">请选择图片</option><option v-for="item in samples" :key="text(item.sample_id)" :value="text(item.sample_id)">{{ text(item.sample_id).slice(0, 12) }} · {{ record(manifest.reviews)[text(item.sample_id)] ? '已有审核记录' : '待审核' }}</option></select></label>
      <div v-if="sample" class="review-grid">
        <div><img :key="reviewImageKey" ref="imageElement" :src="taskSampleImageUrl(task.task_id, workflowId, sampleId)" alt="当前待复核的任务采集帧" @load="imageFinished($event, 'ready')" @error="imageFinished($event, 'failed')" /><p v-if="imageState === 'loading'" role="status">审核图片加载中，暂不能批准。</p><p v-else-if="imageState === 'failed'" role="alert" class="error">审核图片加载失败，不能批准。<button :disabled="busy" @click="imageRetry += 1">重试加载图片</button></p><p class="mono">{{ sample.image_sha256 }}</p><pre>{{ JSON.stringify(sample.identity, null, 2) }}</pre></div>
        <div><p>当前审核：{{ reviewDecision === 'approved' ? '已批准' : reviewDecision === 'rejected' ? '已排除' : '待人工复核' }}</p><details v-if="sample.auto_annotation"><summary>当前自动建议（不等于人工批准）</summary><pre>{{ JSON.stringify(sample.auto_annotation, null, 2) }}</pre></details><p v-if="reviewStale" role="status">流程已更新；保留了你的草稿，但不能用旧表单覆盖新审核。</p><button v-if="reviewStale" :disabled="busy" @click="loadReviewForm">重新载入当前审核</button><label>审核备注<textarea v-model="notes" rows="3" maxlength="4000" :disabled="busy" /></label><label>人工标注（归一化坐标；空 objects 表示人工确认无目标）<textarea v-model="annotation" rows="9" spellcheck="false" :disabled="busy" /></label><button :disabled="busy || reviewStale || imageState !== 'ready'" @click="review('approved')">批准此标注</button><button :disabled="busy || reviewStale" @click="review('rejected')">排除此样本</button><button :disabled="busy || reviewStale" @click="review('needs_review')">恢复待复核</button></div>
      </div>
      <details v-if="sample?.event_origin"><summary>事件关键帧来源（非连续录像、非自动批准）</summary><pre>{{ JSON.stringify(sample.event_origin, null, 2) }}</pre></details>
      <details><summary>CVAT / FiftyOne 离线交换</summary><button v-for="target in (['cvat', 'fiftyone'] as const)" :key="target" :disabled="busy || loading || readFailed" @click="exportDocument(target)">导出 {{ target }} 清单</button><label>回传任务审核 JSON<textarea v-model="importText" rows="6" :disabled="busy" /></label><button :disabled="busy" @click="act(() => importTaskReview(task.task_id, workflowId, { expected_revision: revision, payload: parse(importText) }), '回传已核验；自动标注仍需人工复核')">校验并导入审核</button></details>
      <details v-if="task.template_id !== 'fast_motion.v1'"><summary>外部训练与候选模型</summary><label>训练配方<select v-model="recipeId"><option v-for="recipe in recipes" :key="text(recipe.recipe_id)" :value="text(recipe.recipe_id)">{{ recipe.name || recipe.recipe_id }}</option></select></label><button :disabled="busy || !recipeId" @click="act(() => proposeTaskTraining(task.task_id, workflowId, { expected_revision: revision, recipe_id: recipeId }), '外部训练提案已记录')">创建训练提案</button><button :disabled="busy" @click="act(() => approveTaskTraining(task.task_id, workflowId, { expected_revision: revision }), '外部训练导出已获人工批准')">人工批准导出</button><button :disabled="busy || loading || readFailed" @click="exportDocument('training')">导出训练交接清单</button><label>训练交接摘要<input v-model="packageSha" data-test="manual-package-sha" :disabled="busy" /></label><label>外部训练产物（relative_path + descriptor）<textarea v-model="candidateText" data-test="manual-candidate-json" rows="6" :disabled="busy" /></label><button data-test="prepare-manual-candidate" :disabled="!canRegister || busy || loading || readFailed || candidateSubmitting || !!sharedCandidateDraft || !packageSha" @click="prepareManualCandidate">冻结并核对外部训练候选</button><p v-if="sharedCandidateDraft?.origin === 'training'" role="status">训练闭环入口已保留此任务的候选请求，请在原入口核对；手工入口不会覆盖或重复提交。</p></details>
    </template>
    <section v-if="manualCandidate" aria-label="外部训练候选登记核对" data-test="manual-candidate-draft">
      <h4>手工外部候选：冻结原请求后登记</h4>
      <p>原数据流程 {{ manualCandidate.workflowId }} · r{{ manualCandidate.body.expected_revision }} · 交接 SHA256 {{ manualCandidate.body.expected_package_sha256 }}</p>
      <pre>{{ JSON.stringify(manualCandidate.body, null, 2) }}</pre>
      <p class="note">不伪造训练作业，不自动部署或发布；更换流程和刷新不会重写这个请求。结果未知时只能使用原请求对账。</p>
      <p v-if="manualCandidateBlocked" role="status" data-test="manual-candidate-blocked">{{ manualCandidateBlocked }}</p>
      <p v-if="manualCandidate.error" role="alert" data-test="manual-candidate-error">{{ manualCandidate.error }}；原请求已保留。</p>
      <p v-if="manualCandidate.phase === 'registered'" role="status" data-test="manual-candidate-registered">候选已登记：{{ manualCandidate.artifactId }}，尚未部署或发布。</p>
      <template v-else>
        <label><input v-model="candidateConfirmed" type="checkbox" data-test="confirm-manual-candidate" :disabled="busy || candidateSubmitting || !!manualCandidateBlocked" />我已核对原数据批准、交接和制品，仅确认登记候选</label>
        <button data-test="register-manual-candidate" :disabled="busy || candidateSubmitting || !!manualCandidateBlocked || !candidateConfirmed" @click="registerManualCandidate">{{ manualCandidate.phase === 'unknown' ? '使用原请求核对外部候选' : '确认登记外部候选' }}</button>
      </template>
      <button data-test="clear-manual-candidate" :disabled="busy || candidateSubmitting || manualCandidate.phase === 'unknown'" @click="clearManualCandidate">清除已确认的候选草稿</button>
    </section>
  </section>
</template>

<style scoped>
.task-data { margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--border-default); } h4 { margin: 0; } .note { color: var(--text-secondary); font-size: 12px; line-height: 1.6; } .error { color: var(--status-danger); }
label { display: grid; gap: 5px; margin: 10px 0; font-size: 12px; } input, select, textarea, button { background: var(--bg); color: var(--text-primary); border: 1px solid var(--border-default); border-radius: 6px; padding: 8px; } button { margin: 4px 6px 4px 0; cursor: pointer; } button:disabled { opacity: .6; cursor: not-allowed; }
textarea { width: 100%; box-sizing: border-box; font-family: ui-monospace, monospace; } img { width: 100%; max-height: 380px; object-fit: contain; } pre { max-height: 200px; overflow: auto; font-size: 11px; } .mono { overflow-wrap: anywhere; font-family: ui-monospace, monospace; font-size: 11px; } .review-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; } details { margin-top: 12px; } summary { cursor: pointer; }
button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible, summary:focus-visible { outline: 2px solid var(--status-info); outline-offset: 2px; } @media(max-width: 900px) { .review-grid { grid-template-columns: 1fr; } }
</style>
