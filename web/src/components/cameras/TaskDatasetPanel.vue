<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import {
  getTaskWorkflows, createTaskWorkflow, collectTaskData, stopTaskCollection, reviewTaskSample,
  freezeTaskDataset, exportTaskReview, importTaskReview, getTaskTrainingRecipes,
  proposeTaskTraining, approveTaskTraining, exportTaskTraining, importTaskCandidate,
  taskSampleImageUrl, suggestTaskAnnotations, type AnalysisTask,
} from '../../api/analysisTasks'
import { extractErrorMessage } from '../../utils/error'

const props = defineProps<{ task: AnalysisTask }>()
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
    error.value = ''
  } catch (cause: unknown) {
    if (!disposed && token === generation && sequence === refreshSequence) error.value = extractErrorMessage(cause, '无法读取任务数据流程；下列状态可能过期')
  } finally {
    if (!disposed && token === generation && sequence === refreshSequence) {
      loading.value = false
      timer = setTimeout(() => { void refresh() }, workflow.value?.state === 'collecting' ? 5000 : 60_000)
    }
  }
}
async function act(action: () => Promise<unknown>, message: string, reloadReview = false) {
  if (busy.value) return
  const token = generation; const id = workflowId.value
  busy.value = true; error.value = ''; notice.value = ''
  try {
    await action()
    if (disposed || token !== generation || id !== workflowId.value) return
    notice.value = message
    await refresh()
    if (!disposed && token === generation && id === workflowId.value && reloadReview) loadReviewForm()
  } catch (cause: unknown) {
    if (!disposed && token === generation && id === workflowId.value) error.value = extractErrorMessage(cause, '数据流程操作失败，请刷新后检查版本和前置条件')
  } finally { if (!disposed && token === generation && id === workflowId.value) busy.value = false }
}
function download(payload: Record<string, unknown>, name: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
  const anchor = document.createElement('a')
  // Documents stay local: downloading a manifest performs no external upload.
  anchor.href = url; anchor.download = name; anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}
async function create() {
  await act(async () => { await createTaskWorkflow(props.task.task_id, createKey); createKey = crypto.randomUUID() }, '已建立绑定当前配置的数据流程')
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
  sampleId.value = ''; notice.value = ''; busy.value = false
  notes.value = ''; annotation.value = '{}'; importText.value = '{}'; packageSha.value = ''
  candidateText.value = '{"relative_path":"","descriptor":{}}'
  reviewRevision.value = null; reviewImageSha.value = ''
})
watch(() => props.task.task_id, () => {
  generation += 1; createKey = crypto.randomUUID(); workflowId.value = ''; sampleId.value = ''; recipeId.value = ''
  workflows.value = []; recipes.value = []; error.value = ''; notice.value = ''; busy.value = false
  void refresh()
}, { immediate: true })
onUnmounted(() => { disposed = true; generation += 1; if (timer) clearTimeout(timer) })
</script>

<template>
  <section class="task-data" aria-label="任务采集与审核" :aria-busy="busy || loading">
    <h4>采集 → 复核 → 冻结 → 外部训练 → 候选登记</h4>
    <p class="note">采集使用现有视频源；当前支持定时抽样。CVAT / FiftyOne 导出是离线图片清单。训练导出供外部训练程序衔接，本机不会执行训练；模型候选登记后仍需旁路验证。</p>
    <p v-if="error" role="alert" class="error">{{ error }}</p><p v-if="notice" role="status">{{ notice }}</p>
    <button :disabled="busy" @click="create">新建数据流程</button><button :disabled="busy || loading" @click="refresh">刷新流程</button>
    <label>数据流程<select v-model="workflowId" data-test="workflow-select" :disabled="busy"><option value="">请选择</option><option v-for="row in workflows" :key="text(row.workflow_id)" :value="text(row.workflow_id)">{{ labels[text(row.state)] || row.state }} · r{{ row.task_revision }} · {{ text(row.workflow_id).slice(0, 8) }}</option></select></label>
    <template v-if="workflow">
      <p>状态：{{ labels[text(workflow.state)] || workflow.state }} · 流程版本 {{ revision }} · 样本 {{ samples.length }}</p>
      <p v-if="record(workflow.collection_runtime).state" role="status">采集运行观测：{{ record(workflow.collection_runtime).state }}<span v-if="record(workflow.collection_runtime).error_code"> · {{ record(workflow.collection_runtime).error_code }}</span></p>
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
      <details><summary>CVAT / FiftyOne 离线交换</summary><button v-for="target in (['cvat', 'fiftyone'] as const)" :key="target" :disabled="busy" @click="act(async () => { const response = await exportTaskReview(task.task_id, workflowId, target); download(response.document, `${target}-task-review.json`) }, '离线清单已导出；未上传外部服务')">导出 {{ target }} 清单</button><label>回传任务审核 JSON<textarea v-model="importText" rows="6" :disabled="busy" /></label><button :disabled="busy" @click="act(() => importTaskReview(task.task_id, workflowId, { expected_revision: revision, payload: parse(importText) }), '回传已核验；自动标注仍需人工复核')">校验并导入审核</button></details>
      <details><summary>外部训练与候选模型</summary><label>训练配方<select v-model="recipeId"><option v-for="recipe in recipes" :key="text(recipe.recipe_id)" :value="text(recipe.recipe_id)">{{ recipe.name || recipe.recipe_id }}</option></select></label><button :disabled="busy || !recipeId" @click="act(() => proposeTaskTraining(task.task_id, workflowId, { expected_revision: revision, recipe_id: recipeId }), '外部训练提案已记录')">创建训练提案</button><button :disabled="busy" @click="act(() => approveTaskTraining(task.task_id, workflowId, { expected_revision: revision }), '外部训练导出已获人工批准')">人工批准导出</button><button :disabled="busy" @click="act(async () => { const response = await exportTaskTraining(task.task_id, workflowId); packageSha = text(response.document.package_sha256); download(response.document, 'task-training-manifest.json') }, '训练交接清单已导出；尚未执行训练')">导出训练交接清单</button><label>训练交接摘要<input v-model="packageSha" :disabled="busy" /></label><label>外部训练产物（relative_path + descriptor）<textarea v-model="candidateText" rows="6" :disabled="busy" /></label><button :disabled="busy || !packageSha" @click="act(() => importTaskCandidate(task.task_id, workflowId, { expected_revision: revision, expected_package_sha256: packageSha, candidate: parse(candidateText) }), '候选模型已登记；未发布')">核验并登记候选</button></details>
    </template>
  </section>
</template>

<style scoped>
.task-data { margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--border-default); } h4 { margin: 0; } .note { color: var(--text-secondary); font-size: 12px; line-height: 1.6; } .error { color: var(--status-danger); }
label { display: grid; gap: 5px; margin: 10px 0; font-size: 12px; } input, select, textarea, button { background: var(--bg); color: var(--text-primary); border: 1px solid var(--border-default); border-radius: 6px; padding: 8px; } button { margin: 4px 6px 4px 0; cursor: pointer; } button:disabled { opacity: .6; cursor: not-allowed; }
textarea { width: 100%; box-sizing: border-box; font-family: ui-monospace, monospace; } img { width: 100%; max-height: 380px; object-fit: contain; } pre { max-height: 200px; overflow: auto; font-size: 11px; } .mono { overflow-wrap: anywhere; font-family: ui-monospace, monospace; font-size: 11px; } .review-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; } details { margin-top: 12px; } summary { cursor: pointer; }
button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible, summary:focus-visible { outline: 2px solid var(--status-info); outline-offset: 2px; } @media(max-width: 900px) { .review-grid { grid-template-columns: 1fr; } }
</style>
