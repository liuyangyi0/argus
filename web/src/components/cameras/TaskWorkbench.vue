<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import {
  createAnalysisTask, updateAnalysisTask, getAnalysisTaskTemplates, getTaskModelCompatibility,
  getTaskBundles, freezeTaskBundle, getTaskResults,
  registerTaskModel, archiveAnalysisTask, type AnalysisTask,
} from '../../api/analysisTasks'
import TaskDatasetPanel from './TaskDatasetPanel.vue'
import TaskDeploymentPanel from './TaskDeploymentPanel.vue'
import type { AnalysisTaskTemplateResponse } from '../../types/generated-openapi'
import { extractErrorMessage } from '../../utils/error'

const props = defineProps<{ cameraId: string; tasks: AnalysisTask[] }>()
const emit = defineEmits<{ refresh: [] }>()
const templates = ref<AnalysisTaskTemplateResponse[]>([])
const templateId = ref('object_detection.v1')
const name = ref('')
const configuration = ref('{}')
const editRevision = ref<number | null>(null)
const selectedId = ref('')
const models = ref<Record<string, unknown>[]>([])
const bundles = ref<Record<string, unknown>[]>([])
const modelIds = ref<Record<string, string>>({})
const bundleId = ref('')
const results = ref<Record<string, unknown>[]>([])
const relativePath = ref('')
const defaultDescriptor = '{"name":"Object detector","adapter_id":"onnx_detection.v1","output_schema":"yolov8_detection.v1","labels":["person"],"input_size":[640,640]}'
const descriptor = ref(defaultDescriptor)
const busy = ref(false)
const error = ref('')
const notice = ref('')
let generation = 0
let selectionGeneration = 0
let selectionReadSequence = 0
let operationGeneration = 0
let disposed = false
const commandKeys = new Map<string, string>()
const managed = computed(() => props.tasks.filter(task => task.authority === 'managed' && task.lifecycle !== 'archived'))
const selected = computed(() => managed.value.find(task => task.task_id === selectedId.value))
const configurationStale = computed(() => !!selected.value && editRevision.value !== selected.value.current_revision)
const spec = computed(() => templates.value.find(item => item.template_id === (selected.value?.template_id ?? templateId.value)))
const modelRoles = computed(() => Object.keys(spec.value?.model_roles ?? {}))

function text(value: unknown): string { return typeof value === 'string' ? value : '' }
function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}
function compatible(model: Record<string, unknown>, role: string): boolean {
  return object(object(model.role_compatibility)[role]).compatible === true
}
function roleLabel(role: string): string {
  return ({ detector: '目标检测器', pose: '人体姿态模型', person_detector: '人员检测模型', head_detector: '安全帽 / 裸头检测模型' } as Record<string, string>)[role] ?? role
}
function requiredLabels(role: string): string {
  const labels = object(object(spec.value?.model_role_requirements)[role]).required_labels
  return Array.isArray(labels) && labels.length ? labels.map(text).join(' / ') : '按任务选择类别校验'
}
function incompatibility(model: Record<string, unknown>, role: string): string {
  const reasons = object(object(model.role_compatibility)[role]).reasons
  if (!Array.isArray(reasons)) return '兼容性未核实'
  const messages: Record<string, string> = { CAMERA_SCOPE_MISMATCH: '相机作用域不符', ADAPTER_MISMATCH: '模型适配器不符', REQUIRED_LABELS_MISSING: '缺少角色必需类别', TASK_LABELS_MISSING: '缺少任务选择类别' }
  return reasons.map(reason => messages[text(reason)] ?? text(reason)).join('；') || '兼容性未核实'
}
function objectJson(value: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(value)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('配置必须为 JSON 对象')
  return parsed as Record<string, unknown>
}
function commandKey(action: string, payload: unknown): string {
  const identity = JSON.stringify([props.cameraId, action, payload])
  if (!commandKeys.has(identity)) commandKeys.set(identity, crypto.randomUUID())
  return commandKeys.get(identity)!
}
async function perform(action: () => Promise<unknown>, success: string, refreshSelection = true) {
  if (busy.value) return
  const token = generation
  const selection = selectionGeneration
  const operation = ++operationGeneration
  const taskId = selectedId.value
  busy.value = true; error.value = ''; notice.value = ''
  try {
    const result = await action()
    if (disposed || token !== generation || selection !== selectionGeneration || operation !== operationGeneration || taskId !== selectedId.value) return
    notice.value = success
    if (result && typeof result === 'object' && 'document' in result) {
      const document = result.document as Record<string, unknown>
      const automatic = document.automatic_collection as Record<string, unknown> | undefined
      if (automatic?.state === 'blocked' || automatic?.state === 'cleanup_pending') {
        notice.value += '；自动采集需要处理，请在数据流程中查看状态并重试对账。'
      }
    }
    emit('refresh')
    if (refreshSelection) await loadSelection(token)
  } catch (cause: unknown) {
    if (!disposed && token === generation && selection === selectionGeneration && operation === operationGeneration && taskId === selectedId.value) error.value = extractErrorMessage(cause, '任务操作失败，请刷新后重试')
  } finally {
    if (!disposed && token === generation && selection === selectionGeneration && operation === operationGeneration && taskId === selectedId.value) {
      busy.value = false
    }
  }
}
async function loadSelection(token = generation) {
  const read = ++selectionReadSequence
  const id = selectedId.value
  const selection = selectionGeneration
  const operation = operationGeneration
  if (!id) return
  const currentRead = () => !disposed && token === generation && selection === selectionGeneration
    && operation === operationGeneration && read === selectionReadSequence && id === selectedId.value
  let modelResponse: Awaited<ReturnType<typeof getTaskModelCompatibility>>
  let bundleResponse: Awaited<ReturnType<typeof getTaskBundles>>
  try {
    [modelResponse, bundleResponse] = await Promise.all([getTaskModelCompatibility(id), getTaskBundles(id)])
  } catch (cause: unknown) {
    if (currentRead()) throw cause
    return
  }
  if (!currentRead()) return
  models.value = modelResponse.documents; bundles.value = bundleResponse.documents
  modelIds.value = Object.fromEntries(Object.entries(modelIds.value).filter(([role, modelId]) =>
    models.value.some(model => text(model.model_id) === modelId && compatible(model, role))))
  if (!bundles.value.some(item => text(item.bundle_id) === bundleId.value)) bundleId.value = ''
}
async function refreshDeployment() {
  const token = generation; const selection = selectionGeneration; const id = selectedId.value
  emit('refresh')
  try { await loadSelection(token) } catch (cause: unknown) {
    if (!disposed && token === generation && selection === selectionGeneration && id === selectedId.value) {
      error.value = extractErrorMessage(cause, '任务状态已请求刷新，但执行包列表读取失败')
    }
  }
}
async function create() {
  await perform(async () => {
    const body = { camera_id: props.cameraId, template_id: templateId.value, name: name.value.trim(), config: objectJson(configuration.value) }
    if (!body.name) throw new Error('请输入任务名称')
    await createAnalysisTask(body, commandKey('create', body))
  }, '任务已创建；尚未运行或发布')
}
async function save() {
  const task = selected.value
  if (!task) return
  await perform(() => {
    if (editRevision.value === null || editRevision.value !== task.current_revision) throw new Error('任务配置已更新，请重新载入后编辑')
    const body = { expected_revision: editRevision.value, config: objectJson(configuration.value) }
    return updateAnalysisTask(task.task_id, body, commandKey(`update:${task.task_id}`, body))
  }, '新配置版本已保存；需重新冻结执行包')
}
async function freeze() {
  const task = selected.value
  if (!task) return
  await perform(() => {
    if (!spec.value || spec.value.template_id !== task.template_id) throw new Error('任务模板尚未核实，不能冻结执行包')
    if (modelRoles.value.some(role => !modelIds.value[role] || !models.value.some(model =>
      text(model.model_id) === modelIds.value[role] && compatible(model, role)))) throw new Error('请为每个模型角色选择已核实兼容的模型')
    const body = { expected_revision: task.current_revision, model_ids: Object.fromEntries(modelRoles.value.map(role => [role, modelIds.value[role]!])) }
    return freezeTaskBundle(task.task_id, body, commandKey(`bundle:${task.task_id}`, body))
  }, '执行包已冻结；仅可旁路运行，不代表生产批准')
}
async function registerModel() {
  await perform(() => {
    const body = { relative_path: relativePath.value, descriptor: objectJson(descriptor.value), scope_camera_id: props.cameraId }
    return registerTaskModel(body, commandKey('model', body))
  }, '模型已登记为候选；未加载、未发布')
}
async function showResults() {
  const id = selectedId.value
  const token = generation
  const selection = selectionGeneration
  await perform(async () => {
    const response = await getTaskResults(id)
    if (!disposed && token === generation && selection === selectionGeneration && id === selectedId.value) results.value = response.results
  }, '已读取最近的旁路结果')
}
watch(templateId, () => {
  if (!selectedId.value) configuration.value = JSON.stringify(spec.value?.default_config ?? {}, null, 2)
})
watch(selectedId, () => {
  selectionGeneration += 1
  operationGeneration += 1
  busy.value = false; error.value = ''; notice.value = ''
  models.value = []; bundles.value = []; results.value = []; modelIds.value = Object.fromEntries(modelRoles.value.map(role => [role, ''])); bundleId.value = ''
  configuration.value = JSON.stringify(selected.value?.config ?? spec.value?.default_config ?? {}, null, 2)
  editRevision.value = selected.value?.current_revision ?? null
  const id = selectedId.value; const token = generation; const selection = selectionGeneration
  void loadSelection(token).catch(cause => {
    if (!disposed && token === generation && selection === selectionGeneration && id === selectedId.value) error.value = extractErrorMessage(cause, '无法读取模型和执行包')
  })
})
watch(managed, (tasks) => {
  if (selectedId.value && !tasks.some(task => task.task_id === selectedId.value)) selectedId.value = ''
})
watch(() => props.cameraId, async () => {
  const token = ++generation
  operationGeneration += 1
  selectedId.value = ''; name.value = ''; busy.value = false; error.value = ''; notice.value = ''; commandKeys.clear()
  relativePath.value = ''; descriptor.value = defaultDescriptor; editRevision.value = null
  templates.value = []; configuration.value = '{}'
  models.value = []; bundles.value = []; results.value = []; modelIds.value = {}; bundleId.value = ''
  try {
    const response = await getAnalysisTaskTemplates()
    if (disposed || token !== generation) return
    templates.value = response.templates.filter(item => item.supports_creation)
    configuration.value = JSON.stringify(spec.value?.default_config ?? {}, null, 2)
  } catch (cause: unknown) {
    if (!disposed && token === generation) error.value = extractErrorMessage(cause, '无法读取任务模板')
  }
}, { immediate: true })
onUnmounted(() => { disposed = true; generation += 1 })
</script>

<template>
  <section class="task-workbench" aria-label="多任务工作台" :aria-busy="busy">
    <h3>多任务工作台 · 旁路模式</h3>
    <p class="note">新任务独立运行，不触发生产告警。CPU/CUDA 必须显式选择；没有对应运行依赖或资源时会阻断，不自动降级。</p>
    <p v-if="error" role="alert" class="error">{{ error }}</p>
    <p v-if="notice" role="status">{{ notice }}</p>
    <label>选择任务<select v-model="selectedId" data-test="task-select" :disabled="busy"><option value="">新建任务</option><option v-for="task in managed" :key="task.task_id" :value="task.task_id">{{ task.name || task.template_id }} · r{{ task.current_revision }}</option></select></label>
    <template v-if="!selected">
      <label>功能模板<select v-model="templateId" data-test="task-template" :disabled="busy"><option v-for="template in templates" :key="template.template_id" :value="template.template_id">{{ template.name }}</option></select></label>
      <label>任务名称<input v-model="name" data-test="task-name" maxlength="200" :disabled="busy" /></label>
    </template>
    <details><summary>任务配置（严格校验；修改前须停止任务）</summary><textarea v-model="configuration" aria-label="任务配置 JSON" rows="12" :disabled="busy" spellcheck="false" /></details>
    <p v-if="configurationStale" role="status" class="note">任务版本已变化，当前编辑内容尚未覆盖新配置。请先保存本地草稿，再重新载入。</p>
    <button v-if="configurationStale" :disabled="busy" @click="configuration = JSON.stringify(selected?.config ?? {}, null, 2); editRevision = selected?.current_revision ?? null">重新载入当前配置</button>
    <button v-if="!selected" data-test="create-task" :disabled="busy || !templates.length" @click="create">创建旁路任务</button>
    <button v-else data-test="save-task" :disabled="busy || configurationStale" @click="save">保存新配置版本</button>
    <template v-if="selected">
      <fieldset v-for="role in modelRoles" :key="role" :disabled="busy" class="role-selection">
        <legend>{{ roleLabel(role) }} · {{ role }}</legend>
        <label>角色模型<select v-model="modelIds[role]" :data-test="`model-role-${role}`"><option value="">请选择已登记兼容模型</option><option v-for="model in models" :key="text(model.model_id)" :value="text(model.model_id)" :disabled="!compatible(model, role)">{{ text(model.name) || text(model.model_id) }}{{ compatible(model, role) ? '' : ` — ${incompatibility(model, role)}` }}</option></select></label>
        <p class="note">适配器：{{ spec?.model_roles?.[role] }}；必需类别：{{ requiredLabels(role) }}。所有角色完成校验后才可冻结并整体部署。</p>
      </fieldset>
      <p v-if="!modelRoles.length" class="note">此任务使用固定算法，不需要模型权重。</p>
      <button data-test="freeze-bundle" :disabled="busy || !spec" @click="freeze">冻结当前执行包</button>
      <label>执行包<select v-model="bundleId" data-test="bundle-select" :disabled="busy"><option value="">请选择当前配置的执行包</option><option v-for="bundle in bundles" :key="text(bundle.bundle_id)" :value="text(bundle.bundle_id)">r{{ bundle.task_revision }} · {{ text(bundle.bundle_id).slice(0, 12) }}</option></select></label>
      <div class="actions">
        <button :disabled="busy" @click="showResults">查看旁路结果</button>
        <button disabled title="缺少任务专属评估、签名和发布协议，保持 NO-GO">生产发布：NO-GO</button>
      </div>
      <pre v-if="results.length" aria-label="最近旁路结果">{{ JSON.stringify(results, null, 2) }}</pre>
      <TaskDeploymentPanel :key="selected.task_id" :task="selected" :bundles="bundles" :bundle-id="bundleId" @refresh="refreshDeployment" />
      <TaskDatasetPanel :key="selected.task_id" :task="selected" />
      <details><summary>归档此任务</summary><p class="note">归档前必须停止并完成清理。配置、采集和审核历史将保留。</p><button :disabled="busy" @click="perform(() => archiveAnalysisTask(selectedId, selected!.current_revision, commandKey('archive', { task_id: selectedId, revision: selected!.current_revision })), '任务已归档', false)">确认归档</button></details>
    </template>
    <details><summary>登记本机受控模型（需人工管理权限）</summary><p class="note">只登记 models 目录内已有 ONNX 文件，不上传、不下载、不执行。描述必须与导出模型一致。</p><label>模型相对路径<input v-model="relativePath" placeholder="models 目录内的相对路径" :disabled="busy" /></label><label>模型描述 JSON<textarea v-model="descriptor" rows="6" :disabled="busy" spellcheck="false" /></label><button :disabled="busy" @click="registerModel">核验 SHA 并登记候选</button></details>
  </section>
</template>

<style scoped>
.task-workbench { margin: 16px 0; padding: 16px; background: var(--bg); border: 1px solid var(--border-default); border-radius: var(--r-lg); }
h3 { margin: 0 0 10px; font-size: 14px; } .note { color: var(--text-secondary); font-size: 12px; line-height: 1.6; } .error { color: var(--status-danger); }
label { display: grid; gap: 5px; margin: 10px 0; font-size: 12px; } input, select, textarea, button { color: var(--text-primary); background: var(--bg); border: 1px solid var(--border-default); border-radius: 6px; padding: 8px; }
textarea { width: 100%; box-sizing: border-box; font-family: ui-monospace, monospace; } button { cursor: pointer; margin: 4px 6px 4px 0; } button:disabled { cursor: not-allowed; opacity: .6; }
input:focus-visible, select:focus-visible, textarea:focus-visible, button:focus-visible, summary:focus-visible { outline: 2px solid var(--status-info); outline-offset: 2px; }
details { margin: 12px 0; } summary { cursor: pointer; } pre { max-height: 300px; overflow: auto; font-size: 12px; } .actions { display: flex; flex-wrap: wrap; }
</style>
