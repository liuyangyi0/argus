<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import type { AnalysisTask } from '../../api/analysisTasks'
import {
  getTaskDeployment, getTaskDeploymentOperations, requestTaskDeployment, reconcileTaskDeployment,
  restoreTaskConfiguration,
} from '../../api/taskDeployments'
import type { TaskConfigurationRestoreRequest, TaskDeploymentRequest } from '../../types/generated-openapi'
import { extractErrorMessage, toApiError } from '../../utils/error'

const props = defineProps<{ task: AnalysisTask; bundles: Record<string, unknown>[]; bundleId: string }>()
const emit = defineEmits<{ refresh: [] }>()
const deployment = ref<Record<string, unknown> | null>(null)
const operations = ref<Record<string, unknown>[]>([])
const loading = ref(false)
const sending = ref(false)
const activeAction = ref<TaskDeploymentRequest['action'] | 'reconcile' | 'restore' | null>(null)
const error = ref('')
const notice = ref('')
const retry = ref<{ body: TaskDeploymentRequest; key: string } | null>(null)
const restoreRetry = ref<{ body: TaskConfigurationRestoreRequest; key: string } | null>(null)
let generation = 0
let readSequence = 0
let commandSequence = 0
let disposed = false
let timer: ReturnType<typeof setTimeout> | undefined
const revision = computed(() => {
  const value = deployment.value?.deployment_revision
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null
})
const selectedBundle = computed(() => props.bundles.find(item => item.bundle_id === props.bundleId))
const compatibleBundle = computed(() => !!selectedBundle.value && selectedBundle.value.task_revision === props.task.current_revision)
const state = computed(() => String(deployment.value?.state ?? deployment.value?.runtime_status ?? 'unknown'))
const stoppedReceipt = computed(() => {
  const operation = object(deployment.value?.operation)
  return deployment.value?.desired_state === 'stopped' && state.value === 'stopped'
    && revision.value !== null && deployment.value?.applied_deployment_revision === revision.value
    && operation.task_id === props.task.task_id && operation.action === 'stop'
    && operation.status === 'applied' && operation.deployment_revision === revision.value
})
const appliedTarget = computed(() => {
  const bundle = selectedBundle.value
  if (!bundle || bundle.task_id !== props.task.task_id) return false
  // The authoritative annotation scans all receipts, not just the latest
  // 100 shown in the history panel. An explicit false must beat stale views.
  if (typeof bundle.historically_applied === 'boolean') return bundle.historically_applied
  return deployment.value?.applied_bundle_id === props.bundleId || deployment.value?.previous_bundle_id === props.bundleId
    || operations.value.some(operation => operation.task_id === props.task.task_id
      && operation.bundle_id === props.bundleId && operation.status === 'applied')
})
const canRestore = computed(() => stoppedReceipt.value && appliedTarget.value)
const labels: Record<string, string> = {
  unknown: '未知', stopped: '已停止', running: '运行中', pending: '等待执行', applying: '准备及切换中',
  starting: '启动中', failed: '失败', waiting_source: '等待摄像头源', blocked: '条件受阻',
  preparing: '准备中', recovering: '恢复对账中',
  waiting_scene: '等待场景安全门',
  reconcile_required: '结果待对账', cleanup_pending: '清理未确认', applied: '已核验应用',
}
function current(token: number, taskId: string) { return !disposed && token === generation && taskId === props.task.task_id }
function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}
function schedule() {
  if (timer) clearTimeout(timer)
  timer = setTimeout(() => { void refresh() }, retry.value || restoreRetry.value || !['running', 'stopped', 'failed', 'applied'].includes(state.value) ? 2000 : 60_000)
}
function invalidateReads() {
  readSequence += 1; loading.value = false
  if (timer) clearTimeout(timer)
}
function checkedView(document: Record<string, unknown>, taskId: string) {
  if (document.task_id !== taskId || typeof document.deployment_revision !== 'number'
      || !Number.isInteger(document.deployment_revision) || document.deployment_revision < 0) {
    throw new Error('部署响应身份或版本无效，不能更新当前任务')
  }
  return document
}
function definitelyRejected(cause: unknown) {
  const failure = toApiError(cause)
  const status = failure.httpStatus !== null && failure.httpStatus >= 400 ? failure.httpStatus
    : failure.businessCode !== null && failure.businessCode >= 40000 && failure.businessCode < 50000 ? Math.floor(failure.businessCode / 100) : null
  return status !== null && status >= 400 && status < 500 && ![408, 429].includes(status)
}
async function refresh() {
  if (timer) clearTimeout(timer)
  const token = generation; const taskId = props.task.task_id; const sequence = ++readSequence
  loading.value = true
  try {
    const [view, history] = await Promise.all([getTaskDeployment(taskId), getTaskDeploymentOperations(taskId)])
    if (!current(token, taskId) || sequence !== readSequence) return
    const document = checkedView(view.document, taskId)
    if (history.documents.some(operation => operation.task_id !== taskId)) throw new Error('部署操作记录与当前任务不匹配')
    deployment.value = document; operations.value = history.documents
    error.value = ''
  } catch (cause: unknown) {
    if (current(token, taskId) && sequence === readSequence) error.value = extractErrorMessage(cause, '无法读取部署；显示状态可能过期')
  } finally {
    if (current(token, taskId) && sequence === readSequence) {
      loading.value = false
      schedule()
    }
  }
}
async function send(action: TaskDeploymentRequest['action'], repeat = false) {
  if ((revision.value === null && action !== 'stop' && !repeat)
      || (sending.value && (action !== 'stop' || activeAction.value === 'stop' || activeAction.value === 'restore'))) return
  if (!repeat && retry.value && action !== 'stop') return
  if (restoreRetry.value && action !== 'stop') return
  const original = repeat ? retry.value : null
  if (repeat && !original) return
  let body: TaskDeploymentRequest | null = original?.body ?? null
  const key = original?.key ?? crypto.randomUUID()
  const token = generation; const taskId = props.task.task_id; const command = ++commandSequence
  invalidateReads()
  sending.value = true; activeAction.value = action; error.value = ''; notice.value = ''
  let attempted = false
  try {
    if (body === null) {
      let expected = revision.value
      if (action === 'stop') {
        // An earlier POST can already have committed even while its HTTP
        // response is pending. Read the authoritative CAS before superseding.
        const view = await getTaskDeployment(taskId)
        if (!current(token, taskId) || command !== commandSequence) return
        deployment.value = checkedView(view.document, taskId)
        expected = revision.value
      }
      if (expected === null) throw new Error('部署版本未确认，停止请求尚未登记，请刷新后重试')
      body = { action, expected_deployment_revision: expected, bundle_id: action === 'stop' ? null : props.bundleId }
    }
    // Preserve ownership before awaiting HTTP. A newer stop can supersede
    // this callback, but a failed stop preflight must not lose its exact key.
    retry.value = { body, key }
    attempted = true
    const result = await requestTaskDeployment(taskId, body, key)
    if (!current(token, taskId) || command !== commandSequence) return
    if (result.document.task_id !== taskId || typeof result.document.operation_id !== 'string'
        || !result.document.operation_id) throw new Error('部署回执身份无效，结果仍需按原请求对账')
    retry.value = null
    notice.value = `部署请求已登记，尚不代表已应用：${String(result.document.operation_id ?? '')}`
    await refresh()
    if (current(token, taskId) && command === commandSequence) emit('refresh')
  } catch (cause: unknown) {
    if (!current(token, taskId) || command !== commandSequence) return
    const rejected = attempted && definitelyRejected(cause)
    if (attempted && body !== null) retry.value = rejected ? null : { body, key }
    const message = (!attempted && action === 'stop' ? '停止请求尚未登记，当前运行状态仍需确认：' : '')
      + extractErrorMessage(cause, '请求结果未确认；请使用原请求重试或先对账')
    if (rejected) {
      await refresh()
      if (!current(token, taskId) || command !== commandSequence) return
    }
    error.value = message
  } finally {
    if (current(token, taskId) && command === commandSequence) {
      sending.value = false; activeAction.value = null; schedule()
    }
  }
}
async function restoreConfiguration(repeat = false) {
  if (sending.value || retry.value || (!repeat && (restoreRetry.value || !canRestore.value))) return
  const original = repeat ? restoreRetry.value : null
  if (repeat && !original) return
  const body: TaskConfigurationRestoreRequest = original?.body ?? {
    bundle_id: props.bundleId, expected_revision: props.task.current_revision,
  }
  const key = original?.key ?? crypto.randomUUID()
  const token = generation; const taskId = props.task.task_id; const command = ++commandSequence
  invalidateReads()
  sending.value = true; activeAction.value = 'restore'; error.value = ''; notice.value = ''
  restoreRetry.value = { body, key }
  try {
    const result = await restoreTaskConfiguration(taskId, body, key)
    if (!current(token, taskId) || command !== commandSequence) return
    const restoredTask = object(result.document.task)
    const bundle = object(result.document.bundle)
    const receipt = object(result.document.restore)
    if (restoredTask.task_id !== taskId || bundle.task_id !== taskId
        || typeof bundle.bundle_id !== 'string' || !bundle.bundle_id
        || restoredTask.current_revision !== body.expected_revision + 1
        || bundle.task_revision !== restoredTask.current_revision
        || receipt.new_task_revision !== restoredTask.current_revision
        || receipt.prior_task_revision !== body.expected_revision
        || receipt.restored_from_bundle_id !== body.bundle_id || receipt.production_allowed !== false) {
      throw new Error('配置恢复回执身份无效，请以原请求重试对账')
    }
    restoreRetry.value = null
    notice.value = `配置恢复已确认：新版本 r${String(restoredTask.current_revision)}，新执行包 ${bundle.bundle_id}。未启动、未部署，请刷新后显式选择执行包。`
    emit('refresh')
    await refresh()
  } catch (cause: unknown) {
    if (!current(token, taskId) || command !== commandSequence) return
    const rejected = definitelyRejected(cause)
    if (rejected) {
      restoreRetry.value = null
      emit('refresh')
      await refresh()
      if (!current(token, taskId) || command !== commandSequence) return
    }
    error.value = extractErrorMessage(cause, '配置恢复结果未确认，请以原版本和幂等键重试')
  } finally {
    if (current(token, taskId) && command === commandSequence) {
      sending.value = false; activeAction.value = null; schedule()
    }
  }
}
async function reconcile() {
  if (sending.value) return
  const token = generation; const taskId = props.task.task_id; const command = ++commandSequence
  invalidateReads()
  sending.value = true; activeAction.value = 'reconcile'; error.value = ''
  try {
    await reconcileTaskDeployment(taskId)
    if (!current(token, taskId) || command !== commandSequence) return
    notice.value = '已请求对账；请以期望状态和已应用身份为准'
    await refresh()
    if (current(token, taskId) && command === commandSequence) emit('refresh')
  } catch (cause: unknown) {
    if (current(token, taskId) && command === commandSequence) error.value = extractErrorMessage(cause, '对账失败，未确认的操作不能当作成功')
  } finally {
    if (current(token, taskId) && command === commandSequence) {
      sending.value = false; activeAction.value = null; schedule()
    }
  }
}
watch(() => props.task.task_id, () => {
  generation += 1; commandSequence += 1; readSequence += 1
  deployment.value = null; operations.value = []; retry.value = null; restoreRetry.value = null
  error.value = ''; notice.value = ''; sending.value = false; activeAction.value = null
  void refresh()
}, { immediate: true })
onUnmounted(() => { disposed = true; generation += 1; if (timer) clearTimeout(timer) })
</script>

<template>
  <section class="task-deployment" aria-label="任务持久化部署" :aria-busy="loading || sending">
    <h4>任务部署 · 仅旁路</h4>
    <p class="note">登记请求不等于加载成功。源恢复后按持久化意图对账，不会自动开启摄像头源；生产仍为 NO-GO。</p>
    <p v-if="error" role="alert">{{ error }}</p><p v-if="notice" role="status">{{ notice }}</p>
    <p v-if="retry" role="status">有请求结果尚未确认；保留原版本、请求内容和幂等键。新部署或回滚暂不可提交，可重试对账或明确停止。</p>
    <p v-if="restoreRetry" role="status">配置恢复结果尚未确认，不能按新版本重新提交；请用保存的原请求重试对账。</p>
    <dl v-if="deployment">
      <div><dt>部署版本 / 状态</dt><dd>{{ revision ?? '未知' }} / {{ labels[state] ?? state }}</dd></div>
      <div><dt>期望状态</dt><dd>{{ deployment.desired_state ?? '未知' }}</dd></div>
      <div><dt>期望执行包</dt><dd>{{ deployment.desired_bundle_id ?? '未配置' }}</dd></div>
      <div><dt>已应用执行包（停止后保留历史身份）</dt><dd>{{ deployment.applied_bundle_id ?? '未确认' }}</dd></div>
      <div v-if="deployment.error_code"><dt>受阻原因</dt><dd>{{ deployment.error_code }}</dd></div>
    </dl>
    <p v-if="bundleId && !compatibleBundle" role="status">此执行包不属于当前配置版本；先显式恢复配置并重新冻结，不能绕过配置权威。</p>
    <div class="actions">
      <button data-test="deploy-task" :disabled="sending || !!retry || !!restoreRetry || revision === null || !compatibleBundle" @click="send('deploy')">部署所选执行包</button>
      <button data-test="stop-task" :disabled="sending && (activeAction === 'stop' || activeAction === 'restore')" @click="send('stop')">持久化停止 / 重试清理</button>
      <button data-test="rollback-task" :disabled="sending || !!retry || !!restoreRetry || revision === null || !compatibleBundle" @click="send('rollback')">回滚至所选执行包</button>
      <button :disabled="sending" @click="reconcile">对账运行状态</button>
      <button :disabled="sending || loading" @click="refresh">刷新部署</button>
      <button v-if="retry" data-test="retry-deployment" :disabled="sending" @click="send(retry.body.action, true)">以原版本和幂等键重试</button>
      <button data-test="restore-configuration" :disabled="sending || loading || !!retry || !!restoreRetry || !canRestore" @click="restoreConfiguration()">停止后恢复所选历史配置为新版本</button>
      <button v-if="restoreRetry" data-test="retry-configuration-restore" :disabled="sending" @click="restoreConfiguration(true)">以原请求重试配置恢复</button>
    </div>
    <p class="note">跨配置恢复要求已确认的持久化停止和目标历史应用记录；服务端重新核验全部模型与来源，创建新配置版本和新执行包，不自动部署。旧执行包和旧审批不被改写。</p>
    <details v-if="operations.length"><summary>最近部署操作</summary><ul><li v-for="operation in operations" :key="String(operation.operation_id)">{{ operation.operation_id }} · {{ operation.action }} · {{ operation.status ?? operation.state }}<span v-if="operation.error_code"> · {{ operation.error_code }}</span></li></ul></details>
  </section>
</template>

<style scoped>
.task-deployment { border-top: 1px solid var(--border-default); margin-top: 12px; padding-top: 12px; }
h4 { margin: 0; } .note, dt { color: var(--text-secondary); font-size: 12px; } [role="alert"] { color: var(--status-danger); }
dl { display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 8px; font-size: 12px; } dd { margin: 3px 0; overflow-wrap: anywhere; }
.actions { display: flex; flex-wrap: wrap; gap: 6px; } button { background: var(--bg); color: var(--text-primary); border: 1px solid var(--border-default); border-radius: 6px; padding: 8px; cursor: pointer; } button:disabled { opacity: .6; cursor: not-allowed; } button:focus-visible, summary:focus-visible { outline: 2px solid var(--status-info); outline-offset: 2px; } li { overflow-wrap: anywhere; font-size: 12px; }
</style>
