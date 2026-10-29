<script setup lang="ts">
import { computed, nextTick, onUnmounted, ref, watch } from 'vue'
import { getActivePinia } from 'pinia'
import { useAuthStore } from '../../stores/useAuthStore'
import { getAnalysisTasks, type AnalysisTask } from '../../api/analysisTasks'
import { extractErrorMessage, toApiError } from '../../utils/error'
import TaskWorkbench from './TaskWorkbench.vue'
import CameraSourceControls from './CameraSourceControls.vue'
import TaskLifecyclePanel from './TaskLifecyclePanel.vue'
import TaskGuidanceCard from './TaskGuidanceCard.vue'

const props = defineProps<{ cameraId: string; canManage?: boolean; requestedTaskId?: string; requestedSection?: string; navigationInvalid?: boolean }>()
const emit = defineEmits<{ clearTaskNavigation: [] }>()
const tasks = ref<AnalysisTask[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const loaded = ref(false)
const taskAccessDenied = ref(false)
const eventReviewTaskId = ref('')
const sourceSection = ref<HTMLElement | null>(null)
const taskCards = ref<HTMLElement[]>([])
const pinia = getActivePinia()
const auth = pinia ? useAuthStore(pinia) : null
const authenticationIdentity = computed(() => JSON.stringify([auth?.currentUser?.username, auth?.currentUser?.role]))
const identityReady = computed(() => !auth || !!auth.currentUser?.username)
const canReviewTaskEvents = computed(() => !!auth?.currentUser?.username && auth.currentUser.role === 'operator')
let generation = 0
let disposed = false
let timer: ReturnType<typeof setTimeout> | null = null
let handledNavigation = ''
const missingRequestedTask = computed(() => loaded.value && !error.value && !!props.requestedTaskId
  && !tasks.value.some(task => task.task_id === props.requestedTaskId))
function focusSource() {
  sourceSection.value?.focus({ preventScroll: true })
  sourceSection.value?.scrollIntoView({ block: 'start' })
}
watch(() => [props.requestedTaskId, props.requestedSection, props.cameraId, tasks.value, props.canManage, props.navigationInvalid] as const, async () => {
  if (props.navigationInvalid) { handledNavigation = ''; eventReviewTaskId.value = ''; return }
  const task = tasks.value.find(item => item.task_id === props.requestedTaskId)
  const identity = JSON.stringify([props.cameraId, props.requestedTaskId, props.requestedSection, props.canManage])
  if (!task || (props.canManage && task.authority === 'managed' && task.lifecycle !== 'archived')) {
    handledNavigation = ''
    return
  }
  if (handledNavigation === identity) return
  handledNavigation = identity
  await nextTick()
  if (disposed || props.navigationInvalid || handledNavigation !== identity
    || identity !== JSON.stringify([props.cameraId, props.requestedTaskId, props.requestedSection, props.canManage])) return
  if (props.requestedSection === 'source') { focusSource(); return }
  const element = taskCards.value.find(card => card.dataset.taskId === task.task_id)
  element?.focus({ preventScroll: true }); element?.scrollIntoView({ block: 'start' })
})

const states: Record<string, string> = {
  unknown: '未知', stopped: '已停止', starting: '启动中', running: '运行中',
  stopping: '停止中', failed: '故障', blocked: '已阻断',
  waiting_source: '等待有效源帧', sampling_insufficient: '采样不足',
  cleanup_pending: '等待清理，重启受阻', runtime_dead: '运行线程已退出', cleanup_failed: '清理失败',
  draining: '正在排空旧运行实例',
}
const reasons: Record<string, string> = {
  WAITING_SOURCE: '等待摄像头源恢复', SAMPLING_INSUFFICIENT: '有序帧窗口丢帧',
  CLEANUP_PENDING: '资源尚未确认释放', RUNTIME_DEAD: '需确认清理后重启',
  CLEANUP_FAILED: '清理失败，需重试', RUNTIME_FAILED: '运行失败',
  SCENE_BLOCKED: '场景安全门未放行', MANUAL_STOP: '手动停止',
  DEPLOYMENT_UNAVAILABLE: '部署不可用',
  DRAINING: '旧运行实例尚未退出',
  STALE_TASK: '配置已变化，需停止并重新冻结执行包', RESOURCE_EXHAUSTED: '运行资源不足',
  LOAD_TIMEOUT: '模型加载超时', INFER_TIMEOUT: '模型执行超时', UNAVAILABLE: '运行依赖或设备不可用',
  INVALID_OUTPUT: '模型输出不符合约定', WORKER_FAILED: '独立任务进程故障',
  INVALID_BUNDLE: '执行包校验失败', FRAME_TOO_LARGE: '帧超出任务内存预算',
  LEGACY_NOT_STARTED: '尚未启动异物检测，摄像头源可独立使用',
  LEGACY_INITIALIZATION_FAILED: '异物检测初始化失败，不代表摄像头源不可用',
}
const projectionStates: Record<string, string> = {
  not_synced: '尚未同步', unbound: '未绑定', blocked: '未通过现有发布校验',
  authorized: '注册授权已核验（不代表运行时已加载）',
  stale: '注册记录已变化，快照过期',
  shadow_unconfigured: '尚未配置旁路执行包',
}

function stateLabel(state?: string): string {
  return state ? (states[state] ?? '未知') : '未知'
}

function cancelTimer() {
  if (timer !== null) clearTimeout(timer)
  timer = null
}

async function load() {
  cancelTimer()
  const requestGeneration = ++generation
  const cameraId = props.cameraId
  const requestIdentity = authenticationIdentity.value
  if (!cameraId || !identityReady.value) {
    tasks.value = []
    loaded.value = false
    loading.value = false
    taskAccessDenied.value = !identityReady.value
    error.value = identityReady.value ? null : '当前个人身份尚未就绪，无法读取分析任务'
    return
  }
  loading.value = true
  try {
    const response = await getAnalysisTasks(cameraId)
    if (disposed || generation !== requestGeneration || props.cameraId !== cameraId
      || authenticationIdentity.value !== requestIdentity) return
    if (response.tasks.some(task => task.camera_id !== cameraId)
      || new Set(response.tasks.map(task => task.task_id)).size !== response.tasks.length) throw new Error('任务列表与当前相机不匹配')
    tasks.value = response.tasks
    error.value = null
    loaded.value = true
    taskAccessDenied.value = false
  } catch (cause: unknown) {
    if (disposed || generation !== requestGeneration || props.cameraId !== cameraId
      || authenticationIdentity.value !== requestIdentity) return
    error.value = extractErrorMessage(cause, '无法读取分析任务')
    const apiError = toApiError(cause)
    if ([401, 403].includes(apiError.httpStatus ?? 0) || [40100, 40300].includes(apiError.businessCode ?? 0)) {
      tasks.value = []
      loaded.value = false
      taskAccessDenied.value = true
      eventReviewTaskId.value = ''
    }
  } finally {
    if (!disposed && generation === requestGeneration && props.cameraId === cameraId
      && authenticationIdentity.value === requestIdentity) {
      loading.value = false
      // REST reconciliation remains active even if the surrounding page has WS.
      timer = setTimeout(() => { void load() }, 60_000)
    }
  }
}

watch(() => [props.cameraId, authenticationIdentity.value] as const, () => {
  handledNavigation = ''
  eventReviewTaskId.value = ''
  tasks.value = []
  error.value = null
  loaded.value = false
  taskAccessDenied.value = !identityReady.value
  void load()
}, { immediate: true, flush: 'sync' })

onUnmounted(() => {
  disposed = true
  generation += 1
  cancelTimer()
})
</script>

<template>
  <section class="analysis-tasks" aria-labelledby="analysis-tasks-title" :aria-busy="loading">
    <header>
      <h2 id="analysis-tasks-title">分析任务</h2>
      <button type="button" :disabled="loading" @click="load">{{ loading ? '刷新中…' : '刷新任务' }}</button>
    </header>
    <p class="scope-note">保留现有异物检测与发布安全门。新任务仅支持旁路运行；生产发布保持 NO-GO。legacy 导入快照不是当前生效配置，managed 配置需冻结为执行包后启动。</p>
    <div ref="sourceSection" tabindex="-1" data-test="task-section-source"><CameraSourceControls :camera-id="cameraId" :can-manage="canManage" @refresh="load" /></div>
    <p v-if="missingRequestedTask" role="alert">指定任务不存在或不属于当前摄像头；未自动选择其它任务。</p>
    <p v-if="navigationInvalid" role="alert">任务定位参数无效，未自动选择其它任务。<button @click="emit('clearTaskNavigation')">清除无效定位</button></p>
    <TaskWorkbench v-if="canManage && !taskAccessDenied" :key="authenticationIdentity" :camera-id="cameraId" :tasks="tasks" :initial-task-id="requestedTaskId" :initial-section="requestedSection" :navigation-invalid="navigationInvalid" :task-list-ready="loaded" :stale="!!error" @refresh="load" @open-source="focusSource" @clear-navigation="emit('clearTaskNavigation')" />
    <p v-if="error" class="error" role="alert" data-test="task-error">
      {{ error }}<span v-if="loaded">；下列为上次成功读取的状态，可能已过期。</span>
    </p>
    <p v-else-if="loading && !loaded" role="status">正在加载分析任务…</p>
    <p v-else-if="loaded && tasks.length === 0" data-test="task-empty">尚未登记分析任务；不代表摄像头已经停止。</p>
    <article v-for="task in tasks" :key="task.task_id" ref="taskCards" tabindex="-1" :data-task-id="task.task_id" class="task-card" data-test="analysis-task">
      <h3>{{ task.template_id === 'legacy_anomaly.v1' ? '现有异物检测' : task.name || task.template_id }}<span v-if="task.lifecycle === 'archived'"> · 已归档</span></h3>
      <TaskGuidanceCard :task="task" :stale="!!error" />
      <p v-if="task.runtime?.event_error" class="error" role="alert" data-test="task-event-error">事件证据通道受阻（{{ task.runtime.event_error }}），部分事件可能缺失；推理与预览状态独立显示。</p>
      <dl>
        <div><dt>任务身份</dt><dd class="mono">{{ task.task_id }}</dd></div>
        <div><dt>模板 / 导入快照版本</dt><dd>{{ task.template_id }} / r{{ task.current_revision }}</dd></div>
        <div><dt>任务运行</dt><dd>{{ stateLabel(task.runtime?.state) }}<span v-if="task.runtime?.reason_code && reasons[task.runtime.reason_code]"> · {{ reasons[task.runtime.reason_code] }}</span></dd></div>
        <div v-if="task.authority === 'managed'"><dt>实际配置版本 / 模式</dt><dd>{{ task.runtime?.applied_task_revision ? `r${task.runtime.applied_task_revision}` : '未确认' }} / {{ task.runtime?.mode === 'shadow' ? '旁路' : '未知' }}</dd></div>
        <div><dt>摄像头源</dt><dd>{{ stateLabel(task.source?.state) }} · {{ task.source?.connected === true ? '已连接' : task.source?.connected === false ? '未连接' : '连接状态未知' }}<span v-if="task.source?.reason_code && reasons[task.source.reason_code]"> · {{ reasons[task.source.reason_code] }}</span></dd></div>
        <div><dt>部署管理权</dt><dd>{{ task.deployment?.authority === 'legacy' ? '现有受控发布服务（兼容）' : task.deployment?.authority ?? '尚未登记' }}</dd></div>
        <template v-if="task.authority === 'managed'">
          <div><dt>部署意图 / 版本</dt><dd>{{ task.deployment?.desired_state ?? '未知' }} / {{ task.deployment?.deployment_revision ?? '未知' }}</dd></div>
          <div><dt>期望执行包</dt><dd class="mono">{{ task.deployment?.desired_bundle_id ?? '未记录' }}</dd></div>
          <div><dt>已应用执行包（不代表当前运行）</dt><dd class="mono">{{ task.deployment?.applied_bundle_id ?? '未确认' }}</dd></div>
        </template>
        <template v-else>
          <div><dt>注册授权快照</dt><dd>{{ projectionStates[task.deployment?.projection_status ?? 'not_synced'] ?? '未知' }}<span v-if="task.deployment?.observed_at"> · {{ task.deployment.observed_at }}</span></dd></div>
          <div><dt>期望模型制品</dt><dd class="mono">{{ task.deployment?.desired_artifact_id ?? '未记录' }}</dd></div>
          <div><dt>已应用部署</dt><dd class="mono">{{ task.deployment?.applied_artifact_id ?? '未确认（不能由注册状态推断）' }}</dd></div>
        </template>
        <div><dt>运行时模型观测</dt><dd>{{ task.runtime?.model_version ?? '未知' }} · {{ task.runtime?.model_identity_verified === true ? '制品身份已核验，不代表场景已放行' : '制品身份未确认' }}</dd></div>
      </dl>
      <template v-if="!navigationInvalid && canReviewTaskEvents && task.template_id === 'object_detection.v1'">
        <button data-test="open-task-event-review" @click="eventReviewTaskId = eventReviewTaskId === task.task_id ? '' : task.task_id">{{ eventReviewTaskId === task.task_id ? '收起任务事件复核' : '查看与复核任务事件' }}</button>
        <TaskLifecyclePanel v-if="eventReviewTaskId === task.task_id" :key="task.task_id" :task="task" />
      </template>
    </article>
  </section>
</template>

<style scoped>
.analysis-tasks { margin-top: 12px; padding: 16px; border: 1px solid var(--border-default); border-radius: var(--r-lg); background: var(--bg); color: var(--text-primary); }
header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
h2, h3 { margin: 0; font-size: 14px; }
h3 { margin-bottom: 10px; }
button { border: 1px solid var(--border-default); border-radius: 6px; padding: 6px 10px; background: var(--bg); color: var(--text-primary); cursor: pointer; }
button:disabled { cursor: wait; }
button:focus-visible { outline: 2px solid var(--status-info); outline-offset: 2px; }
[tabindex="-1"]:focus { outline: 2px solid var(--status-info); outline-offset: 4px; }
.scope-note, [role="status"], [data-test="task-empty"] { font-size: 12px; color: var(--text-secondary); line-height: 1.6; }
.error { font-size: 12px; color: var(--status-danger); line-height: 1.6; }
.task-card { border-top: 1px solid var(--border-default); padding-top: 12px; margin-top: 12px; }
dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px 20px; margin: 0; font-size: 12px; }
dt { color: var(--text-secondary); margin-bottom: 3px; }
dd { margin: 0; overflow-wrap: anywhere; }
.mono { font-family: ui-monospace, monospace; }
@media (max-width: 767px) { .analysis-tasks { padding: 12px; } dl { grid-template-columns: 1fr; } }
</style>
