<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { getAnalysisTasks, type AnalysisTask } from '../api/analysisTasks'
import type { TaskGuidanceResponse } from '../types/generated-openapi'
import { useAuthStore } from '../stores/useAuthStore'
import { useWebSocket } from '../composables/useWebSocket'
import { extractErrorMessage, toApiError } from '../utils/error'
import PageShell from '../components/common/PageShell.vue'
import AsyncState from '../components/common/AsyncState.vue'
import TaskGuidanceCard from '../components/cameras/TaskGuidanceCard.vue'

const router = useRouter()
const auth = useAuthStore()
const tasks = ref<AnalysisTask[]>([])
const loading = ref(false)
const loaded = ref(false)
const error = ref('')
const forbidden = ref(false)
const accessDenied = ref(false)
const cameraId = ref('')
const search = ref('')
const showArchived = ref(false)
const updatedAt = ref('')
let sequence = 0
let disposed = false
const identity = computed(() => auth.currentUser ? JSON.stringify([auth.currentUser.username, auth.currentUser.role]) : null)
const canManage = computed(() => !accessDenied.value && auth.hasRole(['admin', 'engineer']))
const cameras = computed(() => [...new Set(tasks.value.map(task => task.camera_id))].sort())
const visible = computed(() => tasks.value.filter(task =>
  (!cameraId.value || task.camera_id === cameraId.value)
  && (showArchived.value || task.lifecycle !== 'archived')
  && [task.name, task.template_id, task.task_id, task.camera_id].join(' ').toLowerCase().includes(search.value.trim().toLowerCase())))

async function load() {
  const owner = identity.value
  if (owner === null) { error.value = '登录身份尚未核实，请重新登录后读取任务'; return }
  const read = ++sequence
  const current = () => !disposed && read === sequence && owner === identity.value
  loading.value = true
  try {
    const response = await getAnalysisTasks()
    if (!current()) return
    if (response.tasks.some(task => !task.task_id || !task.camera_id)
        || new Set(response.tasks.map(task => task.task_id)).size !== response.tasks.length) {
      throw new Error('任务列表身份无效，请重试')
    }
    tasks.value = response.tasks
    error.value = ''; forbidden.value = false; accessDenied.value = false; loaded.value = true
    updatedAt.value = new Date().toLocaleTimeString()
  } catch (cause: unknown) {
    if (!current()) return
    error.value = extractErrorMessage(cause, '无法读取分析任务')
    const failure = toApiError(cause)
    forbidden.value = failure.httpStatus === 403 || failure.businessCode === 40300
    accessDenied.value = [401, 403].includes(failure.httpStatus ?? 0) || [40100, 40300].includes(failure.businessCode ?? 0)
    if (accessDenied.value) { tasks.value = []; loaded.value = false; updatedAt.value = '' }
  } finally {
    if (current()) loading.value = false
  }
}
watch(identity, owner => {
  sequence += 1
  tasks.value = []; loaded.value = false; loading.value = false
  forbidden.value = false; accessDenied.value = false; updatedAt.value = ''
  cameraId.value = ''; search.value = ''; showArchived.value = false
  error.value = owner === null ? '登录身份尚未核实，请重新登录后读取任务' : ''
  if (owner !== null && !disposed) void load()
}, { flush: 'sync' })
function taskLocation(task: AnalysisTask, target: TaskGuidanceResponse['target'] = 'configuration') {
  return { path: `/cameras/${encodeURIComponent(task.camera_id)}`,
    query: { analysis_task: task.task_id, task_section: target } }
}
function open(task: AnalysisTask, target: TaskGuidanceResponse['target']) {
  if (!canManage.value || error.value || target === 'none') return
  if (target === 'legacy_baseline') { void router.push('/models/baseline'); return }
  void router.push(taskLocation(task, target))
}
useWebSocket({ topics: ['cameras'], onMessage: () => { void load() }, fallbackPoll: load })
onMounted(() => { void load() })
onUnmounted(() => { disposed = true; sequence += 1 })
</script>

<template>
  <PageShell title="分析任务" description="按任务查看当前步骤与阻塞原因。已有模型直接验证、按需定制训练、固定算法三条路径分别处理。" :busy="loading">
    <template #actions>
      <button :disabled="loading" @click="load">{{ loading ? '刷新中…' : '刷新任务' }}</button>
      <router-link v-if="canManage" to="/cameras">选择摄像头并创建任务</router-link>
    </template>
    <p class="scope">新任务仅支持验证态；生产保持 NO-GO。原异物检测仍使用原安全门。任务运行、场景适用和业务验收是不同状态。</p>
    <div class="filters">
      <label>摄像头<select v-model="cameraId"><option value="">全部摄像头</option><option v-for="camera in cameras" :key="camera" :value="camera">{{ camera }}</option></select></label>
      <label>查找任务<input v-model="search" type="search" placeholder="任务名称、模板或身份" /></label>
      <label class="checkbox"><input v-model="showArchived" type="checkbox" />显示已归档任务</label>
    </div>
    <p v-if="loaded && !forbidden" class="scope">最近成功读取：{{ updatedAt }}；{{ error ? '状态可能过期' : '每 60 秒 REST 对账，源状态变化时刷新' }}。</p>
    <p v-if="error && loaded && !forbidden" role="alert">{{ error }}；保留上次读取结果，下一步入口暂不可用。</p>
    <AsyncState :loading="loading && !loaded" :refreshing="loading && loaded" :forbidden="forbidden"
      :error="!loaded ? error : ''" :empty="loaded && !error && visible.length === 0"
      empty-title="没有符合条件的任务" empty-description="可清除筛选，或先在摄像头详情创建任务。空列表不代表检测正常。" @retry="load">
      <div class="task-grid">
        <article v-for="task in visible" :key="task.task_id" class="task-card" data-test="task-center-card">
          <h2>{{ task.template_id === 'legacy_anomaly.v1' ? '现有异物检测' : task.name || task.template_id }}</h2>
          <p class="scope">摄像头 {{ task.camera_id }} · {{ task.template_id }} · r{{ task.current_revision }}<span v-if="task.lifecycle === 'archived'"> · 已归档</span></p>
          <TaskGuidanceCard :task="task" :actionable="canManage" :stale="!!error" @open="open(task, $event)" />
          <router-link :to="taskLocation(task)">{{ canManage && task.authority === 'managed' && task.lifecycle !== 'archived' ? '打开该任务工作台' : '查看相机与任务状态' }}</router-link>
          <p v-if="canManage && task.guidance?.target === 'legacy_baseline'" class="scope">进入基线页后请选择摄像头 {{ task.camera_id }}，不会自动批准场景。</p>
          <p class="identity">{{ task.task_id }}</p>
        </article>
      </div>
    </AsyncState>
  </PageShell>
</template>

<style scoped>
.scope { color: var(--text-secondary); font-size: 12px; line-height: 1.6; }
.filters { display: flex; flex-wrap: wrap; align-items: end; gap: 12px; margin: 12px 0; }
label { display: grid; gap: 5px; font-size: 13px; }
.checkbox { display: flex; align-items: center; padding: 8px 0; }
input, select, button { padding: 8px; color: var(--text-primary); background: var(--bg); border: 1px solid var(--border-default); border-radius: 6px; max-width: 100%; }
button { cursor: pointer; } button:disabled { cursor: wait; }
.task-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr)); gap: 16px; }
.task-card { min-width: 0; padding: 16px; border: 1px solid var(--border-default); border-radius: var(--r-lg); }
h2 { font-size: 16px; margin: 0; overflow-wrap: anywhere; }
.identity { font-size: 11px; overflow-wrap: anywhere; color: var(--text-secondary); }
input:focus-visible, select:focus-visible, button:focus-visible, a:focus-visible { outline: 2px solid var(--status-info); outline-offset: 2px; }
[role="alert"] { color: var(--status-danger); }
</style>
