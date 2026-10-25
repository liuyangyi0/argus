<script setup lang="ts">
import { onUnmounted, ref, watch } from 'vue'
import { commandCameraSource, getCameraSource } from '../../api/taskDeployments'
import type { CameraSourceResponse } from '../../types/generated-openapi'
import { extractErrorMessage, toApiError } from '../../utils/error'

const props = defineProps<{ cameraId: string; canManage?: boolean }>()
const emit = defineEmits<{ refresh: [] }>()
const source = ref<CameraSourceResponse | null>(null)
const busy = ref(false)
const activeAction = ref<'start' | 'stop' | null>(null)
const error = ref('')
const commandError = ref('')
let generation = 0
let sequence = 0
let commandSequence = 0
let disposed = false
let commandNeedsRead = false
let timer: ReturnType<typeof setTimeout> | undefined
function schedule() {
  if (timer) clearTimeout(timer)
  timer = setTimeout(() => { void load() }, commandNeedsRead || source.value?.cleanup_pending ? 2000 : 60_000)
}
function checkedSource(value: CameraSourceResponse, cameraId: string) {
  if (value.camera_id !== cameraId) throw new Error('摄像头源响应身份不匹配')
  return value
}
function failureSource(value: unknown, cameraId: string): CameraSourceResponse | null {
  if (!value || typeof value !== 'object') return null
  const details = value as Record<string, unknown>
  if (details.camera_id !== cameraId || typeof details.state !== 'string') return null
  return { camera_id: cameraId, state: details.state,
    connected: typeof details.connected === 'boolean' ? details.connected : null,
    running: typeof details.running === 'boolean' ? details.running : null,
    cleanup_pending: details.cleanup_pending === true,
    reason_code: typeof details.reason_code === 'string' ? details.reason_code : null }
}
async function load() {
  if (timer) clearTimeout(timer)
  const token = generation; const cameraId = props.cameraId; const request = ++sequence
  try {
    const result = await getCameraSource(cameraId)
    if (disposed || token !== generation || request !== sequence) return
    source.value = checkedSource(result, cameraId); error.value = ''; commandNeedsRead = false
  } catch (cause: unknown) {
    if (!disposed && token === generation && request === sequence) error.value = extractErrorMessage(cause, '源状态读取失败，原显示可能过期')
  } finally {
    if (!disposed && token === generation && request === sequence) schedule()
  }
}
async function command(action: 'start' | 'stop') {
  if (!props.canManage || (busy.value && (action !== 'stop' || activeAction.value !== 'start'))) return
  const token = generation; const cameraId = props.cameraId; const command = ++commandSequence
  sequence += 1 // Invalidate any GET before issuing the mutation, even on failure.
  if (timer) clearTimeout(timer)
  busy.value = true; activeAction.value = action; error.value = ''; commandError.value = ''; commandNeedsRead = false
  try {
    const result = await commandCameraSource(cameraId, action)
    if (disposed || token !== generation || command !== commandSequence) return
    sequence += 1 // A pre-command GET must not overwrite this receipt.
    source.value = checkedSource(result, cameraId); emit('refresh')
    void load()
  } catch (cause: unknown) {
    if (!disposed && token === generation && command === commandSequence) {
      const failure = toApiError(cause)
      const observed = failureSource(failure.details, cameraId)
      if (observed) source.value = observed
      commandNeedsRead = true
      commandError.value = extractErrorMessage(cause, '源操作未完成，请刷新并确认清理状态')
    }
  } finally {
    if (!disposed && token === generation && command === commandSequence) {
      busy.value = false; activeAction.value = null; schedule()
    }
  }
}
watch(() => props.cameraId, () => { generation += 1; sequence += 1; commandSequence += 1; busy.value = false; activeAction.value = null; commandNeedsRead = false; source.value = null; error.value = ''; commandError.value = ''; void load() }, { immediate: true })
onUnmounted(() => { disposed = true; generation += 1; if (timer) clearTimeout(timer) })
</script>

<template>
  <section class="source-controls" aria-label="独立摄像头源" :aria-busy="busy">
    <p>摄像头源：{{ source?.state ?? '未知' }} · {{ source?.connected === true ? '已连接' : source?.connected === false ? '未连接' : '连接未知' }}</p>
    <p class="note">启动源不加载异物模型。停止源会中断全部订阅任务、预览和录像；只停止某任务请使用任务部署操作。</p>
    <p v-if="commandError || error" role="alert">{{ commandError || error }}</p>
    <p v-if="source?.reason_code" role="status">源诊断：{{ source.reason_code }}</p>
    <button v-if="canManage" :disabled="busy" data-test="start-source" @click="command('start')">仅启动摄像头源</button>
    <details v-if="canManage"><summary>停止整个摄像头源</summary><p>将中断该源的所有任务、预览和录像；任务持久化意图保留，源再次启动后可能恢复任务。</p><button :disabled="busy && activeAction !== 'start'" data-test="stop-source" @click="command('stop')">确认停止整个源</button></details>
    <button :disabled="busy" @click="load">刷新源状态</button>
  </section>
</template>

<style scoped>
.source-controls { padding: 12px 0; border-top: 1px solid var(--border-default); } p, summary { font-size: 12px; } .note { color: var(--text-secondary); } [role="alert"] { color: var(--status-danger); } button { margin: 4px 8px 4px 0; padding: 7px; background: var(--bg); color: var(--text-primary); border: 1px solid var(--border-default); border-radius: 6px; cursor: pointer; } button:disabled { opacity: .6; cursor: not-allowed; } button:focus-visible, summary:focus-visible { outline: 2px solid var(--status-info); outline-offset: 2px; }
</style>
