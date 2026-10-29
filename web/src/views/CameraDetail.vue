<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Steps, Button, Segmented, message } from 'ant-design-vue'
import { ArrowLeftOutlined } from '@ant-design/icons-vue'
import { getCameraDetail } from '../api'
import { setCameraMode } from '../api/cameras'
import ZoneEditor from '../components/ZoneEditor.vue'
import CalibrationWizard from '../components/calibration/CalibrationWizard.vue'
import AnalysisTaskPanel from '../components/cameras/AnalysisTaskPanel.vue'
import { streamFallbackReasonLabel, useGo2RTC } from '../composables/useGo2RTC'
import { useAuthStore } from '../stores/useAuthStore'
import { extractErrorMessage } from '../utils/error'
import {
  ApiError,
  ErrorCode,
  type CameraDetailPayload,
  type CameraZone,
  type PipelineMode,
  type SceneGuardStatus,
} from '../types/api'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const canOperateCamera = computed(() => auth.hasRole(['admin', 'operator', 'engineer']))
const canEditZones = computed(() => auth.hasRole(['admin', 'engineer']))
const cameraId = computed(() => String(route.params.id || ''))
const taskNavigationInvalid = computed(() => {
  const task = route.query.analysis_task
  const section = route.query.task_section
  const targets = new Set(['source', 'configuration', 'models', 'deployment', 'data', 'events', 'results', 'legacy_baseline', 'legacy_runtime', 'none'])
  return (task !== undefined && (typeof task !== 'string' || !task.trim()))
    || (section !== undefined && (typeof section !== 'string' || !targets.has(section)))
})
function clearTaskNavigation() {
  const { analysis_task: _task, task_section: _section, ...query } = route.query
  void router.replace({ query })
}
const camera = ref<CameraDetailPayload | null>(null)
const loading = ref(true)
const refreshing = ref(false)
const loadError = ref<string | null>(null)
const notFound = ref(false)
const zoneEditorData = ref<CameraZone[]>([])
const zoneEditorCameraId = ref<string | null>(null)
const savingZones = ref(false)
const streamActive = ref(false)
const frontendFrameUpdatedAt = ref<number | null>(null)
let pollTimer: ReturnType<typeof setInterval> | null = null
const cameraRequests = new Map<string, Promise<void>>()

function zonesFromDetail(detail: CameraDetailPayload): CameraZone[] {
  if (Array.isArray(detail.zones)) return detail.zones
  const configured = detail.config?.zones
  return Array.isArray(configured) ? configured as CameraZone[] : []
}

function isNotFoundError(error: unknown): boolean {
  return error instanceof ApiError && (
    error.httpStatus === 404
    || error.businessCode === ErrorCode.NOT_FOUND
    || error.code === 404
  )
}

function fetchCamera(reason: 'initial' | 'poll' | 'manual' = 'manual'): Promise<void> {
  const requestedId = cameraId.value
  const existing = cameraRequests.get(requestedId)
  if (existing) return existing

  if (reason === 'initial' || !camera.value) loading.value = true
  else refreshing.value = true

  const request = getCameraDetail(requestedId)
    .then((detail) => {
      if (cameraId.value !== requestedId) return
      camera.value = detail
      loadError.value = null
      notFound.value = false
      if (zoneEditorCameraId.value !== requestedId) {
        zoneEditorData.value = zonesFromDetail(detail)
        zoneEditorCameraId.value = requestedId
      }
    })
    .catch((error: unknown) => {
      if (cameraId.value !== requestedId) return
      loadError.value = extractErrorMessage(error, '加载摄像头详情失败')
      notFound.value = isNotFoundError(error)
      if (notFound.value) {
        camera.value = null
        stopActiveStream()
      }
    })
    .finally(async () => {
      cameraRequests.delete(requestedId)
      if (cameraId.value === requestedId) {
        loading.value = false
        refreshing.value = false
        // The live-view branch is hidden while the initial request is loading,
        // so its <video> ref does not exist until Vue has rendered this state.
        // Starting before that point skips WebRTC/MSE and delays MJPEG fallback
        // through the factory retry schedule.
        await nextTick()
        if (cameraId.value === requestedId) syncStream()
      }
    })

  cameraRequests.set(requestedId, request)
  return request
}

function retryCameraLoad() {
  void fetchCamera('manual')
}

onMounted(() => {
  void fetchCamera('initial')
  pollTimer = setInterval(() => {
    void fetchCamera('poll')
  }, 5000)
})

onUnmounted(() => {
  if (pollTimer !== null) {
    clearInterval(pollTimer)
    pollTimer = null
  }
  stopActiveStream()
})

// go2rtc WebRTC/MSE player
const _go2rtc = useGo2RTC(cameraId)
// @ts-expect-error TS6133 — used as template ref="videoRef"
const videoRef = _go2rtc.videoRef
// @ts-expect-error TS6133 — used as template ref="mjpegRef"
const mjpegRef = _go2rtc.mjpegRef
const streamStatus = _go2rtc.status
const streamFallbackReason = _go2rtc.fallbackReason
const streamFallbackDetail = computed(() => streamFallbackReasonLabel(streamFallbackReason.value))
const startStream = _go2rtc.start
const stopStream = _go2rtc.stop
const mjpegUrl = computed(() => `/api/cameras/${cameraId.value}/stream`)
const snapshotUrl = computed(() => `/api/cameras/${cameraId.value}/snapshot`)

function markFrontendFrameUpdate() {
  frontendFrameUpdatedAt.value = Date.now() / 1000
}

function stopActiveStream() {
  if (!streamActive.value) return
  stopStream()
  streamActive.value = false
  frontendFrameUpdatedAt.value = null
}

function syncStream() {
  if (camera.value?.connected) {
    if (!streamActive.value) {
      streamActive.value = true
      startStream()
    }
    return
  }
  stopActiveStream()
}

watch(cameraId, async (nextId, previousId) => {
  if (!previousId || nextId === previousId) return
  loading.value = true
  loadError.value = null
  notFound.value = false
  camera.value = null
  zoneEditorCameraId.value = null
  stopActiveStream()
  await fetchCamera('initial')
})

watch(streamStatus, (status) => {
  if (status === 'playing' || status === 'fallback') {
    markFrontendFrameUpdate()
  }
})

async function saveZones() {
  if (savingZones.value) return
  savingZones.value = true
  try {
    const { updateZones } = await import('../api/zones')
    await updateZones(cameraId.value, zoneEditorData.value)
    message.success('区域配置已保存')
  } catch (error) {
    message.error(extractErrorMessage(error, '保存区域配置失败'))
  } finally {
    savingZones.value = false
  }
}

const lifecycleStep = computed(() => {
  if (!camera.value) return 0
  const stages = camera.value.stages
  if (stages && Array.isArray(stages)) {
    const activeIdx = stages.findIndex((s: any) => s.status === 'active')
    if (activeIdx >= 0) return activeIdx
    // All completed
    if (stages.every((s: any) => s.status === 'completed')) return stages.length - 1
    return 0
  }
  // Fallback when stages not available
  const c = camera.value
  if (!c.connected) return 0
  if (!c.stats || c.stats.frames_captured === 0) return 0
  if (c.stats.frames_analyzed === 0) return 2
  return 4
})

function formatLabel(key: string) {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function formatValue(value: any): string {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (Array.isArray(value)) return value.length ? value.map(formatValue).join(', ') : '-'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function flattenEntries(value: any, prefix = ''): Array<{ key: string; label: string; value: string }> {
  if (value === null || value === undefined) return []

  if (typeof value !== 'object' || Array.isArray(value)) {
    return [{ key: prefix, label: formatLabel(prefix || 'value'), value: formatValue(value) }]
  }

  return Object.entries(value).flatMap(([key, nested]) => {
    const nextPrefix = prefix ? `${prefix}.${key}` : key
    if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
      return flattenEntries(nested, nextPrefix)
    }
    return [{ key: nextPrefix, label: formatLabel(nextPrefix), value: formatValue(nested) }]
  })
}

const basicEntries = computed(() => flattenEntries({
  camera_id: camera.value?.camera_id,
  name: camera.value?.name,
  connected: camera.value?.connected,
  running: camera.value?.running,
}))

const runtimeEntries = computed(() => flattenEntries({
  stats: camera.value?.stats,
  runtime: camera.value?.runtime,
  runner: camera.value?.runner,
}))

const healthEntries = computed(() => flattenEntries(camera.value?.health))
const detectorEntries = computed(() => flattenEntries(camera.value?.detector))
const configEntries = computed(() => flattenEntries(camera.value?.config))
const leftTab = ref<'live' | 'info' | 'zones'>('live')

const sourceDiagnostics = computed(() => camera.value?.source_diagnostics || camera.value || {})

function formatTimestampSeconds(value: any): string {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return '-'
  return new Date(value * 1000).toLocaleString('zh-CN', { hour12: false })
}

function formatFps(value: any): string {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return '-'
  return value.toFixed(1)
}

function sourceConsistencyLabel(value: any): string {
  if (value === 'direct') return '直接读取'
  if (value === 'go2rtc_redirect') return 'go2rtc 重定向'
  if (value === 'mismatch') return '不一致'
  return '未知'
}

const sourceConsistencyTone = computed(() => {
  const status = sourceDiagnostics.value?.source_consistency || camera.value?.source_consistency
  if (status === 'direct' || status === 'go2rtc_redirect') return 'ok'
  if (status === 'mismatch') return 'warn'
  return 'muted'
})

const sourceDiagnosticEntries = computed(() => {
  const diag = sourceDiagnostics.value || {}
  return [
    { label: '声明源', value: formatValue(diag.declared_source) },
    { label: '实际检测源', value: formatValue(diag.runtime_source) },
    { label: 'go2rtc stream id', value: formatValue(diag.go2rtc_stream) },
    { label: '当前检测 FPS', value: formatFps(diag.fps_observed) },
    { label: '最新 frame index', value: formatValue(diag.last_frame_index) },
    { label: '前端画面更新时间', value: formatTimestampSeconds(frontendFrameUpdatedAt.value) },
    { label: 'snapshot frame index', value: formatValue(diag.last_snapshot_frame_index) },
    { label: 'snapshot 更新时间', value: formatTimestampSeconds(diag.last_snapshot_ts) },
    { label: '后端预览 frame index', value: formatValue(diag.last_preview_stream_frame_index) },
    { label: '源一致性状态', value: sourceConsistencyLabel(diag.source_consistency) },
  ]
})

function decisionLabel(value: any): string {
  if (value === 'alert') return '告警'
  if (value === 'info') return '信息'
  if (value === 'normal') return '正常'
  if (value === 'suppressed') return '已抑制'
  if (value === 'skipped') return '已跳过'
  if (value === 'candidate') return '候选'
  if (value === 'pending') return '等待确认'
  return formatValue(value)
}

function decisionTone(value: any): string {
  if (value === 'alert' || value === 'candidate') return 'alert'
  if (value === 'info' || value === 'pending') return 'warn'
  if (value === 'suppressed' || value === 'skipped') return 'muted'
  return 'ok'
}

function candidateTypeLabel(value: any): string {
  if (value === 'persistent_change') return '固定场景变化'
  if (value === 'frame_anomaly') return '帧异常'
  if (value === 'frame_change') return '画面变化'
  if (value === 'fast_motion') return '高速飞溅'
  if (value === 'stationary_track') return '静止轨迹'
  if (value === 'person_filter') return '人员过滤'
  if (value === 'classification') return '分类器'
  if (value === 'inference') return '推理'
  return formatValue(value)
}

function formatMetric(value: any, digits = 3): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return formatValue(value)
  return value.toFixed(digits)
}

function extractPersistentEvidence(record: any): any | null {
  const evidence = record?.evidence || {}
  if (record?.candidate_type === 'persistent_change') return evidence
  return evidence.persistent_change || null
}

function candidateSummaryEntries(card: any): Array<{ label: string; value: string }> {
  const evidence = card.record?.evidence || {}
  const persistent = extractPersistentEvidence(card.record)
  const entries: Array<{ label: string; value: string }> = []
  const push = (label: string, value: any, digits = 3) => {
    if (value === null || value === undefined || value === '') return
    entries.push({ label, value: typeof value === 'number' ? formatMetric(value, digits) : formatValue(value) })
  }
  push('frame', card.frame_number, 0)
  push('score', persistent?.score ?? evidence.score ?? evidence.anomaly_score)
  push('变化占比', persistent?.changed_fraction ?? evidence.changed_fraction, 6)
  push('候选数', persistent?.candidate_count ?? evidence.candidate_count, 0)
  push('活动轨迹', persistent?.active_track_count ?? evidence.active_track_count, 0)
  push('MOG2 变化率', evidence.change_ratio, 6)
  push('检测类型', evidence.detection_type)
  push('运行模式', evidence.pipeline_mode)
  return entries
}

function candidateBoxes(record: any): any[] {
  const persistent = extractPersistentEvidence(record)
  const candidates = persistent?.candidates || record?.evidence?.candidates || []
  return Array.isArray(candidates) ? candidates.slice(0, 3) : []
}

const candidateDiagnosticCards = computed(() => {
  const diagnostics = Array.isArray(camera.value?.candidate_diagnostics)
    ? camera.value.candidate_diagnostics
    : []
  const cards: any[] = []
  diagnostics.forEach((diag: any, diagIndex: number) => {
    const records = Array.isArray(diag?.candidate_records) ? diag.candidate_records : []
    records.forEach((record: any, recordIndex: number) => {
      cards.push({
        key: `${diag.frame_number ?? diagIndex}:${record.candidate_id ?? recordIndex}`,
        frame_number: diag.frame_number,
        timestamp: diag.timestamp,
        final_decision: diag.final_decision,
        record,
      })
    })
  })
  return cards.slice(-6).reverse()
})

const inputQualityState = computed(() => {
  const detector = camera.value?.detector
  if (!detector) {
    return {
      label: '等待状态',
      detail: '检测状态未返回',
      tone: 'muted',
    }
  }
  const brightness = typeof detector.last_brightness === 'number'
    ? `亮度 ${detector.last_brightness.toFixed(1)}`
    : '亮度 -'
  if (detector.ssim_calibration_blocked) {
    return {
      label: '低光校准等待',
      detail: `${brightness} · 异常检测校准暂停`,
      tone: 'warn',
    }
  }
  if (detector.low_light) {
    return {
      label: '低光',
      detail: `${brightness} · 高速目标检测受限`,
      tone: 'warn',
    }
  }
  if (detector.exposure_recovering) {
    return {
      label: '曝光恢复',
      detail: `${brightness} · 高速目标检测暂缓`,
      tone: 'warn',
    }
  }
  if (detector.detection_limited) {
    return {
      label: '检测受限',
      detail: `${brightness} · ${detector.detection_limited_reason || '输入质量异常'}`,
      tone: 'warn',
    }
  }
  return {
    label: '稳定',
    detail: `${brightness} · 检测输入正常`,
    tone: 'ok',
  }
})

const modelBackendWarning = computed(() => {
  const actual = camera.value?.model_backend
  const expected = camera.value?.expected_model_backend
  if (expected && expected !== 'auto' && actual !== expected) {
    return {
      label: '模型后端不符合要求',
      detail: `当前 ${actual || '未加载'} · 期望 ${expected}`,
      tone: 'alert',
    }
  }
  if (actual && ['ssim-fallback', 'none', 'disabled'].includes(actual)) {
    return {
      label: '主模型未加载或已回退',
      detail: `当前 ${actual}`,
      tone: 'warn',
    }
  }
  return null
})

const sceneGuard = computed<SceneGuardStatus | null>(() => (
  camera.value?.scene_guard
  ?? camera.value?.runtime?.scene_guard
  ?? null
))

const sceneGuardBlocked = computed(() => (
  Boolean(sceneGuard.value?.enabled) && sceneGuard.value?.usable !== true
))

const sceneGuardTitle = computed(() => {
  const state = sceneGuard.value?.state
  if (state === 'calibrating' || state === 'unknown') return '场景校验中，检测已暂停'
  return '机位不匹配，检测已暂停'
})

const sceneGuardReason = computed(() => (
  sceneGuard.value?.reason || '当前画面尚未通过已批准场景的身份与机位校验'
))

const sceneGuardRecovery = computed(() => {
  const count = sceneGuard.value?.policy_recovery_count
  const sceneConfig = camera.value?.config?.scene_guard
  const required = sceneConfig && typeof sceneConfig === 'object'
    ? (sceneConfig as Record<string, unknown>).consecutive_matches
    : sceneGuard.value?.required_recovery_count
  if (typeof count !== 'number' || typeof required !== 'number' || required <= 0) return null
  return `${Math.min(count, required)}/${required}`
})

const modeOptions = [
  { label: '主动检测', value: 'active' },
  { label: '学习', value: 'learning' },
  { label: '维护', value: 'maintenance' },
]
const currentPipelineMode = computed(() => camera.value?.runtime?.pipeline_mode || null)
const canSwitchMode = computed(() => (
  Boolean(camera.value?.running) && canOperateCamera.value
))

async function handleModeChange(value: string | number) {
  const mode = String(value) as PipelineMode
  if (!camera.value?.camera_id || mode === currentPipelineMode.value) return
  try {
    const result = await setCameraMode(camera.value.camera_id, mode)
    camera.value = {
      ...camera.value,
      runtime: {
        ...(camera.value.runtime || {}),
        pipeline_mode: result.pipeline_mode,
      },
    }
    message.success('运行模式已切换')
  } catch (e) {
    message.error(extractErrorMessage(e, '切换运行模式失败'))
  }
}
</script>

<template>
  <main>
    <div class="topbar glass" style="padding-right: 14px">
      <div style="display:flex;align-items:center;gap:14px;flex:1;overflow:hidden;">
        <Button type="text" size="small" @click="router.push('/cameras')" style="margin-right: 8px; color: var(--ink-3);"><ArrowLeftOutlined /></Button>
        <span class="d" :class="camera?.connected ? 'd-green' : 'd-red'"></span>
        <h1 style="margin:0; font-family: 'JetBrains Mono', monospace">{{ cameraId }}</h1>
        <span class="crumb">{{ camera?.name || '未知设备' }}</span>
        <span v-if="refreshing" class="refreshing-label" aria-live="polite">刷新中…</span>
        <span class="sep"></span>
        <div style="flex:1">
          <Steps :current="lifecycleStep" size="small" style="max-width: 500px; transform: scale(0.9); transform-origin: left center;">
            <Steps.Step title="采集" />
            <Steps.Step title="基线审查" />
            <Steps.Step title="训练" />
            <Steps.Step title="发布" />
            <Steps.Step title="推理" />
          </Steps>
        </div>
        <Segmented
          v-if="canOperateCamera"
          size="small"
          :value="currentPipelineMode || 'learning'"
          :options="modeOptions"
          :disabled="!canSwitchMode"
          @change="handleModeChange"
        />
      </div>
    </div>

    <section v-if="loading && !camera" class="page-state glass" data-test="camera-loading">
      <span class="state-spinner" aria-hidden="true"></span>
      <strong>正在加载摄像头</strong>
      <span>正在读取运行状态、场景校验和视频源信息。</span>
    </section>

    <section v-else-if="notFound" class="page-state glass" data-test="camera-not-found">
      <strong>未找到摄像头 {{ cameraId }}</strong>
      <span>{{ loadError || '该摄像头可能已被删除或当前账号无权访问。' }}</span>
      <div class="state-actions">
        <Button @click="router.push('/cameras')">返回摄像头列表</Button>
        <Button type="primary" @click="retryCameraLoad">重新检查</Button>
      </div>
    </section>

    <section v-else-if="loadError && !camera" class="page-state glass" data-test="camera-error">
      <strong>摄像头详情加载失败</strong>
      <span>{{ loadError }}</span>
      <Button type="primary" :loading="loading" @click="retryCameraLoad">重试</Button>
    </section>

    <template v-else>
      <div v-if="loadError" class="stale-data-banner" role="status" data-test="camera-stale-warning">
        当前显示上一次成功获取的数据；后台刷新失败：{{ loadError }}
        <Button size="small" type="link" @click="retryCameraLoad">立即重试</Button>
      </div>

      <div
        v-if="sceneGuardBlocked"
        class="scene-guard-banner"
        role="alert"
        data-test="scene-guard-blocked"
      >
        <div class="scene-guard-icon" aria-hidden="true">!</div>
        <div class="scene-guard-copy">
          <strong>{{ sceneGuardTitle }}</strong>
          <span>视频预览保持可用；异常评分与告警已暂停。原因：{{ sceneGuardReason }}</span>
          <span class="scene-guard-meta">
            场景 {{ sceneGuard?.profile_id || '未绑定' }}
            · 模型 {{ sceneGuard?.bound_model_version || '未绑定' }}
            <template v-if="sceneGuardRecovery">· 恢复确认 {{ sceneGuardRecovery }}</template>
          </span>
        </div>
      </div>

    <div class="viewport" style="flex:1; overflow:hidden; padding:2px 4px 12px; display:flex; gap:12px; min-height:0;">
      <!-- Main Visual View -->
      <div class="primary-column">
      <section class="main-panel glass" style="flex:1; display:flex; flex-direction:column; min-width:0; border-radius: var(--r-lg);">
        <div class="panel-header">
          <div class="seg" style="display:flex; background:rgba(10,10,15,.05); border-radius:8px; padding:3px; gap:2px">
            <button :class="{ on: leftTab === 'live' }" @click="leftTab = 'live'">实时画面</button>
            <button :class="{ on: leftTab === 'info' }" @click="leftTab = 'info'">全量属性</button>
            <button :class="{ on: leftTab === 'zones' }" @click="leftTab = 'zones'">区域编辑</button>
          </div>
        </div>

        <div class="panel-body">
          <!-- LIVE TAB -->
          <div v-if="leftTab === 'live'" class="live-wrapper">
             <div v-if="camera?.connected" class="video-container">
                <video
                  ref="videoRef"
                  autoplay
                  muted
                  playsinline
                  v-show="streamStatus === 'playing' || streamStatus === 'connecting'"
                  class="player-vid"
                  @playing="markFrontendFrameUpdate"
                  @loadeddata="markFrontendFrameUpdate"
                  @timeupdate="markFrontendFrameUpdate"
                />
                <img
                  ref="mjpegRef"
                  v-if="streamStatus === 'fallback'"
                  :src="mjpegUrl"
                  class="player-vid"
                  alt="实时"
                  @load="markFrontendFrameUpdate"
                />
                <div v-if="streamStatus === 'connecting'" class="overlay-msg">连接中...</div>
                <div
                  v-if="streamStatus === 'fallback'"
                  class="fallback-badge"
                  role="status"
                  :title="streamFallbackDetail"
                >
                  MJPEG fallback · 非实时预览 · {{ streamFallbackDetail }}
                </div>
             </div>
             <div v-else class="video-container offline-state">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="m22 8-6 4 6 4V8Z"/><rect x="2" y="6" width="14" height="12" rx="2"/></svg>
                <span>摄像头离线</span>
             </div>
             <div class="quality-strip" :class="`quality-${inputQualityState.tone}`" data-test="input-quality-strip">
                <span class="quality-title">输入质量</span>
                <strong>{{ inputQualityState.label }}</strong>
                <span class="quality-detail">{{ inputQualityState.detail }}</span>
             </div>
             <div
               v-if="modelBackendWarning"
               class="quality-strip"
               :class="`quality-${modelBackendWarning.tone}`"
               data-test="model-backend-strip"
             >
                <span class="quality-title">模型后端</span>
                <strong>{{ modelBackendWarning.label }}</strong>
                <span class="quality-detail">{{ modelBackendWarning.detail }}</span>
             </div>
          </div>

          <!-- INFO TAB -->
          <div v-else-if="leftTab === 'info'" class="info-wrapper">
             <div class="wide-grid">
               <div class="meta-section" v-if="runtimeEntries.length">
                  <div class="meta-section-hd">运行状态</div>
                  <div class="meta-section-bd">
                     <div class="meta-row" v-for="item in runtimeEntries" :key="item.key">
                       <span class="meta-k">{{ item.label }}</span>
                       <span class="meta-v">{{ item.value }}</span>
                     </div>
                  </div>
               </div>

               <div class="meta-section" v-if="healthEntries.length">
                  <div class="meta-section-hd">外设健康探测</div>
                  <div class="meta-section-bd">
                     <div class="meta-row" v-for="item in healthEntries" :key="item.key">
                       <span class="meta-k">{{ item.label }}</span>
                       <span class="meta-v">{{ item.value }}</span>
                     </div>
                  </div>
               </div>
               
               <div class="meta-section" v-if="detectorEntries.length">
                  <div class="meta-section-hd">计算机视觉参数 (Detector)</div>
                  <div class="meta-section-bd">
                     <div class="meta-row" v-for="item in detectorEntries" :key="item.key">
                       <span class="meta-k">{{ item.label }}</span>
                       <span class="meta-v mono">{{ item.value }}</span>
                     </div>
                  </div>
               </div>

               <div class="meta-section" v-if="configEntries.length">
                  <div class="meta-section-hd">原始基线配置 (Config)</div>
                  <div class="meta-section-bd">
                     <div class="meta-row" v-for="item in configEntries" :key="item.key">
                       <span class="meta-k">{{ item.label }}</span>
                       <span class="meta-v mono">{{ item.value }}</span>
                     </div>
                  </div>
               </div>
             </div>
          </div>

          <!-- ZONES TAB -->
          <div v-else-if="leftTab === 'zones'" class="zones-wrapper">
            <ZoneEditor v-model="zoneEditorData" :image-src="snapshotUrl" :width="640" :height="480" />
            <div v-if="canEditZones" style="margin-top: 16px; text-align:right; flex-shrink: 0;">
               <Button type="primary" :loading="savingZones" :disabled="savingZones" @click="saveZones">
                 {{ savingZones ? '保存中…' : '保存区域配置' }}
               </Button>
            </div>
          </div>
        </div>
      </section>

      <!-- Calibration section (below main content area) -->
      <AnalysisTaskPanel :camera-id="cameraId" :can-manage="auth.hasRole(['admin', 'engineer'])"
        :navigation-invalid="taskNavigationInvalid"
        :requested-task-id="typeof route.query.analysis_task === 'string' ? route.query.analysis_task : undefined"
        :requested-section="typeof route.query.task_section === 'string' ? route.query.task_section : undefined" @clear-task-navigation="clearTaskNavigation" />
      <CalibrationWizard :camera-id="cameraId" style="margin-top: 12px" />
      </div>

      <!-- Right Metadata Sidebar -->
      <aside class="right glass">
        <!-- Stats Grid at top -->
        <div v-if="camera?.stats" class="stats-grid">
          <div class="stat-card">
             <div class="stat-title">已采集帧</div>
             <div class="stat-val mono">{{ camera.stats.frames_captured }}</div>
          </div>
          <div class="stat-card">
             <div class="stat-title">已分析帧</div>
             <div class="stat-val mono">{{ camera.stats.frames_analyzed }}</div>
          </div>
          <div class="stat-card">
             <div class="stat-title">告警触发</div>
             <div class="stat-val mono">{{ camera.stats.alerts_emitted }}</div>
          </div>
          <div class="stat-card">
             <div class="stat-title">平均延迟</div>
             <div class="stat-val mono">{{ camera.stats.avg_latency_ms?.toFixed(1) }}<small>ms</small></div>
          </div>
        </div>

        <!-- Meta Panels -->
        <div class="meta-scroll">
          <div class="meta-section" data-test="source-diagnostics">
             <div class="meta-section-hd diag-hd">
               <span>运行源诊断</span>
               <span class="source-pill" :class="`source-pill-${sourceConsistencyTone}`">
                 {{ sourceConsistencyLabel(sourceDiagnostics?.source_consistency) }}
               </span>
             </div>
             <div class="meta-section-bd">
                <div class="meta-row" v-for="item in sourceDiagnosticEntries" :key="item.label">
                  <span class="meta-k">{{ item.label }}</span>
                  <span class="meta-v mono">{{ item.value }}</span>
                </div>
             </div>
          </div>

          <div class="meta-section" data-test="candidate-diagnostics">
             <div class="meta-section-hd diag-hd">
               <span>候选诊断</span>
               <span class="source-pill source-pill-muted">{{ candidateDiagnosticCards.length }}</span>
             </div>
             <div class="meta-section-bd candidate-diagnostics-bd">
                <div v-if="candidateDiagnosticCards.length === 0" class="candidate-empty">暂无候选记录</div>
                <div
                  v-for="card in candidateDiagnosticCards"
                  :key="card.key"
                  class="candidate-item"
                >
                  <div class="candidate-head">
                    <span class="candidate-type">{{ candidateTypeLabel(card.record?.candidate_type) }}</span>
                    <span class="decision-pill" :class="`decision-${decisionTone(card.record?.decision)}`">
                      {{ decisionLabel(card.record?.decision) }}
                    </span>
                  </div>
                  <div v-if="card.record?.suppression_reason" class="candidate-reason">
                    {{ card.record.suppression_reason }}
                  </div>
                  <div class="candidate-metrics">
                    <span
                      v-for="item in candidateSummaryEntries(card)"
                      :key="`${card.key}:${item.label}`"
                    >
                      <b>{{ item.label }}</b>{{ item.value }}
                    </span>
                  </div>
                  <div v-if="candidateBoxes(card.record).length" class="candidate-boxes">
                    <div
                      v-for="box in candidateBoxes(card.record)"
                      :key="`${card.key}:box:${box.track_id ?? box.bbox?.join('-')}`"
                      class="candidate-box-row"
                    >
                      <span class="mono">#{{ formatValue(box.track_id) }}</span>
                      <span>bbox {{ formatValue(box.bbox) }}</span>
                      <span>{{ formatValue(box.area_px) }}px</span>
                      <span>{{ formatMetric(box.persistence_seconds, 1) }}s</span>
                    </div>
                  </div>
                </div>
             </div>
          </div>

          <div class="meta-section" v-if="basicEntries.length">
             <div class="meta-section-hd">基础信息</div>
             <div class="meta-section-bd">
                <div class="meta-row" v-for="item in basicEntries" :key="item.key">
                  <span class="meta-k">{{ item.label }}</span>
                  <span class="meta-v">{{ item.value }}</span>
                </div>
             </div>
          </div>
        </div>
      </aside>
    </div>
    </template>
  </main>
</template>

<style scoped>
main { flex:1; display:flex; flex-direction:column; min-width:0; gap:12px; height: 100%; min-height: 0; }

.topbar { height:60px; flex-shrink:0; display:flex; align-items:center; justify-content:space-between; padding:0 22px; }
.topbar h1 { font-size:18px; font-weight:700; color:var(--ink); letter-spacing:-0.028em; margin:0; }
.crumb { font-size:12px; color:var(--ink-5); font-weight:500; }
.refreshing-label { font-size: 11px; color: var(--text-muted); white-space: nowrap; }
.sep { width:1px; height:14px; background:var(--line-2); }
.d { width: 8px; height: 8px; border-radius: 50%; }
.d-green { background: var(--green); box-shadow: 0 0 0 3px rgba(21,163,74,.18); animation: breathe 2s infinite; }
.d-red { background: var(--red); box-shadow: 0 0 0 3px rgba(229,72,77,.18); }

.viewport { padding-right:0; gap: 12px; }
.primary-column { flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; overflow: auto; }

.page-state {
  flex: 1;
  min-height: 260px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 32px;
  color: var(--text-secondary);
  text-align: center;
}
.page-state strong { color: var(--text-primary); font-size: 17px; }
.page-state > span:not(.state-spinner) { max-width: 560px; color: var(--text-muted); font-size: 13px; }
.state-actions { display: flex; gap: 8px; margin-top: 4px; }
.state-spinner {
  width: 28px;
  height: 28px;
  border: 3px solid var(--border-default);
  border-top-color: var(--status-info);
  border-radius: 50%;
  animation: camera-spin 0.8s linear infinite;
}
@keyframes camera-spin { to { transform: rotate(360deg); } }

.stale-data-banner {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 7px 12px;
  border: 1px solid rgba(217, 119, 6, 0.28);
  border-radius: var(--r-sm);
  background: rgba(217, 119, 6, 0.07);
  color: #92400e;
  font-size: 12px;
}
.scene-guard-banner {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 11px 14px;
  border: 1px solid rgba(220, 38, 38, 0.28);
  border-radius: var(--r-sm);
  background: rgba(220, 38, 38, 0.07);
  color: #991b1b;
}
.scene-guard-icon {
  width: 22px;
  height: 22px;
  flex: 0 0 22px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: #b91c1c;
  color: #fff;
  font-size: 13px;
  font-weight: 800;
}
.scene-guard-copy { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.scene-guard-copy strong { font-size: 13px; }
.scene-guard-copy > span { font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }
.scene-guard-meta { color: #7f1d1d; opacity: 0.82; }

.main-panel {
  display: flex; flex-direction: column; overflow: hidden;
}
.panel-header {
  padding: 10px 16px; border-bottom: 1px solid var(--line-2); display: flex; align-items: center;
}
.seg button { font-size:11.5px; font-weight:600; color:var(--ink-4); padding:5px 16px; border:none; background:transparent; border-radius:6px; cursor:pointer; transition:all .15s; letter-spacing:.005em; }
.seg button:hover { color:var(--ink-2); }
.seg button.on { background:#fff; color:var(--ink); box-shadow:0 1px 2px rgba(10,10,15,.08),0 0 0 0.5px rgba(10,10,15,.04); }

.panel-body {
  flex: 1; padding: 16px; min-height: 0; overflow-y: auto; display: flex; flex-direction: column;
}
.panel-body::-webkit-scrollbar{width:6px}
.panel-body::-webkit-scrollbar-thumb{background:rgba(10,10,15,.10);border-radius:4px}
.panel-body::-webkit-scrollbar-thumb:hover{background:rgba(10,10,15,.18)}

.live-wrapper, .zones-wrapper { flex: 1; display: flex; flex-direction: column; height: 100%; min-height: 0; }
.info-wrapper { flex: 1; display: flex; flex-direction: column; }
.wide-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; align-items: start; }

.video-container {
  flex: 1; border-radius: 8px; overflow: hidden; background: #000; position: relative; display: flex; align-items: center; justify-content: center;
  box-shadow: 0 4px 20px rgba(0,0,0,.15);
}
.player-vid {
  width: 100%; height: 100%; object-fit: contain; background: #000;
}
.overlay-msg {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #fff; font-size: 14px; background: rgba(0,0,0,.6); padding: 8px 16px; border-radius: 8px;
  backdrop-filter: blur(8px);
}
.fallback-badge {
  position: absolute;
  left: 10px;
  bottom: 10px;
  padding: 4px 9px;
  border-radius: 4px;
  background: rgba(217,119,6,.92);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  pointer-events: none;
}
.offline-state {
  background: var(--bg); border: 1px dashed var(--line-3); color: var(--ink-5); flex-direction: column; gap: 12px;
}
.offline-state svg { width: 32px; height: 32px; }
.offline-state span { font-weight: 500; font-size: 13px; }

.quality-strip {
  margin-top: 10px;
  min-height: 40px;
  border: 1px solid var(--line-2);
  border-radius: 8px;
  background: var(--bg);
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  color: var(--ink-3);
  font-size: 12px;
  flex-shrink: 0;
}
.quality-title {
  color: var(--ink-5);
  font-weight: 600;
}
.quality-strip strong {
  color: var(--ink-2);
  font-size: 13px;
}
.quality-detail {
  color: var(--ink-4);
}
.quality-ok {
  border-color: rgba(21,163,74,.28);
  background: rgba(21,163,74,.06);
}
.quality-warn {
  border-color: rgba(217,119,6,.32);
  background: rgba(217,119,6,.08);
}
.quality-alert {
  border-color: rgba(220,38,38,.34);
  background: rgba(220,38,38,.08);
}
.quality-muted {
  background: rgba(10,10,15,.03);
}

.right { width: 320px; flex-shrink: 0; padding: 16px; display: flex; flex-direction: column; gap: 16px; overflow: hidden; border-radius: var(--r-lg); }
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; flex-shrink: 0; }
.stat-card { background: var(--bg); border: 1px solid var(--line-2); padding: 12px; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; }
.stat-title { font-size: 11px; color: var(--ink-4); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-val { font-size: 18px; color: var(--ink-2); font-weight: 700; display: flex; align-items: baseline; gap: 2px; }
.stat-val small { font-size: 10px; color: var(--ink-5); font-weight: 600; }

.meta-scroll { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; padding-right: 4px; }
.meta-scroll::-webkit-scrollbar{width:6px}
.meta-scroll::-webkit-scrollbar-thumb{background:rgba(10,10,15,.10);border-radius:4px}
.meta-scroll::-webkit-scrollbar-thumb:hover{background:rgba(10,10,15,.18)}

.meta-section { border: 1px solid var(--line-2); background: var(--bg); border-radius: 8px; overflow: hidden; }
.meta-section-hd { padding: 10px 14px; border-bottom: 1px solid var(--line-2); font-size: 11.5px; font-weight: 600; color: var(--ink-3); background: rgba(10,10,15,.02); }
.diag-hd { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.source-pill { font-size: 10.5px; line-height: 1; padding: 4px 6px; border-radius: 999px; border: 1px solid var(--line-2); color: var(--ink-4); background: #fff; white-space: nowrap; }
.source-pill-ok { border-color: rgba(21,163,74,.30); color: #15803d; background: rgba(21,163,74,.08); }
.source-pill-warn { border-color: rgba(217,119,6,.34); color: #a16207; background: rgba(217,119,6,.08); }
.source-pill-muted { color: var(--ink-5); background: rgba(10,10,15,.03); }
.meta-section-bd { padding: 8px 14px; }
.candidate-diagnostics-bd { padding-top: 4px; padding-bottom: 4px; }
.candidate-empty { color: var(--ink-5); font-size: 12px; padding: 8px 0; }
.candidate-item {
  padding: 10px 0;
  border-bottom: 1px dashed var(--line-2);
}
.candidate-item:last-child { border-bottom: none; }
.candidate-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.candidate-type {
  color: var(--ink-2);
  font-size: 12px;
  font-weight: 700;
  min-width: 0;
  overflow-wrap: anywhere;
}
.decision-pill {
  flex-shrink: 0;
  font-size: 10.5px;
  line-height: 1;
  padding: 4px 6px;
  border-radius: 999px;
  border: 1px solid var(--line-2);
  color: var(--ink-4);
  background: #fff;
}
.decision-alert { border-color: rgba(220,38,38,.30); color: #b91c1c; background: rgba(220,38,38,.07); }
.decision-warn { border-color: rgba(217,119,6,.34); color: #a16207; background: rgba(217,119,6,.08); }
.decision-ok { border-color: rgba(21,163,74,.30); color: #15803d; background: rgba(21,163,74,.08); }
.decision-muted { color: var(--ink-5); background: rgba(10,10,15,.03); }
.candidate-reason {
  margin-top: 6px;
  color: var(--ink-4);
  font-size: 11.5px;
  overflow-wrap: anywhere;
}
.candidate-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 10px;
  margin-top: 8px;
  font-size: 11.5px;
  color: var(--ink-3);
}
.candidate-metrics span {
  display: inline-flex;
  gap: 4px;
  align-items: baseline;
}
.candidate-metrics b {
  color: var(--ink-5);
  font-weight: 600;
}
.candidate-boxes {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.candidate-box-row {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) 56px 42px;
  gap: 6px;
  align-items: center;
  color: var(--ink-4);
  font-size: 11px;
  min-width: 0;
}
.candidate-box-row span {
  min-width: 0;
  overflow-wrap: anywhere;
}
.meta-row { display: flex; justify-content: space-between; align-items: flex-start; padding: 6px 0; border-bottom: 1px dashed var(--line-2); font-size: 12px; line-height: 1.6; }
.meta-row:last-child { border-bottom: none; }
.meta-k { color: var(--ink-4); font-size: 11.5px; min-width: 80px; max-width: 120px; }
.meta-v { color: var(--ink-2); font-size: 12px; text-align: right; word-break: break-all; }
.mono { font-family: 'JetBrains Mono', monospace; }

@media (max-width: 1279px) {
  .viewport {
    overflow-y: auto !important;
    flex-direction: column;
  }
  .primary-column { flex: 0 0 auto; min-height: 620px; overflow: visible; }
  .right { width: 100%; flex: 0 0 auto; min-height: 360px; overflow: visible; }
  .meta-scroll { overflow: visible; }
}

@media (max-width: 767px) {
  main { height: auto; min-height: 100%; overflow: visible; }
  .topbar { height: auto; min-height: 52px; padding: 8px 10px; }
  .topbar .sep,
  .topbar :deep(.ant-steps),
  .refreshing-label { display: none; }
  .topbar > div { flex-wrap: wrap; gap: 7px !important; }
  .crumb { max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .scene-guard-banner { margin: 0; }
  .scene-guard-meta { display: none; }
  .viewport { padding: 0 0 8px !important; }
  .primary-column { min-height: 520px; }
  .panel-header { padding: 8px; }
  .seg { width: 100%; }
  .seg button { flex: 1; padding: 6px 8px; }
  .panel-body { padding: 10px; }
  .wide-grid { grid-template-columns: 1fr; }
  .quality-strip { align-items: flex-start; flex-wrap: wrap; }
  .quality-detail { flex-basis: 100%; }
  .right { padding: 10px; }
}
</style>
