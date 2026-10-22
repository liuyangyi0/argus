<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onBeforeUnmount } from 'vue'
import { storeToRefs } from 'pinia'
import {
  Tag, Button, Typography, Tooltip, Segmented, message,
} from 'ant-design-vue'
import {
  CloseOutlined,
  CheckCircleOutlined,
  StopOutlined,
  ExportOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  AppstoreOutlined,
} from '@ant-design/icons-vue'
import { useRouter } from 'vue-router'
import { useAlertStore } from '../../stores/useAlertStore'
import { useAuthStore } from '../../stores/useAuthStore'
import { downloadEvidencePackage } from '../../api/alerts'
import { scoreColor } from '../../utils/colors'
import { formatTimestamp } from '../../utils/time'
import { extractErrorMessage } from '../../utils/error'
import ReplayPlayer from '../ReplayPlayer.vue'
import AnnotationOverlay from '../AnnotationOverlay.vue'
import ImageCompareSlider from '../ImageCompareSlider.vue'
import DangerousActionConfirm from '../common/DangerousActionConfirm.vue'

defineOptions({ name: 'AlertDetailPanel' })

const emit = defineEmits<{
  (e: 'close'): void
}>()

const router = useRouter()
const store = useAlertStore()
const { selectedAlert } = storeToRefs(store)
// Viewer can read alerts but must not mutate workflow state. Backend RBAC
// 403s these endpoints anyway; we just hide the buttons to remove dead clicks.
const auth = useAuthStore()
const canMutateAlert = computed(() => auth.hasRole(['admin', 'operator', 'engineer']))
const canViewModelRegistry = computed(() => auth.hasRole(['admin', 'engineer']))

const imageMode = ref<'composite' | 'snapshot' | 'heatmap' | 'compare'>('composite')
const annotationMode = ref(false)

const viewMode = ref<'recording' | 'snapshot'>('snapshot')
const detailPanelRef = ref<HTMLElement | null>(null)

watch(() => selectedAlert.value?.alert_id, async (alertId) => {
  viewMode.value = 'snapshot'
  imageMode.value = 'composite'
  annotationMode.value = false
  if (alertId) {
    await nextTick()
    detailPanelRef.value?.focus()
  }
})

// Track the compare/annotation layer's rendered size so we aren't hard-coding
// 640×480 for frames that may actually be 1280×720 or larger.
const compareContainerRef = ref<HTMLDivElement | null>(null)
const compareSize = ref({ width: 640, height: 480 })
let compareResizeObs: ResizeObserver | null = null

function updateCompareSize() {
  const el = compareContainerRef.value
  if (!el) return
  const w = el.clientWidth || 640
  // 4:3 fallback when we have no natural dimensions yet — we intentionally
  // don't read the image because it may still be loading.
  const h = Math.round(w * 0.5625) // 16:9 by default (matches most cameras)
  compareSize.value = { width: w, height: h }
}

onMounted(() => {
  updateCompareSize()
  if (typeof ResizeObserver !== 'undefined') {
    compareResizeObs = new ResizeObserver(() => updateCompareSize())
    // Observer is attached lazily once the element actually renders.
    watch(compareContainerRef, (el) => {
      if (el && compareResizeObs) compareResizeObs.observe(el)
    }, { immediate: true })
  }
})

onBeforeUnmount(() => {
  compareResizeObs?.disconnect()
})

const severityColor: Record<string, string> = {
  high: 'red', medium: 'orange', low: 'gold', info: 'blue',
}
const severityLabel: Record<string, string> = {
  high: '高', medium: '中', low: '低', info: '提示',
}
const categoryMeta: Record<string, { label: string; color: string }> = {
  projectile: { label: '抛射物', color: 'red' },
  static_foreign: { label: '静态异物', color: 'orange' },
  water_leak: { label: '可见漏水', color: 'cyan' },
  visible_plume: { label: '可见烟雾/粉尘', color: 'purple' },
  scene_change: { label: '场景变化', color: 'blue' },
  environmental: { label: '环境干扰', color: 'default' },
  person_intrusion: { label: '人员入侵', color: 'purple' },
  equipment_displacement: { label: '设备位移', color: 'gold' },
  unknown: { label: '未分类', color: 'default' },
}
const detectionTypeMeta: Record<string, { label: string; color: string }> = {
  scene_change: { label: '场景变化', color: 'blue' },
  new_object: { label: '新增/遗留物', color: 'orange' },
  persistent_change: { label: '新增/遗留物', color: 'orange' },
  projectile: { label: '高速飞溅', color: 'volcano' },
  water_leak: { label: '可见漏水', color: 'cyan' },
  plume: { label: '可见烟雾/粉尘', color: 'purple' },
  anomaly: { label: '异常变化', color: 'red' },
  object: { label: '目标检测', color: 'geekblue' },
  hybrid: { label: '混合检测', color: 'magenta' },
}
const workflowLabel: Record<string, string> = {
  new: '待处理', acknowledged: '已确认', investigating: '调查中',
  resolved: '已解决', closed: '已关闭', false_positive: '误报', uncertain: '待定',
}
const workflowColor: Record<string, string> = {
  new: 'default', acknowledged: 'green', investigating: 'blue',
  resolved: 'cyan', closed: 'default', false_positive: 'orange', uncertain: 'gold',
}
const candidateTypeMeta: Record<string, { label: string; color: string }> = {
  persistent_change: { label: '固定变化', color: 'orange' },
  water_leak: { label: '可见漏水', color: 'cyan' },
  plume: { label: '可见烟雾', color: 'purple' },
  frame_anomaly: { label: '帧异常', color: 'red' },
  fast_motion: { label: '高速目标', color: 'volcano' },
  scene_mode: { label: '模式判定', color: 'blue' },
}
const candidateDecisionMeta: Record<string, { label: string; color: string }> = {
  alert: { label: '已报警', color: 'red' },
  candidate: { label: '候选', color: 'orange' },
  pending: { label: '等待确认', color: 'gold' },
  suppressed: { label: '已抑制', color: 'default' },
  skipped: { label: '已跳过', color: 'default' },
  normal: { label: '正常', color: 'green' },
}

const candidateEvidence = computed(() => selectedAlert.value?.candidate_evidence || null)
const candidateEvents = computed(() => {
  const events = candidateEvidence.value?.events
  return Array.isArray(events) ? events : []
})
const alertDetectionType = computed(() => (
  selectedAlert.value?.detection_type
  || selectedAlert.value?.evidence_json?.detection_type
  || ''
))
const alertFrameIndex = computed(() => {
  const alert = selectedAlert.value
  const evidence = alert?.evidence_json
  return (
    alert?.frame_number
    ?? evidence?.frame_number
    ?? candidateEvidence.value?.trigger_frame_index
    ?? null
  )
})
const alertReplayPath = computed(() => (
  selectedAlert.value?.alert_id ? `/replay/${selectedAlert.value.alert_id}` : ''
))
const replayReady = computed(() => selectedAlert.value?.replay_ready === true)
const activeViewMode = computed<'recording' | 'snapshot'>(() => {
  if (
    replayReady.value
    && (!selectedAlert.value?.snapshot_path || viewMode.value === 'recording')
  ) {
    return 'recording'
  }
  return 'snapshot'
})
const recordingUnavailableText = computed(() => {
  const reason = selectedAlert.value?.recording_unavailable_reason
  const status = selectedAlert.value?.recording_status
  if (!reason && !status) return ''
  const labels: Record<string, string> = {
    missing_metadata: '录像索引缺失',
    missing_signals: '录像信号缺失',
    missing_video: '录像文件缺失',
    missing_directory: '录像目录缺失',
    recording_save_failed: '录像生成失败',
    recording_store_unavailable: '录像存储不可用',
    ring_buffer_empty: '触发前缓存为空',
    ring_buffer_solidify_failed: '录像片段固化失败',
    recording_store_error: '录像状态读取失败',
  }
  return labels[reason] || (status === 'incomplete' ? '录像数据不完整' : '')
})

// Pixels are hard to eyeball — kilo-pixels give a more scannable magnitude
// for the 分割结果 row. We intentionally don't convert to % of frame area,
// because the frame dimensions aren't persisted on the alert record.
function formatArea(px: number): string {
  if (px < 1000) return `${px} px`
  if (px < 1_000_000) return `${(px / 1000).toFixed(1)} K px`
  return `${(px / 1_000_000).toFixed(2)} M px`
}

function candidateTypeLabel(type: string): string {
  return candidateTypeMeta[type]?.label || type || '候选'
}

function candidateTypeColor(type: string): string {
  return candidateTypeMeta[type]?.color || 'default'
}

function candidateDecisionLabel(decision: string): string {
  return candidateDecisionMeta[decision]?.label || decision || '未知'
}

function candidateDecisionColor(decision: string): string {
  return candidateDecisionMeta[decision]?.color || 'default'
}

function detectionTypeLabel(type: string): string {
  return detectionTypeMeta[type]?.label || type || '未知'
}

function detectionTypeColor(type: string): string {
  return detectionTypeMeta[type]?.color || 'default'
}

function candidateFrameLabel(event: any): string {
  const frame = event?.frame_number ?? event?.frame_index
  return frame === undefined || frame === null ? '-' : `#${frame}`
}

function formatCandidateScore(event: any): string {
  const score = Number(event?.score)
  return Number.isFinite(score) ? score.toFixed(3) : '-'
}

function candidateEvidenceValue(event: any, key: string): any {
  const evidence = event?.evidence
  if (!evidence || typeof evidence !== 'object') return undefined
  if (key in evidence) return evidence[key]
  const persistent = evidence.persistent_change
  if (persistent && typeof persistent === 'object' && key in persistent) {
    return persistent[key]
  }
  return undefined
}

function candidateChangeFraction(event: any): string {
  const value = Number(candidateEvidenceValue(event, 'changed_fraction'))
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : ''
}

function candidateCount(event: any): string {
  const count = Number(candidateEvidenceValue(event, 'candidate_count'))
  return Number.isFinite(count) ? `${count}` : ''
}

function firstCandidate(event: any): any | null {
  const candidates = event?.evidence?.candidates
  return Array.isArray(candidates) && candidates.length ? candidates[0] : null
}

function candidateBbox(event: any): string {
  const bbox = firstCandidate(event)?.bbox
  return Array.isArray(bbox) ? bbox.join(', ') : ''
}

function candidateLeakType(event: any): string {
  const type = firstCandidate(event)?.leak_type
  if (type === 'drip') return '滴水/水流'
  if (type === 'pooling') return '积水扩大'
  return ''
}

function candidatePlumeType(event: any): string {
  const type = firstCandidate(event)?.plume_type
  if (type === 'steam_or_mist') return '蒸汽/雾化'
  if (type === 'smoke_or_dust') return '烟雾/粉尘'
  if (type === 'visible_plume') return '可见羽流'
  return ''
}

function candidateGrowth(event: any): string {
  const growth = Number(firstCandidate(event)?.growth_area_px)
  return Number.isFinite(growth) && growth > 0 ? formatArea(growth) : ''
}

function candidateEdgeDensity(event: any): string {
  const density = Number(firstCandidate(event)?.edge_density)
  return Number.isFinite(density) ? density.toFixed(3) : ''
}

function candidatePersistence(event: any): string {
  const seconds = Number(firstCandidate(event)?.persistence_seconds)
  return Number.isFinite(seconds) ? `${seconds.toFixed(1)}s` : ''
}

// A "多机位回放" storyboard is only meaningful when there's another camera
// we can pair the alert with — either through the cross-camera correlation
// pipeline (correlation_partner) or because the event grouping has picked
// up related alerts from multiple cameras (event_group_count > 1).
const hasMultiCam = computed(() => {
  const a = selectedAlert.value
  if (!a) return false
  if (!replayReady.value) return false
  if (a.correlation_partner) return true
  if ((a.event_group_count ?? 0) > 1) return true
  return false
})

async function handleAcknowledge() {
  if (!selectedAlert.value) return
  try {
    await store.ackAlert(selectedAlert.value.alert_id)
    message.success('已确认')
  } catch (e) {
    message.error(extractErrorMessage(e, '确认失败'))
  }
}

async function handleFalsePositive() {
  if (!selectedAlert.value) return
  try {
    await store.fpAlert(selectedAlert.value.alert_id)
    message.success('已标记误报')
  } catch (e) {
    message.error(extractErrorMessage(e, '标记失败'))
  }
}

const deleting = ref(false)

async function handleDelete() {
  if (!selectedAlert.value) return
  const id = selectedAlert.value.alert_id
  deleting.value = true
  try {
    await store.delAlert(id)
    message.success('已删除')
    // Store clears selectedAlert on successful delete → panel collapses.
  } catch (e) {
    message.error(extractErrorMessage(e, '删除失败'))
  } finally {
    deleting.value = false
  }
}

async function handleEvidenceExport() {
  if (!selectedAlert.value) return
  try {
    await downloadEvidencePackage(selectedAlert.value.alert_id)
    message.success('证据包下载已开始')
  } catch (e) {
    message.error(extractErrorMessage(e, '导出证据包失败'))
  }
}

// C-AlertTrainingLink: jump to model registry page with the version_id as a
// query hint so the registry can pin and highlight the lineage row.
function goToModelVersion() {
  const versionId = selectedAlert.value?.model_version_id
  if (!versionId) return
  router.push({ path: '/models/registry', query: { version_id: versionId } })
}
</script>

<template>
  <div
    v-if="selectedAlert"
    ref="detailPanelRef"
    class="detail-panel"
    role="region"
    :aria-label="`告警 ${selectedAlert.alert_id} 详情`"
    tabindex="-1"
  >
    <div class="detail-header">
      <span :class="['sev-badge', `sev-${selectedAlert.severity}`]">
        {{ severityLabel[selectedAlert.severity] }}
      </span>
      <div class="detail-crumb">
        <b>{{ selectedAlert.camera_id }}</b>
        <span class="detail-crumb-sep">/</span>
        {{ selectedAlert.zone_id || 'DEFAULT' }}
        <span class="detail-crumb-sep">/</span>
        {{ formatTimestamp(selectedAlert.timestamp || selectedAlert.created_at) }}
      </div>

      <div class="detail-header-actions">
        <Tag
          :color="workflowColor[selectedAlert.workflow_status] || 'default'"
          style="margin: 0"
        >
          {{ workflowLabel[selectedAlert.workflow_status] || '待处理' }}
        </Tag>
        <Tooltip v-if="replayReady" title="在独立页面查看录像回放">
          <Button
            size="small"
            type="text"
            style="color: var(--argus-text-muted)"
            @click="router.push(`/replay/${selectedAlert.alert_id}`)"
          >
            <template #icon><PlayCircleOutlined /></template>
            查看录像
          </Button>
        </Tooltip>
        <Tooltip v-if="hasMultiCam" title="在共享时间线上同步播放所有关联摄像头">
          <Button
            size="small"
            type="text"
            style="color: var(--argus-text-muted)"
            @click="router.push(`/replay/${selectedAlert.alert_id}/storyboard`)"
          >
            <template #icon><AppstoreOutlined /></template>
            多机位回放
          </Button>
        </Tooltip>
        <Tooltip title="导出证据包">
          <Button
            size="small"
            type="text"
            style="color: var(--argus-text-muted)"
            :aria-label="`导出告警 ${selectedAlert.alert_id} 证据包`"
            @click="handleEvidenceExport"
          >
            <template #icon><ExportOutlined /></template>
          </Button>
        </Tooltip>
        <Button
          size="small"
          type="text"
          style="color: var(--argus-text-muted)"
          aria-label="关闭告警详情"
          @click="emit('close')"
        >
          <template #icon><CloseOutlined /></template>
        </Button>
      </div>
    </div>

    <div class="detail-content">
      <div
        v-if="recordingUnavailableText && selectedAlert.snapshot_path"
        class="recording-unavailable-banner"
      >
        <Typography.Text type="secondary" style="font-size: 13px">
          {{ recordingUnavailableText }}，已显示触发帧
        </Typography.Text>
      </div>

      <div
        v-if="replayReady && selectedAlert.snapshot_path"
        class="view-mode-bar"
      >
        <Segmented
          v-model:value="viewMode"
          :options="[
            { label: '录像', value: 'recording' },
            { label: '触发帧', value: 'snapshot' },
          ]"
          size="small"
        />
      </div>

      <ReplayPlayer
        v-if="replayReady && activeViewMode === 'recording'"
        :key="selectedAlert.alert_id"
        :alert-id="selectedAlert.alert_id"
        class="detail-section-spacing"
      />

      <div
        v-if="selectedAlert.snapshot_path && activeViewMode === 'snapshot'"
        class="detail-section-spacing"
      >
        <div class="image-mode-bar">
          <Segmented
            v-model:value="imageMode"
            :options="[
              { label: '叠加图', value: 'composite' },
              { label: '原图', value: 'snapshot' },
              { label: '热力图', value: 'heatmap' },
            ]"
            size="small"
          />
          <div class="score-display-wrapper">
            <Typography.Text type="secondary" style="font-size: 11px">异常分数</Typography.Text>
            <div
              :style="{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '2px 10px',
                borderRadius: '12px',
                background: `${scoreColor(selectedAlert.anomaly_score)}22`,
                border: `1px solid ${scoreColor(selectedAlert.anomaly_score)}44`,
              }"
            >
              <span :style="{ color: scoreColor(selectedAlert.anomaly_score), fontWeight: 700, fontSize: '14px' }">
                {{ selectedAlert.anomaly_score?.toFixed(4) }}
              </span>
            </div>
          </div>
        </div>

        <div v-if="imageMode === 'compare'" class="compare-section">
          <div ref="compareContainerRef" class="compare-responsive-wrap">
            <ImageCompareSlider
              :left-src="`/api/alerts/${selectedAlert.alert_id}/image/snapshot`"
              :right-src="`/api/alerts/${selectedAlert.alert_id}/image/heatmap`"
              left-label="原始帧"
              right-label="热力图"
              :width="compareSize.width"
              :height="compareSize.height"
            />
          </div>
          <Typography.Text type="secondary" style="font-size: 11px; margin-top: 4px; display: block">
            按住 Alt 悬停可放大区域
          </Typography.Text>
        </div>
        <div v-else class="image-viewer">
          <img
            :src="`/api/alerts/${selectedAlert.alert_id}/image/${imageMode}`"
            class="alert-detail-img"
            :alt="selectedAlert.alert_id"
          />
          <AnnotationOverlay
            v-if="annotationMode"
            :width="compareSize.width"
            :height="compareSize.height"
            :alert-id="selectedAlert.alert_id"
            :camera-id="selectedAlert.camera_id"
          />
        </div>
        <div class="image-action-bar">
          <Button size="small" :type="imageMode === 'compare' ? 'primary' : 'default'" @click="imageMode = imageMode === 'compare' ? 'composite' : 'compare'">
            {{ imageMode === 'compare' ? '返回普通视图' : '对比滑块' }}
          </Button>
          <Button size="small" :type="annotationMode ? 'primary' : 'default'" @click="annotationMode = !annotationMode">
            {{ annotationMode ? '关闭标注' : '标注画框' }}
          </Button>
        </div>
      </div>

      <div
        v-if="!replayReady && !selectedAlert.snapshot_path"
        class="no-image-placeholder"
      >
        <Typography.Text type="secondary" style="font-size: 14px">
          无快照或录像数据
        </Typography.Text>
      </div>

      <div
        v-if="candidateEvents.length"
        class="candidate-evidence-panel detail-section-spacing"
      >
        <div class="candidate-evidence-hd">
          <span>候选证据</span>
          <b>
            {{ candidateEvidence?.event_count || candidateEvents.length }} 条
            <span v-if="candidateEvidence?.trigger_frame_index !== null && candidateEvidence?.trigger_frame_index !== undefined">
              · 触发帧 #{{ candidateEvidence.trigger_frame_index }}
            </span>
          </b>
        </div>
        <div class="candidate-event-list">
          <div
            v-for="(event, idx) in candidateEvents"
            :key="event.candidate_id || `${event.candidate_type}-${event.frame_index}-${idx}`"
            class="candidate-event"
          >
            <div class="candidate-event-main">
              <Tag :color="candidateTypeColor(event.candidate_type)" style="margin: 0">
                {{ candidateTypeLabel(event.candidate_type) }}
              </Tag>
              <Tag :color="candidateDecisionColor(event.decision)" style="margin: 0">
                {{ candidateDecisionLabel(event.decision) }}
              </Tag>
              <span class="candidate-event-frame">帧 {{ candidateFrameLabel(event) }}</span>
              <span class="candidate-event-score">分数 {{ formatCandidateScore(event) }}</span>
            </div>
            <div class="candidate-event-detail">
              <span v-if="candidateChangeFraction(event)">
                变化面积 {{ candidateChangeFraction(event) }}
              </span>
              <span v-if="candidateCount(event)">
                候选 {{ candidateCount(event) }}
              </span>
              <span v-if="candidateLeakType(event)">
                迹象 {{ candidateLeakType(event) }}
              </span>
              <span v-if="candidatePlumeType(event)">
                类型 {{ candidatePlumeType(event) }}
              </span>
              <span v-if="candidateGrowth(event)">
                增长 {{ candidateGrowth(event) }}
              </span>
              <span v-if="candidateEdgeDensity(event)">
                边缘密度 {{ candidateEdgeDensity(event) }}
              </span>
              <span v-if="candidateBbox(event)">
                框 {{ candidateBbox(event) }}
              </span>
              <span v-if="candidatePersistence(event)">
                持续 {{ candidatePersistence(event) }}
              </span>
              <span v-if="event.suppression_reason">
                抑制 {{ event.suppression_reason }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div class="meta-grid">
        <div class="meta-panel">
          <div class="meta-panel-hd">
            <span>告警信息</span>
            <b>{{ selectedAlert.alert_id?.slice(-8) }}</b>
          </div>
          <div class="meta-panel-bd">
            <div class="meta-row">
              <span class="meta-k">摄像头</span>
              <span class="meta-v">{{ selectedAlert.camera_id }} / {{ selectedAlert.zone_id || '默认' }}</span>
            </div>
            <div class="meta-row">
              <span class="meta-k">严重度</span>
              <span class="meta-v">
                <Tag :color="severityColor[selectedAlert.severity]" style="margin: 0">{{ severityLabel[selectedAlert.severity] }}</Tag>
                <Tooltip v-if="selectedAlert.severity_adjusted_by_classifier" title="严重度已被 AI 分类器根据识别标签上调">
                  <Tag color="purple" style="margin-left: 4px; font-size: 10px">AI↑</Tag>
                </Tooltip>
              </span>
            </div>
            <div v-if="selectedAlert.category" class="meta-row">
              <span class="meta-k">分类</span>
              <span class="meta-v">
                <Tag :color="categoryMeta[selectedAlert.category]?.color || 'default'" style="margin: 0">
                  {{ categoryMeta[selectedAlert.category]?.label || selectedAlert.category }}
                </Tag>
              </span>
            </div>
            <div v-if="alertDetectionType" class="meta-row">
              <span class="meta-k">检测类型</span>
              <span class="meta-v">
                <Tag :color="detectionTypeColor(alertDetectionType)" style="margin: 0">
                  {{ detectionTypeLabel(alertDetectionType) }}
                </Tag>
              </span>
            </div>
            <div class="meta-row">
              <span class="meta-k">置信度</span>
              <span class="meta-v" :style="{ color: scoreColor(selectedAlert.anomaly_score), fontWeight: 600 }">{{ selectedAlert.anomaly_score?.toFixed(4) }}</span>
            </div>
            <div v-if="alertFrameIndex !== null && alertFrameIndex !== undefined" class="meta-row">
              <span class="meta-k">触发帧</span>
              <span class="meta-v">#{{ alertFrameIndex }}</span>
            </div>
            <div v-if="selectedAlert.classification_label" class="meta-row">
              <span class="meta-k">AI 分类</span>
              <span class="meta-v meta-v--classification">
                <Tag color="geekblue" style="margin: 0">{{ selectedAlert.classification_label }}</Tag>
                <span v-if="selectedAlert.classification_confidence != null" class="classification-conf">
                  {{ (selectedAlert.classification_confidence * 100).toFixed(1) }}%
                </span>
              </span>
            </div>
            <div v-if="selectedAlert.corroborated !== undefined && selectedAlert.corroborated !== null" class="meta-row">
              <span class="meta-k">
                <Tooltip title="跨相机相关性：另一个视野重叠的相机是否在同一位置确认了异常。未证实的告警严重度会被自动降级。">
                  <span>跨相机验证</span>
                </Tooltip>
              </span>
              <span class="meta-v meta-v--corroboration">
                <Tag v-if="selectedAlert.corroborated" color="green" style="margin: 0">
                  已证实
                </Tag>
                <Tag v-else color="volcano" style="margin: 0">
                  未证实 (已降级)
                </Tag>
                <span v-if="selectedAlert.correlation_partner" class="correlation-partner">
                  via {{ selectedAlert.correlation_partner }}
                </span>
              </span>
            </div>
            <div v-if="selectedAlert.segmentation_count" class="meta-row">
              <span class="meta-k">
                <Tooltip title="SAM2 实例分割 — 从异常热力图峰值反推每个物体的精确边界。主图上的红框就是分割结果。">
                  <span>分割结果</span>
                </Tooltip>
              </span>
              <span class="meta-v meta-v--segmentation">
                <Tag color="magenta" style="margin: 0">
                  {{ selectedAlert.segmentation_count }} 个对象
                </Tag>
                <span
                  v-if="selectedAlert.segmentation_total_area_px != null"
                  class="segmentation-area"
                >
                  总面积 {{ formatArea(selectedAlert.segmentation_total_area_px) }}
                </span>
              </span>
            </div>
            <div v-if="selectedAlert.speed_ms || selectedAlert.speed_px_per_sec || selectedAlert.trajectory_model" class="meta-row">
              <span class="meta-k">物理数据</span>
              <span class="meta-v">
                <Tag v-if="selectedAlert.speed_ms" color="cyan" style="margin: 0">{{ selectedAlert.speed_ms.toFixed(2) }} m/s</Tag>
                <Tag v-else-if="selectedAlert.speed_px_per_sec" color="cyan" style="margin: 0">{{ selectedAlert.speed_px_per_sec.toFixed(0) }} px/s</Tag>
                <Tag v-if="selectedAlert.trajectory_model" color="green" style="margin: 0; margin-left: 4px">{{ selectedAlert.trajectory_model }}</Tag>
              </span>
            </div>
            <div class="meta-row">
              <span class="meta-k">触发时间</span>
              <span class="meta-v">{{ formatTimestamp(selectedAlert.timestamp || selectedAlert.created_at) }}</span>
            </div>
            <div v-if="selectedAlert.evidence_summary" class="meta-row">
              <span class="meta-k">证据摘要</span>
              <span class="meta-v meta-v--long">{{ selectedAlert.evidence_summary }}</span>
            </div>
            <div v-if="selectedAlert.suppression_reason" class="meta-row">
              <span class="meta-k">抑制原因</span>
              <span class="meta-v meta-v--long">{{ selectedAlert.suppression_reason }}</span>
            </div>
            <div v-if="selectedAlert.assigned_to" class="meta-row">
              <span class="meta-k">指派</span>
              <span class="meta-v">{{ selectedAlert.assigned_to }}</span>
            </div>
            <div v-if="selectedAlert.notes" class="meta-row">
              <span class="meta-k">备注</span>
              <span class="meta-v meta-v--notes">{{ selectedAlert.notes }}</span>
            </div>
            <div v-if="selectedAlert.model_version_id" class="meta-row">
              <span class="meta-k">
                <Tooltip title="触发本告警的模型版本。点击跳转到模型管理页面查看完整谱系。">
                  <span>触发模型</span>
                </Tooltip>
              </span>
              <span class="meta-v meta-v--model-link">
                <Button
                  v-if="canViewModelRegistry"
                  type="link"
                  size="small"
                  style="padding: 0; height: auto; font-family: monospace; font-size: 12px"
                  @click="goToModelVersion"
                >
                  {{ selectedAlert.model_version_id }}
                </Button>
                <span v-else class="model-version-readonly">
                  {{ selectedAlert.model_version_id }}
                </span>
              </span>
            </div>
            <div v-if="replayReady && alertReplayPath" class="meta-row">
              <span class="meta-k">证据回放</span>
              <span class="meta-v">
                <Button
                  type="link"
                  size="small"
                  style="padding: 0; height: auto"
                  @click="router.push(alertReplayPath)"
                >
                  打开回放
                </Button>
              </span>
            </div>
            <div v-if="selectedAlert.workflow_status === 'false_positive'" class="meta-row meta-row--feedback-hint">
              <Tag color="orange" style="margin: 0; white-space: normal; line-height: 1.5">
                此告警已计入 FP 反馈，后续训练任务可在「模型管理 → 训练与评估」查看
              </Tag>
            </div>
          </div>
        </div>

        <div class="meta-panel">
          <div class="meta-panel-hd">
            <span>操作</span>
            <b>{{ workflowLabel[selectedAlert.workflow_status] || '待处理' }}</b>
          </div>
          <div class="meta-panel-bd actions-panel-bd">
            <Button
              v-if="canMutateAlert && selectedAlert.workflow_status === 'new'"
              type="primary"
              block
              @click="handleAcknowledge"
            >
              <template #icon><CheckCircleOutlined /></template>
              确认真实
            </Button>
            <Button
              v-if="canMutateAlert && (selectedAlert.workflow_status === 'new' || selectedAlert.workflow_status === 'acknowledged')"
              block
              @click="handleFalsePositive"
            >
              <template #icon><StopOutlined /></template>
              标记误报
            </Button>
            <div v-if="!canMutateAlert || (selectedAlert.workflow_status !== 'new' && selectedAlert.workflow_status !== 'acknowledged')" class="status-display">
              <Tag :color="workflowColor[selectedAlert.workflow_status]" style="font-size: 13px; padding: 4px 16px">
                {{ workflowLabel[selectedAlert.workflow_status] }}
              </Tag>
            </div>
            <DangerousActionConfirm
              v-if="canMutateAlert"
              title="确认删除告警"
              description="删除后无法恢复，确定要删除此条告警吗？"
              action-label="删除告警"
              :aria-label="`删除告警 ${selectedAlert.alert_id}`"
              :loading="deleting"
              block
              confirm-text="删除"
              @confirm="handleDelete"
            >
              <template #icon><DeleteOutlined /></template>
              删除告警
            </DangerousActionConfirm>

          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.detail-panel {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--argus-surface);
}

.detail-header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 10px 24px;
  background: var(--argus-header-bg);
  border-bottom: 1px solid var(--argus-border);
  flex-shrink: 0;
}
.detail-crumb {
  font-size: 13px;
  color: var(--argus-text-muted);
}
.detail-crumb b {
  color: var(--argus-text);
}
.detail-crumb-sep {
  margin: 0 8px;
  color: var(--argus-text-muted);
  opacity: .5;
}
.detail-header-actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}

.sev-badge {
  display: inline-block;
  padding: 2px 10px;
  font-size: 12px;
  font-weight: 600;
  color: #fff;
  border-radius: 3px;
}
.sev-high { background: #b91c1c; }
.sev-medium { background: #d97706; }
.sev-low { background: #2563eb; }
.sev-info { background: #4b5563; }

.detail-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
}
.detail-section-spacing {
  margin-bottom: 20px;
}

.recording-unavailable-banner {
  padding: 8px 10px;
  margin-bottom: 12px;
  border: 1px solid var(--argus-border);
  border-radius: 6px;
  background: var(--argus-card-bg-solid);
}

.view-mode-bar {
  display: flex;
  align-items: center;
  margin-bottom: 12px;
}
.image-mode-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}
.compare-responsive-wrap {
  width: 100%;
  max-width: 100%;
}
.score-display-wrapper {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}
.compare-section {
  margin-bottom: 12px;
}
.image-viewer {
  border-radius: 8px;
  overflow: hidden;
  background: #000;
  text-align: center;
  position: relative;
}
.alert-detail-img {
  max-width: 100%;
  max-height: 480px;
  object-fit: contain;
}
.image-action-bar {
  display: flex;
  gap: 6px;
  margin-top: 6px;
}

.no-image-placeholder {
  padding: 40px;
  text-align: center;
  background: var(--argus-card-bg-solid);
  border-radius: 8px;
  margin-bottom: 20px;
}

.candidate-evidence-panel {
  border: 1px solid var(--argus-border);
  background: var(--argus-card-bg-solid);
  border-radius: 6px;
  overflow: hidden;
}
.candidate-evidence-hd {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--argus-border);
  color: var(--argus-text-muted);
  font-size: 13px;
  font-weight: 500;
}
.candidate-evidence-hd b {
  color: var(--argus-text);
  font-size: 12px;
  font-weight: 500;
}
.candidate-event-list {
  display: grid;
  gap: 8px;
  padding: 10px 14px;
}
.candidate-event {
  display: grid;
  gap: 5px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--argus-border);
}
.candidate-event:last-child {
  padding-bottom: 0;
  border-bottom: none;
}
.candidate-event-main {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  font-size: 12px;
}
.candidate-event-frame,
.candidate-event-score {
  color: var(--argus-text-muted);
  font-variant-numeric: tabular-nums;
}
.candidate-event-detail {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  color: var(--argus-text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.meta-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 20px;
}
.meta-panel {
  border: 1px solid var(--argus-border);
  background: var(--argus-card-bg-solid);
  border-radius: 6px;
  overflow: hidden;
}
.meta-panel-hd {
  padding: 10px 14px;
  border-bottom: 1px solid var(--argus-border);
  font-size: 13px;
  font-weight: 500;
  color: var(--argus-text-muted);
  display: flex;
  justify-content: space-between;
}
.meta-panel-hd b {
  color: #f59e0b;
}
.meta-panel-bd {
  padding: 12px 14px;
}
.meta-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid var(--argus-border);
  font-size: 13px;
  line-height: 1.8;
}
.meta-row:last-child {
  border-bottom: none;
}
.meta-k {
  color: var(--argus-text-muted);
  font-size: 13px;
  min-width: 70px;
}
.meta-v {
  color: var(--argus-text);
  font-size: 13px;
}
.meta-v--notes {
  max-width: 180px;
  text-align: right;
}
.meta-v--long {
  max-width: 260px;
  text-align: right;
  overflow-wrap: anywhere;
  line-height: 1.5;
}
.meta-v--classification,
.meta-v--segmentation {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.meta-v--corroboration {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.correlation-partner {
  font-size: 11px;
  color: var(--argus-text-muted);
  font-variant-numeric: tabular-nums;
}
.classification-conf,
.segmentation-area {
  font-size: 11px;
  color: var(--argus-text-muted);
  font-variant-numeric: tabular-nums;
}

.actions-panel-bd {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.status-display {
  text-align: center;
  padding: 12px 0;
}

/* C-AlertTrainingLink: model link + FP feedback hint row */
.meta-v--model-link {
  display: inline-flex;
  align-items: center;
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.model-version-readonly {
  font-family: monospace;
  font-size: 12px;
}
.meta-row--feedback-hint {
  justify-content: center;
  padding-top: 10px;
}

</style>
