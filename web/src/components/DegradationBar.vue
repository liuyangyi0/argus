<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Tag, Typography } from 'ant-design-vue'
import { DownOutlined } from '@ant-design/icons-vue'
import { getDegradationSummary } from '../api'
import { getModelsStatus } from '../api/system'
import type { ModelHealthStatus } from '../types/api'
import { formatDuration } from '../utils/time'
import { useWebSocket } from '../composables/useWebSocket'

interface DegradationEvent {
  event_id: string
  level: 'info' | 'warning' | 'moderate' | 'severe'
  category: string
  camera_id: string | null
  title: string
  impact: string
  action: string
  started_at: number
}

const events = ref<DegradationEvent[]>([])
const modelEvents = ref<DegradationEvent[]>([])
const expanded = ref(false)
const summaryFailureStartedAt = ref<number | null>(null)
const modelFailureStartedAt = ref<number | null>(null)
const summaryLoaded = ref(false)
const modelsLoaded = ref(false)
let summaryRequestRevision = 0
let modelRequestRevision = 0

async function fetchSummary() {
  const revision = ++summaryRequestRevision
  try {
    const res = await getDegradationSummary()
    if (revision !== summaryRequestRevision) return
    events.value = res.events || []
    summaryLoaded.value = true
    summaryFailureStartedAt.value = null
  } catch {
    if (revision !== summaryRequestRevision) return
    summaryFailureStartedAt.value ??= Date.now() / 1000
  }
}

function expectedBackend(model: ModelHealthStatus): string {
  const expected = model.extra?.expected_backend
  return typeof expected === 'string' && expected ? expected : 'auto'
}

function modelDegradationEvent(model: ModelHealthStatus): DegradationEvent | null {
  if (model.name !== 'anomaly') return null
  const expected = expectedBackend(model)
  const actual = model.loaded ? model.backend : '未加载'
  const mismatch = expected !== 'auto' && model.backend !== expected
  const fallback = ['ssim-fallback', 'none', 'disabled'].includes(model.backend) || !model.loaded
  if (!mismatch && !fallback) return null
  return {
    event_id: `model:${model.camera_id}:${model.name}`,
    level: mismatch ? 'severe' : 'warning',
    category: mismatch ? 'model_backend_mismatch' : 'model_fallback',
    camera_id: model.camera_id,
    title: mismatch ? '模型后端不符合要求' : '主模型未加载或已回退',
    impact: mismatch
      ? `${model.name} 当前 ${actual}，期望 ${expected}`
      : `${model.name} 当前 ${actual}`,
    action: '检查模型发布',
    started_at: model.last_error_ts || model.last_success_ts || Date.now() / 1000,
  }
}

async function fetchModelDegradations() {
  const revision = ++modelRequestRevision
  try {
    const models = await getModelsStatus()
    if (revision !== modelRequestRevision) return
    modelEvents.value = models
      .map(modelDegradationEvent)
      .filter((event): event is DegradationEvent => Boolean(event))
    modelsLoaded.value = true
    modelFailureStartedAt.value = null
  } catch {
    if (revision !== modelRequestRevision) return
    // A failed query is not evidence of recovery. Keep the last successful
    // snapshot and explicitly mark it stale, including on the first failure.
    modelFailureStartedAt.value ??= Date.now() / 1000
  }
}

async function refreshAll() {
  await Promise.allSettled([fetchSummary(), fetchModelDegradations()])
}

useWebSocket<{ type?: string }>({
  topics: ['degradation', 'models', 'model_release'],
  onMessage(topic, data) {
    if (
      data?.type === 'degradation.new'
      || data?.type === 'degradation.resolved'
      || topic === 'models'
      || topic === 'model_release'
    ) {
      refreshAll()
    }
  },
  fallbackPoll: refreshAll,
  fallbackInterval: 30000,
})

onMounted(refreshAll)

const levelColor: Record<string, string> = {
  info: '#2563eb',
  warning: '#d97706',
  moderate: '#d97706',
  severe: '#e5484d',
}

const levelLabel: Record<string, string> = {
  info: '提示',
  warning: '警告',
  moderate: '中度',
  severe: '严重',
}

const queryStatusEvents = computed<DegradationEvent[]>(() => {
  const unknown: DegradationEvent[] = []
  if (modelFailureStartedAt.value !== null) {
    unknown.push({
      event_id: 'query:model-status', level: 'warning', category: 'model_status_unknown',
      camera_id: null, title: '模型状态未知',
      impact: modelsLoaded.value
        ? '查询失败，保留上次模型状态；数据可能已过期，不能视为恢复'
        : '尚未取得模型状态，无法确认主模型及 CUDA 后端是否正常',
      action: '检查连接并重试', started_at: modelFailureStartedAt.value,
    })
  }
  if (summaryFailureStartedAt.value !== null) {
    unknown.push({
      event_id: 'query:degradation-summary', level: 'warning', category: 'degradation_status_unknown',
      camera_id: null, title: '系统降级状态未知',
      impact: summaryLoaded.value
        ? '查询失败，保留上次降级状态；数据可能已过期'
        : '尚未取得系统降级状态，不能确认系统已恢复',
      action: '检查连接并重试', started_at: summaryFailureStartedAt.value,
    })
  }
  return unknown
})
const allEvents = computed(() => [...queryStatusEvents.value, ...modelEvents.value, ...events.value])
const activeCount = computed(() => allEvents.value.length)
const maxLevel = computed(() => {
  const severity: Record<string, number> = { info: 1, warning: 2, moderate: 3, severe: 4 }
  const highest = allEvents.value.reduce<DegradationEvent | null>((best, event) => {
    if (best === null) return event
    return severity[event.level] > severity[best.level] ? event : best
  }, null)
  return highest?.level ?? null
})

const barBackground = computed(() => {
  if (!maxLevel.value) return 'transparent'
  const colors: Record<string, string> = {
    info: 'rgba(37, 99, 235, 0.05)',
    warning: 'rgba(217, 119, 6, 0.05)',
    moderate: 'rgba(217, 119, 6, 0.08)',
    severe: 'rgba(229, 72, 77, 0.06)',
  }
  return colors[maxLevel.value] || 'transparent'
})

const barBorderColor = computed(() => {
  if (!maxLevel.value) return 'transparent'
  return levelColor[maxLevel.value] || 'transparent'
})

const displayEvents = computed(() => expanded.value ? allEvents.value : allEvents.value.slice(0, 3))
const hasMore = computed(() => activeCount.value > 3)

</script>

<template>
  <div
    v-if="activeCount > 0"
    role="status"
    aria-live="polite"
    :class="['degradation-bar', maxLevel === 'severe' && 'degradation-bar--severe']"
    :style="{
      background: barBackground,
      borderBottom: `1px solid ${barBorderColor}`,
      padding: '8px 24px',
      cursor: hasMore ? 'pointer' : 'default',
    }"
    @click="hasMore && (expanded = !expanded)"
  >
    <div v-for="evt in displayEvents" :key="evt.event_id" style="margin-bottom: 4px; display: flex; align-items: center; gap: 10px; flex-wrap: nowrap">
      <!-- Severity tag -->
      <Tag :color="levelColor[evt.level]" style="margin: 0; min-width: 40px; text-align: center; font-size: 11px; flex-shrink: 0">
        {{ levelLabel[evt.level] || evt.level }}
      </Tag>
      <!-- Title -->
      <Typography.Text strong style="flex-shrink: 0; font-size: 13px">{{ evt.title }}</Typography.Text>
      <!-- Affected camera -->
      <Tag v-if="evt.camera_id" size="small" style="margin: 0; font-size: 10px; background: rgba(255,255,255,0.06); border-color: var(--line-2); flex-shrink: 0">
        {{ evt.camera_id }}
      </Tag>
      <Tag v-else size="small" style="margin: 0; font-size: 10px; background: rgba(255,255,255,0.06); border-color: var(--line-2); flex-shrink: 0">
        全系统
      </Tag>
      <!-- Duration -->
      <Typography.Text type="secondary" style="font-size: 11px; flex-shrink: 0">
        {{ formatDuration(evt.started_at) }}
      </Typography.Text>
      <!-- Impact -->
      <Typography.Text type="secondary" style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px">
        {{ evt.impact }}
      </Typography.Text>
      <!-- Action link -->
      <Typography.Text style="color: #60a5fa; flex-shrink: 0; font-size: 12px">{{ evt.action }}</Typography.Text>
    </div>
    <!-- Expand/collapse -->
    <div v-if="hasMore" style="text-align: center; color: #9ca3af; font-size: 12px; margin-top: 4px; display: flex; align-items: center; justify-content: center; gap: 4px">
      <template v-if="!expanded">+{{ activeCount - 3 }} 项降级</template>
      <template v-else>收起</template>
      <DownOutlined :style="{ fontSize: '10px', transform: expanded ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.2s' }" />
    </div>
  </div>
</template>

<style scoped>
.degradation-bar {
  transition: max-height 0.3s ease, padding 0.3s ease;
}
.degradation-bar--severe {
  animation: degrade-pulse 2s ease-in-out infinite;
}
@keyframes degrade-pulse {
  0%, 100% { border-bottom-color: rgba(229, 72, 77, 0.18); }
  50% { border-bottom-color: rgba(229, 72, 77, 0.6); }
}
</style>
