<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Tag, Typography } from 'ant-design-vue'
import { DownOutlined } from '@ant-design/icons-vue'
import { getDegradationSummary } from '../api'
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
const activeCount = ref(0)
const maxLevel = ref<string | null>(null)
const expanded = ref(false)

async function fetchSummary() {
  try {
    const res = await getDegradationSummary()
    events.value = res.events || []
    activeCount.value = res.active_count || 0
    maxLevel.value = res.max_level
  } catch { /* silent */ }
}

useWebSocket({
  topics: ['degradation'],
  onMessage(_topic, data) {
    if (data?.type === 'degradation.new' || data?.type === 'degradation.resolved') {
      fetchSummary()
    }
  },
  fallbackPoll: fetchSummary,
  fallbackInterval: 30000,
})

onMounted(fetchSummary)

const levelColor: Record<string, string> = {
  info: '#3b82f6',
  warning: '#f59e0b',
  moderate: '#f97316',
  severe: '#ef4444',
}

const levelLabel: Record<string, string> = {
  info: '提示',
  warning: '警告',
  moderate: '中度',
  severe: '严重',
}

const barBackground = computed(() => {
  if (!maxLevel.value) return 'transparent'
  const colors: Record<string, string> = {
    info: 'rgba(59, 130, 246, 0.15)',
    warning: 'rgba(245, 158, 11, 0.15)',
    moderate: 'rgba(249, 115, 22, 0.15)',
    severe: 'rgba(239, 68, 68, 0.15)',
  }
  return colors[maxLevel.value] || 'transparent'
})

const barBorderColor = computed(() => {
  if (!maxLevel.value) return 'transparent'
  return levelColor[maxLevel.value] || 'transparent'
})

const displayEvents = computed(() => expanded.value ? events.value : events.value.slice(0, 3))
const hasMore = computed(() => activeCount.value > 3)

</script>

<template>
  <div
    v-if="activeCount > 0"
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
      <Tag v-if="evt.camera_id" size="small" style="margin: 0; font-size: 10px; background: rgba(255,255,255,0.06); border-color: var(--argus-border); flex-shrink: 0">
        {{ evt.camera_id }}
      </Tag>
      <Tag v-else size="small" style="margin: 0; font-size: 10px; background: rgba(255,255,255,0.06); border-color: var(--argus-border); flex-shrink: 0">
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
  0%, 100% { border-bottom-color: #ef4444; }
  50% { border-bottom-color: #991b1b; }
}
</style>
