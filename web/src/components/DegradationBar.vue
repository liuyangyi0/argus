<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Tag, Typography } from 'ant-design-vue'
import { getDegradationSummary } from '../api'
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
    :style="{
      background: barBackground,
      borderBottom: `1px solid ${barBorderColor}`,
      padding: '8px 24px',
      cursor: hasMore ? 'pointer' : 'default',
    }"
    @click="hasMore && (expanded = !expanded)"
  >
    <div v-for="evt in displayEvents" :key="evt.event_id" style="margin-bottom: 4px; display: flex; align-items: center; gap: 12px">
      <Tag :color="levelColor[evt.level]" style="margin: 0; min-width: 48px; text-align: center">
        {{ evt.level === 'severe' ? '严重' : evt.level === 'moderate' ? '中度' : evt.level === 'warning' ? '警告' : '提示' }}
      </Tag>
      <Typography.Text strong style="flex-shrink: 0">{{ evt.title }}</Typography.Text>
      <Typography.Text type="secondary" style="flex: 1">{{ evt.impact }}</Typography.Text>
      <Typography.Text style="color: #60a5fa; flex-shrink: 0">{{ evt.action }}</Typography.Text>
    </div>
    <div v-if="hasMore && !expanded" style="text-align: center; color: #9ca3af; font-size: 12px">
      +{{ activeCount - 3 }} 项降级... 点击展开
    </div>
  </div>
</template>
