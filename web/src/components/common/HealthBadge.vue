<script setup lang="ts">
import { computed } from 'vue'
import type { CameraHealth } from '../../types/api'

const props = defineProps<{
  health?: CameraHealth | null
  /** Fallback when health is unavailable (e.g. health monitor disabled). */
  connected?: boolean
}>()

const palette: Record<string, { color: string; label: string }> = {
  healthy: { color: '#52c41a', label: '正常' },
  ok: { color: '#52c41a', label: '正常' },
  degraded: { color: '#faad14', label: '降级' },
  warning: { color: '#faad14', label: '警告' },
  unhealthy: { color: '#ff4d4f', label: '异常' },
  error: { color: '#ff4d4f', label: '异常' },
  offline: { color: '#bfbfbf', label: '离线' },
  unknown: { color: '#bfbfbf', label: '未知' },
}

const meta = computed(() => {
  const status = (props.health?.status ?? '').toLowerCase()
  if (status && palette[status]) return palette[status]
  if (props.connected) return palette.ok
  if (props.connected === false) return palette.offline
  return palette.unknown
})

const tooltip = computed(() => {
  const lines: string[] = []
  if (props.health?.status) lines.push(`状态：${props.health.status}`)
  if (props.health?.last_heartbeat) {
    lines.push(`最近心跳：${props.health.last_heartbeat}`)
  }
  if (props.health?.fps !== undefined && props.health?.fps !== null) {
    lines.push(`FPS：${props.health.fps}`)
  }
  const errs = props.health?.recent_errors
  if (Array.isArray(errs) && errs.length > 0) {
    lines.push('最近错误：')
    for (const e of errs.slice(0, 3)) lines.push(`  · ${e}`)
  }
  return lines.join('\n') || meta.value.label
})
</script>

<template>
  <span class="health-badge" :title="tooltip">
    <span class="dot" :style="{ backgroundColor: meta.color }" />
    <span>{{ meta.label }}</span>
  </span>
</template>

<style scoped>
.health-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}
.dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
</style>
