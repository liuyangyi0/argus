<script setup lang="ts">
import { computed } from 'vue'
import { Badge, Typography } from 'ant-design-vue'
import Sparkline from './Sparkline.vue'

export interface CameraTileData {
  camera_id: string
  name: string
  status: string
  model_version?: string
  current_score: number
  score_sparkline: number[]
  alert_count_today: number
  active_alert: { alert_id: string; severity: string } | null
  degradation: string | null
}

const props = defineProps<{
  camera: CameraTileData
  layout: '1x1' | '2x2' | '3x3' | 'focus-main' | 'focus-side'
}>()

const emit = defineEmits<{
  (e: 'dblclick', cameraId: string): void
}>()

const borderStyle = computed(() => {
  if (props.camera.active_alert) {
    const sev = props.camera.active_alert.severity
    if (sev === 'high') return { border: '3px solid #ef4444', animation: 'pulse-red 2s ease-in-out infinite' }
    if (sev === 'medium') return { border: '2px solid #f97316' }
    return { border: '2px solid #f59e0b' }
  }
  if (props.camera.current_score > 0.5) return { border: '2px solid #f59e0b' }
  return { border: '1px solid #2d2d4a' }
})

const statusBadge = computed(() => {
  if (props.camera.active_alert) return 'error'
  if (props.camera.degradation) return 'warning'
  if (props.camera.status === 'online') return 'success'
  return 'default'
})

const streamUrl = computed(() => `/api/cameras/${props.camera.camera_id}/stream`)
</script>

<template>
  <div
    :style="{
      ...borderStyle,
      borderRadius: '6px',
      overflow: 'hidden',
      background: '#1a1a2e',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
    }"
    @dblclick="emit('dblclick', camera.camera_id)"
  >
    <!-- Header -->
    <div style="display: flex; align-items: center; gap: 6px; padding: 6px 10px; background: #0f0f1a; flex-shrink: 0">
      <Badge :status="statusBadge" />
      <Typography.Text strong style="font-size: 12px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">
        {{ camera.name || camera.camera_id }}
      </Typography.Text>
      <Typography.Text v-if="camera.model_version" type="secondary" style="font-size: 10px">
        {{ camera.model_version.slice(0, 8) }}
      </Typography.Text>
    </div>

    <!-- Video stream -->
    <div style="flex: 1; min-height: 0; position: relative; background: #000">
      <img
        v-if="camera.status === 'online'"
        :src="streamUrl"
        style="width: 100%; height: 100%; object-fit: contain; display: block"
        :alt="camera.camera_id"
      />
      <div
        v-else
        style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #4a5568; font-size: 13px"
      >
        {{ camera.degradation === 'rtsp_broken' ? '重连中...' : '离线' }}
      </div>
    </div>

    <!-- Footer -->
    <div style="display: flex; align-items: center; gap: 8px; padding: 4px 10px; background: #0f0f1a; flex-shrink: 0">
      <Sparkline :data="camera.score_sparkline" :width="80" :height="20" :color="camera.current_score > 0.7 ? '#ef4444' : '#3b82f6'" />
      <Typography.Text type="secondary" style="font-size: 11px; margin-left: auto">
        {{ camera.alert_count_today }} 告警
      </Typography.Text>
    </div>
  </div>
</template>

<style scoped>
@keyframes pulse-red {
  0%, 100% { border-color: #ef4444; }
  50% { border-color: #991b1b; }
}
</style>
