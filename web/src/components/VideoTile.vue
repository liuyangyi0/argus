<script setup lang="ts">
import { computed, onMounted, onUnmounted, watch, ref } from 'vue'
import { Badge, Typography } from 'ant-design-vue'
import Sparkline from './Sparkline.vue'
import { useGo2RTC } from '../composables/useGo2RTC'

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

// go2rtc WebRTC/MSE player
const cameraIdRef = computed(() => props.camera.camera_id)
const { videoRef, status: streamStatus, start, stop } = useGo2RTC(cameraIdRef)

// Legacy MJPEG fallback URL
const mjpegUrl = computed(() => `/api/cameras/${props.camera.camera_id}/stream`)

// Visibility-based streaming: stop when tile is not visible
const tileRef = ref<HTMLElement | null>(null)
let observer: IntersectionObserver | null = null

onMounted(() => {
  // IntersectionObserver is the sole trigger for start/stop —
  // it fires immediately for visible elements on mount.
  if (tileRef.value) {
    observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && props.camera.status === 'online') {
          if (streamStatus.value === 'idle') start()
        } else if (!entry.isIntersecting) {
          stop()
        }
      },
      { threshold: 0.1 },
    )
    observer.observe(tileRef.value)
  }
})

watch(() => props.camera.status, (newStatus, oldStatus) => {
  if (newStatus === oldStatus) return
  if (newStatus === 'online') {
    start()
  } else {
    stop()
  }
})

onUnmounted(() => {
  observer?.disconnect()
})
</script>

<template>
  <div
    ref="tileRef"
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
      <!-- WebRTC / MSE via go2rtc -->
      <video
        v-if="camera.status === 'online' && (streamStatus === 'playing' || streamStatus === 'connecting')"
        ref="videoRef"
        autoplay
        muted
        playsinline
        style="width: 100%; height: 100%; object-fit: contain; display: block"
      />
      <!-- MJPEG fallback when go2rtc is unavailable -->
      <img
        v-else-if="camera.status === 'online' && streamStatus === 'fallback'"
        :src="mjpegUrl"
        style="width: 100%; height: 100%; object-fit: contain; display: block"
        :alt="camera.camera_id"
      />
      <!-- Offline state -->
      <div
        v-else-if="camera.status !== 'online'"
        style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #4a5568; font-size: 13px"
      >
        {{ camera.degradation === 'rtsp_broken' ? '重连中...' : '离线' }}
      </div>
      <!-- Connecting indicator -->
      <div
        v-if="streamStatus === 'connecting'"
        style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #64748b; font-size: 12px"
      >
        连接中...
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
