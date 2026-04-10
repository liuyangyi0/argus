<script setup lang="ts">
import { computed, onMounted, onUnmounted, onActivated, onDeactivated, watch, ref } from 'vue'
import { Badge, Typography, Tooltip } from 'ant-design-vue'
import {
  WarningOutlined,
  BellOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons-vue'
import Sparkline from './Sparkline.vue'
import { useGo2RTC } from '../composables/useGo2RTC'

export interface CameraTileData {
  camera_id: string
  name: string
  status: string
  model_version?: string
  fps?: number
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
  (e: 'alert-click', alertId: string): void
}>()

const borderStyle = computed(() => {
  if (props.camera.active_alert) {
    const sev = props.camera.active_alert.severity
    if (sev === 'high') return { border: '3px solid #ef4444', animation: 'pulse-red 2s ease-in-out infinite' }
    if (sev === 'medium') return { border: '2px solid #f97316' }
    return { border: '2px solid #f59e0b' }
  }
  if (props.camera.current_score > 0.5) return { border: '2px solid #f59e0b' }
  return { border: '1px solid var(--argus-border)' }
})

const statusBadge = computed(() => {
  if (props.camera.active_alert) return 'error'
  if (props.camera.degradation) return 'warning'
  if (props.camera.status === 'online') return 'success'
  return 'default'
})

// go2rtc WebRTC/MSE player
const cameraIdRef = computed(() => props.camera.camera_id)
const { videoRef, mjpegRef, status: streamStatus, start, stop } = useGo2RTC(cameraIdRef)

// Legacy MJPEG fallback URL
const mjpegUrl = computed(() => `/api/cameras/${props.camera.camera_id}/stream`)

// Visibility-based streaming: stop when tile is not visible
const tileRef = ref<HTMLElement | null>(null)
let observer: IntersectionObserver | null = null

onMounted(() => {
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

// KeepAlive support: pause MJPEG fallback when page is cached (saves HTTP connections),
// but keep WebRTC/MSE alive (they use go2rtc's separate port, no resource cost).
onDeactivated(() => {
  if (streamStatus.value === 'fallback' && mjpegRef.value) {
    mjpegRef.value.src = ''
  }
})

onActivated(() => {
  // Re-attach MJPEG src if was in fallback mode
  if (streamStatus.value === 'fallback' && mjpegRef.value) {
    mjpegRef.value.src = mjpegUrl.value
  }
  // Restart stream if idle or error (e.g., WebRTC died while page was cached)
  if ((streamStatus.value === 'idle' || streamStatus.value === 'error') && props.camera.status === 'online') {
    start()
  }
})

function handleAlertBadgeClick(e: MouseEvent) {
  e.stopPropagation()
  if (props.camera.active_alert) {
    emit('alert-click', props.camera.active_alert.alert_id)
  }
}
</script>

<template>
  <div
    ref="tileRef"
    :style="{
      ...borderStyle,
      borderRadius: '6px',
      overflow: 'hidden',
      background: 'var(--argus-card-bg-solid)',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
    }"
    @dblclick="emit('dblclick', camera.camera_id)"
  >
    <!-- Header: 20px status bar -->
    <div style="display: flex; align-items: center; gap: 6px; padding: 3px 10px; background: var(--argus-header-bg); flex-shrink: 0; min-height: 20px">
      <Typography.Text strong style="font-size: 12px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">
        {{ camera.name || camera.camera_id }}
      </Typography.Text>
      <Typography.Text v-if="camera.model_version" type="secondary" style="font-size: 9px; flex-shrink: 0">
        {{ camera.model_version.slice(0, 8) }}
      </Typography.Text>
      <!-- Status icons cluster -->
      <div style="display: flex; align-items: center; gap: 4px; flex-shrink: 0">
        <!-- Online/offline -->
        <Tooltip :title="camera.status === 'online' ? '在线' : '离线'">
          <CheckCircleOutlined
            v-if="camera.status === 'online'"
            style="font-size: 12px; color: #22c55e"
          />
          <span v-else style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #6b7280" />
        </Tooltip>
        <!-- Degradation warning -->
        <Tooltip v-if="camera.degradation" :title="`降级: ${camera.degradation}`">
          <WarningOutlined style="font-size: 12px; color: #f59e0b" />
        </Tooltip>
        <!-- Active alert -->
        <Tooltip v-if="camera.active_alert" :title="`活跃告警 (${camera.active_alert.severity})`">
          <BellOutlined
            :style="{
              fontSize: '12px',
              color: camera.active_alert.severity === 'high' ? '#ef4444' : '#f97316',
              cursor: 'pointer',
            }"
            @click="handleAlertBadgeClick"
          />
        </Tooltip>
      </div>
    </div>

    <!-- Video stream -->
    <div style="flex: 1; min-height: 0; position: relative; background: #000">
      <!-- WebRTC / MSE via go2rtc -->
      <video
        ref="videoRef"
        autoplay
        muted
        playsinline
        v-show="camera.status === 'online' && (streamStatus === 'playing' || streamStatus === 'connecting')"
        style="width: 100%; height: 100%; object-fit: contain; display: block"
      />
      <!-- MJPEG fallback -->
      <img
        ref="mjpegRef"
        v-if="camera.status === 'online' && streamStatus === 'fallback'"
        :src="mjpegUrl"
        style="width: 100%; height: 100%; object-fit: contain; display: block"
        :alt="camera.camera_id"
      />
      <!-- Offline state -->
      <div
        v-if="camera.status !== 'online'"
        style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #4a5568; font-size: 13px"
      >
        {{ camera.degradation === 'rtsp_broken' ? '重连中...' : '离线' }}
      </div>
      <!-- Connecting indicator -->
      <div
        v-if="camera.status === 'online' && streamStatus === 'connecting'"
        style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #64748b; font-size: 12px"
      >
        连接中...
      </div>
      <!-- Error state: stream failed -->
      <div
        v-if="camera.status === 'online' && streamStatus === 'error'"
        style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: #ef4444; font-size: 12px; cursor: pointer"
        @click.stop="start()"
      >
        <div style="margin-bottom: 4px">连接失败</div>
        <div style="color: #64748b; font-size: 11px">点击重试</div>
      </div>
      <!-- FPS label overlay -->
      <div
        v-if="camera.status === 'online'"
        style="position: absolute; top: 4px; right: 4px; background: rgba(0,0,0,0.6); padding: 1px 6px; border-radius: 3px; font-size: 10px; color: #9ca3af; pointer-events: none"
      >
        {{ camera.fps || '--' }} FPS
      </div>
    </div>

    <!-- Footer: 28px status bar -->
    <div style="display: flex; align-items: center; gap: 8px; padding: 4px 10px; background: var(--argus-footer-bg); flex-shrink: 0; min-height: 28px">
      <div style="flex: 1; min-width: 0">
        <Sparkline
          :data="camera.score_sparkline"
          :width="0"
          :height="22"
          :color="camera.current_score > 0.7 ? '#ef4444' : '#3b82f6'"
          style="width: 100%"
        />
      </div>
      <Badge
        v-if="camera.alert_count_today > 0"
        :count="camera.alert_count_today"
        :number-style="{ backgroundColor: '#ef4444', fontSize: '10px', minWidth: '16px', height: '16px', lineHeight: '16px', padding: '0 4px', boxShadow: 'none' }"
        style="flex-shrink: 0"
      />
      <Typography.Text v-else type="secondary" style="font-size: 10px; flex-shrink: 0">
        0
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
