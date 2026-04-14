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
  active_alert: {
    alert_id: string;
    severity: string;
    anomaly_score?: number;
    timestamp?: number;
    created_at?: number;
    description?: string;
  } | null
  degradation: string | null
  frames_dropped?: number
  backpressured?: boolean
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
    if (sev === 'high') return { border: '3px solid var(--red)', animation: 'pulse-red 2s ease-in-out infinite' }
    if (sev === 'medium') return { border: '2px solid var(--amber)' }
    return { border: '2px solid var(--amber)' }
  }
  if (props.camera.current_score > 0.5) return { border: '2px solid var(--amber)' }
  return { border: '1px solid var(--line-2)' }
})

// go2rtc WebRTC/MSE player
const cameraIdRef = computed(() => props.camera.camera_id)
const _go2rtc = useGo2RTC(cameraIdRef)
// @ts-expect-error TS6133 — used as template ref="videoRef"
const videoRef = _go2rtc.videoRef
const mjpegRef = _go2rtc.mjpegRef
const streamStatus = _go2rtc.status
const { start, stop } = _go2rtc

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
    class="rounded-md overflow-hidden bg-bg flex flex-col h-full"
    :style="borderStyle"
    @dblclick="emit('dblclick', camera.camera_id)"
  >
    <!-- Header: 20px status bar -->
    <div class="flex items-center gap-1.5 px-2.5 py-0.5 bg-transparent shrink-0 min-h-[20px]">
      <Typography.Text strong class="text-xs flex-1 overflow-hidden text-ellipsis whitespace-nowrap">
        {{ camera.name || camera.camera_id }}
      </Typography.Text>
      <Typography.Text v-if="camera.model_version" type="secondary" class="text-[9px] shrink-0">
        {{ camera.model_version.slice(0, 8) }}
      </Typography.Text>
      <!-- Status icons cluster -->
      <div class="flex items-center gap-1 shrink-0">
        <!-- Online/offline -->
        <Tooltip :title="camera.status === 'online' ? '在线' : '离线'">
          <CheckCircleOutlined
            v-if="camera.status === 'online'"
            class="text-xs text-green-500"
          />
          <span v-else class="inline-block w-2 h-2 rounded-full bg-slate-500" />
        </Tooltip>
        <!-- Degradation warning -->
        <Tooltip v-if="camera.degradation" :title="`降级: ${camera.degradation}`">
          <WarningOutlined class="text-xs text-amber-500" />
        </Tooltip>
        <!-- Active alert -->
        <Tooltip v-if="camera.active_alert" :title="`活跃告警 (${camera.active_alert.severity})`">
          <BellOutlined
            class="text-xs cursor-pointer"
            :style="{
              color: camera.active_alert.severity === 'high' ? 'var(--red)' : 'var(--amber)',
            }"
            @click="handleAlertBadgeClick"
          />
        </Tooltip>
      </div>
    </div>

    <!-- Video stream -->
    <div class="flex-1 min-h-0 relative bg-black">
      <!-- WebRTC / MSE via go2rtc -->
      <video
        ref="videoRef"
        autoplay
        muted
        playsinline
        v-show="camera.status === 'online' && (streamStatus === 'playing' || streamStatus === 'connecting')"
        class="w-full h-full object-cover block"
      />
      <!-- MJPEG fallback -->
      <img
        ref="mjpegRef"
        v-if="camera.status === 'online' && streamStatus === 'fallback'"
        :src="mjpegUrl"
        class="w-full h-full object-cover block"
        :alt="camera.camera_id"
      />
      <!-- Offline state -->
      <div
        v-if="camera.status !== 'online'"
        class="w-full h-full flex items-center justify-center text-slate-600 text-[13px]"
      >
        {{ camera.degradation === 'rtsp_broken' ? '重连中...' : '离线' }}
      </div>
      <!-- Connecting indicator -->
      <div
        v-if="camera.status === 'online' && streamStatus === 'connecting'"
        class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-slate-500 text-xs"
      >
        连接中...
      </div>
      <!-- Error state: stream failed -->
      <div
        v-if="camera.status === 'online' && streamStatus === 'error'"
        class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center text-red-500 text-xs cursor-pointer"
        @click.stop="start()"
      >
        <div class="mb-1">连接失败</div>
        <div class="text-[11px] text-slate-500">点击重试</div>
      </div>
      <!-- FPS label overlay -->
      <div
        v-if="camera.status === 'online'"
        class="absolute top-1 right-1 bg-black/60 px-1.5 py-px rounded-[3px] text-[10px] text-gray-400 pointer-events-none"
      >
        {{ camera.fps || '--' }} FPS
      </div>
      <!-- Backpressure warning badge -->
      <Tooltip v-if="camera.frames_dropped && camera.frames_dropped > 0" :title="`丢帧: ${camera.frames_dropped}`">
        <div class="absolute top-1 left-1 px-1.5 py-px rounded-[3px] text-[10px] text-white pointer-events-none" :class="camera.backpressured ? 'bg-red-500/90 animate-[pulse-red_2s_ease-in-out_infinite]' : 'bg-amber-500/85'">
          {{ camera.frames_dropped }}
        </div>
      </Tooltip>
    </div>

    <!-- Footer: 28px status bar -->
    <div class="flex items-center gap-2 px-2.5 py-1 bg-transparent shrink-0 min-h-[28px]">
      <div class="flex-1 min-w-0">
        <Sparkline
          :data="camera.score_sparkline"
          :width="0"
          :height="22"
          :color="camera.current_score > 0.7 ? '#e5484d' : '#2563eb'"
          class="w-full"
        />
      </div>
      <Badge
        v-if="camera.alert_count_today > 0"
        :count="camera.alert_count_today"
        :number-style="{ backgroundColor: '#e5484d', fontSize: '10px', minWidth: '16px', height: '16px', lineHeight: '16px', padding: '0 4px', boxShadow: 'none' }"
        class="shrink-0"
      />
      <Typography.Text v-else type="secondary" class="text-[10px] shrink-0">
        0
      </Typography.Text>
    </div>
  </div>
</template>

<style scoped>
@keyframes pulse-red {
  0%, 100% { border-color: var(--red); }
  50% { border-color: #991b1b; }
}
</style>
