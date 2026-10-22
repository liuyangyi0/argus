<script setup lang="ts">
import { computed, onActivated, onBeforeUnmount, onDeactivated, onMounted, watch } from 'vue'
import { streamFallbackReasonLabel, useGo2RTC } from '../composables/useGo2RTC'

const props = defineProps<{
  cameraId: string
  online: boolean
}>()

const cameraIdRef = computed(() => props.cameraId)
const go2rtc = useGo2RTC(cameraIdRef)
// @ts-expect-error TS6133 — used as template ref="videoRef"
const videoRef = go2rtc.videoRef
// @ts-expect-error TS6133 — used as template ref="mjpegRef"
const mjpegRef = go2rtc.mjpegRef
const streamStatus = go2rtc.status
const fallbackReason = go2rtc.fallbackReason
const fallbackDetail = computed(() => streamFallbackReasonLabel(fallbackReason.value))
const mjpegUrl = computed(() => `/api/cameras/${props.cameraId}/stream`)

function syncStream() {
  if (props.online) {
    go2rtc.start()
  } else {
    go2rtc.stop()
  }
}

onMounted(syncStream)
onActivated(syncStream)
onDeactivated(go2rtc.stop)
onBeforeUnmount(go2rtc.stop)

watch([cameraIdRef, () => props.online], syncStream)
</script>

<template>
  <div class="feed-video">
    <video
      ref="videoRef"
      autoplay
      muted
      playsinline
      v-show="online && (streamStatus === 'playing' || streamStatus === 'connecting')"
      class="stream-media"
    />
    <img
      ref="mjpegRef"
      v-if="online && streamStatus === 'fallback'"
      :src="mjpegUrl"
      class="stream-media"
      :alt="cameraId"
    />
    <div v-if="online && streamStatus === 'connecting'" class="stream-status">
      连接中...
    </div>
    <div
      v-if="online && streamStatus === 'fallback'"
      class="fallback-badge"
      role="status"
      :title="fallbackDetail"
    >
      MJPEG fallback · 非实时预览 · {{ fallbackDetail }}
    </div>
    <div v-else-if="!online" class="stream-status">
      OFFLINE
    </div>
  </div>
</template>

<style scoped>
.feed-video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 1;
  background: #000;
}

.stream-media {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.stream-status {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(226, 232, 240, 0.72);
  font-size: 12px;
  background: rgba(0, 0, 0, 0.35);
}

.fallback-badge {
  position: absolute;
  left: 8px;
  bottom: 8px;
  padding: 3px 8px;
  border-radius: 4px;
  background: rgba(217, 119, 6, 0.92);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  pointer-events: none;
}
</style>
