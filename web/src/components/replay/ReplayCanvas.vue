<script setup lang="ts">
import { inject, ref, onMounted, onBeforeUnmount, watch } from 'vue'
import type { useReplayController } from '../../composables/useReplayController'
import { useHeatmapCache } from '../../composables/useHeatmapCache'
import { useCanvasCompositor } from '../../composables/useCanvasCompositor'

const ctrl = inject<ReturnType<typeof useReplayController>>('replayCtrl')!

const containerRef = ref<HTMLDivElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const videoRef = ref<HTMLVideoElement | null>(null)

// Bind the video element to the controller so playback controls work
watch(videoRef, (el) => {
  ctrl.videoEl.value = el
}, { immediate: true })

// Create heatmap cache (auto-starts preloading)
const alertId = ctrl.metadata.value?.alert_id ?? ''
const frameCount = ctrl.metadata.value?.frame_count ?? 0
const heatmapCacheEnabled = ctrl.hasHeatmaps.value && frameCount > 0
const hmCache = heatmapCacheEnabled
  ? useHeatmapCache(alertId, frameCount)
  : { cache: ref(new Map()), loading: ref(false), progress: ref(1), getFrame: () => null, dispose: () => {} }

// Create compositor
const compositor = useCanvasCompositor({
  canvas: canvasRef,
  videoEl: videoRef,
  fps: ctrl.fps,
  frameCount: ref(ctrl.metadata.value?.frame_count ?? 0),
  heatmapCache: hmCache,
  signals: ctrl.signals,
  showHeatmap: ctrl.showHeatmap,
  showBoxes: ctrl.showBoxes,
  showTrajectory: ctrl.showTrajectory,
  showHud: ctrl.showHud,
  heatmapOpacity: ctrl.heatmapOpacity,
  metadata: ctrl.metadata,
  trajectoryFits: ctrl.trajectoryFits,
})

// Sync compositor currentIndex back to the controller
watch(() => compositor.currentIndex.value, (idx) => {
  ctrl.currentIndex.value = idx
})

// Sync playing state back to the controller
watch(() => compositor.playing.value, (v) => {
  ctrl.playing.value = v
})

// ResizeObserver: track container size, update canvas dimensions
let resizeObs: ResizeObserver | null = null

onMounted(() => {
  if (containerRef.value) {
    const update = () => {
      if (!containerRef.value) return
      const w = containerRef.value.clientWidth
      const h = containerRef.value.clientHeight
      compositor.updateCanvasSize(w, h)
      // Re-render if paused
      if (!ctrl.playing.value) {
        compositor.renderOnce()
      }
    }
    resizeObs = new ResizeObserver(update)
    resizeObs.observe(containerRef.value)
    update()
  }
})

onBeforeUnmount(() => {
  compositor.stop()
  // hmCache disposes itself via its own onBeforeUnmount
  resizeObs?.disconnect()
})

/* ── Video event handlers ── */

function onVideoCanPlay() {
  ctrl.videoError.value = ''
  if (ctrl.pendingSeekIndex.value !== null && videoRef.value) {
    const seekTime = ctrl.pendingSeekIndex.value / ctrl.fps.value
    videoRef.value.currentTime = seekTime
    ctrl.pendingSeekIndex.value = null
  }
  // Start the compositor render loop
  compositor.start()
  compositor.renderOnce()
}

function onVideoPlay() {
  compositor.playing.value = true
  ctrl.videoError.value = ''
  // Re-start RVFC / RAF loop
  compositor.start()
}

function onVideoPause() {
  compositor.playing.value = false
  compositor.renderOnce()
}

function onVideoEnded() {
  compositor.playing.value = false
  compositor.renderOnce()
}

function onVideoError() {
  const el = videoRef.value
  if (!el?.error) return
  ctrl.videoError.value = `视频错误 (code=${el.error.code})`
}

function onLoadedMetadata() {
  // Initial render once metadata is available
  compositor.renderOnce()
}

// When controller seeks (e.g. from timeline scrubber), re-render
watch(() => ctrl.currentIndex.value, (idx) => {
  if (!ctrl.playing.value && videoRef.value) {
    videoRef.value.currentTime = idx / ctrl.fps.value
    compositor.renderOnce()
  }
})

// Speed is handled by the controller's own watcher (useReplayController.ts)
</script>

<template>
  <div ref="containerRef" class="replay-canvas-container">
    <!-- Hidden video: decode only, not displayed -->
    <video
      ref="videoRef"
      :src="ctrl.videoUrl.value"
      preload="auto"
      playsinline
      style="display: none"
      @canplay="onVideoCanPlay"
      @play="onVideoPlay"
      @pause="onVideoPause"
      @ended="onVideoEnded"
      @loadedmetadata="onLoadedMetadata"
      @error="onVideoError"
    />

    <!-- Single visible canvas: all layers composited here -->
    <canvas
      ref="canvasRef"
      class="replay-canvas"
      @click="ctrl.togglePlay"
    />

    <!-- Error overlay (DOM, only shown on error) -->
    <div v-if="ctrl.videoError.value" class="replay-error">
      {{ ctrl.videoError.value }}
    </div>
  </div>
</template>

<style scoped>
.replay-canvas-container {
  position: relative;
  background: #000;
  aspect-ratio: 16/9;
  overflow: hidden;
}
.replay-canvas {
  width: 100%;
  height: 100%;
  display: block;
  cursor: pointer;
}
.replay-error {
  position: absolute;
  bottom: 8px; left: 8px; right: 8px;
  background: rgba(239,68,68,0.9);
  color: #fff;
  padding: 6px 10px;
  border-radius: 4px;
  font-size: 12px;
}
</style>
