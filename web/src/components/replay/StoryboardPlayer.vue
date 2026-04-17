<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch } from 'vue'
import { useCanvasCompositor } from '../../composables/useCanvasCompositor'
import { getReplayMetadata, getReplaySignals, getReplayVideoUrl } from '../../api'
import type { StoryboardCamera } from '../../types/api'

/**
 * Renders a single camera inside the storyboard grid.
 *
 * Drives its <video> element from the shared masterTime:
 *   localTime = masterTime - camera.trigger_offset_s
 * When localTime falls outside [0, duration] the canvas is blanked and a
 * "out of range" badge is shown.
 *
 * Reuses `useCanvasCompositor` so heatmap / box / trajectory overlays all
 * work identically to the single-camera ReplayCanvas. No render code is
 * duplicated.
 */
const props = defineProps<{
  camera: StoryboardCamera
  masterTime: number
  playing: boolean
  speed: number
  isPrimary?: boolean
}>()

const emit = defineEmits<{
  (e: 'duration', alertId: string, seconds: number): void
}>()

const containerRef = ref<HTMLDivElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const videoRef = ref<HTMLVideoElement | null>(null)

const metadata = ref<any>(null)
const signals = ref<any>(null)
const loading = ref(true)
const loadError = ref<string | null>(null)
/** Local time the <video> is *supposed* to be at, in seconds. */
const inRange = ref(true)

const videoUrl = ref(getReplayVideoUrl(props.camera.alert_id))

// Static overlay toggles for the storyboard — kept simple; the single-cam
// ReplayPlayer exposes its own fine-grained controls.
const showHeatmap = ref(false)
const showBoxes = ref(false)
const showTrajectory = ref(true)
const showHud = ref(true)
const heatmapOpacity = ref(0.4)

const fps = ref(15)
const frameCount = ref(0)

/* ── Data loading ── */

async function loadCameraData(): Promise<void> {
  loading.value = true
  loadError.value = null
  try {
    const [meta, sig] = await Promise.all([
      getReplayMetadata(props.camera.alert_id),
      getReplaySignals(props.camera.alert_id).catch(() => null),
    ])
    metadata.value = meta
    signals.value = sig
    fps.value = meta?.fps || 15
    frameCount.value = meta?.frame_count || 0
    if (fps.value > 0 && frameCount.value > 0) {
      emit('duration', props.camera.alert_id, frameCount.value / fps.value)
    }
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : '加载失败'
  } finally {
    loading.value = false
  }
}

/* ── Compositor ── */

// Heatmap cache is intentionally a no-op here — the storyboard view favours
// raw video sync over heavy heatmap preload. Users can open a single camera
// in full ReplayView to see heatmaps.
const hmCache = {
  cache: ref(new Map()),
  loading: ref(false),
  progress: ref(1),
  getFrame: () => null,
  dispose: () => {},
}

const compositor = useCanvasCompositor({
  canvas: canvasRef,
  videoEl: videoRef,
  fps,
  frameCount,
  heatmapCache: hmCache,
  signals,
  showHeatmap,
  showBoxes,
  showTrajectory,
  showHud,
  heatmapOpacity,
  metadata,
})

/* ── Resize tracking ── */

let resizeObs: ResizeObserver | null = null

onMounted(() => {
  loadCameraData()
  if (containerRef.value) {
    const update = (): void => {
      if (!containerRef.value) return
      const w = containerRef.value.clientWidth
      const h = containerRef.value.clientHeight
      compositor.updateCanvasSize(w, h)
      compositor.renderOnce()
    }
    resizeObs = new ResizeObserver(update)
    resizeObs.observe(containerRef.value)
    update()
  }
})

onBeforeUnmount(() => {
  compositor.stop()
  resizeObs?.disconnect()
})

/* ── Video lifecycle ── */

function onVideoLoadedMetadata(): void {
  if (videoRef.value && isFinite(videoRef.value.duration)) {
    // Reconcile duration with what we inferred from metadata.
    emit('duration', props.camera.alert_id, videoRef.value.duration)
  }
  compositor.start()
  applyMasterTime()
}

function onVideoError(): void {
  const el = videoRef.value
  loadError.value = el?.error ? `视频错误 (code=${el.error.code})` : '视频加载失败'
}

/* ── Sync: master -> video ── */

function applyMasterTime(): void {
  const vid = videoRef.value
  if (!vid) return
  const dur = isFinite(vid.duration) ? vid.duration : frameCount.value / Math.max(fps.value, 1)
  if (!dur) return

  const local = props.masterTime - props.camera.trigger_offset_s
  const within = local >= 0 && local <= dur
  inRange.value = within

  if (!within) {
    if (!vid.paused) vid.pause()
    // Clamp currentTime to the closer edge so the next resume picks up
    // smoothly when masterTime re-enters range.
    vid.currentTime = local < 0 ? 0 : dur
    return
  }

  // Only seek if drift exceeds ~1 frame; tiny drifts self-correct during
  // normal playback and avoid stuttering seeks.
  const driftThreshold = 1 / Math.max(fps.value, 1) * 1.5
  if (Math.abs(vid.currentTime - local) > driftThreshold) {
    vid.currentTime = local
  }
  vid.playbackRate = Math.max(0.0625, Math.min(16, props.speed))

  if (props.playing && vid.paused) {
    // safePlay: ignore AbortError from user toggles during buffering.
    vid.play().catch(() => { /* ignored */ })
  } else if (!props.playing && !vid.paused) {
    vid.pause()
  }
}

watch(
  () => [props.masterTime, props.playing, props.speed],
  () => applyMasterTime(),
)

// Ensure the overlay canvas re-renders whenever master time advances, even
// during pause (so scrubbing shows the right frame immediately).
watch(() => props.masterTime, () => {
  if (!props.playing) compositor.renderOnce()
})
</script>

<template>
  <div ref="containerRef" class="storyboard-player" :class="{ 'is-primary': isPrimary }">
    <video
      ref="videoRef"
      :src="videoUrl"
      preload="auto"
      muted
      playsinline
      style="display: none"
      @loadedmetadata="onVideoLoadedMetadata"
      @error="onVideoError"
    />
    <canvas ref="canvasRef" class="storyboard-canvas" />
    <div class="badge camera-badge">
      <span v-if="isPrimary" class="primary-dot" />
      {{ camera.camera_id }}
      <span v-if="camera.trigger_offset_s !== 0" class="offset-chip">
        {{ camera.trigger_offset_s > 0 ? '+' : '' }}{{ camera.trigger_offset_s.toFixed(2) }}s
      </span>
    </div>
    <div v-if="loading" class="state-overlay">加载中…</div>
    <div v-else-if="loadError" class="state-overlay error">{{ loadError }}</div>
    <div v-else-if="!inRange" class="state-overlay ghost">无此时刻画面</div>
  </div>
</template>

<style scoped>
.storyboard-player {
  position: relative;
  background: #000;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  border: 1px solid var(--line-2, #2a2a2a);
  border-radius: 4px;
}
.storyboard-player.is-primary {
  border-color: #f59e0b;
  box-shadow: 0 0 0 1px rgba(245, 158, 11, 0.35);
}
.storyboard-canvas {
  width: 100%;
  height: 100%;
  display: block;
}
.badge.camera-badge {
  position: absolute;
  top: 8px;
  left: 8px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 9px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 11px;
  font-weight: 600;
  color: #fff;
  background: rgba(0, 0, 0, 0.6);
  border-radius: 3px;
  letter-spacing: 0.02em;
  pointer-events: none;
}
.primary-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #f59e0b;
  box-shadow: 0 0 6px rgba(245, 158, 11, 0.8);
}
.offset-chip {
  font-size: 10px;
  color: #bdf;
  opacity: 0.9;
  margin-left: 2px;
}
.state-overlay {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: rgba(255, 255, 255, 0.75);
  background: rgba(0, 0, 0, 0.55);
  font-size: 12px;
  pointer-events: none;
}
.state-overlay.error {
  color: #fecaca;
  background: rgba(127, 29, 29, 0.7);
}
.state-overlay.ghost {
  background: rgba(0, 0, 0, 0.35);
  color: rgba(255, 255, 255, 0.55);
  font-style: italic;
}
</style>
