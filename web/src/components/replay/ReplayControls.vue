<script setup lang="ts">
import { inject, computed, ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { CameraOutlined, CaretRightOutlined, PauseOutlined } from '@ant-design/icons-vue'
import { formatPlaybackTime } from '../../utils/time'
import type { useReplayController } from '../../composables/useReplayController'

const ctrl = inject<ReturnType<typeof useReplayController>>('replayCtrl')!

const speeds = [0.25, 0.5, 1, 2, 4]

// ── Confidence sparkline under the scrubber ──
const sparkCanvas = ref<HTMLCanvasElement | null>(null)
const SPARK_HEIGHT = 20
let sparkRaf = 0

function drawSparkline() {
  const canvas = sparkCanvas.value
  if (!canvas) return
  const scores: number[] = ctrl.signals.value?.anomaly_scores || []
  const cssWidth = canvas.clientWidth || canvas.parentElement?.clientWidth || 0
  if (cssWidth <= 0) return
  const dpr = Math.max(1, window.devicePixelRatio || 1)
  if (canvas.width !== cssWidth * dpr || canvas.height !== SPARK_HEIGHT * dpr) {
    canvas.width = cssWidth * dpr
    canvas.height = SPARK_HEIGHT * dpr
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cssWidth, SPARK_HEIGHT)

  if (scores.length === 0) return

  const maxScore = Math.max(...scores, 0.01)
  const n = scores.length
  const innerH = SPARK_HEIGHT - 2

  // Baseline guide
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(0, SPARK_HEIGHT - 1)
  ctx.lineTo(cssWidth, SPARK_HEIGHT - 1)
  ctx.stroke()

  // Polyline of scores, normalized to [0, maxScore]
  ctx.strokeStyle = 'rgba(59, 130, 246, 0.4)'
  ctx.fillStyle = 'rgba(59, 130, 246, 0.12)'
  ctx.lineWidth = 1.25
  ctx.beginPath()
  for (let i = 0; i < n; i++) {
    const x = n === 1 ? cssWidth / 2 : (i / (n - 1)) * cssWidth
    const norm = Math.max(0, Math.min(1, scores[i] / maxScore))
    const y = SPARK_HEIGHT - 1 - norm * innerH
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()
  ctx.lineTo(cssWidth, SPARK_HEIGHT - 1)
  ctx.lineTo(0, SPARK_HEIGHT - 1)
  ctx.closePath()
  ctx.fill()

  // Current-frame marker (vertical red line)
  const total = Math.max((ctrl.metadata.value?.frame_count || n) - 1, 1)
  const cursorX = total > 0 ? (ctrl.currentIndex.value / total) * cssWidth : 0
  ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)'
  ctx.lineWidth = 1.25
  ctx.beginPath()
  ctx.moveTo(cursorX, 0)
  ctx.lineTo(cursorX, SPARK_HEIGHT)
  ctx.stroke()
}

function scheduleDraw() {
  if (sparkRaf) cancelAnimationFrame(sparkRaf)
  sparkRaf = requestAnimationFrame(drawSparkline)
}

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  nextTick(drawSparkline)
  if (sparkCanvas.value && typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(() => scheduleDraw())
    resizeObserver.observe(sparkCanvas.value)
  }
})

onUnmounted(() => {
  if (sparkRaf) cancelAnimationFrame(sparkRaf)
  if (resizeObserver) resizeObserver.disconnect()
})

watch(
  () => [ctrl.currentIndex.value, ctrl.signals.value?.anomaly_scores?.length],
  () => scheduleDraw(),
)

// Also redraw when signals is swapped wholesale — e.g. initial arrival or an
// alert switch that reuses this component. The length-only watch above won't
// fire if the number of scores happens to match the previous value.
watch(
  () => ctrl.signals.value,
  () => scheduleDraw(),
  { flush: 'post' },
)

// ── Clip markers ──
function clipLeft(idx: number) {
  const total = Math.max((ctrl.metadata.value?.frame_count || 1) - 1, 1)
  return (Math.max(0, Math.min(idx, total)) / total) * 100
}

function clipWidth(start: number, end: number) {
  const total = Math.max((ctrl.metadata.value?.frame_count || 1) - 1, 1)
  const a = Math.max(0, Math.min(start, total))
  const b = Math.max(0, Math.min(end, total))
  return Math.max(((b - a) / total) * 100, 0.6)
}

function clipTitle(c: { start_index: number; end_index: number; label: string }) {
  const base = `#${c.start_index}–${c.end_index}`
  return c.label ? `${c.label} · ${base}` : base
}

const hasYoloData = computed(() => {
  const boxes = ctrl.signals.value?.yolo_boxes
  return Array.isArray(boxes) && boxes.length > 0
})

const currentTime = computed(() => {
  const fps = ctrl.fps.value || 15
  return formatPlaybackTime(ctrl.currentIndex.value / fps)
})

const totalDuration = computed(() => {
  const fps = ctrl.fps.value || 15
  const frames = ctrl.metadata.value?.frame_count || 0
  return formatPlaybackTime(frames / fps)
})

const progressPct = computed(() => {
  const total = Math.max((ctrl.metadata.value?.frame_count || 1) - 1, 1)
  return (ctrl.currentIndex.value / total) * 100
})

const triggerProgressPct = computed(() => {
  if (!ctrl.metadata.value) return 50
  const triggerIdx = ctrl.metadata.value.trigger_frame_index || 0
  const total = ctrl.metadata.value.frame_count || 1
  return Math.round((triggerIdx / total) * 100)
})

const remainingRecordingSeconds = computed(() => {
  if (!ctrl.metadata.value || ctrl.metadata.value.status !== 'recording') return 0
  const triggerTs = ctrl.metadata.value.trigger_timestamp || 0
  const postSeconds = ctrl.metadata.value.severity === 'low' ? 10 : 30
  const deadline = triggerTs + postSeconds
  const now = Date.now() / 1000
  return Math.max(0, Math.round(deadline - now))
})
</script>

<template>
  <div class="replay-controls-wrapper">
    <!-- Controls bar -->
    <div class="replay-controls">
      <button class="ctrl-btn" @click="ctrl.goToStart" title="Start">&#9198;</button>
      <button class="ctrl-btn" @click="ctrl.stepFrame(-1)" title="-1 frame">&#9664;&#9664;</button>
      <button class="ctrl-btn ctrl-play" @click="ctrl.togglePlay">
        <PauseOutlined v-if="ctrl.playing.value" />
        <CaretRightOutlined v-else />
      </button>
      <button class="ctrl-btn" @click="ctrl.stepFrame(1)" title="+1 frame">&#9654;&#9654;</button>
      <button class="ctrl-btn" @click="ctrl.goToEnd" title="End">&#9197;</button>

      <div class="ctrl-speeds">
        <button
          v-for="s in speeds" :key="s"
          :class="['speed-btn', { on: ctrl.speed.value === s }]"
          @click="ctrl.speed.value = s"
        >{{ s }}x</button>
      </div>

      <!-- Overlay toggles -->
      <div class="ctrl-toggles">
        <button
          :class="['toggle-btn', { on: ctrl.showHeatmap.value }]"
          :disabled="!ctrl.hasHeatmaps.value"
          @click="ctrl.showHeatmap.value = !ctrl.showHeatmap.value"
          title="热力图叠加"
        >热力</button>
        <button
          :class="['toggle-btn', { on: ctrl.showBoxes.value }]"
          :disabled="!hasYoloData"
          @click="ctrl.showBoxes.value = !ctrl.showBoxes.value"
          title="YOLO 检测框"
        >框选</button>
        <button
          :class="['toggle-btn', { on: ctrl.showTrajectory.value }]"
          :disabled="!hasYoloData"
          @click="ctrl.showTrajectory.value = !ctrl.showTrajectory.value"
          title="轨迹叠加"
        >轨迹</button>
        <!-- 会把当前可见画面导出为 PNG 并触发浏览器下载，文件名包含摄像头、告警/回放 id、帧号和时间戳。 -->
        <button
          class="toggle-btn capture-btn"
          :disabled="!ctrl.canvasEl.value"
          @click="ctrl.captureCurrentFrame"
          title="抓取当前画面并下载 PNG"
        >
          <CameraOutlined />
          <span>抓图</span>
        </button>
      </div>

      <span class="ctrl-frame-info">
        <b>{{ currentTime }}</b> / {{ totalDuration }} · FRAME {{ ctrl.currentIndex.value + 1 }} / {{ ctrl.metadata.value.frame_count }}
      </span>
    </div>

    <!-- Timeline scrubber -->
    <div class="replay-timeline">
      <div class="tl-bar">
        <div class="tl-progress" :style="{ width: progressPct + '%' }"></div>
        <!-- Persisted clip markers (bracket spans survive reload) -->
        <div
          v-for="(clip, idx) in ctrl.persistedClips.value"
          :key="`clip-${idx}-${clip.created_at}`"
          class="tl-clip"
          :style="{
            left: clipLeft(clip.start_index) + '%',
            width: clipWidth(clip.start_index, clip.end_index) + '%',
          }"
          :title="clipTitle(clip)"
        >
          <span
            class="tl-clip-start"
            :title="clipTitle(clip)"
            @click.stop="ctrl.seekTo(clip.start_index)"
          ></span>
          <span
            class="tl-clip-end"
            :title="clipTitle(clip)"
            @click.stop="ctrl.seekTo(clip.end_index)"
          ></span>
          <button
            class="tl-clip-del"
            type="button"
            title="删除片段"
            @click.stop="ctrl.removeClip(idx)"
          >×</button>
        </div>
        <div class="tl-head" :style="{ left: progressPct + '%' }"></div>
      </div>

      <!-- Confidence sparkline: faint primary curve + red current-frame line -->
      <canvas
        ref="sparkCanvas"
        class="tl-sparkline"
        :style="{ height: SPARK_HEIGHT + 'px' }"
      ></canvas>

      <input
        type="range"
        :min="0"
        :max="(ctrl.metadata.value.frame_count || 1) - 1"
        :value="ctrl.currentIndex.value"
        @input="ctrl.seekTo(Number(($event.target as HTMLInputElement).value))"
        class="tl-input"
      />
      <!-- Recording-in-progress indicator -->
      <div v-if="ctrl.metadata.value.status === 'recording'" class="replay-recording-bar">
        <div style="flex: 1; display: flex; align-items: center; gap: 4px">
          <div :style="{ width: triggerProgressPct + '%' }" />
          <div class="rec-dot" />
          <div style="flex: 1; height: 2px; border-top: 2px dashed var(--line-2)" />
        </div>
        <span class="rec-text">&#9210; 录制中 · 剩余 {{ remainingRecordingSeconds }}s</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.replay-controls-wrapper {
  display: flex;
  flex-direction: column;
}
.replay-controls {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-top: 1px solid var(--line-2);
  background: var(--bg);
}
.ctrl-btn {
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border: 1px solid var(--line-2);
  background: transparent;
  color: var(--ink-2);
  cursor: pointer;
  font-size: 10px;
  transition: all .12s;
}
.ctrl-btn:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}
.ctrl-play {
  width: 34px;
  height: 34px;
  background: #3b82f6;
  border-color: #3b82f6;
  color: #fff;
  border-radius: 50%;
  font-size: 14px;
}
.ctrl-play:hover {
  background: #2563eb;
  border-color: #2563eb;
  color: #fff;
}
.ctrl-speeds {
  display: flex;
  margin-left: 8px;
}
.speed-btn {
  padding: 4px 8px;
  border: 1px solid var(--line-2);
  border-right: none;
  background: transparent;
  color: var(--ink-4);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 10px;
  cursor: pointer;
  transition: all .12s;
}
.speed-btn:last-child {
  border-right: 1px solid var(--line-2);
}
.speed-btn:hover {
  color: var(--ink-2);
}
.speed-btn.on {
  background: var(--ink-2);
  color: var(--bg);
  border-color: var(--ink-2);
}
.ctrl-toggles {
  display: flex;
  gap: 4px;
  margin-left: 8px;
}
.toggle-btn {
  padding: 4px 10px;
  border: 1px solid var(--line-2);
  background: transparent;
  color: var(--ink-4);
  font-size: 11px;
  cursor: pointer;
  transition: all .12s;
}
.toggle-btn:hover {
  border-color: #3b82f6;
  color: var(--ink-2);
}
.toggle-btn.on {
  border-color: #3b82f6;
  color: #3b82f6;
  background: rgba(59,130,246,.1);
}
.toggle-btn:disabled {
  opacity: .4;
  cursor: not-allowed;
}
.capture-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.capture-btn:hover:not(:disabled) {
  border-color: #f59e0b;
  color: #f59e0b;
}
.ctrl-frame-info {
  margin-left: auto;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 10px;
  color: var(--ink-4);
  letter-spacing: .08em;
}
.ctrl-frame-info b {
  color: #f59e0b;
}

/* ── Timeline ── */
.replay-timeline {
  padding: 12px 16px 6px;
  position: relative;
  background: var(--bg);
}
.tl-bar {
  height: 3px;
  background: var(--line-2);
  position: relative;
  border-radius: 2px;
  pointer-events: none;
}
.tl-progress {
  position: absolute;
  left: 0; top: 0; bottom: 0;
  background: #3b82f6;
  border-radius: 2px;
  transition: width .05s linear;
}
.tl-head {
  position: absolute;
  top: -4px;
  width: 2px;
  height: 11px;
  background: #f59e0b;
  transform: translateX(-1px);
  transition: left .05s linear;
}
.tl-clip {
  position: absolute;
  top: -3px;
  height: 9px;
  border-top: 1px solid rgba(34, 197, 94, 0.75);
  border-bottom: 1px solid rgba(34, 197, 94, 0.75);
  background: rgba(34, 197, 94, 0.08);
  pointer-events: none;
}
.tl-clip-start,
.tl-clip-end {
  position: absolute;
  top: -1px;
  width: 6px;
  height: 11px;
  pointer-events: auto;
  cursor: pointer;
}
.tl-clip-start {
  left: -1px;
  border-left: 2px solid rgba(34, 197, 94, 0.9);
  border-top: 2px solid rgba(34, 197, 94, 0.9);
  border-bottom: 2px solid rgba(34, 197, 94, 0.9);
  border-right: none;
  border-radius: 2px 0 0 2px;
  background: rgba(34, 197, 94, 0.25);
}
.tl-clip-end {
  right: -1px;
  border-right: 2px solid rgba(34, 197, 94, 0.9);
  border-top: 2px solid rgba(34, 197, 94, 0.9);
  border-bottom: 2px solid rgba(34, 197, 94, 0.9);
  border-left: none;
  border-radius: 0 2px 2px 0;
  background: rgba(34, 197, 94, 0.25);
}
.tl-clip-start:hover,
.tl-clip-end:hover {
  background: rgba(34, 197, 94, 0.55);
}
.tl-clip-del {
  position: absolute;
  top: -9px;
  right: -6px;
  width: 12px;
  height: 12px;
  border-radius: 6px;
  border: 1px solid var(--line-2);
  background: var(--bg);
  color: var(--ink-4);
  font-size: 9px;
  line-height: 10px;
  padding: 0;
  cursor: pointer;
  display: none;
  pointer-events: auto;
}
.tl-clip:hover .tl-clip-del {
  display: block;
}
.tl-clip-del:hover {
  color: #ef4444;
  border-color: #ef4444;
}
.tl-sparkline {
  display: block;
  width: 100%;
  margin-top: 4px;
  pointer-events: none;
}
.tl-input {
  position: absolute;
  left: 16px; right: 16px;
  top: 6px;
  width: calc(100% - 32px);
  height: 20px;
  opacity: 0;
  cursor: pointer;
  margin: 0;
}
.replay-recording-bar {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  margin-top: 4px;
}
.rec-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #f59e0b;
  flex-shrink: 0;
  animation: hud-blink 1.5s infinite;
}
.rec-text {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 11px;
  margin-left: 8px;
}
</style>
