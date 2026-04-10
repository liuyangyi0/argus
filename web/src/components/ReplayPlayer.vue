<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { Select } from 'ant-design-vue'
import {
  CaretRightOutlined,
  PauseOutlined,
} from '@ant-design/icons-vue'
import SignalTrack from './SignalTrack.vue'
import { getReplayMetadata, getReplaySignals, getReplayVideoUrl, getReplayHeatmapUrl, getReplayReference, pinReplayFrame } from '../api'

function videoErrorMessage(err: MediaError | DOMException): string {
  if (err instanceof MediaError) {
    const msgs: Record<number, string> = { 1: '视频加载被中止', 2: '网络错误', 3: '视频解码失败', 4: '视频格式不支持' }
    return msgs[err.code] || `视频错误 (code=${err.code})`
  }
  if (err instanceof DOMException) {
    if (err.name === 'NotAllowedError') return '浏览器阻止自动播放，请点击视频区域后重试'
    if (err.name === 'NotSupportedError') return '视频格式不支持'
    return `播放失败: ${err.message || err.name}`
  }
  return `播放失败`
}

const props = defineProps<{
  alertId: string
}>()

// State
const metadata = ref<any>(null)
const signals = ref<any>(null)
const currentIndex = ref(0)
const playing = ref(false)
const speed = ref(1)
const referenceFrame = ref<string | null>(null)
const referenceDate = ref('')
const loadingRef = ref(false)
const videoEl = ref<HTMLVideoElement | null>(null)
const videoError = ref('')
const pendingSeekIndex = ref<number | null>(null)

const selectedRefOption = ref('yesterday')
const clipStart = ref<number | null>(null)
const clipEnd = ref<number | null>(null)

const showHeatmap = ref(true)
const showBoxes = ref(false)
const hasHeatmaps = computed(() => signals.value?.has_heatmaps || false)

const fps = computed(() => metadata.value?.fps || 15)
const videoUrl = computed(() => getReplayVideoUrl(props.alertId))
// Freeze heatmap index when playing to avoid per-frame HTTP requests
const heatmapIndex = ref(0)
const heatmapUrl = computed(() => getReplayHeatmapUrl(props.alertId, heatmapIndex.value))

function onRefOptionChange(value: any) {
  selectedRefOption.value = value
  if (value === 'custom') {
    const dateStr = prompt('输入日期 (YYYY-MM-DD):')
    if (dateStr && /^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
      loadReference(dateStr)
    }
    return
  }
  const triggerTs = metadata.value?.trigger_timestamp
  if (!triggerTs) return
  const trigger = new Date(triggerTs * 1000)
  let refDate: Date
  if (value === 'yesterday') {
    refDate = new Date(trigger)
    refDate.setDate(refDate.getDate() - 1)
  } else if (value === 'last_week') {
    refDate = new Date(trigger)
    refDate.setDate(refDate.getDate() - 7)
  } else if (value === 'prev_week') {
    refDate = new Date(trigger)
    refDate.setDate(refDate.getDate() - 14)
  } else {
    refDate = new Date(trigger)
    refDate.setDate(refDate.getDate() - 1)
  }
  const dateStr = refDate.toISOString().slice(0, 10)
  loadReference(dateStr)
}

function safePlay() {
  if (!videoEl.value) return
  videoEl.value.play().catch((err: DOMException) => {
    if (err.name === 'AbortError') return
    videoError.value = videoErrorMessage(err)
    playing.value = false
  })
}

// Load data
async function loadData() {
  try {
    const [metaRes, sigRes] = await Promise.all([
      getReplayMetadata(props.alertId),
      getReplaySignals(props.alertId),
    ])
    metadata.value = metaRes
    signals.value = sigRes

    // Defer seek to trigger frame until video is ready (canplay event)
    if (metadata.value?.trigger_frame_index !== undefined) {
      pendingSeekIndex.value = metadata.value.trigger_frame_index
      currentIndex.value = metadata.value.trigger_frame_index
      heatmapIndex.value = metadata.value.trigger_frame_index
    }

    loadReference()
  } catch (e) {
    console.error('Replay load error', e)
    metadata.value = null
  }
}

async function loadReference(date?: string) {
  loadingRef.value = true
  try {
    const params: any = {}
    if (date) params.date = date
    const res = await getReplayReference(props.alertId, params)
    if (res.available && res.frame_base64) {
      referenceFrame.value = `data:image/jpeg;base64,${res.frame_base64}`
    } else {
      referenceFrame.value = null
    }
    referenceDate.value = res.source_date || ''
  } catch {
    referenceFrame.value = null
  } finally {
    loadingRef.value = false
  }
}

onMounted(loadData)

// Video event handlers
function onTimeUpdate() {
  if (!videoEl.value) return
  const idx = Math.min(
    Math.floor(videoEl.value.currentTime * fps.value),
    (metadata.value?.frame_count || 1) - 1,
  )
  if (idx !== currentIndex.value) currentIndex.value = idx
}

function onVideoPlay() {
  playing.value = true
  videoError.value = ''
}
function onVideoPause() {
  playing.value = false
  heatmapIndex.value = currentIndex.value
}
function onVideoEnded() {
  playing.value = false
  heatmapIndex.value = currentIndex.value
}
function onVideoCanPlay() {
  videoError.value = ''
  // Execute pending seek (trigger frame) now that video is ready
  if (pendingSeekIndex.value !== null && videoEl.value) {
    const seekTime = pendingSeekIndex.value / fps.value
    videoEl.value.currentTime = seekTime
    pendingSeekIndex.value = null
  }
}
function onVideoError() {
  const el = videoEl.value
  if (!el?.error) return
  videoError.value = videoErrorMessage(el.error)
}

// Playback controls — delegate to <video> element
function togglePlay() {
  if (!videoEl.value) return
  if (videoEl.value.paused) {
    safePlay()
  } else {
    videoEl.value.pause()
  }
}

function stepFrame(delta: number) {
  if (!videoEl.value) return
  videoEl.value.pause()
  videoEl.value.currentTime = Math.max(0, videoEl.value.currentTime + delta / fps.value)
}

function seekTo(index: number) {
  if (!videoEl.value) return
  videoEl.value.pause()
  videoEl.value.currentTime = index / fps.value
  currentIndex.value = index
}

function goToStart() { seekTo(0) }
function goToEnd() { seekTo((metadata.value?.frame_count || 1) - 1) }

// Speed control — set video.playbackRate
watch(speed, (s) => {
  if (videoEl.value) {
    videoEl.value.playbackRate = s
  }
})

// Keyboard shortcuts
function handleKeydown(e: KeyboardEvent) {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
  switch (e.key) {
    case ' ': e.preventDefault(); togglePlay(); break
    case 'ArrowLeft': stepFrame(-1); break
    case 'ArrowRight': stepFrame(1); break
    case 'k': case 'K': if (videoEl.value) videoEl.value.pause(); break
    case 'j': case 'J': speed.value = Math.max(0.25, speed.value / 2); if (videoEl.value?.paused) safePlay(); break
    case 'l': case 'L': speed.value = Math.min(4, speed.value * 2); if (videoEl.value?.paused) safePlay(); break
    case 'Home': goToStart(); break
    case 'End': goToEnd(); break
    case '[': clipStart.value = currentIndex.value; break
    case ']': clipEnd.value = currentIndex.value; break
  }
}

onMounted(() => window.addEventListener('keydown', handleKeydown))
onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
})

// Computed
const currentBoxes = computed(() => {
  if (!showBoxes.value || !signals.value?.yolo_boxes) return []
  return signals.value.yolo_boxes[currentIndex.value] || []
})
const keyFrames = computed(() => signals.value?.key_frames || [])
const currentTimestamp = computed(() => {
  const ts = signals.value?.timestamps?.[currentIndex.value]
  if (!ts) return ''
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString('zh-CN')
})

const speeds = [0.25, 0.5, 1, 2, 4]

// Timeline progress (used for both progress bar and playhead)
const progressPct = computed(() => {
  const total = Math.max((metadata.value?.frame_count || 1) - 1, 1)
  return (currentIndex.value / total) * 100
})

// Signal track data (extract .map() out of template to avoid new arrays per render)
const simplexData = computed(() => signals.value?.simplex_scores?.map((s: any) => s ?? 0) || [])
const hasSimplexData = computed(() => signals.value?.simplex_scores?.some((s: any) => s != null) || false)
const yoloPersonsData = computed(() => signals.value?.yolo_persons?.map((p: any) => p.count || 0) || [])

const triggerProgressPct = computed(() => {
  if (!metadata.value) return 50
  const triggerIdx = metadata.value.trigger_frame_index || 0
  const total = metadata.value.frame_count || 1
  return Math.round((triggerIdx / total) * 100)
})

const remainingRecordingSeconds = computed(() => {
  if (!metadata.value || metadata.value.status !== 'recording') return 0
  const triggerTs = metadata.value.trigger_timestamp || 0
  const postSeconds = metadata.value.severity === 'low' ? 10 : 30
  const deadline = triggerTs + postSeconds
  const now = Date.now() / 1000
  return Math.max(0, Math.round(deadline - now))
})

async function handlePinFrame() {
  const label = prompt('帧标签:')
  if (label) {
    await pinReplayFrame(props.alertId, { index: currentIndex.value, label })
  }
}
</script>

<template>
  <div v-if="metadata" class="replay-root">
    <!-- Video area: main + reference -->
    <div class="replay-viewport-row">
      <!-- Main playback window -->
      <div class="replay-main">
        <div class="replay-player">
          <video
            ref="videoEl"
            :src="videoUrl"
            preload="auto"
            class="replay-video"
            @timeupdate="onTimeUpdate"
            @play="onVideoPlay"
            @pause="onVideoPause"
            @ended="onVideoEnded"
            @canplay="onVideoCanPlay"
            @error="onVideoError"
          />
          <!-- Heatmap overlay -->
          <img
            v-if="showHeatmap && hasHeatmaps"
            :src="heatmapUrl"
            class="replay-heatmap"
          />
          <!-- YOLO detection boxes overlay -->
          <svg
            v-if="showBoxes && currentBoxes.length > 0"
            class="replay-boxes"
            :viewBox="`0 0 ${metadata.width || 1920} ${metadata.height || 1080}`"
            preserveAspectRatio="xMidYMid meet"
          >
            <template v-for="(box, idx) in currentBoxes" :key="idx">
              <rect
                :x="box.bbox?.[0] || 0" :y="box.bbox?.[1] || 0"
                :width="(box.bbox?.[2] || 0) - (box.bbox?.[0] || 0)"
                :height="(box.bbox?.[3] || 0) - (box.bbox?.[1] || 0)"
                fill="none" stroke="#10b981" stroke-width="2"
              />
              <text
                :x="(box.bbox?.[0] || 0) + 4" :y="(box.bbox?.[1] || 0) - 6"
                fill="#fff" font-family="monospace" font-size="12" font-weight="600"
                style="text-shadow: 0 0 4px rgba(0,0,0,.9), 0 0 8px rgba(0,0,0,.5)"
              >{{ box.class }} {{ box.confidence?.toFixed?.(2) }}</text>
            </template>
          </svg>
          <!-- HUD: top -->
          <div class="replay-hud replay-hud-top">
            <span v-if="metadata.status === 'recording'" class="hud-rec">&#9679; REC</span>
            <span v-else class="hud-cam">{{ metadata.camera_id || '' }}</span>
            <span>{{ metadata.width || 1920 }}&#215;{{ metadata.height || 1080 }} // {{ fps }} FPS</span>
          </div>
          <!-- HUD: bottom -->
          <div class="replay-hud replay-hud-bottom">
            <span>{{ currentTimestamp }}</span>
            <span>FRAME {{ currentIndex + 1 }} / {{ metadata.frame_count }}</span>
          </div>
          <!-- Video error -->
          <div v-if="videoError" class="replay-error">{{ videoError }}</div>
        </div>
      </div>

      <!-- Reference window -->
      <div class="replay-ref">
        <div class="replay-ref-label">
          <span>历史对照 · Reference</span>
          <Select
            :value="selectedRefOption"
            size="small"
            style="flex: 1; min-width: 100px"
            @change="onRefOptionChange"
          >
            <Select.Option value="yesterday">昨天</Select.Option>
            <Select.Option value="last_week">上周</Select.Option>
            <Select.Option value="prev_week">两周前</Select.Option>
            <Select.Option value="custom">手动...</Select.Option>
          </Select>
        </div>
        <div class="replay-ref-img">
          <img v-if="referenceFrame" :src="referenceFrame" style="width: 100%; height: 100%; object-fit: contain; display: block" />
          <span v-else class="replay-ref-empty">{{ loadingRef ? '...' : '无数据' }}</span>
        </div>
        <div v-if="referenceDate" class="replay-ref-date">{{ referenceDate }}</div>
      </div>
    </div>

    <!-- DVR Controls -->
    <div class="replay-controls">
      <button class="ctrl-btn" @click="goToStart" title="Start">&#9198;</button>
      <button class="ctrl-btn" @click="stepFrame(-1)" title="-1 frame">&#9664;&#9664;</button>
      <button class="ctrl-btn ctrl-play" @click="togglePlay">
        <PauseOutlined v-if="playing" />
        <CaretRightOutlined v-else />
      </button>
      <button class="ctrl-btn" @click="stepFrame(1)" title="+1 frame">&#9654;&#9654;</button>
      <button class="ctrl-btn" @click="goToEnd" title="End">&#9197;</button>

      <div class="ctrl-speeds">
        <button
          v-for="s in speeds" :key="s"
          :class="['speed-btn', { on: speed === s }]"
          @click="speed = s"
        >{{ s }}x</button>
      </div>

      <!-- Overlay toggles -->
      <div class="ctrl-toggles">
        <button
          :class="['toggle-btn', { on: showHeatmap }]"
          :disabled="!hasHeatmaps"
          @click="showHeatmap = !showHeatmap"
          title="热力图叠加"
        >热力</button>
        <button
          :class="['toggle-btn', { on: showBoxes }]"
          @click="showBoxes = !showBoxes"
          title="YOLO 检测框"
        >框选</button>
      </div>

      <span class="ctrl-frame-info">
        FRAME <b>{{ currentIndex + 1 }}</b> / {{ metadata.frame_count }} · {{ currentTimestamp }}
      </span>
    </div>

    <!-- Timeline scrubber -->
    <div class="replay-timeline">
      <div class="tl-bar">
        <div class="tl-progress" :style="{ width: progressPct + '%' }"></div>
        <div class="tl-head" :style="{ left: progressPct + '%' }"></div>
      </div>
      <input
        type="range"
        :min="0"
        :max="(metadata.frame_count || 1) - 1"
        :value="currentIndex"
        @input="seekTo(Number(($event.target as HTMLInputElement).value))"
        class="tl-input"
      />
      <!-- Recording-in-progress indicator -->
      <div v-if="metadata.status === 'recording'" class="replay-recording-bar">
        <div style="flex: 1; display: flex; align-items: center; gap: 4px">
          <div :style="{ width: triggerProgressPct + '%' }" />
          <div class="rec-dot" />
          <div style="flex: 1; height: 2px; border-top: 2px dashed var(--argus-border)" />
        </div>
        <span class="rec-text">&#9210; 录制中 · 剩余 {{ remainingRecordingSeconds }}s</span>
      </div>
    </div>

    <!-- Signal tracks -->
    <div v-if="signals && metadata.severity !== 'low' && metadata.severity !== 'info'" class="replay-signals">
      <div v-if="signals.anomaly_scores" class="sig-wrap" style="border-left-color: #ef4444">
        <SignalTrack
          :data="signals.anomaly_scores"
          :current-index="currentIndex"
          label="Dinomaly"
          color="#ef4444"
          :height="32"
          @seek="seekTo"
        />
      </div>
      <div v-if="hasSimplexData" class="sig-wrap" style="border-left-color: #f97316">
        <SignalTrack
          :data="simplexData"
          :current-index="currentIndex"
          label="Simplex"
          color="#f97316"
          :height="32"
          @seek="seekTo"
        />
      </div>
      <template v-for="(values, zone) in (signals.cusum_evidence || {})" :key="zone">
        <div class="sig-wrap" style="border-left-color: #8b5cf6">
          <SignalTrack
            :data="values"
            :current-index="currentIndex"
            :label="'CUSUM'"
            color="#8b5cf6"
            :height="32"
            @seek="seekTo"
          />
        </div>
      </template>
      <div v-if="signals.yolo_persons" class="sig-wrap" style="border-left-color: #10b981">
        <SignalTrack
          :data="yoloPersonsData"
          :current-index="currentIndex"
          label="YOLO人员"
          color="#10b981"
          :height="24"
          @seek="seekTo"
        />
      </div>
      <!-- Operator action track -->
      <div v-if="signals.operator_actions?.length" class="sig-wrap" style="border-left-color: #f59e0b; height: 20px; position: relative; overflow: hidden">
        <span style="font-size: 10px; position: absolute; left: 8px; top: 2px; color: var(--argus-text-muted); font-family: monospace; letter-spacing: .1em">操作员</span>
        <div
          v-for="(action, idx) in signals.operator_actions"
          :key="idx"
          :style="{
            position: 'absolute',
            left: ((action.timestamp - signals.timestamps[0]) / (signals.timestamps[signals.timestamps.length-1] - signals.timestamps[0]) * 100) + '%',
            top: '2px',
            width: '8px',
            height: '16px',
            background: '#f59e0b',
            borderRadius: '2px',
            cursor: 'pointer',
          }"
          :title="`${action.user}: ${action.action}`"
        />
      </div>
    </div>

    <!-- Key frames -->
    <div v-if="keyFrames.length > 0" class="replay-keyframes">
      <span class="kf-label">关键帧</span>
      <button
        v-for="kf in keyFrames" :key="kf.index"
        :class="['kf-tag', kf.type === 'trigger' ? 'kf-purple' : '']"
        @click="seekTo(kf.index)"
      >{{ kf.label }} · #{{ kf.index }}</button>
      <button class="kf-tag kf-add" @click="handlePinFrame">+ 标记当前帧</button>
    </div>
  </div>

  <!-- Loading / no recording -->
  <div v-else style="padding: 24px; text-align: center; color: var(--argus-text-muted)">
    加载回放数据...
  </div>
</template>

<style scoped>
.replay-root {
  background: var(--argus-surface);
  border-radius: 6px;
  border: 1px solid var(--argus-border);
  padding: 0;
  overflow: hidden;
}

/* ── Viewport ── */
.replay-viewport-row {
  display: flex;
  gap: 0;
}
.replay-main {
  flex: 1;
  min-width: 0;
}
.replay-player {
  position: relative;
  background: #000;
  aspect-ratio: 16/9;
  overflow: hidden;
}
.replay-video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.replay-heatmap {
  position: absolute;
  top: 0; left: 0; width: 100%; height: 100%;
  object-fit: contain;
  opacity: 0.4;
  pointer-events: none;
  mix-blend-mode: screen;
}
.replay-boxes {
  position: absolute;
  top: 0; left: 0; width: 100%; height: 100%;
  pointer-events: none;
}

/* ── HUD overlays ── */
.replay-hud {
  position: absolute;
  left: 12px; right: 12px;
  display: flex;
  justify-content: space-between;
  font-family: var(--argus-font-mono);
  font-size: 10px;
  letter-spacing: .12em;
  pointer-events: none;
  text-shadow: 0 1px 3px rgba(0,0,0,.7);
}
.replay-hud-top {
  top: 10px;
  color: rgba(255,255,255,.5);
}
.replay-hud-bottom {
  bottom: 10px;
  color: rgba(255,255,255,.6);
}
.hud-rec {
  color: #f59e0b;
  animation: hud-blink 1.5s infinite;
}
.hud-cam {
  color: rgba(255,255,255,.5);
}
@keyframes hud-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: .4; }
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

/* ── Reference panel ── */
.replay-ref {
  width: 220px;
  flex-shrink: 0;
  background: var(--argus-card-bg-solid);
  border-left: 1px solid var(--argus-border);
  display: flex;
  flex-direction: column;
}
.replay-ref-label {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 10px;
  font-family: var(--argus-font-mono);
  font-size: 9px;
  color: var(--argus-text-muted);
  letter-spacing: .12em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--argus-border);
}
.replay-ref-img {
  flex: 1;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 120px;
}
.replay-ref-empty {
  font-size: 11px;
  color: var(--argus-text-muted);
}
.replay-ref-date {
  padding: 4px 10px;
  font-family: var(--argus-font-mono);
  font-size: 10px;
  color: var(--argus-text-muted);
  letter-spacing: .1em;
  border-top: 1px solid var(--argus-border);
}

/* ── Controls bar ── */
.replay-controls {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-top: 1px solid var(--argus-border);
  background: var(--argus-card-bg-solid);
}
.ctrl-btn {
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border: 1px solid var(--argus-border);
  background: transparent;
  color: var(--argus-text);
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
  border: 1px solid var(--argus-border);
  border-right: none;
  background: transparent;
  color: var(--argus-text-muted);
  font-family: var(--argus-font-mono);
  font-size: 10px;
  cursor: pointer;
  transition: all .12s;
}
.speed-btn:last-child {
  border-right: 1px solid var(--argus-border);
}
.speed-btn:hover {
  color: var(--argus-text);
}
.speed-btn.on {
  background: var(--argus-text);
  color: var(--argus-surface);
  border-color: var(--argus-text);
}
.ctrl-toggles {
  display: flex;
  gap: 4px;
  margin-left: 8px;
}
.toggle-btn {
  padding: 4px 10px;
  border: 1px solid var(--argus-border);
  background: transparent;
  color: var(--argus-text-muted);
  font-size: 11px;
  cursor: pointer;
  transition: all .12s;
}
.toggle-btn:hover {
  border-color: #3b82f6;
  color: var(--argus-text);
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
.ctrl-frame-info {
  margin-left: auto;
  font-family: var(--argus-font-mono);
  font-size: 10px;
  color: var(--argus-text-muted);
  letter-spacing: .08em;
}
.ctrl-frame-info b {
  color: #f59e0b;
}

/* ── Timeline ── */
.replay-timeline {
  padding: 12px 16px 6px;
  position: relative;
}
.tl-bar {
  height: 3px;
  background: var(--argus-border);
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
  font-family: var(--argus-font-mono);
  font-size: 11px;
  margin-left: 8px;
  flex-shrink: 0;
  color: #f59e0b;
}

/* ── Signal tracks ── */
.replay-signals {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 4px 16px 8px;
}
.sig-wrap {
  border-left: 3px solid transparent;
  padding-left: 6px;
  background: var(--argus-card-bg-solid);
  border-radius: 2px;
}

/* ── Keyframes ── */
.replay-keyframes {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding: 10px 16px;
  border-top: 1px solid var(--argus-border);
}
.kf-label {
  font-family: var(--argus-font-mono);
  font-size: 10px;
  color: var(--argus-text-muted);
  letter-spacing: .15em;
  text-transform: uppercase;
}
.kf-tag {
  padding: 3px 10px;
  border: 1px solid #3b82f6;
  color: #3b82f6;
  font-family: var(--argus-font-mono);
  font-size: 10px;
  background: rgba(59,130,246,.06);
  cursor: pointer;
  transition: all .12s;
}
.kf-tag:hover {
  background: rgba(59,130,246,.15);
}
.kf-purple {
  border-color: #8b5cf6;
  color: #8b5cf6;
  background: rgba(139,92,246,.06);
}
.kf-purple:hover {
  background: rgba(139,92,246,.15);
}
.kf-add {
  border-style: dashed;
  border-color: var(--argus-border);
  color: var(--argus-text-muted);
  background: transparent;
}
.kf-add:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}

/* ── Mobile ── */
@media (max-width: 768px) {
  .replay-viewport-row {
    flex-direction: column;
  }
  .replay-ref {
    width: 100%;
    border-left: none;
    border-top: 1px solid var(--argus-border);
  }
  .ctrl-speeds {
    display: none;
  }
}
</style>
