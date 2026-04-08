<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { Button, Space, Tag, Typography, Select } from 'ant-design-vue'
import {
  StepBackwardOutlined,
  StepForwardOutlined,
  CaretRightOutlined,
  PauseOutlined,
  FastBackwardOutlined,
  FastForwardOutlined,
} from '@ant-design/icons-vue'
import SignalTrack from './SignalTrack.vue'
import { getReplayMetadata, getReplaySignals, getReplayFrameUrl, getReplayHeatmapUrl, getReplayReference, pinReplayFrame } from '../api'

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

const selectedRefOption = ref('yesterday')
const clipStart = ref<number | null>(null)
const clipEnd = ref<number | null>(null)

// §1.3: Overlay toggles
const showHeatmap = ref(true)  // default on
const showBoxes = ref(false)   // default off
const hasHeatmaps = computed(() => signals.value?.has_heatmaps || false)

let playTimer: ReturnType<typeof setInterval> | null = null

function onRefOptionChange(value: any) {
  selectedRefOption.value = value
  if (value === 'custom') {
    const dateStr = prompt('输入日期 (YYYY-MM-DD):')
    if (dateStr && /^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
      loadReference(dateStr)
    }
    return
  }
  // Compute the reference date based on trigger timestamp
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

// Load data
async function loadData() {
  try {
    const [metaRes, sigRes] = await Promise.all([
      getReplayMetadata(props.alertId),
      getReplaySignals(props.alertId),
    ])
    metadata.value = metaRes.data
    signals.value = sigRes.data

    // Default to trigger frame index (don't auto-play)
    if (metadata.value?.trigger_frame_index !== undefined) {
      currentIndex.value = metadata.value.trigger_frame_index
    }

    // Load default reference (yesterday)
    loadReference()
  } catch (e) {
    console.error('Replay load error', e)
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

// Playback controls
function togglePlay() {
  playing.value = !playing.value
}

watch(playing, (isPlaying) => {
  if (playTimer) {
    clearInterval(playTimer)
    playTimer = null
  }
  if (isPlaying && metadata.value) {
    const fps = metadata.value.fps || 5
    const intervalMs = 1000 / (fps * speed.value)
    playTimer = setInterval(() => {
      if (currentIndex.value < (metadata.value?.frame_count || 0) - 1) {
        currentIndex.value++
      } else {
        playing.value = false
      }
    }, intervalMs)
  }
})

watch(speed, () => {
  if (playing.value) {
    playing.value = false
    playing.value = true // restart with new speed
  }
})

function stepFrame(delta: number) {
  playing.value = false
  const max = (metadata.value?.frame_count || 1) - 1
  currentIndex.value = Math.max(0, Math.min(max, currentIndex.value + delta))
}

function seekTo(index: number) {
  playing.value = false
  currentIndex.value = index
}

function goToStart() { seekTo(0) }
function goToEnd() { seekTo((metadata.value?.frame_count || 1) - 1) }

// Keyboard shortcuts
function handleKeydown(e: KeyboardEvent) {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
  switch (e.key) {
    case ' ': e.preventDefault(); togglePlay(); break
    case 'ArrowLeft': stepFrame(-1); break
    case 'ArrowRight': stepFrame(1); break
    case 'k': case 'K': playing.value = false; break
    case 'j': case 'J': speed.value = Math.max(0.25, speed.value / 2); if (!playing.value) playing.value = true; break
    case 'l': case 'L': speed.value = Math.min(4, speed.value * 2); if (!playing.value) playing.value = true; break
    case 'Home': goToStart(); break
    case 'End': goToEnd(); break
    case '[': clipStart.value = currentIndex.value; break
    case ']': clipEnd.value = currentIndex.value; break
  }
}

onMounted(() => window.addEventListener('keydown', handleKeydown))
onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
  if (playTimer) clearInterval(playTimer)
})

// Computed
const frameUrl = computed(() => getReplayFrameUrl(props.alertId, currentIndex.value))
const heatmapUrl = computed(() => getReplayHeatmapUrl(props.alertId, currentIndex.value))

// Current frame's YOLO boxes from signals
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

const statusText = computed(() => {
  if (!metadata.value) return ''
  if (metadata.value.status === 'recording') {
    return '录制中...'
  }
  return `${metadata.value.frame_count} 帧 / ${metadata.value.fps} FPS`
})

const speeds = [0.25, 0.5, 1, 2, 4]

// §1.2.2: Recording-in-progress computations
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
  <div v-if="metadata" style="background: #0f0f1a; border-radius: 8px; padding: 12px">
    <!-- Video area: main + reference -->
    <div style="display: flex; gap: 12px; margin-bottom: 12px">
      <!-- Main playback window -->
      <div style="flex: 1; min-width: 0">
        <div style="position: relative; background: #000; border-radius: 4px; overflow: hidden; aspect-ratio: 16/9">
          <img :src="frameUrl" style="width: 100%; height: 100%; object-fit: contain; display: block" />
          <!-- §1.3: Heatmap overlay (toggle) -->
          <img
            v-if="showHeatmap && hasHeatmaps"
            :src="heatmapUrl"
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; opacity: 0.4; pointer-events: none; mix-blend-mode: screen"
          />
          <!-- §1.3: YOLO detection boxes overlay (toggle) -->
          <svg
            v-if="showBoxes && currentBoxes.length > 0"
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none"
            viewBox="0 0 1920 1080"
            preserveAspectRatio="xMidYMid meet"
          >
            <rect
              v-for="(box, idx) in currentBoxes" :key="idx"
              :x="box.bbox?.[0] || 0" :y="box.bbox?.[1] || 0"
              :width="(box.bbox?.[2] || 0) - (box.bbox?.[0] || 0)"
              :height="(box.bbox?.[3] || 0) - (box.bbox?.[1] || 0)"
              fill="none" stroke="#10b981" stroke-width="3"
            />
            <text
              v-for="(box, idx) in currentBoxes" :key="'t'+idx"
              :x="(box.bbox?.[0] || 0) + 4" :y="(box.bbox?.[1] || 0) - 4"
              fill="#10b981" font-size="24"
            >{{ box.class }} {{ box.confidence }}</text>
          </svg>
          <div v-if="metadata.status === 'recording'" style="position: absolute; top: 8px; right: 8px">
            <Tag color="red">录制中</Tag>
          </div>
        </div>
      </div>
      <!-- Reference window with date selector (§1.3) -->
      <div style="width: 240px; flex-shrink: 0">
        <div style="display: flex; align-items: center; gap: 4px; margin-bottom: 4px">
          <Typography.Text type="secondary" style="font-size: 12px; flex-shrink: 0">历史对照</Typography.Text>
          <Select
            :value="selectedRefOption"
            size="small"
            style="flex: 1; font-size: 11px"
            @change="onRefOptionChange"
          >
            <Select.Option value="yesterday">昨天同一时刻</Select.Option>
            <Select.Option value="last_week">上周同一日</Select.Option>
            <Select.Option value="prev_week">本月上一周</Select.Option>
            <Select.Option value="custom">手动选择...</Select.Option>
          </Select>
        </div>
        <div style="background: #000; border-radius: 4px; overflow: hidden; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center">
          <img v-if="referenceFrame" :src="referenceFrame" style="width: 100%; height: 100%; object-fit: contain; display: block" />
          <Typography.Text v-else type="secondary" style="font-size: 12px">
            {{ loadingRef ? '加载中...' : '无历史数据' }}
          </Typography.Text>
        </div>
        <Typography.Text v-if="referenceDate" type="secondary" style="font-size: 11px; margin-top: 2px; display: block">
          {{ referenceDate }}
        </Typography.Text>
      </div>
    </div>

    <!-- DVR Controls -->
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap">
      <Space size="small">
        <Button size="small" @click="goToStart"><template #icon><StepBackwardOutlined /></template></Button>
        <Button size="small" @click="stepFrame(-1)"><template #icon><FastBackwardOutlined /></template></Button>
        <Button size="small" type="primary" @click="togglePlay">
          <template #icon>
            <PauseOutlined v-if="playing" />
            <CaretRightOutlined v-else />
          </template>
        </Button>
        <Button size="small" @click="stepFrame(1)"><template #icon><FastForwardOutlined /></template></Button>
        <Button size="small" @click="goToEnd"><template #icon><StepForwardOutlined /></template></Button>
      </Space>

      <Space size="small">
        <Button
          v-for="s in speeds" :key="s"
          size="small"
          :type="speed === s ? 'primary' : 'default'"
          @click="speed = s"
        >
          {{ s }}x
        </Button>
      </Space>

      <!-- §1.3: Overlay toggles -->
      <Space size="small" style="margin-left: 8px">
        <Button
          size="small"
          :type="showHeatmap ? 'primary' : 'default'"
          :disabled="!hasHeatmaps"
          @click="showHeatmap = !showHeatmap"
          title="热力图叠加"
        >
          &#128293; 热力
        </Button>
        <Button
          size="small"
          :type="showBoxes ? 'primary' : 'default'"
          @click="showBoxes = !showBoxes"
          title="YOLO 检测框"
        >
          &#128230; 框选
        </Button>
      </Space>

      <Typography.Text type="secondary" style="margin-left: auto; font-size: 12px">
        {{ currentIndex + 1 }}/{{ metadata.frame_count }} · {{ currentTimestamp }} · {{ statusText }}
      </Typography.Text>
    </div>

    <!-- Timeline scrubber (§1.2.2: recording-in-progress shows dashed incomplete segment) -->
    <div style="margin-bottom: 8px; position: relative">
      <input
        type="range"
        :min="0"
        :max="(metadata.frame_count || 1) - 1"
        :value="currentIndex"
        @input="seekTo(Number(($event.target as HTMLInputElement).value))"
        style="width: 100%; accent-color: #3b82f6"
      />
      <!-- Recording-in-progress indicator -->
      <div v-if="metadata.status === 'recording'" style="display: flex; align-items: center; justify-content: flex-end; margin-top: 2px">
        <div style="flex: 1; display: flex; align-items: center; gap: 4px">
          <!-- Trigger point marker -->
          <div :style="{ width: triggerProgressPct + '%' }" />
          <div style="width: 8px; height: 8px; border-radius: 50%; background: #ef4444; flex-shrink: 0" />
          <!-- Dashed incomplete segment -->
          <div style="flex: 1; height: 2px; border-top: 2px dashed #4a5568" />
        </div>
        <Typography.Text type="secondary" style="font-size: 11px; margin-left: 8px; flex-shrink: 0; color: #ef4444">
          &#9210; 录制中 · 剩余 {{ remainingRecordingSeconds }}s
        </Typography.Text>
      </div>
    </div>

    <!-- Signal tracks (MEDIUM/HIGH only — LOW shows video + trigger point only per §1.2.3) -->
    <div v-if="signals && metadata.severity !== 'low' && metadata.severity !== 'info'" style="display: flex; flex-direction: column; gap: 2px; margin-bottom: 8px">
      <SignalTrack
        v-if="signals.anomaly_scores"
        :data="signals.anomaly_scores"
        :current-index="currentIndex"
        label="Dinomaly"
        color="#ef4444"
        :height="32"
        @seek="seekTo"
      />
      <SignalTrack
        v-if="signals.simplex_scores?.some((s: any) => s != null)"
        :data="signals.simplex_scores.map((s: any) => s ?? 0)"
        :current-index="currentIndex"
        label="Simplex"
        color="#f97316"
        :height="32"
        @seek="seekTo"
      />
      <SignalTrack
        v-for="(values, zone) in (signals.cusum_evidence || {})"
        :key="zone"
        :data="values"
        :current-index="currentIndex"
        :label="'CUSUM'"
        color="#8b5cf6"
        :height="32"
        @seek="seekTo"
      />
      <SignalTrack
        v-if="signals.yolo_persons"
        :data="signals.yolo_persons.map((p: any) => p.count || 0)"
        :current-index="currentIndex"
        label="YOLO人员"
        color="#10b981"
        :height="24"
        @seek="seekTo"
      />
      <!-- Operator action track (§1.3 — 5th track) -->
      <div v-if="signals.operator_actions?.length" style="height: 20px; position: relative; background: #1a1a2e; border-radius: 2px; overflow: hidden">
        <Typography.Text type="secondary" style="font-size: 10px; position: absolute; left: 4px; top: 2px">操作员</Typography.Text>
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
    <div v-if="keyFrames.length > 0" style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap">
      <Typography.Text type="secondary" style="font-size: 12px">关键帧:</Typography.Text>
      <Tag
        v-for="kf in keyFrames" :key="kf.index"
        :color="kf.type === 'trigger' ? 'red' : kf.type === 'evidence_threshold' ? 'orange' : 'blue'"
        style="cursor: pointer"
        @click="seekTo(kf.index)"
      >
        {{ kf.label }} (#{{ kf.index }})
      </Tag>
      <Button size="small" type="dashed" @click="handlePinFrame" style="font-size: 11px">
        + 标记当前帧
      </Button>
    </div>
  </div>

  <!-- Loading / no recording -->
  <div v-else style="padding: 24px; text-align: center; color: #6b7280">
    加载回放数据...
  </div>
</template>
