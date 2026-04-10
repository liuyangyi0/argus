<template>
  <div class="timeline-container">
    <div class="timeline-header">
      <a-typography-text strong style="color: var(--argus-text); font-size: 14px">24h 告警时间线</a-typography-text>
      <a-date-picker
        v-model:value="selectedDate"
        :allow-clear="false"
        size="small"
        style="width: 140px"
        @change="fetchTimeline"
      />
    </div>

    <div v-if="loading" class="timeline-loading">
      <a-spin size="small" />
    </div>

    <div v-else-if="cameras.length === 0" class="timeline-empty">
      <a-typography-text type="secondary">暂无告警数据</a-typography-text>
    </div>

    <div v-else class="timeline-body">
      <!-- Time axis -->
      <div class="time-axis">
        <div class="camera-label-spacer" />
        <div class="time-ticks">
          <span v-for="h in hours" :key="h" class="tick" :style="{ left: `${(h / 24) * 100}%` }">
            {{ String(h).padStart(2, '0') }}
          </span>
        </div>
      </div>

      <!-- Camera rows -->
      <div v-for="cam in cameras" :key="cam.camera_id" class="camera-row">
        <div class="camera-label" :title="cam.name">{{ cam.name }}</div>
        <div class="track" ref="trackRefs">
          <!-- Alert segments -->
          <div
            v-for="(seg, idx) in cam.segments"
            :key="idx"
            class="segment"
            :class="seg.severity"
            :style="segmentStyle(seg)"
            :title="`${seg.start.slice(11, 16)} - ${seg.end.slice(11, 16)} | ${severityLabel(seg.severity)} | ${seg.alert_count} 条告警`"
            @click="$emit('segment-click', { camera_id: cam.camera_id, segment: seg })"
          />
          <!-- Current time line -->
          <div
            v-if="isToday"
            class="now-line"
            :style="{ left: `${nowPercent}%` }"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { getAlertTimeline } from '../api/alerts'
import dayjs from 'dayjs'

interface Segment {
  start: string
  end: string
  severity: string
  alert_count: number
  first_alert_id?: string
}

interface CameraTimeline {
  camera_id: string
  name: string
  segments: Segment[]
}

defineEmits<{ (e: 'segment-click', payload: { camera_id: string; segment: Segment }): void }>()

const selectedDate = ref(dayjs())
const cameras = ref<CameraTimeline[]>([])
const loading = ref(false)
const hours = Array.from({ length: 25 }, (_, i) => i)

const isToday = computed(() => selectedDate.value.isSame(dayjs(), 'day'))
const nowPercent = computed(() => {
  const now = dayjs()
  return ((now.hour() * 60 + now.minute()) / 1440) * 100
})

function segmentStyle(seg: Segment) {
  const start = dayjs(seg.start)
  const end = dayjs(seg.end)
  const startMin = start.hour() * 60 + start.minute()
  const endMin = end.hour() * 60 + end.minute()
  const left = (startMin / 1440) * 100
  const width = Math.max(((endMin - startMin) / 1440) * 100, 0.3)
  return { left: `${left}%`, width: `${width}%` }
}

function severityLabel(sev: string) {
  return { high: '高', medium: '中', low: '低', info: '信息' }[sev] || sev
}

async function fetchTimeline() {
  loading.value = true
  try {
    const data = await getAlertTimeline(selectedDate.value.format('YYYY-MM-DD'))
    cameras.value = data.cameras || []
  } catch {
    cameras.value = []
  } finally {
    loading.value = false
  }
}

let timer: ReturnType<typeof setInterval>
onMounted(() => {
  fetchTimeline()
  timer = setInterval(fetchTimeline, 60000)
})
onUnmounted(() => clearInterval(timer))
</script>

<style scoped>
.timeline-container {
  background: var(--argus-card-bg, linear-gradient(135deg, #1a1a2e 0%, #16162a 100%));
  border: 1px solid var(--argus-border, #2d2d4a);
  border-radius: 8px;
  padding: 12px 16px;
}
.timeline-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.timeline-loading, .timeline-empty {
  text-align: center;
  padding: 20px 0;
}
.time-axis {
  display: flex;
  align-items: flex-end;
  margin-bottom: 4px;
}
.camera-label-spacer { width: 100px; flex-shrink: 0; }
.time-ticks {
  flex: 1;
  position: relative;
  height: 18px;
}
.tick {
  position: absolute;
  transform: translateX(-50%);
  font-size: 10px;
  color: #6b7280;
}
.camera-row {
  display: flex;
  align-items: center;
  margin-bottom: 4px;
}
.camera-label {
  width: 100px;
  flex-shrink: 0;
  font-size: 12px;
  color: #9ca3af;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding-right: 8px;
}
.track {
  flex: 1;
  height: 20px;
  background: var(--argus-surface);
  border-radius: 3px;
  position: relative;
  overflow: hidden;
}
.segment {
  position: absolute;
  top: 2px;
  height: 16px;
  border-radius: 2px;
  cursor: pointer;
  min-width: 3px;
  transition: opacity 0.15s;
}
.segment:hover { opacity: 0.8; }
.segment.high { background: #ef4444; }
.segment.medium { background: #f59e0b; }
.segment.low { background: #3b82f6; }
.segment.info { background: #6b7280; }
.now-line {
  position: absolute;
  top: 0;
  width: 2px;
  height: 100%;
  background: #22c55e;
  z-index: 2;
}

/* Light theme */
:global(.light-theme) .timeline-container {
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border-color: #e5e7eb;
}
:global(.light-theme) .track { background: #f1f5f9; }
:global(.light-theme) .camera-label { color: #374151; }
:global(.light-theme) .tick { color: #9ca3af; }

/* Mobile */
@media (max-width: 768px) {
  .camera-label { width: 70px; font-size: 11px; }
  .timeline-container { padding: 8px 10px; }
}
</style>
