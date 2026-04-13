<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { Tooltip, Tag } from 'ant-design-vue'
import { use } from 'echarts/core'
import { GaugeChart, LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { GridComponent, TooltipComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import type { CameraTileData } from './VideoTile.vue'
import { useThemeStore } from '../stores/theme'

use([CanvasRenderer, GaugeChart, LineChart, GridComponent, TooltipComponent])

const themeStore = useThemeStore()

interface HealthData {
  status: string
  uptime_seconds: number
  total_alerts: number
  cameras: Array<{
    camera_id: string
    connected: boolean
    frames_captured: number
    avg_latency_ms: number
  }>
}

const props = defineProps<{
  health: HealthData | null
  cameras: CameraTileData[]
}>()

// --- System Status ---
const systemStatus = computed(() => props.health?.status ?? 'unknown')
const statusLabel = computed(() => {
  const m: Record<string, string> = { healthy: '运行正常', degraded: '降级运行', unhealthy: '系统异常' }
  return m[systemStatus.value] || '未知'
})
const statusColor = computed(() => {
  const m: Record<string, string> = { healthy: '#22c55e', degraded: '#f59e0b', unhealthy: '#ef4444' }
  return m[systemStatus.value] || '#6b7280'
})
const uptimeFormatted = computed(() => {
  const s = props.health?.uptime_seconds ?? 0
  if (s < 3600) return `${Math.floor(s / 60)}m`
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`
})

// --- Cameras ---
const connectedCount = computed(() =>
  props.health?.cameras?.filter(c => c.connected).length ?? 0)
const totalCameras = computed(() =>
  props.health?.cameras?.length ?? 0)
const cameraColor = computed(() =>
  connectedCount.value === totalCameras.value && totalCameras.value > 0 ? '#22c55e' : '#f59e0b')

// --- Today's Alerts ---
const todayAlerts = computed(() =>
  props.cameras.reduce((sum, c) => sum + (c.alert_count_today || 0), 0))
const severityBreakdown = computed(() => {
  let high = 0, medium = 0, low = 0
  for (const c of props.cameras) {
    if (c.active_alert) {
      const sev = c.active_alert.severity
      if (sev === 'high') high++
      else if (sev === 'medium') medium++
      else low++
    }
  }
  return { high, medium, low }
})
const alertColor = computed(() => {
  if (severityBreakdown.value.high > 0) return '#ef4444'
  if (severityBreakdown.value.medium > 0) return '#f97316'
  if (todayAlerts.value > 0) return '#f59e0b'
  return '#22c55e'
})

// --- Peak Anomaly ---
const peakAnomaly = computed(() => {
  if (!props.cameras.length) return { score: 0, camera: '--' }
  const sorted = [...props.cameras].sort((a, b) => (b.current_score ?? 0) - (a.current_score ?? 0))
  return { score: sorted[0].current_score ?? 0, camera: sorted[0].name || sorted[0].camera_id }
})
const anomalyColor = computed(() => {
  const s = peakAnomaly.value.score
  if (s >= 0.7) return '#ef4444'
  if (s >= 0.3) return '#f59e0b'
  return '#22c55e'
})

// --- Avg Latency ---
const avgLatency = computed(() => {
  const cams = props.health?.cameras ?? []
  if (!cams.length) return 0
  return cams.reduce((sum, c) => sum + (c.avg_latency_ms ?? 0), 0) / cams.length
})
const latencyColor = computed(() => {
  if (avgLatency.value > 500) return '#ef4444'
  if (avgLatency.value > 100) return '#f59e0b'
  return '#22c55e'
})

// --- ECharts: shared gauge factory ---
function makeGauge(opts: { min: number; max: number; value: number; color: string; label: string; fontSize?: number }) {
  return {
    series: [{
      type: 'gauge',
      startAngle: 220,
      endAngle: -40,
      radius: '100%',
      center: ['50%', '60%'],
      min: opts.min,
      max: opts.max,
      pointer: { show: false },
      progress: { show: true, width: 6, roundCap: true, itemStyle: { color: opts.color } },
      axisLine: { lineStyle: { width: 6, color: [[1, themeStore.isDark ? '#1e1e36' : '#e5e7eb']] } },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { show: false },
      detail: {
        valueAnimation: true,
        formatter: opts.label,
        fontSize: opts.fontSize ?? 16,
        fontWeight: 600,
        color: opts.color,
        offsetCenter: [0, '10%'],
      },
      data: [{ value: opts.value }],
    }],
  }
}

const gaugeOption = computed(() =>
  makeGauge({ min: 0, max: 1, value: +peakAnomaly.value.score.toFixed(2), color: anomalyColor.value, label: '{value}', fontSize: 18 })
)

const cameraRingOption = computed(() => {
  const connected = connectedCount.value
  const total = totalCameras.value || 1
  return makeGauge({ min: 0, max: total, value: connected, color: cameraColor.value, label: connected + '/' + total })
})

// --- ECharts: Latency sparkline ---
const latencyHistory = ref<number[]>([])
watch(() => avgLatency.value, (v) => {
  const rounded = Math.round(v)
  if (latencyHistory.value.length > 0 && latencyHistory.value[latencyHistory.value.length - 1] === rounded) return
  latencyHistory.value.push(rounded)
  if (latencyHistory.value.length > 30) latencyHistory.value.shift()
})

const latencySparkOption = computed(() => ({
  grid: { left: 0, right: 0, top: 2, bottom: 0 },
  xAxis: { show: false, type: 'category', data: latencyHistory.value.map((_, i) => i) },
  yAxis: { show: false, type: 'value', min: 0 },
  series: [{
    type: 'line',
    data: latencyHistory.value,
    smooth: true,
    symbol: 'none',
    lineStyle: { width: 1.5, color: latencyColor.value },
    areaStyle: {
      color: {
        type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [
          { offset: 0, color: latencyColor.value + '40' },
          { offset: 1, color: latencyColor.value + '05' },
        ],
      },
    },
  }],
}))

// --- Calm state ---
const hasActiveAlerts = computed(() =>
  props.cameras.some(c => c.active_alert !== null && c.active_alert !== undefined))
</script>

<template>
  <div class="status-strip">
    <template v-if="!health">
      <div v-for="i in 5" :key="i" class="status-card status-card--loading">
        <div class="card-shimmer" />
      </div>
    </template>
    <template v-else>
      <!-- System Status -->
      <Tooltip :title="`运行 ${uptimeFormatted}`">
        <div class="status-card" :style="{ '--accent': statusColor }">
          <div class="card-top">
            <span class="card-label">系统状态</span>
            <span class="uptime-badge">{{ uptimeFormatted }}</span>
          </div>
          <div class="card-main">
            <span class="glow-dot" :style="{ background: statusColor, boxShadow: `0 0 10px ${statusColor}88` }" />
            <span class="card-value" :style="{ color: statusColor }">{{ statusLabel }}</span>
          </div>
        </div>
      </Tooltip>

      <!-- Cameras Online -->
      <Tooltip title="在线摄像头数量">
        <div class="status-card" :style="{ '--accent': cameraColor }">
          <div class="card-top">
            <span class="card-label">摄像头</span>
          </div>
          <div class="card-chart-row">
            <VChart :option="cameraRingOption" :autoresize="true" class="mini-gauge" />
            <span class="card-sub" :style="{ color: cameraColor }">在线</span>
          </div>
        </div>
      </Tooltip>

      <!-- Today's Alerts -->
      <Tooltip title="今日累计告警">
        <div class="status-card" :style="{ '--accent': alertColor }">
          <div class="card-top">
            <span class="card-label">今日告警</span>
          </div>
          <div class="card-main">
            <span class="card-value card-value--large" :style="{ color: alertColor }">{{ todayAlerts }}</span>
          </div>
          <div class="severity-tags">
            <Tag v-if="severityBreakdown.high" color="red" class="mini-tag">{{ severityBreakdown.high }} 高</Tag>
            <Tag v-if="severityBreakdown.medium" color="orange" class="mini-tag">{{ severityBreakdown.medium }} 中</Tag>
            <Tag v-if="severityBreakdown.low" color="gold" class="mini-tag">{{ severityBreakdown.low }} 低</Tag>
            <span v-if="!severityBreakdown.high && !severityBreakdown.medium && !severityBreakdown.low" class="card-sub" style="color: #4a5568">无活跃告警</span>
          </div>
        </div>
      </Tooltip>

      <!-- Peak Anomaly Score -->
      <Tooltip :title="`最高异常分来自 ${peakAnomaly.camera}`">
        <div class="status-card" :style="{ '--accent': anomalyColor }">
          <div class="card-top">
            <span class="card-label">峰值异常</span>
          </div>
          <div class="card-chart-row">
            <VChart :option="gaugeOption" :autoresize="true" class="mini-gauge" />
            <span class="card-sub" :title="peakAnomaly.camera" style="max-width: 80px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
              {{ peakAnomaly.camera }}
            </span>
          </div>
        </div>
      </Tooltip>

      <!-- Avg Latency with sparkline -->
      <Tooltip title="摄像头平均处理延迟">
        <div class="status-card" :style="{ '--accent': latencyColor }">
          <div class="card-top">
            <span class="card-label">平均延迟</span>
          </div>
          <div class="card-main">
            <span class="card-value" :style="{ color: latencyColor }">{{ Math.round(avgLatency) }}</span>
            <span class="card-unit">ms</span>
          </div>
          <div class="spark-container">
            <VChart
              v-if="latencyHistory.length > 1"
              :option="latencySparkOption"
              :autoresize="true"
              style="height: 28px; width: 100%"
            />
            <div v-else class="spark-placeholder" />
          </div>
        </div>
      </Tooltip>
    </template>
  </div>

  <!-- Calm state indicator -->
  <div v-if="health && !hasActiveAlerts" class="calm-indicator">
    <span class="calm-dot" />
    系统正常运行中
  </div>
</template>

<style scoped>
.status-strip {
  display: flex;
  gap: 10px;
  flex-shrink: 0;
  margin-bottom: 4px;
}

.status-card {
  flex: 1;
  background: var(--glass);
  border: 1px solid var(--line-2);
  border-radius: 8px;
  padding: 10px 14px;
  min-width: 0;
  transition: border-color 0.3s, box-shadow 0.3s, transform 0.2s;
  position: relative;
  overflow: hidden;
}

.status-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--accent, #3b82f6);
  opacity: 0.8;
}

.status-card:hover {
  border-color: var(--line-2);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  transform: translateY(-1px);
}

.status-card--loading {
  min-height: 80px;
}

.card-shimmer {
  height: 100%;
  background: linear-gradient(90deg, rgba(10, 10, 15, 0.05) 25%, var(--glass) 50%, rgba(10, 10, 15, 0.05) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

.card-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}

.card-label {
  font-size: 11px;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 500;
}

.uptime-badge {
  font-size: 10px;
  color: #4a5568;
  background: rgba(10, 10, 15, 0.05);
  padding: 1px 6px;
  border-radius: 3px;
}

.card-main {
  display: flex;
  align-items: baseline;
  gap: 6px;
}

.card-value {
  font-size: 22px;
  font-weight: 700;
  line-height: 1.2;
  font-variant-numeric: tabular-nums;
}

.card-value--large {
  font-size: 28px;
}

.card-unit {
  font-size: 12px;
  color: #6b7280;
  font-weight: 400;
}

.card-chart-row {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.mini-gauge {
  width: 80px;
  height: 52px;
}

.glow-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
  animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.card-sub {
  font-size: 11px;
  color: #6b7280;
}

.severity-tags {
  display: flex;
  gap: 4px;
  margin-top: 4px;
  flex-wrap: wrap;
}

.mini-tag {
  font-size: 10px;
  line-height: 16px;
  padding: 0 4px;
  margin: 0;
  border-radius: 3px;
}

.spark-container {
  margin-top: 4px;
  height: 28px;
}

.spark-placeholder {
  height: 28px;
  background: linear-gradient(90deg, rgba(10, 10, 15, 0.05) 0%, var(--glass) 50%, rgba(10, 10, 15, 0.05) 100%);
  border-radius: 3px;
}

.calm-indicator {
  text-align: center;
  padding: 5px 0;
  color: rgba(34, 197, 94, 0.5);
  font-size: 12px;
  letter-spacing: 2px;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.calm-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: rgba(34, 197, 94, 0.4);
  animation: pulse-glow 3s ease-in-out infinite;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
  .status-strip {
    flex-wrap: wrap;
    gap: 6px;
  }
  .status-card {
    flex: 1 1 calc(50% - 6px);
    min-width: 140px;
    padding: 8px 10px;
  }
}

@media (max-width: 480px) {
  .status-card {
    flex: 1 1 100%;
  }
}

/* ── Light theme overrides ── */
:global(.light-theme) .status-card {
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border-color: #e5e7eb;
}
:global(.light-theme) .status-card .card-label {
  color: #6b7280;
}
:global(.light-theme) .status-card .card-value {
  color: #111827;
}
</style>
