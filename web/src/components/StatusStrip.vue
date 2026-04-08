<script setup lang="ts">
import { computed } from 'vue'
import { Statistic, Skeleton, Tag, Tooltip, Typography } from 'ant-design-vue'
import type { CameraTileData } from './VideoTile.vue'

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

// --- Calm state ---
const hasActiveAlerts = computed(() =>
  props.cameras.some(c => c.active_alert !== null && c.active_alert !== undefined))
</script>

<template>
  <div class="status-strip">
    <template v-if="!health">
      <div v-for="i in 5" :key="i" class="status-card">
        <Skeleton active :paragraph="false" :title="{ width: '60%' }" />
      </div>
    </template>
    <template v-else>
      <!-- System Status -->
      <Tooltip :title="`运行 ${uptimeFormatted}`">
        <div class="status-card" :style="{ '--accent': statusColor }">
          <Typography.Text type="secondary" class="card-label">系统状态</Typography.Text>
          <div class="card-value-row">
            <span class="glow-dot" :style="{ background: statusColor, boxShadow: `0 0 8px ${statusColor}88` }" />
            <span class="card-value" :style="{ color: statusColor }">{{ statusLabel }}</span>
          </div>
          <Typography.Text type="secondary" class="card-sub">已运行 {{ uptimeFormatted }}</Typography.Text>
        </div>
      </Tooltip>

      <!-- Cameras Online -->
      <Tooltip title="在线摄像头数量">
        <div class="status-card" :style="{ '--accent': cameraColor }">
          <Typography.Text type="secondary" class="card-label">摄像头</Typography.Text>
          <Statistic
            :value="connectedCount"
            :value-style="{ color: cameraColor, fontSize: '22px', fontWeight: 600 }"
          >
            <template #suffix>
              <span style="font-size: 14px; color: #6b7280"> / {{ totalCameras }}</span>
            </template>
          </Statistic>
          <Typography.Text type="secondary" class="card-sub">在线</Typography.Text>
        </div>
      </Tooltip>

      <!-- Today's Alerts -->
      <Tooltip title="今日累计告警">
        <div class="status-card" :style="{ '--accent': alertColor }">
          <Typography.Text type="secondary" class="card-label">今日告警</Typography.Text>
          <Statistic
            :value="todayAlerts"
            :value-style="{ color: alertColor, fontSize: '22px', fontWeight: 600 }"
          />
          <div class="severity-tags">
            <Tag v-if="severityBreakdown.high" color="red" class="mini-tag">{{ severityBreakdown.high }} 高</Tag>
            <Tag v-if="severityBreakdown.medium" color="orange" class="mini-tag">{{ severityBreakdown.medium }} 中</Tag>
            <Tag v-if="severityBreakdown.low" color="gold" class="mini-tag">{{ severityBreakdown.low }} 低</Tag>
          </div>
        </div>
      </Tooltip>

      <!-- Peak Anomaly Score -->
      <Tooltip :title="`最高异常分来自 ${peakAnomaly.camera}`">
        <div class="status-card" :style="{ '--accent': anomalyColor }">
          <Typography.Text type="secondary" class="card-label">峰值异常</Typography.Text>
          <Statistic
            :value="peakAnomaly.score"
            :precision="2"
            :value-style="{ color: anomalyColor, fontSize: '22px', fontWeight: 600 }"
          />
          <Typography.Text type="secondary" class="card-sub" :title="peakAnomaly.camera" style="max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: block;">
            {{ peakAnomaly.camera }}
          </Typography.Text>
        </div>
      </Tooltip>

      <!-- Avg Latency -->
      <Tooltip title="摄像头平均处理延迟">
        <div class="status-card" :style="{ '--accent': latencyColor }">
          <Typography.Text type="secondary" class="card-label">平均延迟</Typography.Text>
          <Statistic
            :value="Math.round(avgLatency)"
            :value-style="{ color: latencyColor, fontSize: '22px', fontWeight: 600 }"
          >
            <template #suffix>
              <span style="font-size: 13px; color: #6b7280">ms</span>
            </template>
          </Statistic>
          <Typography.Text type="secondary" class="card-sub">处理延迟</Typography.Text>
        </div>
      </Tooltip>
    </template>
  </div>

  <!-- Calm state indicator -->
  <div v-if="health && !hasActiveAlerts" class="calm-indicator">
    系统正常运行中
  </div>
</template>

<style scoped>
.status-strip {
  display: flex;
  gap: 10px;
  flex-shrink: 0;
  margin-bottom: 12px;
}

.status-card {
  flex: 1;
  background: #1a1a2e;
  border: 1px solid #2d2d4a;
  border-top: 2px solid var(--accent, #3b82f6);
  border-radius: 6px;
  padding: 10px 14px;
  min-width: 0;
  transition: border-color 0.3s, box-shadow 0.3s;
}

.status-card:hover {
  border-color: #3d3d5c;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.card-label {
  font-size: 12px;
  display: block;
  margin-bottom: 4px;
}

.card-value-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.card-value {
  font-size: 22px;
  font-weight: 600;
  line-height: 1.2;
}

.glow-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.card-sub {
  font-size: 11px;
  margin-top: 2px;
}

.severity-tags {
  display: flex;
  gap: 4px;
  margin-top: 2px;
  flex-wrap: wrap;
}

.mini-tag {
  font-size: 10px;
  line-height: 16px;
  padding: 0 4px;
  margin: 0;
  border-radius: 3px;
}

.calm-indicator {
  text-align: center;
  padding: 4px 0;
  color: rgba(34, 197, 94, 0.5);
  font-size: 12px;
  letter-spacing: 2px;
  border-top: 1px solid rgba(34, 197, 94, 0.1);
  margin-bottom: 8px;
}
</style>
