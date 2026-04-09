<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { Button, Space, Typography, Tooltip } from 'ant-design-vue'
import {
  AppstoreOutlined,
  BlockOutlined,
  BorderOutlined,
  TableOutlined,
  BellOutlined,
} from '@ant-design/icons-vue'
import VideoTile, { type CameraTileData } from '../components/VideoTile.vue'
import AlertSidebar from '../components/AlertSidebar.vue'
import StatusStrip from '../components/StatusStrip.vue'
import { getWallStatus, getHealth } from '../api'
import { useWebSocket } from '../composables/useWebSocket'

const router = useRouter()
const cameras = ref<CameraTileData[]>([])
const health = ref<any>(null)
const loading = ref(true)
const layout = ref<'1x1' | '2x2' | '3x3' | 'focus'>('2x2')
const focusCamera = ref<string | null>(null)

const systemStatus = computed(() => health.value?.status ?? 'unknown')
const statusLabel = computed(() => {
  const m: Record<string, string> = { healthy: '运行正常', degraded: '降级运行', unhealthy: '系统异常' }
  return m[systemStatus.value] || ''
})
const statusColor = computed(() => {
  const m: Record<string, string> = { healthy: '#22c55e', degraded: '#f59e0b', unhealthy: '#ef4444' }
  return m[systemStatus.value] || '#6b7280'
})

// High alert detection from camera tiles
const highAlertCameras = computed(() =>
  cameras.value.filter(c => c.active_alert?.severity === 'high')
)
const hasHighAlert = computed(() => highAlertCameras.value.length > 0)

// Last alert time indicator
const lastAlertTime = computed(() => {
  const withAlerts = cameras.value.filter(c => c.active_alert)
  if (withAlerts.length === 0) return null
  return '活跃中'
})

async function fetchWallStatus() {
  try {
    const data = await getWallStatus()
    cameras.value = data.cameras || []
  } catch {
    // Fallback: empty
  } finally {
    loading.value = false
  }
}

async function fetchHealth() {
  try {
    health.value = await getHealth()
  } catch { /* silent */ }
}

useWebSocket({
  topics: ['wall', 'alerts', 'health'],
  onMessage(topic, data) {
    if (topic === 'wall' && data?.cameras) {
      for (const update of data.cameras) {
        const cam = cameras.value.find(c => c.camera_id === update.camera_id)
        if (cam) {
          cam.current_score = update.current_score ?? cam.current_score
          cam.score_sparkline = update.score_sparkline ?? cam.score_sparkline
          if (update.active_alert !== undefined) cam.active_alert = update.active_alert
        }
      }
    }
    if (topic === 'health') {
      health.value = data
    }
  },
  fallbackPoll: () => Promise.all([fetchWallStatus(), fetchHealth()]),
  fallbackInterval: 5000,
})

onMounted(() => Promise.all([fetchWallStatus(), fetchHealth()]))

const gridStyle = computed(() => {
  if (layout.value === '1x1') return { gridTemplateColumns: '1fr', gridTemplateRows: '1fr' }
  if (layout.value === '3x3') return { gridTemplateColumns: 'repeat(3, 1fr)', gridTemplateRows: 'repeat(3, 1fr)' }
  if (layout.value === 'focus') return { gridTemplateColumns: '2fr 1fr', gridTemplateRows: '1fr 1fr' }
  return { gridTemplateColumns: 'repeat(2, 1fr)', gridTemplateRows: 'repeat(2, 1fr)' }
})

const displayCameras = computed(() => {
  if (layout.value === '1x1' && focusCamera.value) {
    return cameras.value.filter(c => c.camera_id === focusCamera.value)
  }
  if (layout.value === 'focus' && focusCamera.value) {
    const main = cameras.value.find(c => c.camera_id === focusCamera.value)
    const others = cameras.value.filter(c => c.camera_id !== focusCamera.value).slice(0, 3)
    return main ? [main, ...others] : cameras.value.slice(0, 4)
  }
  const maxTiles: Record<string, number> = { '1x1': 1, '2x2': 4, '3x3': 9, focus: 4 }
  return cameras.value.slice(0, maxTiles[layout.value] || 4)
})

function handleTileDblClick(cameraId: string) {
  if (layout.value === '1x1' && focusCamera.value === cameraId) {
    layout.value = '2x2'
    focusCamera.value = null
  } else {
    layout.value = '1x1'
    focusCamera.value = cameraId
  }
}

function handleAlertClick(alertId: string) {
  router.push(`/alerts?id=${alertId}`)
}

function setLayout(l: typeof layout.value) {
  layout.value = l
  if (l !== '1x1' && l !== 'focus') focusCamera.value = null
}
</script>

<template>
  <div style="display: flex; gap: 0; height: calc(100vh - 72px); margin: -24px; padding: 0">
    <!-- Video Wall Area -->
    <div style="flex: 1; min-width: 0; display: flex; flex-direction: column; padding: 16px">
      <!-- High alert banner -->
      <div
        v-if="hasHighAlert"
        style="display: flex; align-items: center; gap: 10px; padding: 8px 16px; margin-bottom: 10px; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.4); border-radius: 6px"
      >
        <BellOutlined style="color: #ef4444; font-size: 16px" />
        <Typography.Text strong style="color: #ef4444; font-size: 13px">
          {{ highAlertCameras.length }} 个高级告警需要处理
        </Typography.Text>
        <Typography.Text type="secondary" style="font-size: 12px">
          ({{ highAlertCameras.map(c => c.camera_id).join(', ') }})
        </Typography.Text>
        <Button
          type="primary"
          danger
          size="small"
          style="margin-left: auto"
          @click="router.push('/alerts?severity=high')"
        >
          查看
        </Button>
      </div>

      <!-- Toolbar -->
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid rgba(59, 130, 246, 0.12); flex-shrink: 0">
        <div style="display: flex; align-items: center; gap: 10px">
          <span
            :class="['status-glow', systemStatus !== 'healthy' && 'status-glow--pulse']"
            :style="{ background: statusColor, boxShadow: `0 0 8px ${statusColor}88` }"
          />
          <Typography.Title :level="4" style="margin: 0">值班台</Typography.Title>
          <Typography.Text v-if="statusLabel" type="secondary" style="font-size: 12px">{{ statusLabel }}</Typography.Text>
          <!-- Last alert indicator -->
          <Typography.Text
            v-if="lastAlertTime"
            type="secondary"
            style="font-size: 11px; margin-left: 8px; color: #ef4444"
          >
            告警 {{ lastAlertTime }}
          </Typography.Text>
        </div>
        <Space>
          <Tooltip title="1x1">
            <Button
              :type="layout === '1x1' ? 'primary' : 'default'"
              size="small"
              @click="setLayout('1x1')"
            >
              <template #icon><BorderOutlined /></template>
            </Button>
          </Tooltip>
          <Tooltip title="2x2">
            <Button
              :type="layout === '2x2' ? 'primary' : 'default'"
              size="small"
              @click="setLayout('2x2')"
            >
              <template #icon><AppstoreOutlined /></template>
            </Button>
          </Tooltip>
          <Tooltip title="3x3">
            <Button
              :type="layout === '3x3' ? 'primary' : 'default'"
              size="small"
              @click="setLayout('3x3')"
            >
              <template #icon><TableOutlined /></template>
            </Button>
          </Tooltip>
          <Tooltip title="焦点模式">
            <Button
              :type="layout === 'focus' ? 'primary' : 'default'"
              size="small"
              @click="setLayout('focus')"
            >
              <template #icon><BlockOutlined /></template>
            </Button>
          </Tooltip>
        </Space>
      </div>

      <!-- Status Strip -->
      <StatusStrip :health="health" :cameras="cameras" />

      <!-- Grid -->
      <div
        :style="{
          display: 'grid',
          ...gridStyle,
          gap: '8px',
          flex: '1',
          minHeight: 0,
        }"
      >
        <VideoTile
          v-for="(cam, idx) in displayCameras"
          :key="cam.camera_id"
          :camera="cam"
          :layout="layout === 'focus' && idx === 0 ? 'focus-main' : layout === 'focus' ? 'focus-side' : layout"
          :style="layout === 'focus' && idx === 0 ? { gridRow: '1 / 3' } : {}"
          @dblclick="handleTileDblClick"
          @alert-click="handleAlertClick"
        />
        <!-- Empty slots -->
        <div
          v-for="n in Math.max(0, (layout === '2x2' ? 4 : layout === '3x3' ? 9 : 0) - displayCameras.length)"
          :key="'empty-' + n"
          style="border: 1px dashed #2d2d4a; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #4a5568; font-size: 13px"
        >
          无摄像头
        </div>
      </div>
    </div>

    <!-- Alert Sidebar -->
    <div style="border-left: 1px solid #1f2937; padding: 16px 12px 16px 0; flex-shrink: 0">
      <AlertSidebar :health="health" />
    </div>
  </div>
</template>

<style scoped>
.status-glow {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.status-glow--pulse {
  animation: pulse-glow 2s ease-in-out infinite;
}
@keyframes pulse-glow {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
</style>
