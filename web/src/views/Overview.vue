<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'

defineOptions({ name: 'OverviewPage' })
import { Button, Space, Typography, Tooltip } from 'ant-design-vue'
import {
  AppstoreOutlined,
  BlockOutlined,
  BorderOutlined,
  TableOutlined,
  BellOutlined,
  VideoCameraOutlined,
} from '@ant-design/icons-vue'
import VideoTile, { type CameraTileData } from '../components/VideoTile.vue'
import AlertSidebar from '../components/AlertSidebar.vue'
import StatusStrip from '../components/StatusStrip.vue'
import ScoreTrendPanel from '../components/ScoreTrendPanel.vue'
import AlertTimeline from '../components/AlertTimeline.vue'
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

// Layout button configs for cleaner template
const layoutButtons = [
  { key: '1x1' as const, icon: BorderOutlined, tip: '单画面' },
  { key: '2x2' as const, icon: AppstoreOutlined, tip: '四分屏' },
  { key: '3x3' as const, icon: TableOutlined, tip: '九分屏' },
  { key: 'focus' as const, icon: BlockOutlined, tip: '焦点模式' },
]
</script>

<template>
  <div class="overview-root">
    <!-- Main content area -->
    <div class="overview-main">
      <!-- High alert banner -->
      <div v-if="hasHighAlert" class="high-alert-banner">
        <div class="high-alert-inner">
          <div class="high-alert-pulse" />
          <BellOutlined style="color: #ef4444; font-size: 16px" />
          <Typography.Text strong style="color: #fca5a5; font-size: 13px">
            {{ highAlertCameras.length }} 个高级告警需要立即处理
          </Typography.Text>
          <Typography.Text type="secondary" style="font-size: 12px; color: #ef444488">
            {{ highAlertCameras.map(c => c.camera_id).join(', ') }}
          </Typography.Text>
          <Button
            type="primary"
            danger
            size="small"
            style="margin-left: auto"
            @click="router.push('/alerts?severity=high')"
          >
            查看告警
          </Button>
        </div>
      </div>

      <!-- Toolbar -->
      <div class="overview-toolbar">
        <div class="toolbar-left">
          <span
            class="status-glow"
            :class="{ 'status-glow--pulse': systemStatus !== 'healthy' }"
            :style="{ background: statusColor, boxShadow: `0 0 10px ${statusColor}88` }"
          />
          <Typography.Title :level="4" style="margin: 0; color: #e2e8f0">值班台</Typography.Title>
          <Typography.Text v-if="statusLabel" style="font-size: 12px; color: #6b7280">{{ statusLabel }}</Typography.Text>
        </div>
        <div class="toolbar-right">
          <Tooltip v-for="btn in layoutButtons" :key="btn.key" :title="btn.tip">
            <Button
              :type="layout === btn.key ? 'primary' : 'text'"
              size="small"
              class="layout-btn"
              :class="{ 'layout-btn--active': layout === btn.key }"
              @click="setLayout(btn.key)"
            >
              <template #icon><component :is="btn.icon" /></template>
            </Button>
          </Tooltip>
        </div>
      </div>

      <!-- Status Strip -->
      <StatusStrip :health="health" :cameras="cameras" />

      <!-- Video Grid -->
      <div class="video-grid" :style="gridStyle">
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
          class="empty-tile"
        >
          <div class="empty-tile-content">
            <VideoCameraOutlined style="font-size: 24px; color: #2d2d4a" />
            <span>未配置</span>
          </div>
        </div>
      </div>

      <!-- Score Trend Panel -->
      <ScoreTrendPanel :cameras="cameras" />

      <!-- 24h Alert Timeline -->
      <AlertTimeline @segment-click="(p) => router.push(
        p.segment.first_alert_id
          ? `/cameras/${p.camera_id}?replay=${p.segment.first_alert_id}`
          : `/alerts?camera=${p.camera_id}`
      )" />
    </div>

    <!-- Alert Sidebar -->
    <div class="overview-sidebar">
      <AlertSidebar :health="health" />
    </div>
  </div>
</template>

<style scoped>
.overview-root {
  display: flex;
  gap: 0;
  height: calc(100vh - 72px);
  margin: -24px;
  padding: 0;
}

.overview-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 10px;
}

.overview-sidebar {
  border-left: 1px solid #1f2937;
  padding: 16px 12px 16px 0;
  flex-shrink: 0;
}

/* High alert banner */
.high-alert-banner {
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 8px;
  overflow: hidden;
  position: relative;
}

.high-alert-inner {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  position: relative;
  z-index: 1;
}

.high-alert-pulse {
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent 0%, rgba(239, 68, 68, 0.05) 50%, transparent 100%);
  animation: alert-sweep 3s ease-in-out infinite;
}

@keyframes alert-sweep {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

/* Toolbar */
.overview-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}

.toolbar-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.toolbar-right {
  display: flex;
  gap: 4px;
}

.layout-btn {
  width: 32px;
  height: 32px;
  border-radius: 6px;
  transition: all 0.2s;
}

.layout-btn:not(.layout-btn--active) {
  color: #6b7280;
}

.layout-btn:not(.layout-btn--active):hover {
  background: #1e1e36;
  color: #e2e8f0;
}

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

/* Video grid */
.video-grid {
  display: grid;
  gap: 8px;
  flex: 1;
  min-height: 0;
}

.empty-tile {
  border: 1px dashed #2d2d4a;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #12121e;
  transition: border-color 0.3s;
}

.empty-tile:hover {
  border-color: #3b82f644;
}

.empty-tile-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: #2d2d4a;
  font-size: 12px;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
  .overview-root {
    flex-direction: column;
    height: auto;
  }
  .overview-sidebar {
    border-left: none;
    border-top: 1px solid #1f2937;
    padding: 12px;
    max-height: 40vh;
    overflow-y: auto;
  }
  .overview-main {
    padding: 10px;
  }
  .overview-toolbar {
    flex-wrap: wrap;
    gap: 8px;
  }
  .toolbar-right {
    flex-wrap: wrap;
  }
  .video-grid {
    grid-template-columns: 1fr !important;
    grid-template-rows: auto !important;
  }
}

@media (max-width: 480px) {
  .overview-main {
    padding: 8px;
    gap: 6px;
  }
  .layout-btn {
    width: 28px;
    height: 28px;
  }
}
</style>
