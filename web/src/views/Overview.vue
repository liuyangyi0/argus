<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Button, Space, Typography, Tooltip } from 'ant-design-vue'
import {
  AppstoreOutlined,
  BlockOutlined,
  BorderOutlined,
  TableOutlined,
} from '@ant-design/icons-vue'
import VideoTile, { type CameraTileData } from '../components/VideoTile.vue'
import AlertSidebar from '../components/AlertSidebar.vue'
import { getWallStatus } from '../api'
import { useWebSocket } from '../composables/useWebSocket'

const cameras = ref<CameraTileData[]>([])
const loading = ref(true)
const layout = ref<'1x1' | '2x2' | '3x3' | 'focus'>('2x2')
const focusCamera = ref<string | null>(null)

async function fetchWallStatus() {
  try {
    const res = await getWallStatus()
    cameras.value = res.data.cameras || []
  } catch {
    // Fallback: empty
  } finally {
    loading.value = false
  }
}

useWebSocket({
  topics: ['wall', 'alerts'],
  onMessage(topic, data) {
    if (topic === 'wall' && data?.cameras) {
      // Merge wall updates into existing camera data
      for (const update of data.cameras) {
        const cam = cameras.value.find(c => c.camera_id === update.camera_id)
        if (cam) {
          cam.current_score = update.current_score ?? cam.current_score
          cam.score_sparkline = update.score_sparkline ?? cam.score_sparkline
          if (update.active_alert !== undefined) cam.active_alert = update.active_alert
          if (update.border_state) {
            // Map border_state to active_alert for UI
          }
        }
      }
    }
  },
  fallbackPoll: fetchWallStatus,
  fallbackInterval: 5000,
})

onMounted(fetchWallStatus)

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
    // Return to 2x2
    layout.value = '2x2'
    focusCamera.value = null
  } else {
    layout.value = '1x1'
    focusCamera.value = cameraId
  }
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
      <!-- Toolbar -->
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-shrink: 0">
        <Typography.Title :level="4" style="margin: 0">值班台</Typography.Title>
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
      <AlertSidebar />
    </div>
  </div>
</template>
