<script setup lang="ts">
import { computed } from 'vue'
import { VideoCameraOutlined } from '@ant-design/icons-vue'
import VideoTile from './VideoTile.vue'
import ContentSkeleton from './ContentSkeleton.vue'
import EmptyState from './EmptyState.vue'
import type { CameraTileData } from './VideoTile.vue'
import { useRouter } from 'vue-router'

const props = defineProps<{
  cameras: CameraTileData[]
  loading: boolean
  layoutMode: '1x1' | '2x2' | '3x3' | 'focus'
  focusCamera: string | null
}>()

const emit = defineEmits<{
  (e: 'update:focusCamera', cameraId: string | null): void
  (e: 'update:layoutMode', mode: '1x1' | '2x2' | '3x3' | 'focus'): void
}>()

const router = useRouter()

// Calculate N-Split Grid styles natively
const gridStyle = computed(() => {
  if (props.layoutMode === '1x1') return { gridTemplateColumns: '1fr', gridTemplateRows: '1fr' }
  if (props.layoutMode === '3x3') return { gridTemplateColumns: 'repeat(3, 1fr)', gridTemplateRows: 'repeat(3, 1fr)' }
  if (props.layoutMode === 'focus') return { gridTemplateColumns: '2fr 1fr', gridTemplateRows: '1fr 1fr' }
  return { gridTemplateColumns: 'repeat(2, 1fr)', gridTemplateRows: 'repeat(2, 1fr)' }
})

// Calculate displayed cameras based on layout and focus
const displayCameras = computed(() => {
  if (props.layoutMode === '1x1' && props.focusCamera) {
    return props.cameras.filter(c => c.camera_id === props.focusCamera)
  }
  if (props.layoutMode === 'focus' && props.focusCamera) {
    const main = props.cameras.find(c => c.camera_id === props.focusCamera)
    const others = props.cameras.filter(c => c.camera_id !== props.focusCamera).slice(0, 3)
    return main ? [main, ...others] : props.cameras.slice(0, 4)
  }
  const maxTiles: Record<string, number> = { '1x1': 1, '2x2': 4, '3x3': 9, focus: 4 }
  return props.cameras.slice(0, maxTiles[props.layoutMode] || 4)
})

function handleTileDblClick(cameraId: string) {
  if (props.layoutMode === '1x1' && props.focusCamera === cameraId) {
    emit('update:layoutMode', '2x2')
    emit('update:focusCamera', null)
  } else {
    emit('update:layoutMode', '1x1')
    emit('update:focusCamera', cameraId)
  }
}

function handleAlertClick(alertId: string) {
  router.push(`/alerts?id=${alertId}`)
}

</script>

<template>
  <div class="video-wall-root">
    <div v-if="loading" class="video-grid" :style="gridStyle">
      <ContentSkeleton v-for="n in 4" :key="'skel-' + n" type="card" :rows="6" />
    </div>
    
    <EmptyState
      v-else-if="cameras.length === 0"
      icon="camera"
      title="未配置视频流"
      description="前端尚未接收到任何摄像头配置"
      action-text="前往设置"
      action-route="/cameras"
    />
    
    <div v-else class="video-grid" :style="gridStyle">
      <VideoTile
        v-for="(cam, idx) in displayCameras"
        :key="cam.camera_id"
        :camera="cam"
        :layout="layoutMode === 'focus' && idx === 0 ? 'focus-main' : layoutMode === 'focus' ? 'focus-side' : layoutMode"
        :style="layoutMode === 'focus' && idx === 0 ? { gridRow: '1 / 3' } : {}"
        @dblclick="handleTileDblClick"
        @alert-click="handleAlertClick"
      />
      <!-- Empty placeholders -->
      <div
        v-for="n in Math.max(0, (layoutMode === '2x2' ? 4 : layoutMode === '3x3' ? 9 : 0) - displayCameras.length)"
        :key="'empty-' + n"
        class="empty-tile"
      >
        <div class="empty-tile-content">
          <VideoCameraOutlined style="font-size: 24px; color: var(--argus-icon-muted, #4a5568)" />
          <span>待接入</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.video-wall-root {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 320px;
}

.video-grid {
  display: grid;
  gap: 8px;
  flex: 1;
}

.empty-tile {
  border: 1px dashed var(--argus-border, #2d2d4a);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--argus-surface, #0f0f13);
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
  color: var(--argus-icon-muted, #4a5568);
  font-size: 12px;
  opacity: 0.6;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
  .video-grid {
    grid-template-columns: 1fr !important;
    grid-template-rows: auto !important;
  }
}
</style>
