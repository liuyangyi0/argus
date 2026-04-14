<script setup lang="ts">
import { provide, onMounted, onUnmounted } from 'vue'
import { useReplayController } from '../composables/useReplayController'
import ReplayVideoView from './replay/ReplayVideoView.vue'
import ReplayControls from './replay/ReplayControls.vue'
import ReplayReference from './replay/ReplayReference.vue'
import ReplaySignalTracks from './replay/ReplaySignalTracks.vue'

const props = defineProps<{
  alertId: string
}>()

const ctrl = useReplayController(props.alertId)
provide('replayCtrl', ctrl)

onMounted(() => {
  ctrl.loadData()
  window.addEventListener('keydown', ctrl.handleKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', ctrl.handleKeydown)
})
</script>

<template>
  <div v-if="ctrl.metadata.value" class="replay-root glass-panel">
    <!-- Viewport area -->
    <div class="replay-viewport-row">
      <div class="replay-main">
        <ReplayVideoView />
      </div>
      <ReplayReference />
    </div>

    <!-- Controls -->
    <ReplayControls />

    <!-- Details/Tracks -->
    <ReplaySignalTracks />
  </div>

  <div v-else style="padding: 24px; text-align: center; color: var(--ink-4)">
    {{ ctrl.loading.value ? '加载回放数据...' : '获取回放数据失败...' }}
  </div>
</template>

<style scoped>
.replay-root {
  border-radius: 6px;
  padding: 0;
  overflow: hidden;
  border: 1px solid var(--line-2);
  display: flex;
  flex-direction: column;
}

.replay-viewport-row {
  display: flex;
  gap: 0;
  min-height: 0;
}

.replay-main {
  flex: 1;
  min-width: 0;
}
</style>
