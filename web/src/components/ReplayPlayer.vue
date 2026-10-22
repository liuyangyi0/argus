<script setup lang="ts">
import { provide, onMounted, onUnmounted } from 'vue'
import { useReplayController } from '../composables/useReplayController'
import ReplayCanvas from './replay/ReplayCanvas.vue'
import ReplayControls from './replay/ReplayControls.vue'
import ReplayReference from './replay/ReplayReference.vue'
import ReplaySignalTracks from './replay/ReplaySignalTracks.vue'
import AsyncState from './common/AsyncState.vue'
import StatusBanner from './common/StatusBanner.vue'
import { Button } from 'ant-design-vue'

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
  <AsyncState
    :loading="ctrl.metadataLoading.value && !ctrl.metadata.value"
    :refreshing="ctrl.metadataLoading.value && !!ctrl.metadata.value"
    :error="!ctrl.metadata.value && ![403, 404].includes(ctrl.metadataErrorStatus.value ?? 0) ? ctrl.metadataError.value : ''"
    :forbidden="!ctrl.metadata.value && ctrl.metadataErrorStatus.value === 403"
    :not-found="!ctrl.metadata.value && ctrl.metadataErrorStatus.value === 404"
    :empty="ctrl.metadataState.value !== 'loading' && !ctrl.metadataError.value && !ctrl.metadata.value"
    skeleton-type="card"
    :skeleton-rows="8"
    empty-title="没有可回放的数据"
    empty-description="该告警尚未生成录像或录像数据已被清理。"
    @retry="ctrl.loadData"
  >
    <StatusBanner
      v-if="ctrl.metadata.value && ctrl.metadataError.value"
      class="replay-stale-banner"
      status="error"
      title="回放元数据刷新失败，当前显示上次成功加载的数据"
      :description="ctrl.metadataError.value"
    >
      <template #action><Button size="small" @click="ctrl.retryMetadata">重试</Button></template>
    </StatusBanner>

    <StatusBanner
      v-if="ctrl.signalsLoading.value && !ctrl.signals.value"
      class="replay-stale-banner"
      status="loading"
      title="正在加载信号与时间轴"
    />
    <StatusBanner
      v-else-if="ctrl.signalsState.value === 'error'"
      class="replay-stale-banner"
      status="error"
      :title="ctrl.signals.value ? '信号刷新失败，当前显示上次成功加载的数据' : '信号与时间轴加载失败'"
      :description="ctrl.signalsError.value"
    >
      <template #action>
        <Button size="small" aria-label="重试信号与时间轴" @click="ctrl.loadSignals">重试</Button>
      </template>
    </StatusBanner>

    <StatusBanner
      v-if="ctrl.trajectoryState.value === 'error'"
      class="replay-stale-banner"
      status="warning"
      title="物理轨迹加载失败，视频和其他信号仍可使用"
      :description="ctrl.trajectoryError.value"
    >
      <template #action><Button size="small" @click="ctrl.loadTrajectory">重试</Button></template>
    </StatusBanner>

    <StatusBanner
      v-if="ctrl.videoState.value === 'error'"
      class="replay-stale-banner"
      status="error"
      title="录像视频加载失败，信号和轨迹仍可查看"
      :description="ctrl.videoError.value"
    >
      <template #action><Button size="small" @click="ctrl.reloadVideo">重新加载视频</Button></template>
    </StatusBanner>

    <div
      v-if="ctrl.metadata.value"
      class="replay-root glass-panel"
      data-test="replay-root"
      :data-metadata-state="ctrl.metadataState.value"
      :data-signals-state="ctrl.signalsState.value"
      :data-trajectory-state="ctrl.trajectoryState.value"
      :data-video-state="ctrl.videoState.value"
    >
      <!-- Viewport area -->
      <div class="replay-viewport-row">
        <div class="replay-main">
          <ReplayCanvas
            :key="`${ctrl.metadata.value.alert_id}:${ctrl.videoUrl.value}:${ctrl.hasHeatmaps.value ? 'heatmaps' : 'plain'}`"
          />
        </div>
        <ReplayReference />
      </div>

      <!-- Controls -->
      <ReplayControls />

      <!-- Details/Tracks -->
      <ReplaySignalTracks />
    </div>
  </AsyncState>
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

.replay-stale-banner {
  margin-bottom: 10px;
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
