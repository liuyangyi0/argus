<script setup lang="ts">
import { inject, computed } from 'vue'
import { CaretRightOutlined, PauseOutlined } from '@ant-design/icons-vue'
import type { useReplayController } from '../../composables/useReplayController'

const ctrl = inject<ReturnType<typeof useReplayController>>('replayCtrl')!

const speeds = [0.25, 0.5, 1, 2, 4]

const progressPct = computed(() => {
  const total = Math.max((ctrl.metadata.value?.frame_count || 1) - 1, 1)
  return (ctrl.currentIndex.value / total) * 100
})

const triggerProgressPct = computed(() => {
  if (!ctrl.metadata.value) return 50
  const triggerIdx = ctrl.metadata.value.trigger_frame_index || 0
  const total = ctrl.metadata.value.frame_count || 1
  return Math.round((triggerIdx / total) * 100)
})

const remainingRecordingSeconds = computed(() => {
  if (!ctrl.metadata.value || ctrl.metadata.value.status !== 'recording') return 0
  const triggerTs = ctrl.metadata.value.trigger_timestamp || 0
  const postSeconds = ctrl.metadata.value.severity === 'low' ? 10 : 30
  const deadline = triggerTs + postSeconds
  const now = Date.now() / 1000
  return Math.max(0, Math.round(deadline - now))
})
</script>

<template>
  <div class="replay-controls-wrapper">
    <!-- Controls bar -->
    <div class="replay-controls">
      <button class="ctrl-btn" @click="ctrl.goToStart" title="Start">&#9198;</button>
      <button class="ctrl-btn" @click="ctrl.stepFrame(-1)" title="-1 frame">&#9664;&#9664;</button>
      <button class="ctrl-btn ctrl-play" @click="ctrl.togglePlay">
        <PauseOutlined v-if="ctrl.playing.value" />
        <CaretRightOutlined v-else />
      </button>
      <button class="ctrl-btn" @click="ctrl.stepFrame(1)" title="+1 frame">&#9654;&#9654;</button>
      <button class="ctrl-btn" @click="ctrl.goToEnd" title="End">&#9197;</button>

      <div class="ctrl-speeds">
        <button
          v-for="s in speeds" :key="s"
          :class="['speed-btn', { on: ctrl.speed.value === s }]"
          @click="ctrl.speed.value = s"
        >{{ s }}x</button>
      </div>

      <!-- Overlay toggles -->
      <div class="ctrl-toggles">
        <button
          :class="['toggle-btn', { on: ctrl.showHeatmap.value }]"
          :disabled="!ctrl.hasHeatmaps.value"
          @click="ctrl.showHeatmap.value = !ctrl.showHeatmap.value"
          title="热力图叠加"
        >热力</button>
        <button
          :class="['toggle-btn', { on: ctrl.showBoxes.value }]"
          @click="ctrl.showBoxes.value = !ctrl.showBoxes.value"
          title="YOLO 检测框"
        >框选</button>
      </div>

      <span class="ctrl-frame-info">
        FRAME <b>{{ ctrl.currentIndex.value + 1 }}</b> / {{ ctrl.metadata.value.frame_count }} · {{ ctrl.currentTimestamp.value }}
      </span>
    </div>

    <!-- Timeline scrubber -->
    <div class="replay-timeline">
      <div class="tl-bar">
        <div class="tl-progress" :style="{ width: progressPct + '%' }"></div>
        <div class="tl-head" :style="{ left: progressPct + '%' }"></div>
      </div>
      <input
        type="range"
        :min="0"
        :max="(ctrl.metadata.value.frame_count || 1) - 1"
        :value="ctrl.currentIndex.value"
        @input="ctrl.seekTo(Number(($event.target as HTMLInputElement).value))"
        class="tl-input"
      />
      <!-- Recording-in-progress indicator -->
      <div v-if="ctrl.metadata.value.status === 'recording'" class="replay-recording-bar">
        <div style="flex: 1; display: flex; align-items: center; gap: 4px">
          <div :style="{ width: triggerProgressPct + '%' }" />
          <div class="rec-dot" />
          <div style="flex: 1; height: 2px; border-top: 2px dashed var(--line-2)" />
        </div>
        <span class="rec-text">&#9210; 录制中 · 剩余 {{ remainingRecordingSeconds }}s</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.replay-controls-wrapper {
  display: flex;
  flex-direction: column;
}
.replay-controls {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-top: 1px solid var(--line-2);
  background: var(--bg);
}
.ctrl-btn {
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border: 1px solid var(--line-2);
  background: transparent;
  color: var(--ink-2);
  cursor: pointer;
  font-size: 10px;
  transition: all .12s;
}
.ctrl-btn:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}
.ctrl-play {
  width: 34px;
  height: 34px;
  background: #3b82f6;
  border-color: #3b82f6;
  color: #fff;
  border-radius: 50%;
  font-size: 14px;
}
.ctrl-play:hover {
  background: #2563eb;
  border-color: #2563eb;
  color: #fff;
}
.ctrl-speeds {
  display: flex;
  margin-left: 8px;
}
.speed-btn {
  padding: 4px 8px;
  border: 1px solid var(--line-2);
  border-right: none;
  background: transparent;
  color: var(--ink-4);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 10px;
  cursor: pointer;
  transition: all .12s;
}
.speed-btn:last-child {
  border-right: 1px solid var(--line-2);
}
.speed-btn:hover {
  color: var(--ink-2);
}
.speed-btn.on {
  background: var(--ink-2);
  color: var(--bg);
  border-color: var(--ink-2);
}
.ctrl-toggles {
  display: flex;
  gap: 4px;
  margin-left: 8px;
}
.toggle-btn {
  padding: 4px 10px;
  border: 1px solid var(--line-2);
  background: transparent;
  color: var(--ink-4);
  font-size: 11px;
  cursor: pointer;
  transition: all .12s;
}
.toggle-btn:hover {
  border-color: #3b82f6;
  color: var(--ink-2);
}
.toggle-btn.on {
  border-color: #3b82f6;
  color: #3b82f6;
  background: rgba(59,130,246,.1);
}
.toggle-btn:disabled {
  opacity: .4;
  cursor: not-allowed;
}
.ctrl-frame-info {
  margin-left: auto;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 10px;
  color: var(--ink-4);
  letter-spacing: .08em;
}
.ctrl-frame-info b {
  color: #f59e0b;
}

/* ── Timeline ── */
.replay-timeline {
  padding: 12px 16px 6px;
  position: relative;
  background: var(--bg);
}
.tl-bar {
  height: 3px;
  background: var(--line-2);
  position: relative;
  border-radius: 2px;
  pointer-events: none;
}
.tl-progress {
  position: absolute;
  left: 0; top: 0; bottom: 0;
  background: #3b82f6;
  border-radius: 2px;
  transition: width .05s linear;
}
.tl-head {
  position: absolute;
  top: -4px;
  width: 2px;
  height: 11px;
  background: #f59e0b;
  transform: translateX(-1px);
  transition: left .05s linear;
}
.tl-input {
  position: absolute;
  left: 16px; right: 16px;
  top: 6px;
  width: calc(100% - 32px);
  height: 20px;
  opacity: 0;
  cursor: pointer;
  margin: 0;
}
.replay-recording-bar {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  margin-top: 4px;
}
.rec-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #f59e0b;
  flex-shrink: 0;
  animation: hud-blink 1.5s infinite;
}
.rec-text {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 11px;
  margin-left: 8px;
}
</style>
