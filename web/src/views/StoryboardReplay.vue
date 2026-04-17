<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { CaretRightOutlined, PauseOutlined } from '@ant-design/icons-vue'
import StoryboardPlayer from '../components/replay/StoryboardPlayer.vue'
import { useStoryboardController } from '../composables/useStoryboardController'
import { formatPlaybackTime } from '../utils/time'

defineOptions({ name: 'StoryboardReplayPage' })

const route = useRoute()
const router = useRouter()
const alertId = route.params.alertId as string

const ctrl = useStoryboardController(alertId)
const speeds = [0.25, 0.5, 1, 2, 4]

const gridClass = computed(() => {
  const n = ctrl.cameras.value.length
  if (n <= 1) return 'grid-1'
  if (n === 2) return 'grid-2'
  return 'grid-4' // 3 or 4 cameras laid out in a 2x2 (spec: 3-4 cameras = 2x2)
})

const progressPct = computed(() => {
  const dur = ctrl.timelineDuration.value
  if (!dur) return 0
  return ((ctrl.masterTime.value - ctrl.timelineStart.value) / dur) * 100
})

const elapsedLabel = computed(() =>
  formatPlaybackTime(ctrl.masterTime.value - ctrl.timelineStart.value),
)
const totalLabel = computed(() => formatPlaybackTime(ctrl.timelineDuration.value))

function onScrubInput(e: Event): void {
  const target = e.target as HTMLInputElement
  const pct = Number(target.value) / 100
  const absolute =
    ctrl.timelineStart.value +
    pct * (ctrl.timelineEnd.value - ctrl.timelineStart.value)
  ctrl.seek(absolute)
}

function onDurationReport(id: string, seconds: number): void {
  ctrl.reportDuration(id, seconds)
}

onMounted(() => {
  ctrl.load()
  window.addEventListener('keydown', ctrl.handleKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', ctrl.handleKeydown)
})
</script>

<template>
  <main class="storyboard-page">
    <div class="topbar glass">
      <button class="ibtn" @click="router.back()" title="返回">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
        </svg>
      </button>
      <h1>多机位回放</h1>
      <span class="sep"></span>
      <span class="crumb mono">{{ alertId }}</span>
      <span v-if="ctrl.cameras.value.length" class="cam-count">
        {{ ctrl.cameras.value.length }} 路同步
      </span>

      <div class="topbar-right">
        <button
          class="ibtn ghost"
          @click="router.push(`/replay/${alertId}`)"
          title="返回单机位"
        >单机位</button>
      </div>
    </div>

    <div v-if="ctrl.loading.value" class="state-msg">加载多机位回放…</div>
    <div v-else-if="ctrl.error.value" class="state-msg error">{{ ctrl.error.value }}</div>
    <div v-else-if="ctrl.cameras.value.length === 0" class="state-msg">
      未找到关联摄像头
    </div>

    <template v-else>
      <div class="grid" :class="gridClass">
        <StoryboardPlayer
          v-for="(cam, idx) in ctrl.cameras.value"
          :key="cam.alert_id"
          :camera="cam"
          :master-time="ctrl.masterTime.value"
          :playing="ctrl.playing.value"
          :speed="ctrl.speed.value"
          :is-primary="idx === 0"
          @duration="onDurationReport"
        />
      </div>

      <div class="controls">
        <button class="ctrl-btn ctrl-play" @click="ctrl.togglePlay" title="空格">
          <PauseOutlined v-if="ctrl.playing.value" />
          <CaretRightOutlined v-else />
        </button>
        <button class="ctrl-btn" @click="ctrl.seek(ctrl.masterTime.value - 1)" title="← 后退 1s">&#9664;&#9664;</button>
        <button class="ctrl-btn" @click="ctrl.seek(ctrl.masterTime.value + 1)" title="→ 前进 1s">&#9654;&#9654;</button>

        <div class="speed-group">
          <button
            v-for="s in speeds"
            :key="s"
            :class="['speed-btn', { on: ctrl.speed.value === s }]"
            @click="ctrl.setSpeed(s)"
          >{{ s }}x</button>
        </div>

        <span class="time-label mono">
          <b>{{ elapsedLabel }}</b> / {{ totalLabel }}
          <span class="master-t">· T{{ ctrl.masterTime.value >= 0 ? '+' : '' }}{{ ctrl.masterTime.value.toFixed(2) }}s</span>
        </span>

        <div class="timeline">
          <div class="tl-bar">
            <div class="tl-progress" :style="{ width: progressPct + '%' }" />
            <!-- Marker for the primary alert's trigger (master t=0) -->
            <div
              class="tl-trigger"
              :style="{
                left: ctrl.timelineDuration.value > 0
                  ? ((0 - ctrl.timelineStart.value) / ctrl.timelineDuration.value) * 100 + '%'
                  : '50%'
              }"
              title="主告警触发点"
            />
          </div>
          <input
            type="range"
            min="0"
            max="100"
            step="0.1"
            :value="progressPct"
            @input="onScrubInput"
            class="tl-input"
          />
        </div>
      </div>
    </template>
  </main>
</template>

<style scoped>
.storyboard-page {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  gap: 12px;
}
.topbar {
  height: 60px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 0 22px;
}
.topbar h1 {
  font-size: 18px;
  font-weight: 700;
  color: var(--ink, #f3f3f3);
  letter-spacing: -0.028em;
  margin: 0;
}
.sep {
  width: 1px;
  height: 14px;
  background: var(--line-2, #333);
}
.crumb {
  font-size: 12px;
  color: var(--ink-5, #888);
  font-weight: 500;
}
.cam-count {
  margin-left: 6px;
  padding: 2px 8px;
  font-size: 11px;
  color: #f59e0b;
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.35);
  border-radius: 3px;
}
.topbar-right {
  margin-left: auto;
}
.mono {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}

.ibtn {
  width: 34px;
  height: 34px;
  border-radius: var(--r-sm, 4px);
  display: grid;
  place-items: center;
  background: rgba(255, 255, 255, 0.06);
  border: 0.5px solid var(--line-2, #333);
  color: var(--ink-3, #ccc);
  cursor: pointer;
  transition: all 0.18s;
  font-size: 11px;
  padding: 0 10px;
}
.ibtn.ghost {
  width: auto;
  font-weight: 600;
}
.ibtn:hover {
  background: #fff;
  color: var(--ink, #111);
}
.ibtn svg {
  width: 15px;
  height: 15px;
  stroke-width: 1.9;
}

.state-msg {
  flex: 1;
  display: grid;
  place-items: center;
  color: var(--ink-5, #888);
  font-size: 14px;
}
.state-msg.error {
  color: #f87171;
}

.grid {
  flex: 1;
  min-height: 0;
  display: grid;
  gap: 6px;
  padding: 0 12px;
}
.grid-1 {
  grid-template-columns: 1fr;
  grid-template-rows: 1fr;
}
.grid-2 {
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr;
}
.grid-4 {
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
}

.controls {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: var(--bg-1, #121212);
  border-top: 1px solid var(--line-2, #2a2a2a);
}
.ctrl-btn {
  width: 32px;
  height: 32px;
  display: grid;
  place-items: center;
  border: 1px solid var(--line-2, #333);
  background: transparent;
  color: var(--ink-2, #ccc);
  cursor: pointer;
  font-size: 11px;
}
.ctrl-btn:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}
.ctrl-play {
  width: 38px;
  height: 38px;
  background: #3b82f6;
  border-color: #3b82f6;
  color: #fff;
  border-radius: 50%;
  font-size: 15px;
}
.ctrl-play:hover {
  background: #2563eb;
  border-color: #2563eb;
  color: #fff;
}
.speed-group {
  display: flex;
  margin-left: 6px;
}
.speed-btn {
  padding: 4px 8px;
  border: 1px solid var(--line-2, #333);
  border-right: none;
  background: transparent;
  color: var(--ink-4, #aaa);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 10px;
  cursor: pointer;
}
.speed-btn:last-child {
  border-right: 1px solid var(--line-2, #333);
}
.speed-btn.on {
  background: var(--ink-2, #ddd);
  color: var(--bg, #000);
  border-color: var(--ink-2, #ddd);
}
.time-label {
  font-size: 11px;
  color: var(--ink-4, #aaa);
  min-width: 150px;
  text-align: center;
}
.time-label b {
  color: #f59e0b;
}
.master-t {
  margin-left: 4px;
  color: var(--ink-5, #777);
}
.timeline {
  flex: 1;
  position: relative;
  height: 22px;
  display: flex;
  align-items: center;
}
.tl-bar {
  position: absolute;
  left: 0;
  right: 0;
  height: 4px;
  background: var(--line-2, #333);
  border-radius: 2px;
}
.tl-progress {
  height: 100%;
  background: #3b82f6;
  border-radius: 2px;
}
.tl-trigger {
  position: absolute;
  top: -4px;
  width: 2px;
  height: 12px;
  background: #f59e0b;
  transform: translateX(-1px);
}
.tl-input {
  position: absolute;
  left: 0;
  right: 0;
  width: 100%;
  height: 22px;
  opacity: 0;
  cursor: pointer;
  margin: 0;
}
</style>
