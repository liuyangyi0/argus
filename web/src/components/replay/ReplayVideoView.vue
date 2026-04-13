<script setup lang="ts">
import { inject } from 'vue'
import type { useReplayController } from '../../composables/useReplayController'

const ctrl = inject<ReturnType<typeof useReplayController>>('replayCtrl')!

import VideoOverlayCanvas from './VideoOverlayCanvas.vue'

function onTimeUpdate() {
  if (!ctrl.videoEl.value) return
  const idx = Math.min(
    Math.floor(ctrl.videoEl.value.currentTime * ctrl.fps.value),
    (ctrl.metadata.value?.frame_count || 1) - 1,
  )
  if (idx !== ctrl.currentIndex.value) ctrl.currentIndex.value = idx
}

function onVideoPlay() {
  ctrl.playing.value = true
  ctrl.videoError.value = ''
}

function onVideoPause() {
  ctrl.playing.value = false
  ctrl.heatmapIndex.value = ctrl.currentIndex.value
}

function onVideoEnded() {
  ctrl.playing.value = false
  ctrl.heatmapIndex.value = ctrl.currentIndex.value
}

function onVideoCanPlay() {
  ctrl.videoError.value = ''
  if (ctrl.pendingSeekIndex.value !== null && ctrl.videoEl.value) {
    const seekTime = ctrl.pendingSeekIndex.value / ctrl.fps.value
    ctrl.videoEl.value.currentTime = seekTime
    ctrl.pendingSeekIndex.value = null
  }
}

function onVideoError() {
  const el = ctrl.videoEl.value
  if (!el?.error) return
  ctrl.videoError.value = `视频错误 (code=${el.error.code})`
}
</script>

<template>
  <div class="replay-player">
    <video
      ref="videoRef"
      :src="ctrl.videoUrl.value"
      preload="auto"
      class="replay-video"
      @timeupdate="onTimeUpdate"
      @play="onVideoPlay"
      @pause="onVideoPause"
      @ended="onVideoEnded"
      @canplay="onVideoCanPlay"
      @error="onVideoError"
    />
    
    <!-- 热力图 -->
    <img
      v-if="ctrl.showHeatmap.value && ctrl.hasHeatmaps.value"
      :src="ctrl.heatmapUrl.value"
      class="replay-heatmap"
    />
    
    <!-- YOLO 检测框 (Canvas 60FPS) -->
    <VideoOverlayCanvas
      :video-el="ctrl.videoEl.value"
      :fps="ctrl.fps.value"
      :frame-count="ctrl.metadata.value.frame_count || 1"
      :video-width="ctrl.metadata.value.width || 1920"
      :video-height="ctrl.metadata.value.height || 1080"
      :all-boxes="ctrl.signals.value?.yolo_boxes"
      :show-boxes="ctrl.showBoxes.value"
      :current-index="ctrl.currentIndex.value"
      @update:index="(idx) => ctrl.currentIndex.value = idx"
    />
    
    <!-- HUD: top -->
    <div class="replay-hud replay-hud-top">
      <span v-if="ctrl.metadata.value.status === 'recording'" class="hud-rec">&#9679; REC</span>
      <span v-else class="hud-cam">{{ ctrl.metadata.value.camera_id || '' }}</span>
      <span>{{ ctrl.metadata.value.width || 1920 }}&#215;{{ ctrl.metadata.value.height || 1080 }} // {{ ctrl.fps.value }} FPS</span>
    </div>
    
    <!-- HUD: bottom -->
    <div class="replay-hud replay-hud-bottom">
      <span>{{ ctrl.currentTimestamp.value }}</span>
      <span>FRAME {{ ctrl.currentIndex.value + 1 }} / {{ ctrl.metadata.value.frame_count }}</span>
    </div>
    
    <!-- 错误信息 -->
    <div v-if="ctrl.videoError.value" class="replay-error">{{ ctrl.videoError.value }}</div>
  </div>
</template>

<style scoped>
.replay-player {
  position: relative;
  background: #000;
  aspect-ratio: 16/9;
  overflow: hidden;
}
.replay-video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.replay-heatmap {
  position: absolute;
  top: 0; left: 0; width: 100%; height: 100%;
  object-fit: contain;
  opacity: 0.4;
  pointer-events: none;
  mix-blend-mode: screen;
}
.replay-hud {
  position: absolute;
  left: 12px; right: 12px;
  display: flex;
  justify-content: space-between;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 10px;
  letter-spacing: .12em;
  pointer-events: none;
  text-shadow: 0 1px 3px rgba(0,0,0,.7);
}
.replay-hud-top {
  top: 10px;
  color: rgba(255,255,255,.5);
}
.replay-hud-bottom {
  bottom: 10px;
  color: rgba(255,255,255,.6);
}
.hud-rec {
  color: #f59e0b;
  animation: hud-blink 1.5s infinite;
}
.hud-cam {
  color: rgba(255,255,255,.5);
}
@keyframes hud-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: .4; }
}
.replay-error {
  position: absolute;
  bottom: 8px; left: 8px; right: 8px;
  background: rgba(239,68,68,0.9);
  color: #fff;
  padding: 6px 10px;
  border-radius: 4px;
  font-size: 12px;
}
</style>
