<template>
  <div
    class="compare-container"
    ref="containerRef"
    :style="{ width: width + 'px', height: height + 'px' }"
    @mousemove="onMove"
    @touchmove.prevent="onTouch"
    @mousedown="startDrag"
    @mouseup="stopDrag"
    @mouseleave="stopDrag"
  >
    <!-- Left image (snapshot) -->
    <img :src="leftSrc" class="compare-img" :alt="leftLabel" draggable="false" />

    <!-- Right image (heatmap), clipped by slider -->
    <div class="compare-overlay" :style="{ clipPath: `inset(0 0 0 ${sliderPos}%)` }">
      <img :src="rightSrc" class="compare-img" :alt="rightLabel" draggable="false" />
    </div>

    <!-- Slider line -->
    <div class="slider-line" :style="{ left: `${sliderPos}%` }">
      <div class="slider-handle">
        <span class="arrow">&#9664;</span>
        <span class="arrow">&#9654;</span>
      </div>
    </div>

    <!-- Labels -->
    <div class="label left-label">{{ leftLabel }}</div>
    <div class="label right-label">{{ rightLabel }}</div>

    <!-- Zoom lens -->
    <div
      v-if="zooming"
      class="zoom-lens"
      :style="zoomLensStyle"
    >
      <div
        class="zoom-content"
        :style="zoomContentStyle"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

const props = withDefaults(defineProps<{
  leftSrc: string
  rightSrc: string
  leftLabel?: string
  rightLabel?: string
  width?: number
  height?: number
  zoomFactor?: number
}>(), {
  leftLabel: '原始帧',
  rightLabel: '热力图',
  width: 640,
  height: 480,
  zoomFactor: 3,
})

const containerRef = ref<HTMLElement>()
const sliderPos = ref(50)
const dragging = ref(false)
const zooming = ref(false)
const mouseX = ref(0)
const mouseY = ref(0)

const lensSize = 120

function startDrag() { dragging.value = true }
function stopDrag() { dragging.value = false }

function onMove(e: MouseEvent) {
  const rect = containerRef.value?.getBoundingClientRect()
  if (!rect) return
  const x = e.clientX - rect.left
  const y = e.clientY - rect.top
  mouseX.value = x
  mouseY.value = y

  if (dragging.value) {
    sliderPos.value = Math.max(0, Math.min(100, (x / rect.width) * 100))
  }

  // Alt+hover for zoom
  zooming.value = e.altKey
}

function onTouch(e: TouchEvent) {
  const rect = containerRef.value?.getBoundingClientRect()
  if (!rect || !e.touches[0]) return
  const x = e.touches[0].clientX - rect.left
  sliderPos.value = Math.max(0, Math.min(100, (x / rect.width) * 100))
}

const zoomLensStyle = computed(() => ({
  left: `${mouseX.value - lensSize / 2}px`,
  top: `${mouseY.value - lensSize / 2}px`,
  width: `${lensSize}px`,
  height: `${lensSize}px`,
}))

const zoomContentStyle = computed(() => {
  const f = props.zoomFactor
  const bgX = -(mouseX.value * f - lensSize / 2)
  const bgY = -(mouseY.value * f - lensSize / 2)
  return {
    backgroundImage: `url(${props.leftSrc})`,
    backgroundSize: `${props.width * f}px ${props.height * f}px`,
    backgroundPosition: `${bgX}px ${bgY}px`,
    width: `${lensSize}px`,
    height: `${lensSize}px`,
  }
})
</script>

<style scoped>
.compare-container {
  position: relative;
  overflow: hidden;
  border-radius: 6px;
  cursor: col-resize;
  user-select: none;
  background: var(--glass);
}
.compare-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  pointer-events: none;
}
.compare-overlay {
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
}
.slider-line {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 3px;
  background: #ffffff;
  transform: translateX(-50%);
  z-index: 5;
}
.slider-handle {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: #1a1a2eee;
  border: 2px solid #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 2px;
}
.arrow { color: #fff; font-size: 10px; }
.label {
  position: absolute;
  top: 8px;
  padding: 2px 8px;
  background: #1a1a2ecc;
  color: #e2e8f0;
  font-size: 11px;
  border-radius: 4px;
  z-index: 4;
}
.left-label { left: 8px; }
.right-label { right: 8px; }
.zoom-lens {
  position: absolute;
  border: 2px solid #3b82f6;
  border-radius: 50%;
  overflow: hidden;
  pointer-events: none;
  z-index: 10;
  box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
}
.zoom-content {
  width: 100%;
  height: 100%;
  background-repeat: no-repeat;
}

:global(.light-theme) .label { background: #ffffffcc; color: #111827; }
:global(.light-theme) .slider-handle { background: #ffffffee; }
:global(.light-theme) .compare-container { background: #f1f5f9; }

@media (max-width: 768px) {
  .compare-container { cursor: grab; touch-action: pan-y; }
  .slider-handle { width: 40px; height: 40px; }
  .label { font-size: 10px; padding: 1px 6px; }
}
@media (max-width: 480px) {
  .slider-handle { width: 48px; height: 48px; }
}
</style>
