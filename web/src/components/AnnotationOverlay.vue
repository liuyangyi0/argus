<template>
  <div class="annotation-root" :style="{ width: width + 'px', height: height + 'px' }">
    <canvas
      ref="canvasRef"
      :width="width"
      :height="height"
      class="annotation-canvas"
      @mousedown="onMouseDown"
      @mousemove="onMouseMove"
      @mouseup="onMouseUp"
    />

    <!-- Toolbar -->
    <div class="annotation-toolbar">
      <a-select v-model:value="currentLabel" size="small" style="width: 100px">
        <a-select-option value="anomaly">异物</a-select-option>
        <a-select-option value="false_positive">误报</a-select-option>
        <a-select-option value="suspicious">可疑</a-select-option>
      </a-select>
      <a-button size="small" @click="clearAll">清除</a-button>
      <a-button size="small" type="primary" @click="submit" :loading="submitting">
        提交 ({{ annotations.length }})
      </a-button>
    </div>

    <!-- Annotation list -->
    <div v-if="annotations.length" class="annotation-list">
      <div v-for="(a, idx) in annotations" :key="idx" class="annotation-item">
        <span class="dot" :style="{ background: labelColor(a.label) }" />
        <span>{{ labelText(a.label) }} {{ a.width }}x{{ a.height }}</span>
        <a-button size="small" type="text" danger @click="removeAnnotation(idx)">x</a-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { saveAnnotations } from '../api/alerts'
import { message } from 'ant-design-vue'

interface Annotation {
  x: number; y: number; width: number; height: number; label: string
}

const props = defineProps<{
  width: number
  height: number
  alertId: string
  cameraId: string
}>()

const canvasRef = ref<HTMLCanvasElement>()
const annotations = ref<Annotation[]>([])
const currentLabel = ref('anomaly')
const submitting = ref(false)

// Drawing state
const drawing = ref(false)
const startX = ref(0)
const startY = ref(0)
const currentX = ref(0)
const currentY = ref(0)

const COLORS: Record<string, string> = {
  anomaly: '#ef4444',
  false_positive: '#22c55e',
  suspicious: '#f59e0b',
}

function labelColor(label: string) { return COLORS[label] || '#6b7280' }
function labelText(label: string) {
  return { anomaly: '异物', false_positive: '误报', suspicious: '可疑' }[label] || label
}

function onMouseDown(e: MouseEvent) {
  const rect = canvasRef.value!.getBoundingClientRect()
  startX.value = e.clientX - rect.left
  startY.value = e.clientY - rect.top
  drawing.value = true
}

function onMouseMove(e: MouseEvent) {
  if (!drawing.value) return
  const rect = canvasRef.value!.getBoundingClientRect()
  currentX.value = e.clientX - rect.left
  currentY.value = e.clientY - rect.top
  redraw()
}

function onMouseUp() {
  if (!drawing.value) return
  drawing.value = false
  const x = Math.min(startX.value, currentX.value)
  const y = Math.min(startY.value, currentY.value)
  const w = Math.abs(currentX.value - startX.value)
  const h = Math.abs(currentY.value - startY.value)
  if (w > 5 && h > 5) {
    annotations.value.push({
      x: Math.round(x), y: Math.round(y),
      width: Math.round(w), height: Math.round(h),
      label: currentLabel.value,
    })
  }
  redraw()
}

function redraw() {
  const ctx = canvasRef.value?.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, props.width, props.height)

  // Draw existing annotations
  for (const a of annotations.value) {
    ctx.strokeStyle = labelColor(a.label)
    ctx.lineWidth = 2
    ctx.strokeRect(a.x, a.y, a.width, a.height)
    ctx.fillStyle = labelColor(a.label) + '33'
    ctx.fillRect(a.x, a.y, a.width, a.height)
    ctx.fillStyle = labelColor(a.label)
    ctx.font = '11px sans-serif'
    ctx.fillText(labelText(a.label), a.x + 3, a.y + 13)
  }

  // Draw current rectangle being drawn
  if (drawing.value) {
    const x = Math.min(startX.value, currentX.value)
    const y = Math.min(startY.value, currentY.value)
    const w = Math.abs(currentX.value - startX.value)
    const h = Math.abs(currentY.value - startY.value)
    ctx.strokeStyle = labelColor(currentLabel.value)
    ctx.lineWidth = 2
    ctx.setLineDash([5, 3])
    ctx.strokeRect(x, y, w, h)
    ctx.setLineDash([])
  }
}

function removeAnnotation(idx: number) {
  annotations.value.splice(idx, 1)
  redraw()
}

function clearAll() {
  annotations.value = []
  redraw()
}

async function submit() {
  if (!annotations.value.length) return
  submitting.value = true
  try {
    await saveAnnotations(props.alertId, annotations.value)
    message.success(`已提交 ${annotations.value.length} 条标注`)
    annotations.value = []
    redraw()
  } catch {
    message.error('提交失败')
  } finally {
    submitting.value = false
  }
}

watch(() => annotations.value.length, () => nextTick(redraw))
</script>

<style scoped>
.annotation-root {
  position: relative;
}
.annotation-canvas {
  position: absolute;
  top: 0;
  left: 0;
  cursor: crosshair;
  z-index: 10;
}
.annotation-toolbar {
  position: absolute;
  bottom: 8px;
  left: 8px;
  display: flex;
  gap: 6px;
  z-index: 11;
  background: rgba(26, 26, 46, 0.8);
  padding: 4px 8px;
  border-radius: 6px;
  border: 1px solid rgba(45, 45, 74, 0.6);
}
.annotation-list {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 11;
  background: rgba(26, 26, 46, 0.8);
  padding: 6px;
  border-radius: 6px;
  border: 1px solid rgba(45, 45, 74, 0.6);
  max-height: 200px;
  overflow-y: auto;
}
.annotation-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: #e2e8f0;
  margin-bottom: 2px;
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

:global(.light-theme) .annotation-toolbar {
  background: #ffffffcc;
  border-color: #e5e7eb;
}
:global(.light-theme) .annotation-list {
  background: #ffffffcc;
  border-color: #e5e7eb;
}
:global(.light-theme) .annotation-item { color: #111827; }

@media (max-width: 768px) {
  .annotation-toolbar { flex-wrap: wrap; gap: 4px; }
}
</style>
