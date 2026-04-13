<template>
  <div class="zone-editor">
    <div class="editor-canvas-wrapper" :style="{ aspectRatio: `${width} / ${height}` }">
      <!-- Background image -->
      <img v-if="imageSrc" :src="imageSrc" class="editor-bg" draggable="false" />
      <div v-else class="editor-placeholder">摄像头快照加载中...</div>

      <!-- SVG overlay for zones -->
      <svg class="editor-svg" :viewBox="`0 0 ${width} ${height}`">
        <!-- Existing zones -->
        <g v-for="(zone, zIdx) in zones" :key="zone.zone_id">
          <polygon
            :points="polygonPoints(zone.vertices)"
            :fill="zoneFill(zone)"
            :stroke="zoneStroke(zone)"
            stroke-width="2"
            :opacity="zone === selectedZone ? 0.5 : 0.3"
            style="cursor: pointer"
            @click="selectZone(zone)"
          />
          <!-- Draggable vertices -->
          <circle
            v-for="(v, vIdx) in zone.vertices"
            :key="vIdx"
            :cx="v.x"
            :cy="v.y"
            r="6"
            :fill="zoneStroke(zone)"
            stroke="#fff"
            stroke-width="1.5"
            style="cursor: grab"
            @mousedown.stop="startDragVertex(zIdx, vIdx, $event)"
          />
          <!-- Label -->
          <text
            :x="centroid(zone.vertices).x"
            :y="centroid(zone.vertices).y"
            fill="#fff"
            font-size="12"
            text-anchor="middle"
            dominant-baseline="middle"
            style="pointer-events: none"
          >
            {{ zone.zone_id }}
          </text>
        </g>

        <!-- Drawing new polygon -->
        <g v-if="drawingVertices.length > 0">
          <polyline
            :points="polygonPoints(drawingVertices)"
            fill="none"
            stroke="#3b82f6"
            stroke-width="2"
            stroke-dasharray="5,3"
          />
          <circle
            v-for="(v, i) in drawingVertices"
            :key="i"
            :cx="v.x"
            :cy="v.y"
            r="5"
            fill="#3b82f6"
            stroke="#fff"
            stroke-width="1"
          />
        </g>
      </svg>

      <!-- Click handler for adding vertices -->
      <div
        v-if="isDrawing"
        class="editor-click-area"
        @click="addVertex($event)"
        @dblclick="finishDrawing"
        @contextmenu.prevent="finishDrawing"
      />
    </div>

    <!-- Controls -->
    <div class="editor-controls">
      <div style="display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap;">
        <a-space>
          <a-button
            :type="isDrawing ? 'primary' : 'default'"
            size="small"
            @click="toggleDrawing"
          >
            {{ isDrawing ? '完成绘制 (双击)' : '添加区域' }}
          </a-button>
          <a-select v-model:value="newZoneType" size="small" style="width: 90px">
            <a-select-option value="include">包含区</a-select-option>
            <a-select-option value="exclude">排除区</a-select-option>
          </a-select>
          <a-button
            v-if="selectedZone"
            size="small"
            danger
            @click="deleteSelected"
          >
            删除选中
          </a-button>
        </a-space>

        <!-- Zone list -->
        <div class="zone-list" style="display:flex; flex-wrap:wrap; gap:8px;">
          <div
            v-for="zone in zones"
            :key="zone.zone_id"
            class="zone-item"
            :class="{ active: zone === selectedZone }"
            @click="selectZone(zone)"
          >
            <span class="zone-dot" :style="{ background: zoneStroke(zone) }" />
            <span>{{ zone.zone_id }}</span>
            <a-tag :color="zone.zone_type === 'include' ? 'blue' : 'orange'" size="small" style="margin-left: 6px">
              {{ zone.zone_type === 'include' ? '包含' : '排除' }}
            </a-tag>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

interface Vertex { x: number; y: number }
interface Zone {
  zone_id: string
  zone_type: 'include' | 'exclude'
  vertices: Vertex[]
  priority?: string
}

const props = withDefaults(defineProps<{
  modelValue: Zone[]
  imageSrc?: string
  width?: number
  height?: number
}>(), {
  width: 640,
  height: 480,
})

const emit = defineEmits<{
  (e: 'update:modelValue', zones: Zone[]): void
}>()

const zones = ref<Zone[]>([...props.modelValue])
const selectedZone = ref<Zone | null>(null)
const isDrawing = ref(false)
const drawingVertices = ref<Vertex[]>([])
const newZoneType = ref<'include' | 'exclude'>('include')

// Drag state
let dragZoneIdx = -1
let dragVertexIdx = -1
let dragStartX = 0
let dragStartY = 0

function polygonPoints(vertices: Vertex[]) {
  return vertices.map(v => `${v.x},${v.y}`).join(' ')
}

function centroid(vertices: Vertex[]) {
  const n = vertices.length || 1
  return {
    x: vertices.reduce((s, v) => s + v.x, 0) / n,
    y: vertices.reduce((s, v) => s + v.y, 0) / n,
  }
}

function zoneFill(zone: Zone) {
  return zone.zone_type === 'include' ? '#3b82f6' : '#f59e0b'
}
function zoneStroke(zone: Zone) {
  return zone.zone_type === 'include' ? '#60a5fa' : '#fbbf24'
}

function selectZone(zone: Zone) {
  selectedZone.value = zone === selectedZone.value ? null : zone
}

function toggleDrawing() {
  if (isDrawing.value) {
    finishDrawing()
  } else {
    isDrawing.value = true
    drawingVertices.value = []
    selectedZone.value = null
  }
}

function addVertex(e: MouseEvent) {
  const el = e.target as HTMLElement
  const rect = el.getBoundingClientRect()
  const scaleX = props.width / rect.width
  const scaleY = props.height / rect.height
  drawingVertices.value.push({
    x: Math.round((e.clientX - rect.left) * scaleX),
    y: Math.round((e.clientY - rect.top) * scaleY),
  })
}

function finishDrawing() {
  if (drawingVertices.value.length >= 3) {
    const newZone: Zone = {
      zone_id: `zone_${Date.now().toString(36)}`,
      zone_type: newZoneType.value,
      vertices: [...drawingVertices.value],
    }
    zones.value.push(newZone)
    emit('update:modelValue', zones.value)
  }
  drawingVertices.value = []
  isDrawing.value = false
}

function deleteSelected() {
  if (!selectedZone.value) return
  zones.value = zones.value.filter(z => z !== selectedZone.value)
  selectedZone.value = null
  emit('update:modelValue', zones.value)
}

let dragScaleX = 1
let dragScaleY = 1

function startDragVertex(zIdx: number, vIdx: number, e: MouseEvent) {
  const svg = (e.target as HTMLElement).closest('svg')
  if (svg) {
    const rect = svg.getBoundingClientRect()
    dragScaleX = props.width / rect.width
    dragScaleY = props.height / rect.height
  } else {
    dragScaleX = 1
    dragScaleY = 1
  }

  dragZoneIdx = zIdx
  dragVertexIdx = vIdx
  dragStartX = e.clientX
  dragStartY = e.clientY
  document.addEventListener('mousemove', onDragVertex)
  document.addEventListener('mouseup', stopDragVertex)
}

function onDragVertex(e: MouseEvent) {
  if (dragZoneIdx < 0) return
  const dx = (e.clientX - dragStartX) * dragScaleX
  const dy = (e.clientY - dragStartY) * dragScaleY
  dragStartX = e.clientX
  dragStartY = e.clientY
  const v = zones.value[dragZoneIdx].vertices[dragVertexIdx]
  v.x = Math.max(0, Math.min(props.width, v.x + dx))
  v.y = Math.max(0, Math.min(props.height, v.y + dy))
}

function stopDragVertex() {
  dragZoneIdx = -1
  dragVertexIdx = -1
  document.removeEventListener('mousemove', onDragVertex)
  document.removeEventListener('mouseup', stopDragVertex)
  emit('update:modelValue', zones.value)
}
</script>

<style scoped>
.zone-editor {
  display: flex;
  flex-direction: column;
  gap: 16px;
  flex: 1;
  min-height: 0;
}
.editor-canvas-wrapper {
  position: relative;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,.15);
  flex: 1;
  min-height: 0;
  min-width: 0;
  max-width: 100%;
  max-height: 100%;
  margin: 0 auto;
  display: flex;
}
.editor-bg {
  width: 100%;
  height: 100%;
  object-fit: cover;
  pointer-events: none;
}
.editor-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  font-size: 14px;
}
.editor-svg {
  position: absolute;
  top: 0; left: 0;
  width: 100%;
  height: 100%;
}
.editor-click-area {
  position: absolute;
  top: 0; left: 0;
  width: 100%;
  height: 100%;
  cursor: crosshair;
  z-index: 5;
}
.editor-controls {
  flex-shrink: 0;
  width: 100%;
}
.zone-list {
  margin-top: 0;
}
.zone-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12.5px;
  color: var(--ink-2);
  margin-bottom: 0px;
  background: var(--glass);
  border: 1px solid var(--line-2);
  transition: all .2s;
}
.zone-item:hover { background: #fff; border-color: rgba(37,99,235,.2); }
.zone-item.active { background: #fff; border-color: #3b82f6; box-shadow: 0 2px 8px rgba(37,99,235,.15); font-weight: 600; }
.zone-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

:global(.light-theme) .editor-canvas-wrapper { background: #f1f5f9; }
:global(.light-theme) .zone-item { color: #111827; }
:global(.light-theme) .zone-item:hover { background: #f1f5f9; }
:global(.light-theme) .zone-item.active { background: #e2e8f0; }

@media (max-width: 768px) {
  .zone-editor { flex-direction: column; }
  .editor-canvas-wrapper { width: 100% !important; height: auto !important; aspect-ratio: 4/3; }
}
</style>
