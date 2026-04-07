<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'

const props = withDefaults(defineProps<{
  data: number[]
  currentIndex: number
  color?: string
  label: string
  height?: number
  threshold?: number
}>(), {
  color: '#3b82f6',
  height: 36,
})

const emit = defineEmits<{
  (e: 'seek', index: number): void
}>()

const canvas = ref<HTMLCanvasElement | null>(null)

function draw() {
  const el = canvas.value
  if (!el) return
  const ctx = el.getContext('2d')
  if (!ctx) return
  const w = el.clientWidth
  const h = props.height

  const dpr = window.devicePixelRatio || 1
  el.width = w * dpr
  el.height = h * dpr
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, w, h)

  const values = props.data
  if (values.length < 2) return

  const max = Math.max(...values, 0.01)
  const min = 0
  const range = max - min || 1
  const stepX = w / (values.length - 1)
  const pad = 2

  // Label
  ctx.fillStyle = '#6b7280'
  ctx.font = '10px sans-serif'
  ctx.fillText(props.label, 4, 11)

  // Threshold zone
  if (props.threshold !== undefined && props.threshold > 0) {
    const ty = h - pad - ((props.threshold - min) / range) * (h - pad * 2)
    ctx.fillStyle = 'rgba(239, 68, 68, 0.08)'
    ctx.fillRect(0, 0, w, ty)
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)'
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(0, ty)
    ctx.lineTo(w, ty)
    ctx.stroke()
    ctx.setLineDash([])
  }

  // Data line
  ctx.beginPath()
  ctx.strokeStyle = props.color
  ctx.lineWidth = 1.5
  ctx.lineJoin = 'round'

  for (let i = 0; i < values.length; i++) {
    const x = i * stepX
    const y = h - pad - ((values[i] - min) / range) * (h - pad * 2)
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()

  // Current position indicator
  if (props.currentIndex >= 0 && props.currentIndex < values.length) {
    const cx = props.currentIndex * stepX
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(cx, 0)
    ctx.lineTo(cx, h)
    ctx.stroke()

    // Dot at current value
    const cy = h - pad - ((values[props.currentIndex] - min) / range) * (h - pad * 2)
    ctx.fillStyle = '#ffffff'
    ctx.beginPath()
    ctx.arc(cx, cy, 3, 0, Math.PI * 2)
    ctx.fill()
  }
}

function handleClick(e: MouseEvent) {
  const el = canvas.value
  if (!el) return
  const rect = el.getBoundingClientRect()
  const x = e.clientX - rect.left
  const index = Math.round((x / rect.width) * (props.data.length - 1))
  if (index >= 0 && index < props.data.length) {
    emit('seek', index)
  }
}

watch(() => props.data, draw)
watch(() => props.currentIndex, draw)
onMounted(draw)
</script>

<template>
  <canvas
    ref="canvas"
    :style="{ width: '100%', height: height + 'px', display: 'block', cursor: 'pointer' }"
    @click="handleClick"
  />
</template>
