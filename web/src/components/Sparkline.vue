<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'

const props = withDefaults(defineProps<{
  data: number[]
  width?: number
  height?: number
  color?: string
  lineWidth?: number
}>(), {
  width: 120,
  height: 32,
  color: '#3b82f6',
  lineWidth: 1.5,
})

const canvas = ref<HTMLCanvasElement | null>(null)

function draw() {
  const el = canvas.value
  if (!el) return
  const ctx = el.getContext('2d')
  if (!ctx) return

  const dpr = window.devicePixelRatio || 1
  el.width = props.width * dpr
  el.height = props.height * dpr
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, props.width, props.height)

  const values = props.data
  if (values.length < 2) return

  const max = Math.max(...values, 0.01)
  const min = Math.min(...values, 0)
  const range = max - min || 1
  const stepX = props.width / (values.length - 1)
  const pad = 2

  ctx.beginPath()
  ctx.strokeStyle = props.color
  ctx.lineWidth = props.lineWidth
  ctx.lineJoin = 'round'

  for (let i = 0; i < values.length; i++) {
    const x = i * stepX
    const y = props.height - pad - ((values[i] - min) / range) * (props.height - pad * 2)
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()

  // Fill area under the line with low opacity
  ctx.lineTo((values.length - 1) * stepX, props.height)
  ctx.lineTo(0, props.height)
  ctx.closePath()
  ctx.fillStyle = props.color + '1a'
  ctx.fill()
}

watch(() => props.data, draw, { deep: true })
onMounted(draw)
</script>

<template>
  <canvas
    ref="canvas"
    :style="{ width: width + 'px', height: height + 'px', display: 'block' }"
  />
</template>
