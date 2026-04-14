<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'

const props = defineProps<{
  /** Active tracks from the physics API */
  tracks: Array<{
    track_id: number
    centroid_x: number
    centroid_y: number
    trajectory_length: number
    max_score: number
  }>
  /** Full trajectory history points (pixel coords) for selected track */
  trajectoryHistory?: Array<{ x: number; y: number }>
  /** Canvas dimensions matching the video feed */
  width: number
  height: number
  /** Speed info to display */
  speedMs?: number | null
  /** Origin point (pixel coords) */
  originPx?: { x: number; y: number } | null
  /** Landing point (pixel coords) */
  landingPx?: { x: number; y: number } | null
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)

function draw() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, props.width, props.height)

  // Draw trajectory history as a path
  if (props.trajectoryHistory && props.trajectoryHistory.length > 1) {
    ctx.strokeStyle = '#ff6600'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.beginPath()
    ctx.moveTo(props.trajectoryHistory[0].x, props.trajectoryHistory[0].y)
    for (let i = 1; i < props.trajectoryHistory.length; i++) {
      ctx.lineTo(props.trajectoryHistory[i].x, props.trajectoryHistory[i].y)
    }
    ctx.stroke()
  }

  // Draw active track centroids
  for (const track of props.tracks) {
    // Circle at centroid
    ctx.beginPath()
    ctx.arc(track.centroid_x, track.centroid_y, 8, 0, Math.PI * 2)
    ctx.fillStyle = track.max_score > 0.8 ? 'rgba(255,0,0,0.7)' : 'rgba(255,165,0,0.7)'
    ctx.fill()
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 1.5
    ctx.stroke()

    // Track ID label
    ctx.fillStyle = '#fff'
    ctx.font = '11px monospace'
    ctx.fillText(`#${track.track_id}`, track.centroid_x + 12, track.centroid_y - 4)
  }

  // Draw origin point marker (green diamond)
  if (props.originPx) {
    const { x, y } = props.originPx
    ctx.fillStyle = 'rgba(0,255,0,0.8)'
    ctx.beginPath()
    ctx.moveTo(x, y - 10)
    ctx.lineTo(x + 8, y)
    ctx.lineTo(x, y + 10)
    ctx.lineTo(x - 8, y)
    ctx.closePath()
    ctx.fill()
    ctx.fillStyle = '#0f0'
    ctx.font = '11px monospace'
    ctx.fillText('起点', x + 12, y + 4)
  }

  // Draw landing point marker (blue circle)
  if (props.landingPx) {
    const { x, y } = props.landingPx
    ctx.beginPath()
    ctx.arc(x, y, 10, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(0,100,255,0.7)'
    ctx.fill()
    ctx.strokeStyle = '#0af'
    ctx.lineWidth = 2
    ctx.stroke()
    ctx.fillStyle = '#0af'
    ctx.font = '11px monospace'
    ctx.fillText('落点', x + 14, y + 4)
  }

  // Draw speed label
  if (props.speedMs != null && props.tracks.length > 0) {
    const t = props.tracks[0]
    ctx.fillStyle = 'rgba(0,0,0,0.6)'
    ctx.fillRect(t.centroid_x + 12, t.centroid_y + 6, 80, 18)
    ctx.fillStyle = '#fff'
    ctx.font = '12px monospace'
    ctx.fillText(`${props.speedMs.toFixed(1)} m/s`, t.centroid_x + 16, t.centroid_y + 20)
  }
}

watch(() => [props.tracks, props.trajectoryHistory, props.originPx, props.landingPx], draw, { deep: true })
onMounted(draw)
</script>

<template>
  <canvas
    ref="canvasRef"
    :width="width"
    :height="height"
    class="trajectory-overlay"
  />
</template>

<style scoped>
.trajectory-overlay {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
  z-index: 10;
}
</style>
