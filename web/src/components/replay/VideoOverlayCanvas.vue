<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch } from 'vue'

const props = defineProps<{
  videoEl?: HTMLVideoElement | null
  fps: number
  frameCount: number
  videoWidth: number
  videoHeight: number
  allBoxes?: any[] // array of frames containing boxes e.g., signals.value.yolo_boxes
  showBoxes: boolean
  currentIndex: number // Fallback if videoEl is not playing or pending seek
}>()

const emit = defineEmits(['update:index'])

const canvasRef = ref<HTMLCanvasElement | null>(null)
let rafId = 0
let lastDrawnIndex = -1 // Avoid redrawing the same frame repeatedly

function drawLoop() {
  if (!canvasRef.value) {
    rafId = requestAnimationFrame(drawLoop)
    return
  }
  
  const ctx = canvasRef.value.getContext('2d')
  if (!ctx) {
    rafId = requestAnimationFrame(drawLoop)
    return
  }

  if (!props.showBoxes || !props.allBoxes) {
    if (lastDrawnIndex !== -2) {
      ctx.clearRect(0, 0, props.videoWidth || 1920, props.videoHeight || 1080)
      lastDrawnIndex = -2
    }
    rafId = requestAnimationFrame(drawLoop)
    return
  }

  // Calculate current index based on video time to achieve 60FPS sync
  let targetIdx = props.currentIndex
  if (props.videoEl && !props.videoEl.paused && !props.videoEl.seeking) {
    targetIdx = Math.min(
      Math.floor(props.videoEl.currentTime * props.fps),
      props.frameCount > 0 ? props.frameCount - 1 : 0
    )
    if (targetIdx !== props.currentIndex) {
      emit('update:index', targetIdx)
    }
  }

  if (targetIdx !== lastDrawnIndex) {
    lastDrawnIndex = targetIdx
    
    ctx.clearRect(0, 0, props.videoWidth || 1920, props.videoHeight || 1080)

    const boxes = props.allBoxes[targetIdx] || []
    if (boxes.length > 0) {
      ctx.lineWidth = 2
      ctx.font = '600 12px monospace'
      ctx.textBaseline = 'bottom' // adjust for better positioning

      for (const box of boxes) {
        const [x1, y1, x2, y2] = box.bbox || [0, 0, 0, 0]
        const bw = x2 - x1
        const bh = y2 - y1

        // Draw bounding box
        ctx.strokeStyle = '#10b981' // emerald-500
        ctx.strokeRect(x1, y1, bw, bh)

        // Draw label text with drop shadow (replaces CSS text-shadow)
        const text = `${box.class} ${box.confidence?.toFixed?.(2) ?? ''}`
        
        ctx.fillStyle = '#fff'
        ctx.shadowColor = 'rgba(0, 0, 0, 0.9)'
        ctx.shadowBlur = 4
        // Draw text slightly above top-left corner
        ctx.fillText(text, x1 + 4, y1 - 4)
        
        // Reset shadow so the rect next iteration isn't blurred
        ctx.shadowBlur = 0
      }
    }
  }

  rafId = requestAnimationFrame(drawLoop)
}

// Force a redraw when props change (especially useful when paused and seeking)
watch(
  () => [props.currentIndex, props.showBoxes, props.allBoxes],
  () => {
    // If paused, the rAF loop sees videoEl.paused = true and uses props.currentIndex.
    // It will trigger a redraw organically in the next frame.
    lastDrawnIndex = -1 // clear cache to force redraw
  },
  { deep: true }
)

onMounted(() => {
  rafId = requestAnimationFrame(drawLoop)
})

onBeforeUnmount(() => {
  cancelAnimationFrame(rafId)
})
</script>

<template>
  <div class="video-overlay-wrapper">
    <canvas
      ref="canvasRef"
      class="video-overlay-canvas"
      :width="videoWidth || 1920"
      :height="videoHeight || 1080"
    ></canvas>
  </div>
</template>

<style scoped>
.video-overlay-wrapper {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none; /* Let clicks pass through to the video */
  overflow: hidden;
}

.video-overlay-canvas {
  width: 100%;
  height: 100%;
  display: block;
  /* Make canvas scale to fit without distorting aspect ratio, exactly like video's object-fit: contain */
  object-fit: contain; 
}
</style>
