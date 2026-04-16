import { ref, type Ref, watch, onBeforeUnmount } from 'vue'
import type { useHeatmapCache } from './useHeatmapCache'

/* ── Types ── */

interface YoloBox {
  bbox: [number, number, number, number]
  class: string
  confidence?: number
}

interface SignalsData {
  yolo_boxes?: YoloBox[][]
  trajectory_points?: ({ x: number; y: number } | null)[]
  speed_px_per_sec?: (number | null)[]
  origin?: { x: number; y: number }
  landing?: { x: number; y: number }
  timestamps?: number[]
  [key: string]: unknown
}

interface ReplayMeta {
  camera_id?: string
  width?: number
  height?: number
  fps?: number
  frame_count?: number
  status?: string
  [key: string]: unknown
}

interface CompositorOptions {
  canvas: Ref<HTMLCanvasElement | null>
  videoEl: Ref<HTMLVideoElement | null>
  fps: Ref<number>
  frameCount: Ref<number>
  heatmapCache: ReturnType<typeof useHeatmapCache>
  signals: Ref<SignalsData | null>
  showHeatmap: Ref<boolean>
  showBoxes: Ref<boolean>
  showTrajectory: Ref<boolean>
  showHud: Ref<boolean>
  heatmapOpacity: Ref<number>
  metadata: Ref<ReplayMeta | null>
}

/* ── Compositor ── */

export function useCanvasCompositor(options: CompositorOptions) {
  const {
    canvas,
    videoEl,
    fps,
    frameCount,
    heatmapCache,
    signals,
    showHeatmap,
    showBoxes,
    showTrajectory,
    showHud,
    heatmapOpacity,
    metadata,
  } = options

  const currentIndex = ref(0)
  const playing = ref(false)

  let lastRenderedIndex = -1
  let rafId = 0
  let vfcId = 0 // requestVideoFrameCallback handle
  let running = false

  // Display dimensions (CSS px, set by ResizeObserver)
  let displayWidth = 0
  let displayHeight = 0

  /* ── DPR-aware canvas sizing ── */

  function updateCanvasSize(w: number, h: number): void {
    displayWidth = w
    displayHeight = h
    const cvs = canvas.value
    if (!cvs) return
    const dpr = window.devicePixelRatio || 1
    cvs.width = Math.round(w * dpr)
    cvs.height = Math.round(h * dpr)
    cvs.style.width = w + 'px'
    cvs.style.height = h + 'px'
    // Force re-render after resize
    lastRenderedIndex = -1
  }

  /* ── Drawing helpers ── */

  function drawVideo(
    ctx: CanvasRenderingContext2D,
    vid: HTMLVideoElement,
  ): void {
    ctx.drawImage(vid, 0, 0, displayWidth, displayHeight)
  }

  function drawHeatmap(
    ctx: CanvasRenderingContext2D,
    bitmap: ImageBitmap,
    opacity: number,
  ): void {
    ctx.save()
    ctx.globalAlpha = opacity
    ctx.globalCompositeOperation = 'screen'
    ctx.drawImage(bitmap, 0, 0, displayWidth, displayHeight)
    ctx.restore()
  }

  function drawYoloBoxes(
    ctx: CanvasRenderingContext2D,
    boxes: YoloBox[],
    videoWidth: number,
    videoHeight: number,
  ): void {
    if (boxes.length === 0) return

    const scaleX = displayWidth / videoWidth
    const scaleY = displayHeight / videoHeight

    ctx.lineWidth = 2
    ctx.font = '600 12px monospace'
    ctx.textBaseline = 'bottom'

    for (const box of boxes) {
      const [x1, y1, x2, y2] = box.bbox || [0, 0, 0, 0]
      const sx = x1 * scaleX
      const sy = y1 * scaleY
      const sw = (x2 - x1) * scaleX
      const sh = (y2 - y1) * scaleY

      // Bounding box
      ctx.strokeStyle = '#10b981' // emerald-500
      ctx.strokeRect(sx, sy, sw, sh)

      // Label with drop shadow
      const text = `${box.class} ${box.confidence?.toFixed?.(2) ?? ''}`
      ctx.fillStyle = '#fff'
      ctx.shadowColor = 'rgba(0, 0, 0, 0.9)'
      ctx.shadowBlur = 4
      ctx.fillText(text, sx + 4, sy - 4)
      ctx.shadowBlur = 0
    }
  }

  function drawTrajectory(
    ctx: CanvasRenderingContext2D,
    points: ({ x: number; y: number } | null)[],
    frameIndex: number,
    videoWidth: number,
    videoHeight: number,
  ): void {
    const scaleX = displayWidth / videoWidth
    const scaleY = displayHeight / videoHeight

    // Draw polyline up to current frame
    ctx.beginPath()
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.7)' // blue-500
    ctx.lineWidth = 2
    let started = false
    for (let i = 0; i <= frameIndex && i < points.length; i++) {
      const pt = points[i]
      if (!pt) continue
      const px = pt.x * scaleX
      const py = pt.y * scaleY
      if (!started) {
        ctx.moveTo(px, py)
        started = true
      } else {
        ctx.lineTo(px, py)
      }
    }
    ctx.stroke()

    // Draw centroid markers
    for (let i = 0; i <= frameIndex && i < points.length; i++) {
      const pt = points[i]
      if (!pt) continue
      const px = pt.x * scaleX
      const py = pt.y * scaleY
      ctx.beginPath()
      ctx.arc(px, py, 3, 0, Math.PI * 2)
      ctx.fillStyle =
        i === frameIndex ? '#f59e0b' : 'rgba(59, 130, 246, 0.5)'
      ctx.fill()
    }
  }

  function drawOriginLanding(
    ctx: CanvasRenderingContext2D,
    sig: SignalsData,
    videoWidth: number,
    videoHeight: number,
  ): void {
    const scaleX = displayWidth / videoWidth
    const scaleY = displayHeight / videoHeight

    // Origin: diamond shape
    if (sig.origin) {
      const ox = sig.origin.x * scaleX
      const oy = sig.origin.y * scaleY
      ctx.save()
      ctx.translate(ox, oy)
      ctx.rotate(Math.PI / 4)
      ctx.fillStyle = 'rgba(34, 197, 94, 0.7)' // green-500
      ctx.fillRect(-6, -6, 12, 12)
      ctx.restore()
    }

    // Landing: circle
    if (sig.landing) {
      const lx = sig.landing.x * scaleX
      const ly = sig.landing.y * scaleY
      ctx.beginPath()
      ctx.arc(lx, ly, 8, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(239, 68, 68, 0.7)' // red-500
      ctx.fill()
    }
  }

  function drawSpeedLabel(
    ctx: CanvasRenderingContext2D,
    sig: SignalsData,
    frameIndex: number,
    videoWidth: number,
    videoHeight: number,
  ): void {
    const speeds = sig.speed_px_per_sec
    if (!speeds || speeds[frameIndex] == null) return
    const points = sig.trajectory_points
    if (!points) return

    const pt = points[frameIndex]
    if (!pt) return

    const scaleX = displayWidth / videoWidth
    const scaleY = displayHeight / videoHeight
    const px = pt.x * scaleX
    const py = pt.y * scaleY
    const speed = speeds[frameIndex]!

    const text = `${speed.toFixed(1)} px/s`
    ctx.font = '600 11px monospace'
    const metrics = ctx.measureText(text)
    const pad = 4
    const bgW = metrics.width + pad * 2
    const bgH = 16

    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
    ctx.fillRect(px + 10, py - bgH / 2, bgW, bgH)
    ctx.fillStyle = '#fff'
    ctx.textBaseline = 'middle'
    ctx.fillText(text, px + 10 + pad, py)
  }

  function drawHud(
    ctx: CanvasRenderingContext2D,
    meta: ReplayMeta,
    frameIndex: number,
    timestamp: string,
  ): void {
    const hudFont = '10px "JetBrains Mono", ui-monospace, monospace'
    ctx.font = hudFont
    ctx.textBaseline = 'top'
    ctx.shadowColor = 'rgba(0, 0, 0, 0.7)'
    ctx.shadowBlur = 3

    // Top-left: camera ID or REC indicator
    const topLeft =
      meta.status === 'recording'
        ? '\u25CF REC'
        : meta.camera_id || ''
    ctx.fillStyle =
      meta.status === 'recording'
        ? '#f59e0b'
        : 'rgba(255, 255, 255, 0.5)'
    ctx.fillText(topLeft, 12, 10)

    // Top-right: resolution + FPS
    const topRight = `${meta.width || 1920}\u00D7${meta.height || 1080} // ${meta.fps || 15} FPS`
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'
    const trMetrics = ctx.measureText(topRight)
    ctx.fillText(topRight, displayWidth - 12 - trMetrics.width, 10)

    // Bottom-left: timestamp
    ctx.textBaseline = 'bottom'
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'
    ctx.fillText(timestamp, 12, displayHeight - 10)

    // Bottom-right: frame counter
    const frameText = `FRAME ${frameIndex + 1} / ${meta.frame_count || 0}`
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'
    const frMetrics = ctx.measureText(frameText)
    ctx.fillText(
      frameText,
      displayWidth - 12 - frMetrics.width,
      displayHeight - 10,
    )

    ctx.shadowBlur = 0
  }

  /* ── Core render function ── */

  function renderFrame(frameIndex: number): void {
    const cvs = canvas.value
    const vid = videoEl.value
    if (!cvs || !vid || displayWidth === 0 || displayHeight === 0) return

    const dpr = window.devicePixelRatio || 1
    const ctx = cvs.getContext('2d')
    if (!ctx) return

    ctx.save()
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, displayWidth, displayHeight)

    // Layer 1: Video
    drawVideo(ctx, vid)

    // Layer 2: Heatmap
    if (showHeatmap.value) {
      const bitmap = heatmapCache.getFrame(frameIndex)
      if (bitmap) {
        drawHeatmap(ctx, bitmap, heatmapOpacity.value)
      }
    }

    const meta = metadata.value
    const vw = meta?.width || 1920
    const vh = meta?.height || 1080
    const sig = signals.value

    // Layer 3: YOLO boxes
    if (showBoxes.value && sig?.yolo_boxes) {
      const boxes: YoloBox[] = sig.yolo_boxes[frameIndex] || []
      drawYoloBoxes(ctx, boxes, vw, vh)
    }

    // Layer 4: Trajectory
    if (showTrajectory.value && sig) {
      // Derive trajectory from YOLO boxes if no explicit trajectory data
      let points = sig.trajectory_points
      if (!points && sig.yolo_boxes) {
        points = sig.yolo_boxes.map((frameBoxes: YoloBox[]) => {
          if (!frameBoxes || frameBoxes.length === 0) return null
          const box = frameBoxes[0]
          const [x1, y1, x2, y2] = box.bbox
          return { x: (x1 + x2) / 2, y: (y1 + y2) / 2 }
        })
      }
      if (points) {
        drawTrajectory(ctx, points, frameIndex, vw, vh)
        drawOriginLanding(ctx, sig, vw, vh)
        drawSpeedLabel(ctx, sig, frameIndex, vw, vh)
      }
    }

    // Layer 5: HUD
    if (showHud.value && meta) {
      const ts = sig?.timestamps?.[frameIndex]
      const timestamp = ts
        ? new Date(ts * 1000).toLocaleTimeString('zh-CN')
        : ''
      drawHud(ctx, meta, frameIndex, timestamp)
    }

    ctx.restore()
    currentIndex.value = frameIndex
    lastRenderedIndex = frameIndex
  }

  /* ── Frame sync mechanisms ── */

  const hasRVFC =
    typeof HTMLVideoElement !== 'undefined' &&
    'requestVideoFrameCallback' in HTMLVideoElement.prototype

  function startRVFC(): void {
    const vid = videoEl.value
    if (!vid || !hasRVFC) return

    function onVideoFrame(
      _now: DOMHighResTimeStamp,
      meta: { mediaTime: number },
    ): void {
      if (!running) return
      const frameIndex = Math.min(
        Math.floor(meta.mediaTime * fps.value),
        Math.max(frameCount.value - 1, 0),
      )
      renderFrame(frameIndex)
      if (playing.value && videoEl.value) {
        vfcId = (videoEl.value as any).requestVideoFrameCallback(
          onVideoFrame,
        )
      }
    }

    vfcId = (vid as any).requestVideoFrameCallback(onVideoFrame)
  }

  function startRAF(): void {
    function onAnimationFrame(): void {
      if (!running) return
      const vid = videoEl.value
      if (vid && !vid.paused && !vid.ended) {
        const frameIndex = Math.min(
          Math.floor(vid.currentTime * fps.value),
          Math.max(frameCount.value - 1, 0),
        )
        if (frameIndex !== lastRenderedIndex) {
          renderFrame(frameIndex)
        }
      }
      rafId = requestAnimationFrame(onAnimationFrame)
    }
    rafId = requestAnimationFrame(onAnimationFrame)
  }

  /* ── Lifecycle ── */

  function start(): void {
    if (running) return
    running = true
    if (hasRVFC) {
      startRVFC()
    } else {
      startRAF()
    }
  }

  function stop(): void {
    running = false
    if (rafId) {
      cancelAnimationFrame(rafId)
      rafId = 0
    }
    if (vfcId && videoEl.value && hasRVFC) {
      ;(videoEl.value as any).cancelVideoFrameCallback(vfcId)
      vfcId = 0
    }
  }

  function renderOnce(): void {
    const vid = videoEl.value
    if (!vid) return
    const frameIndex = Math.min(
      Math.floor(vid.currentTime * fps.value),
      Math.max(frameCount.value - 1, 0),
    )
    renderFrame(frameIndex)
  }

  /* ── Watchers for overlay toggles (re-render while paused) ── */

  watch(
    [showHeatmap, showBoxes, showTrajectory, showHud, heatmapOpacity],
    () => {
      if (!playing.value) {
        lastRenderedIndex = -1
        renderOnce()
      }
    },
  )

  /* ── Re-start RVFC loop when play resumes ── */

  watch(playing, (isPlaying) => {
    if (isPlaying && running && hasRVFC) {
      startRVFC()
    }
  })

  onBeforeUnmount(() => {
    stop()
  })

  return {
    currentIndex,
    playing,
    displayWidth: () => displayWidth,
    displayHeight: () => displayHeight,
    updateCanvasSize,
    start,
    stop,
    renderOnce,
  }
}
