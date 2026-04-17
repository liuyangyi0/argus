import { ref, type Ref, watch, onBeforeUnmount } from 'vue'
import type { useHeatmapCache } from './useHeatmapCache'
import type { TrajectoryFit } from '../types/api'

/* ── Types ── */

interface YoloBox {
  bbox: [number, number, number, number]
  class: string
  confidence?: number
}

type TrackPoints = ({ x: number; y: number } | null)[]

interface SignalsData {
  yolo_boxes?: YoloBox[][]
  trajectory_points?: TrackPoints
  trajectory_points_by_track?: Record<string, TrackPoints>
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
  // Optional: per-track origin/landing + speed fits fetched from
  // /physics/{alert_id}/trajectory. When absent the compositor still works
  // using only the per-frame trajectory_points_by_track signal.
  trajectoryFits?: Ref<TrajectoryFit[]>
}

/* ── Multi-track colour helpers ── */

// Deterministic HSL hue per track_id using the golden ratio for even spread.
function trackHue(trackId: number | string): number {
  const n = typeof trackId === 'number' ? trackId : parseInt(String(trackId), 10) || 0
  // 137.508° is the golden angle — successive tracks land far apart on the wheel.
  return (n * 137.508) % 360
}

function trackColor(trackId: number | string, alpha = 1, isPrimary = false): string {
  const h = trackHue(trackId)
  const s = isPrimary ? 90 : 70
  const l = isPrimary ? 55 : 60
  return `hsla(${h.toFixed(1)}, ${s}%, ${l}%, ${alpha})`
}

/* Clamp (px,py) to the visible canvas rect; returns the clamped coordinate
 * plus whether the original point was outside the rect and its direction. */
function clampToEdge(
  px: number,
  py: number,
  w: number,
  h: number,
  pad = 8,
): { x: number; y: number; outside: boolean; angle: number } {
  const outside = px < pad || px > w - pad || py < pad || py > h - pad
  if (!outside) {
    return { x: px, y: py, outside: false, angle: 0 }
  }
  const cx = w / 2
  const cy = h / 2
  const dx = px - cx
  const dy = py - cy
  const angle = Math.atan2(dy, dx)
  // Clamp to the padded rect
  const x = Math.max(pad, Math.min(w - pad, px))
  const y = Math.max(pad, Math.min(h - pad, py))
  return { x, y, outside: true, angle }
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

  function drawTrackPolyline(
    ctx: CanvasRenderingContext2D,
    trackId: number | string,
    points: TrackPoints,
    frameIndex: number,
    videoWidth: number,
    videoHeight: number,
    isPrimary: boolean,
  ): void {
    const scaleX = displayWidth / videoWidth
    const scaleY = displayHeight / videoHeight
    const lineColor = trackColor(trackId, isPrimary ? 0.9 : 0.55, isPrimary)

    ctx.beginPath()
    ctx.strokeStyle = lineColor
    ctx.lineWidth = isPrimary ? 3 : 2
    let started = false
    let lastPt: { x: number; y: number } | null = null
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
      lastPt = { x: px, y: py }
    }
    ctx.stroke()

    // Centroid dots — smaller for secondary tracks
    const dotRadius = isPrimary ? 3 : 2
    for (let i = 0; i <= frameIndex && i < points.length; i++) {
      const pt = points[i]
      if (!pt) continue
      const px = pt.x * scaleX
      const py = pt.y * scaleY
      ctx.beginPath()
      ctx.arc(px, py, dotRadius, 0, Math.PI * 2)
      ctx.fillStyle = i === frameIndex
        ? '#f59e0b'
        : trackColor(trackId, isPrimary ? 0.6 : 0.35, isPrimary)
      ctx.fill()
    }

    // Track id label on the most recent point
    if (lastPt && isPrimary) {
      ctx.save()
      ctx.font = '600 10px monospace'
      ctx.fillStyle = lineColor
      ctx.shadowColor = 'rgba(0, 0, 0, 0.8)'
      ctx.shadowBlur = 3
      ctx.fillText(`#${trackId}`, lastPt.x + 6, lastPt.y - 6)
      ctx.restore()
    }
  }

  function drawOffscreenMarker(
    ctx: CanvasRenderingContext2D,
    clamped: { x: number; y: number; angle: number },
    label: string,
    color: string,
    originalPx: number,
    originalPy: number,
  ): void {
    // Draw a triangular arrow pointing toward the offscreen point
    const { x, y, angle } = clamped
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(angle)
    ctx.beginPath()
    ctx.moveTo(10, 0)
    ctx.lineTo(-6, 6)
    ctx.lineTo(-6, -6)
    ctx.closePath()
    ctx.fillStyle = color
    ctx.fill()
    ctx.restore()

    // Distance label next to the arrow
    const dist = Math.round(Math.hypot(originalPx - x, originalPy - y))
    ctx.save()
    ctx.font = '600 10px monospace'
    ctx.fillStyle = '#fff'
    ctx.shadowColor = 'rgba(0, 0, 0, 0.85)'
    ctx.shadowBlur = 3
    // Place text just inside the canvas, on the opposite side of the arrow
    const tx = x - Math.cos(angle) * 16
    const ty = y - Math.sin(angle) * 16
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(`${label} ${dist}px`, tx, ty)
    ctx.restore()
  }

  function drawFitEndpoints(
    ctx: CanvasRenderingContext2D,
    fit: TrajectoryFit,
    videoWidth: number,
    videoHeight: number,
  ): void {
    const scaleX = displayWidth / videoWidth
    const scaleY = displayHeight / videoHeight
    const color = trackColor(fit.track_id, fit.is_primary ? 0.95 : 0.6, fit.is_primary)

    // Origin (release point) — diamond when on-screen, arrow when off
    if (fit.origin.x_px != null && fit.origin.y_px != null) {
      const ox = fit.origin.x_px * scaleX
      const oy = fit.origin.y_px * scaleY
      const c = clampToEdge(ox, oy, displayWidth, displayHeight)
      if (c.outside) {
        drawOffscreenMarker(ctx, c, `↑#${fit.track_id}`, color, ox, oy)
      } else {
        ctx.save()
        ctx.translate(ox, oy)
        ctx.rotate(Math.PI / 4)
        ctx.fillStyle = color
        ctx.strokeStyle = 'rgba(0,0,0,0.6)'
        ctx.lineWidth = 1
        const s = fit.is_primary ? 7 : 5
        ctx.fillRect(-s, -s, s * 2, s * 2)
        ctx.strokeRect(-s, -s, s * 2, s * 2)
        ctx.restore()
      }
    }

    // Landing (impact point) — circle when on-screen, arrow when off
    if (fit.landing.x_px != null && fit.landing.y_px != null) {
      const lx = fit.landing.x_px * scaleX
      const ly = fit.landing.y_px * scaleY
      const c = clampToEdge(lx, ly, displayWidth, displayHeight)
      if (c.outside) {
        drawOffscreenMarker(ctx, c, `↓#${fit.track_id}`, color, lx, ly)
      } else {
        ctx.beginPath()
        ctx.arc(lx, ly, fit.is_primary ? 9 : 6, 0, Math.PI * 2)
        ctx.fillStyle = color
        ctx.fill()
        ctx.strokeStyle = 'rgba(0,0,0,0.6)'
        ctx.lineWidth = 1
        ctx.stroke()
      }
    }
  }

  function drawSpeedLabelAt(
    ctx: CanvasRenderingContext2D,
    pt: { x: number; y: number },
    speedPxPerSec: number,
    videoWidth: number,
    videoHeight: number,
  ): void {
    const scaleX = displayWidth / videoWidth
    const scaleY = displayHeight / videoHeight
    const px = pt.x * scaleX
    const py = pt.y * scaleY

    const text = `${speedPxPerSec.toFixed(1)} px/s`
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

    // Layer 4: Trajectory (multi-track)
    if (showTrajectory.value && sig) {
      const fits = options.trajectoryFits?.value ?? []
      const primaryId = fits.find((f) => f.is_primary)?.track_id
      const byTrack = sig.trajectory_points_by_track

      if (byTrack && Object.keys(byTrack).length > 0) {
        // Draw each track's polyline. Primary last so it sits on top.
        const entries = Object.entries(byTrack)
        const ordered = entries.sort(([a], [b]) => {
          if (primaryId != null && Number(a) === primaryId) return 1
          if (primaryId != null && Number(b) === primaryId) return -1
          return 0
        })
        for (const [tid, points] of ordered) {
          const isPrimary = primaryId != null && Number(tid) === primaryId
          drawTrackPolyline(ctx, tid, points, frameIndex, vw, vh, isPrimary)

          // Per-track speed label at the current point
          const fit = fits.find((f) => String(f.track_id) === tid)
          const spd = fit?.speed_px_per_sec
          const currentPt = points[frameIndex]
          if (isPrimary && currentPt && spd != null) {
            drawSpeedLabelAt(ctx, currentPt, spd, vw, vh)
          }
        }
      } else if (sig.trajectory_points) {
        // Legacy single-track path kept for backward compatibility
        drawTrackPolyline(ctx, 'legacy', sig.trajectory_points, frameIndex, vw, vh, true)
      } else if (sig.yolo_boxes) {
        // Fallback: derive a synthetic trajectory from YOLO first box per frame
        const derived: TrackPoints = sig.yolo_boxes.map((frameBoxes: YoloBox[]) => {
          if (!frameBoxes || frameBoxes.length === 0) return null
          const [x1, y1, x2, y2] = frameBoxes[0].bbox
          return { x: (x1 + x2) / 2, y: (y1 + y2) / 2 }
        })
        drawTrackPolyline(ctx, 'yolo', derived, frameIndex, vw, vh, true)
      }

      // Origin / landing endpoints per track (with offscreen arrow fallback)
      for (const fit of fits) {
        drawFitEndpoints(ctx, fit, vw, vh)
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

  // Re-render when trajectory fits arrive asynchronously (API returns after signals)
  if (options.trajectoryFits) {
    watch(options.trajectoryFits, () => {
      if (!playing.value) {
        lastRenderedIndex = -1
        renderOnce()
      }
    })
  }

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
