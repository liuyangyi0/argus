import { ref, computed, watch } from 'vue'
import { message } from 'ant-design-vue'
import {
  getReplayMetadata,
  getReplaySignals,
  getReplayVideoUrl,
  getReplayReference,
  pinReplayFrame,
  addReplayClip,
  deleteReplayClip,
  getAlertTrajectory,
} from '../api'
import type { ReplayClip, TrajectoryFit } from '../types/api'

export function useReplayController(alertId: string) {
  // core state
  const metadata = ref<any>(null)
  const signals = ref<any>(null)
  const trajectoryFits = ref<TrajectoryFit[]>([])
  const currentIndex = ref(0)
  const playing = ref(false)
  const speed = ref(1)
  const loading = ref(true)

  // video/canvas dom refs
  const videoEl = ref<HTMLVideoElement | null>(null)
  const canvasEl = ref<HTMLCanvasElement | null>(null)
  const videoError = ref('')
  const pendingSeekIndex = ref<number | null>(null)

  // overlays
  const showHeatmap = ref(true)
  const showBoxes = ref(false)
  const showTrajectory = ref(false)
  const showHud = ref(true)
  const heatmapOpacity = ref(0.4)

  // reference frames
  const referenceFrame = ref<string | null>(null)
  const referenceDate = ref('')
  const loadingRef = ref(false)
  const selectedRefOption = ref('yesterday')
  const referenceOffsetSeconds = ref(0)
  const lastRefDate = ref<string | undefined>(undefined)

  // clips
  const clipStart = ref<number | null>(null)
  const clipEnd = ref<number | null>(null)
  const persistedClips = ref<ReplayClip[]>([])

  // computed properties
  const fps = computed(() => metadata.value?.fps || 15)
  const videoUrl = computed(() => getReplayVideoUrl(alertId))
  const hasHeatmaps = computed(() => signals.value?.has_heatmaps || false)

  const currentTimestamp = computed(() => {
    const ts = signals.value?.timestamps?.[currentIndex.value]
    if (!ts) return ''
    return new Date(ts * 1000).toLocaleTimeString('zh-CN')
  })

  // data loading
  async function loadData() {
    loading.value = true
    try {
      const [metaRes, sigRes, trajRes] = await Promise.all([
        getReplayMetadata(alertId),
        getReplaySignals(alertId),
        // Older alerts have no trajectory record — swallow 404 but log real errors
        getAlertTrajectory(alertId).catch((err) => {
          if (!/404|not found/i.test(String(err?.message ?? err))) {
            console.warn('[replay] trajectory fetch failed:', err)
          }
          return null
        }),
      ])
      metadata.value = metaRes
      signals.value = sigRes
      trajectoryFits.value = trajRes?.trajectories ?? []

      // Transform backend's {track_id: [{frame_index, x, y, t}]} into a
      // frame-indexed array per track for the canvas compositor.
      const tracksMap = (sigRes as any)?.trajectory_points ?? {}
      const frameCount = Array.isArray(sigRes?.timestamps) ? sigRes.timestamps.length : 0
      if (frameCount > 0 && tracksMap && typeof tracksMap === 'object') {
        const byTrack: Record<string, Array<{ x: number; y: number } | null>> = {}
        for (const [tid, pts] of Object.entries<any>(tracksMap)) {
          const arr: Array<{ x: number; y: number } | null> = new Array(frameCount).fill(null)
          for (const p of (pts as any[])) {
            const idx = p?.frame_index
            if (Number.isInteger(idx) && idx >= 0 && idx < frameCount) {
              arr[idx] = { x: p.x, y: p.y }
            }
          }
          byTrack[tid] = arr
        }
        ;(signals.value as any).trajectory_points_by_track = byTrack
      }

      if (metadata.value?.trigger_frame_index !== undefined) {
        pendingSeekIndex.value = metadata.value.trigger_frame_index
        currentIndex.value = metadata.value.trigger_frame_index
      }

      // Rehydrate operator-persisted clip ranges so they survive reload.
      const loadedClips = Array.isArray((sigRes as any)?.clips) ? (sigRes as any).clips : []
      persistedClips.value = loadedClips as ReplayClip[]

      loadReference()
    } catch (e) {
      message.error('回放数据加载失败')
      metadata.value = null
    } finally {
      loading.value = false
    }
  }

  async function loadReference(date?: string) {
    loadingRef.value = true
    lastRefDate.value = date ?? lastRefDate.value
    try {
      const params: Record<string, string | number> = {}
      const effectiveDate = date ?? lastRefDate.value
      if (effectiveDate) params.date = effectiveDate
      if (referenceOffsetSeconds.value) {
        params.frame_offset_seconds = referenceOffsetSeconds.value
      }
      const res = await getReplayReference(alertId, params)
      if (res.available && res.frame_base64) {
        referenceFrame.value = `data:image/jpeg;base64,${res.frame_base64}`
      } else {
        referenceFrame.value = null
      }
      referenceDate.value = res.source_date || ''
    } catch {
      referenceFrame.value = null
    } finally {
      loadingRef.value = false
    }
  }

  // Clip persistence (POST on [ + ], DELETE via marker menu)
  async function commitClip(startIdx: number, endIdx: number, label?: string) {
    try {
      const res = await addReplayClip(alertId, {
        start_index: startIdx,
        end_index: endIdx,
        label: label || '',
      })
      if (Array.isArray(res?.clips)) {
        persistedClips.value = res.clips as ReplayClip[]
      }
      message.success(`片段已保存 #${startIdx}–${endIdx}`)
    } catch {
      message.error('片段保存失败')
    }
  }

  async function removeClip(index: number) {
    if (index < 0 || index >= persistedClips.value.length) return
    try {
      const res = await deleteReplayClip(alertId, index)
      if (Array.isArray(res?.clips)) {
        persistedClips.value = res.clips as ReplayClip[]
      } else {
        persistedClips.value = persistedClips.value.filter((_, i) => i !== index)
      }
    } catch {
      message.error('删除失败')
    }
  }

  // reference option change
  function onRefOptionChange(value: any) {
    selectedRefOption.value = value
    if (value === 'custom') {
      const dateStr = prompt('输入日期 (YYYY-MM-DD):')
      if (dateStr && /^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
        loadReference(dateStr)
      }
      return
    }
    const triggerTs = metadata.value?.trigger_timestamp
    if (!triggerTs) return
    const trigger = new Date(triggerTs * 1000)
    let refDate = new Date(trigger)
    if (value === 'yesterday') {
      refDate.setDate(refDate.getDate() - 1)
    } else if (value === 'last_week') {
      refDate.setDate(refDate.getDate() - 7)
    } else if (value === 'prev_week') {
      refDate.setDate(refDate.getDate() - 14)
    } else {
      refDate.setDate(refDate.getDate() - 1)
    }
    const dateStr = refDate.toISOString().slice(0, 10)
    loadReference(dateStr)
  }

  // playback controls
  function safePlay() {
    if (!videoEl.value) return
    // Bail out early when the <video> has no usable source — otherwise the
    // browser quietly resolves / rejects the play() promise and the user sees
    // "no response" from the play button.
    const el = videoEl.value
    // networkState 3 = NETWORK_NO_SOURCE, readyState 0 = HAVE_NOTHING
    if (el.error || (el.readyState === 0 && el.networkState === 3)) {
      const msg = el.error
        ? videoErrorMessage(el.error)
        : '录像文件无法加载，请切换到"触发帧"查看'
      videoError.value = msg
      message.warning(msg)
      playing.value = false
      return
    }
    el.play().catch((err: DOMException) => {
      if (err.name === 'AbortError') return
      const msg = videoErrorMessage(err)
      videoError.value = msg
      message.error(msg)
      playing.value = false
    })
  }

  function togglePlay() {
    if (!videoEl.value) {
      // ref not yet populated — almost certainly a race during mount; surface
      // it rather than swallowing the click silently.
      message.warning('播放器还未就绪，请稍候')
      return
    }
    if (videoEl.value.paused) {
      safePlay()
    } else {
      videoEl.value.pause()
    }
  }

  function stepFrame(delta: number) {
    if (!videoEl.value) return
    videoEl.value.pause()
    videoEl.value.currentTime = Math.max(0, videoEl.value.currentTime + delta / fps.value)
  }

  function seekTo(index: number) {
    if (!videoEl.value) return
    videoEl.value.pause()
    videoEl.value.currentTime = index / fps.value
    currentIndex.value = index
  }

  function goToStart() { seekTo(0) }
  function goToEnd() { seekTo((metadata.value?.frame_count || 1) - 1) }

  function snapshotFilename(): string {
    const sanitize = (value: unknown, fallback: string) => {
      const text = String(value || fallback).trim() || fallback
      return text.replace(/[^\w.-]+/g, '_')
    }
    const cameraId = sanitize(metadata.value?.camera_id, 'camera')
    const replayId = sanitize(metadata.value?.alert_id || alertId, 'replay')
    const frameNo = String(currentIndex.value + 1).padStart(6, '0')
    const frameTs = signals.value?.timestamps?.[currentIndex.value]
    const date = typeof frameTs === 'number' ? new Date(frameTs * 1000) : new Date()
    const stamp = Number.isNaN(date.getTime())
      ? new Date().toISOString()
      : date.toISOString()
    return `snapshot_${cameraId}_${replayId}_f${frameNo}_${stamp.replace(/[:.]/g, '-')}.png`
  }

  async function captureCurrentFrame() {
    const canvas = canvasEl.value
    if (!canvas || canvas.width === 0 || canvas.height === 0) {
      message.warning('当前画面还未就绪，请稍候')
      return
    }

    try {
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((result) => {
          if (result) resolve(result)
          else reject(new Error('empty snapshot'))
        }, 'image/png')
      })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = snapshotFilename()
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(url)
      message.success('抓图已保存')
    } catch (err) {
      console.error('[replay] snapshot failed:', err)
      message.error('抓图失败，请确认视频允许当前页面截图')
    }
  }

  async function handlePinFrame() {
    const label = prompt('帧标签:')
    if (!label) return
    try {
      await pinReplayFrame(alertId, { index: currentIndex.value, label })
      if (signals.value) {
        if (!signals.value.key_frames) signals.value.key_frames = []
        const idx = currentIndex.value
        if (!signals.value.key_frames.some((kf: any) => kf.index === idx)) {
          signals.value.key_frames.push({ index: idx, label, type: 'user' })
        }
      }
      message.success('已标记帧')
    } catch {
      message.error('标记失败')
    }
  }

  function videoErrorMessage(err: MediaError | DOMException): string {
    if (err instanceof MediaError) {
      const msgs: Record<number, string> = { 1: '视频加载被中止', 2: '网络错误', 3: '视频解码失败', 4: '视频格式不支持' }
      return msgs[err.code] || `视频错误 (code=${err.code})`
    }
    if (err instanceof DOMException) {
      if (err.name === 'NotAllowedError') return '浏览器阻止自动播放，请点击视频区域后重试'
      if (err.name === 'NotSupportedError') return '视频格式不支持'
      return `播放失败: ${err.message || err.name}`
    }
    return `播放失败`
  }

  watch(speed, (s) => {
    if (videoEl.value) {
      videoEl.value.playbackRate = s
    }
  })

  // Debounce reference-frame re-fetching while the operator drags the slider.
  let refOffsetTimer: ReturnType<typeof setTimeout> | null = null
  watch(referenceOffsetSeconds, () => {
    if (refOffsetTimer) clearTimeout(refOffsetTimer)
    refOffsetTimer = setTimeout(() => {
      loadReference()
    }, 150)
  })

  // Shortcuts logic (exposed so component can addEventListener on mount)
  function handleKeydown(e: KeyboardEvent) {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
    switch (e.key) {
      case ' ': e.preventDefault(); togglePlay(); break
      case 'ArrowLeft': stepFrame(-1); break
      case 'ArrowRight': stepFrame(1); break
      case 'k': case 'K': if (videoEl.value) videoEl.value.pause(); break
      case 'j': case 'J': speed.value = Math.max(0.25, speed.value / 2); if (videoEl.value?.paused) safePlay(); break
      case 'l': case 'L': speed.value = Math.min(4, speed.value * 2); if (videoEl.value?.paused) safePlay(); break
      case 'Home': goToStart(); break
      case 'End': goToEnd(); break
      case '[': {
        clipStart.value = currentIndex.value
        clipEnd.value = null
        break
      }
      case ']': {
        clipEnd.value = currentIndex.value
        // If a [ has been set, auto-persist the range so it survives reload.
        if (clipStart.value !== null && clipEnd.value !== null) {
          const a = Math.min(clipStart.value, clipEnd.value)
          const b = Math.max(clipStart.value, clipEnd.value)
          commitClip(a, b)
        }
        break
      }
    }
  }

  return {
    metadata, signals, trajectoryFits, currentIndex, playing, speed, loading,
    videoEl, canvasEl, videoError, pendingSeekIndex,
    showHeatmap, showBoxes, showTrajectory, showHud, heatmapOpacity,
    referenceFrame, referenceDate, loadingRef, selectedRefOption,
    referenceOffsetSeconds,
    clipStart, clipEnd, persistedClips,
    fps, videoUrl, hasHeatmaps, currentTimestamp,
    loadData, loadReference, togglePlay, stepFrame, seekTo, goToStart, goToEnd,
    captureCurrentFrame, handlePinFrame, handleKeydown, onRefOptionChange,
    commitClip, removeClip,
  }
}
