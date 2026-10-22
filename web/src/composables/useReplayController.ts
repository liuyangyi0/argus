import { ref, computed, watch, onUnmounted } from 'vue'
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
import type { ReplayMetadataPayload, ReplaySignalsPayload } from '../api/replay'
import type { TrajectoryFit } from '../types/api'
import { toApiError } from '../utils/error'

export type ReplayResourceState = 'idle' | 'loading' | 'ready' | 'error' | 'unavailable'

type ReferenceOption = 'yesterday' | 'last_week' | 'prev_week' | 'custom'

interface ReplayClipState {
  start_index: number
  end_index: number
  label: string
  created_at?: string
}

export interface ReplayYoloBox {
  bbox: [number, number, number, number]
  class: string
  confidence?: number
}

export interface ReplayCanvasPoint {
  x: number
  y: number
}

export type ReplayTrackPoints = Array<ReplayCanvasPoint | null>

export type ReplaySignalsState = Omit<
  ReplaySignalsPayload,
  'trajectory_points' | 'yolo_boxes'
> & {
  // The API stores trajectory samples grouped by track. The compositor needs
  // fixed-length, frame-indexed arrays, so expose the normalized shape here.
  trajectory_points?: ReplayTrackPoints
  trajectory_points_by_track?: Record<string, ReplayTrackPoints>
  yolo_boxes?: ReplayYoloBox[][]
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isNumberTuple4(value: unknown): value is [number, number, number, number] {
  return Array.isArray(value)
    && value.length === 4
    && value.every(isFiniteNumber)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function normalizeYoloBoxes(
  frames: ReplaySignalsPayload['yolo_boxes'],
): ReplayYoloBox[][] | undefined {
  if (!frames) return undefined
  return frames.map((frame) => frame.flatMap((rawBox) => {
    if (!isRecord(rawBox) || !isNumberTuple4(rawBox.bbox) || typeof rawBox.class !== 'string') {
      return []
    }
    const box: ReplayYoloBox = {
      bbox: rawBox.bbox,
      class: rawBox.class,
    }
    if (isFiniteNumber(rawBox.confidence)) box.confidence = rawBox.confidence
    return [box]
  }))
}

function isReferenceOption(value: unknown): value is ReferenceOption {
  return value === 'yesterday'
    || value === 'last_week'
    || value === 'prev_week'
    || value === 'custom'
}

export function useReplayController(alertId: string) {
  // core state
  const metadata = ref<ReplayMetadataPayload | null>(null)
  const signals = ref<ReplaySignalsState | null>(null)
  const trajectoryFits = ref<TrajectoryFit[]>([])
  const currentIndex = ref(0)
  const playing = ref(false)
  const speed = ref(1)
  // Start in loading so the first render cannot flash an empty-state before
  // ReplayPlayer's mounted hook starts the initial requests.
  const metadataState = ref<ReplayResourceState>('loading')
  const signalsState = ref<ReplayResourceState>('loading')
  const trajectoryState = ref<ReplayResourceState>('loading')
  const videoState = ref<ReplayResourceState>('idle')
  const metadataError = ref('')
  const metadataErrorStatus = ref<number | null>(null)
  const signalsError = ref('')
  const signalsErrorStatus = ref<number | null>(null)
  const trajectoryError = ref('')
  const trajectoryErrorStatus = ref<number | null>(null)

  const metadataLoading = computed(() => metadataState.value === 'loading')
  const signalsLoading = computed(() => signalsState.value === 'loading')
  const trajectoryLoading = computed(() => trajectoryState.value === 'loading')
  const videoLoading = computed(() => videoState.value === 'loading')
  // Compatibility aliases for existing callers. The page shell is gated only
  // by metadata; signals, trajectory and video expose their own states below.
  const loading = computed(() => (
    metadataLoading.value || signalsLoading.value || trajectoryLoading.value
  ))
  const loadError = computed(() => metadataError.value)
  const loadErrorStatus = computed(() => metadataErrorStatus.value)

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
  const selectedRefOption = ref<ReferenceOption>('yesterday')
  const referenceOffsetSeconds = ref(0)
  const lastRefDate = ref<string | undefined>(undefined)

  // clips
  const clipStart = ref<number | null>(null)
  const clipEnd = ref<number | null>(null)
  const persistedClips = ref<ReplayClipState[]>([])
  let recordingRefreshTimer: ReturnType<typeof setTimeout> | null = null
  let loadDataRevision = 0
  let metadataLoadRevision = 0
  let signalsLoadRevision = 0
  let trajectoryLoadRevision = 0

  function clearRecordingRefresh() {
    if (recordingRefreshTimer) {
      clearTimeout(recordingRefreshTimer)
      recordingRefreshTimer = null
    }
  }

  function scheduleRecordingRefresh() {
    clearRecordingRefresh()
    if (metadata.value?.status !== 'recording') return
    recordingRefreshTimer = setTimeout(() => {
      recordingRefreshTimer = null
      loadData()
    }, 2000)
  }

  // computed properties
  const fps = computed(() => metadata.value?.fps || 15)
  const videoUrl = computed(() => {
    const base = metadata.value?.video_url || getReplayVideoUrl(alertId)
    if (!metadata.value) return base
    // In-progress media is served no-store and keeps a stable source so each
    // 2-second metadata refresh does not reset playback. Once finalized, move
    // to an immutable metadata-derived URL and remount exactly once.
    if (metadata.value.status !== 'complete') return base
    const artifactRevision = [
      'complete',
      metadata.value.frame_count ?? 0,
      metadata.value.created_at || '',
    ].join(':')
    const separator = base.includes('?') ? '&' : '?'
    return `${base}${separator}v=${encodeURIComponent(artifactRevision)}`
  })
  const hasHeatmaps = computed(() => signals.value?.has_heatmaps || false)

  const currentTimestamp = computed(() => {
    const ts = signals.value?.timestamps?.[currentIndex.value]
    if (!ts) return ''
    return new Date(ts * 1000).toLocaleTimeString('zh-CN')
  })

  // data loading. Each resource commits independently so a missing signal or
  // trajectory document never discards usable metadata/video state.
  async function loadMetadata(): Promise<void> {
    const revision = ++metadataLoadRevision
    metadataState.value = 'loading'
    metadataError.value = ''
    metadataErrorStatus.value = null
    try {
      const metaRes = await getReplayMetadata(alertId)
      if (revision !== metadataLoadRevision) return
      metadata.value = metaRes
      metadataState.value = 'ready'
      const triggerFrameIndex = metadata.value?.trigger_frame_index
      if (triggerFrameIndex !== undefined) {
        pendingSeekIndex.value = triggerFrameIndex
        currentIndex.value = triggerFrameIndex
      }
      void loadReference()
    } catch (err) {
      if (revision !== metadataLoadRevision) return
      const apiError = toApiError(err, '回放元数据加载失败')
      metadataError.value = apiError.message
      metadataErrorStatus.value = apiError.httpStatus
      metadataState.value = 'error'
    }
  }

  async function loadSignals(): Promise<void> {
    const revision = ++signalsLoadRevision
    signalsState.value = 'loading'
    signalsError.value = ''
    signalsErrorStatus.value = null
    try {
      const sigRes = await getReplaySignals(alertId)
      if (revision !== signalsLoadRevision) return

      // Transform backend's {track_id: [{frame_index, x, y, t}]} into a
      // frame-indexed array per track for the canvas compositor.
      const {
        trajectory_points: tracksMap = {},
        yolo_boxes: rawYoloBoxes,
        ...signalFields
      } = sigRes
      const frameCount = Array.isArray(sigRes?.timestamps) ? sigRes.timestamps.length : 0
      const enrichedSignals: ReplaySignalsState = {
        ...signalFields,
        ...(rawYoloBoxes ? { yolo_boxes: normalizeYoloBoxes(rawYoloBoxes) } : {}),
      }
      if (frameCount > 0 && tracksMap && typeof tracksMap === 'object') {
        const byTrack: Record<string, ReplayTrackPoints> = {}
        for (const [tid, pts] of Object.entries(tracksMap)) {
          const arr: ReplayTrackPoints = new Array(frameCount).fill(null)
          for (const p of pts) {
            const idx = p?.frame_index
            if (
              Number.isInteger(idx)
              && idx >= 0
              && idx < frameCount
              && isFiniteNumber(p.x)
              && isFiniteNumber(p.y)
            ) {
              arr[idx] = { x: p.x, y: p.y }
            }
          }
          byTrack[tid] = arr
        }
        enrichedSignals.trajectory_points_by_track = byTrack
        enrichedSignals.trajectory_points = Object.values(byTrack)[0]
      }
      signals.value = enrichedSignals
      signalsState.value = 'ready'
      // Rehydrate operator-persisted clip ranges so they survive reload.
      persistedClips.value = Array.isArray(sigRes.clips) ? sigRes.clips : []
    } catch (err) {
      if (revision !== signalsLoadRevision) return
      const apiError = toApiError(err, '回放信号加载失败')
      signalsError.value = apiError.message
      signalsErrorStatus.value = apiError.httpStatus
      signalsState.value = 'error'
    }
  }

  async function loadTrajectory(): Promise<void> {
    const revision = ++trajectoryLoadRevision
    trajectoryState.value = 'loading'
    trajectoryError.value = ''
    trajectoryErrorStatus.value = null
    try {
      const response = await getAlertTrajectory(alertId)
      if (revision !== trajectoryLoadRevision) return
      trajectoryFits.value = response?.trajectories ?? []
      trajectoryState.value = response ? 'ready' : 'unavailable'
    } catch (err) {
      if (revision !== trajectoryLoadRevision) return
      const apiError = toApiError(err, '轨迹数据加载失败')
      if (apiError.httpStatus === 404 || /404|not found|不存在/i.test(apiError.message)) {
        trajectoryFits.value = []
        trajectoryState.value = 'unavailable'
        return
      }
      trajectoryError.value = apiError.message
      trajectoryErrorStatus.value = apiError.httpStatus
      trajectoryState.value = 'error'
    }
  }

  async function loadData(): Promise<void> {
    const revision = ++loadDataRevision
    clearRecordingRefresh()
    await Promise.allSettled([
      loadMetadata(),
      loadSignals(),
      loadTrajectory(),
    ])
    if (revision !== loadDataRevision) return
    scheduleRecordingRefresh()
  }

  async function retryMetadata(): Promise<void> {
    clearRecordingRefresh()
    await loadMetadata()
    scheduleRecordingRefresh()
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
        persistedClips.value = res.clips
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
        persistedClips.value = res.clips
      } else {
        persistedClips.value = persistedClips.value.filter((_, i) => i !== index)
      }
    } catch {
      message.error('删除失败')
    }
  }

  // reference option change
  function onRefOptionChange(value: unknown) {
    if (!isReferenceOption(value)) return
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
  function markVideoLoading() {
    videoState.value = 'loading'
    videoError.value = ''
  }

  function markVideoReady() {
    videoState.value = 'ready'
    videoError.value = ''
  }

  function markVideoError(error: string) {
    videoState.value = 'error'
    videoError.value = error
    playing.value = false
  }

  function reloadVideo() {
    const el = videoEl.value
    if (!el) {
      videoState.value = 'idle'
      return
    }
    markVideoLoading()
    el.load()
  }

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
      markVideoError(msg)
      message.warning(msg)
      return
    }
    el.play().catch((err: DOMException) => {
      if (err.name === 'AbortError') return
      const msg = videoErrorMessage(err)
      markVideoError(msg)
      message.error(msg)
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
      const currentSignals = signals.value
      if (currentSignals) {
        if (!currentSignals.key_frames) currentSignals.key_frames = []
        const idx = currentIndex.value
        if (!currentSignals.key_frames.some((kf) => kf.index === idx)) {
          currentSignals.key_frames.push({ index: idx, label, type: 'user' })
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

  onUnmounted(() => {
    clearRecordingRefresh()
    if (refOffsetTimer) clearTimeout(refOffsetTimer)
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
    metadata, signals, trajectoryFits, currentIndex, playing, speed,
    metadataState, signalsState, trajectoryState, videoState,
    metadataLoading, signalsLoading, trajectoryLoading, videoLoading,
    metadataError, metadataErrorStatus, signalsError, signalsErrorStatus,
    trajectoryError, trajectoryErrorStatus,
    loading, loadError, loadErrorStatus,
    videoEl, canvasEl, videoError, pendingSeekIndex,
    showHeatmap, showBoxes, showTrajectory, showHud, heatmapOpacity,
    referenceFrame, referenceDate, loadingRef, selectedRefOption,
    referenceOffsetSeconds,
    clipStart, clipEnd, persistedClips,
    fps, videoUrl, hasHeatmaps, currentTimestamp,
    loadData, loadMetadata, retryMetadata, loadSignals, loadTrajectory, loadReference,
    markVideoLoading, markVideoReady, markVideoError, reloadVideo,
    togglePlay, stepFrame, seekTo, goToStart, goToEnd,
    captureCurrentFrame, handlePinFrame, handleKeydown, onRefOptionChange,
    commitClip, removeClip,
  }
}
